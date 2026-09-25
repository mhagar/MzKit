"""
Auto-annotate DIA ensembles with molecular formulae via find-mfs
`annotate_analyte_dia`.

Hands each ensemble's composite (representative) MS1 + MS2 spectrum to find-mfs,
which does its own base-envelope/precursor selection, charge + adduct resolution
and ranked formula search. Produces one FormulaAssignment per ensemble and, when
enabled, attaches the resolved per-envelope adduct labels to the ensemble's MS1
signals as 'auto_adduct'-tagged generic annotations.

DIA / MS1-only only (composite_spectrum is DIA-only); DDA ensembles are skipped.

ProcessController-compatible: accepts `progress_callback` + `cancel_event` and
accepts an iterable of ensembles, so the same function backs the single-ensemble
GUI button and the whole-sample headless batch.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field, fields
from typing import Callable, Iterable, Optional, TYPE_CHECKING

from core.formula.assign_formula import (
    _get_scorer,
    results_to_assigned_candidates,
    search_provenance,
)
from core.formula.params import FindMfsParams
from core.data_structs.formula_assignment import FormulaAssignment

if TYPE_CHECKING:
    from configparser import ConfigParser
    from core.data_structs import Ensemble

logger = logging.getLogger(__name__)

# The generic-annotation source tag for adduct labels we attach,
# so re-runs can clear+replace only these (leaving user annotations alone).
ADDUCT_ANNOT_SOURCE = 'auto_adduct'

SELECTION_SECTION = 'auto_find_mfs'


@dataclass
class EnsembleSelection:
    """
    Handles deciding which ensembles a batch find-mfs run annotates.

    Both limits rank by `Ensemble.base_intsy`,
    so a whole-sample run can skip the long tail of low-intensity ensembles.
     The values are kept while a limit is off, so the GUI can restore them.
    """
    limit_count: bool = False
    max_ensembles: int = 200
    limit_intensity: bool = False
    min_base_intsy: float = 100_000.0

    @classmethod
    def from_config(cls, config: Optional['ConfigParser']) -> 'EnsembleSelection':
        """
        Read `[auto_find_mfs]`; missing or unparsable keys keep defaults.
        """
        sel = cls()
        if config is None or not config.has_section(SELECTION_SECTION):
            return sel
        for f in fields(cls):
            if not config.has_option(SELECTION_SECTION, f.name):
                continue
            getter = {
                bool: config.getboolean,
                int: config.getint,
                float: config.getfloat,
            }[type(getattr(sel, f.name))]
            try:
                setattr(sel, f.name, getter(SELECTION_SECTION, f.name))
            except ValueError:
                pass
        return sel

    def to_config(self, config: 'ConfigParser') -> None:
        """
        Write into `[auto_find_mfs]` (does not save to disk)
        """
        if not config.has_section(SELECTION_SECTION):
            config.add_section(SELECTION_SECTION)
        for f in fields(self):
            config.set(SELECTION_SECTION, f.name, str(getattr(self, f.name)))

    def select(
        self,
        ensembles: list['Ensemble'],
    ) -> tuple[list['Ensemble'], int, int]:
        """
        Apply the limits, most intense first.

        :return: (selected ensembles in descending base_intsy order,
                  number dropped by the intensity floor,
                  number dropped by the count limit)
        """
        ranked = sorted(ensembles, key=lambda e: e.base_intsy, reverse=True)

        n_below = 0
        if self.limit_intensity:
            kept = [e for e in ranked if e.base_intsy >= self.min_base_intsy]
            n_below = len(ranked) - len(kept)
            ranked = kept

        n_beyond = 0
        if self.limit_count and len(ranked) > self.max_ensembles:
            n_beyond = len(ranked) - self.max_ensembles
            ranked = ranked[:self.max_ensembles]

        return ranked, n_below, n_beyond


@dataclass
class BatchAnnotationResult:
    """
    What a find-mfs annotation run did, including what it skipped and why
    """
    assignments: list[FormulaAssignment] = field(default_factory=list)
    n_ensembles: int = 0            # handed in
    n_dda: int = 0                  # skipped: TODO DDA has no composite spectrum yet
    n_below_intensity: int = 0      # skipped: below the base-peak intensity floor
    n_beyond_count: int = 0         # skipped: outside the N most intense
    n_no_envelope: int = 0          # skipped: find-mfs found no resolvable envelope
    n_cancelled: int = 0            # not reached before cancellation

    @property
    def n_no_candidates(self) -> int:
        """
        Annotated, but the search found no formula.
        """
        return sum(1 for a in self.assignments if not a.candidates)

    def summary(self) -> str:
        """
        One liner for status bar / log, e.g.
        'Annotated 212 of 340 ensembles; skipped 100 below min. intensity,
         28 with no resolvable envelope'.
         """
        noun = 'ensemble' if self.n_ensembles == 1 else 'ensembles'
        text = f"Annotated {len(self.assignments)} of {self.n_ensembles} {noun}"
        if self.n_no_candidates:
            text += f" ({self.n_no_candidates} with no candidate formula)"

        skipped = [
            f"{n} {why}" for n, why in (
                (self.n_below_intensity, "below min. intensity"),
                (self.n_beyond_count, "beyond the N most intense"),
                (self.n_no_envelope, "with no resolvable envelope"),
                (self.n_dda, "DDA (not supported yet)"),
                (self.n_cancelled, "not reached (cancelled)"),
            ) if n
        ]
        if skipped:
            text += "; skipped " + ", ".join(skipped)
        return text


def annotate_ensembles_dia(
    ensembles: Iterable['Ensemble'],
    *,
    params: Optional[FindMfsParams] = None,
    selection: Optional[EnsembleSelection] = None,
    model_path: Optional[str] = None,
    attach_adduct_labels: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> BatchAnnotationResult:
    """
    Run find-mfs `annotate_analyte_dia` over each (DIA) ensemble.

    For each ensemble: builds a FormulaAssignment (top candidate pre-selected,
    `chosen_idx=0`), sets `ensemble.proposed_formula`, and — if
    `attach_adduct_labels` — replaces prior 'auto_adduct' generic annotations
    with the resolved per-envelope adduct labels. DDA ensembles are skipped.

    Ensembles are annotated most intense first (so a cancelled run has done
    the ones that matter), limited by `selection` if given.

    :param params: find-mfs constraints/scoring; defaults to FindMfsParams().
    :param selection: which ensembles to annotate; None annotates all of them.
    :return: the assignments, plus counts of what was skipped and why.
    :raises ValueError: up front, if the count constraints are invalid (rather
        than once per ensemble, where it would be indistinguishable from an
        ensemble with no resolvable envelope).
    """
    # Lazy import: find-mfs pulls in MistNet + scorers.
    from find_mfs import annotate_analyte_dia

    params = params if params is not None else FindMfsParams()
    params.validate()
    selection = selection if selection is not None else EnsembleSelection()

    ensembles = list(ensembles)
    result = BatchAnnotationResult(n_ensembles=len(ensembles))

    # Composite is DIA-only for now; nothing to do for DDA.
    dia = [e for e in ensembles if not e.is_dda]
    result.n_dda = len(ensembles) - len(dia)

    todo, result.n_below_intensity, result.n_beyond_count = selection.select(dia)

    scorer = _get_scorer(model_path)  # MistNet warmed once, cached across calls

    n = len(todo)
    for i, ensemble in enumerate(todo):
        if cancel_event is not None and cancel_event.is_set():
            logger.info("annotate_ensembles_dia cancelled at %d/%d", i, n)
            result.n_cancelled = n - i
            break

        if progress_callback is not None:
            progress_callback(100.0 * i / n, f"Annotating {i + 1}/{n}")

        try:
            assignment = _annotate_one(
                ensemble,
                annotate_analyte_dia=annotate_analyte_dia,
                scorer=scorer,
                params=params,
                attach_adduct_labels=attach_adduct_labels,
            )
        except ValueError as exc:
            # find-mfs raises on an empty scan / no resolvable envelope. Skip
            # this ensemble rather than aborting the whole batch.
            logger.warning(
                "Skipping ensemble %s: %s", ensemble.uuid, exc
            )
            result.n_no_envelope += 1
            continue

        result.assignments.append(assignment)

    logger.info(result.summary())
    if progress_callback is not None:
        progress_callback(100.0, result.summary())

    return result


def _annotate_one(
    ensemble: 'Ensemble',
    *,
    annotate_analyte_dia,
    scorer,
    params: FindMfsParams,
    attach_adduct_labels: bool,
    ms1_peaks=None,
    ms2_peaks=None,
    precursor_mz: Optional[float] = None,
    ms2_mode: str = 'tallest',
) -> FormulaAssignment:
    # DIA default: pull the composite (representative) spectra and let find-mfs
    # pick the base envelope (precursor_mz=None). When the caller passes explicit
    # spectra (the temporary DDA path), use those + the given precursor instead.
    if ms1_peaks is None:
        composite = ensemble.composite_spectrum
        ms1_peaks = composite.ms1
        ms2_peaks = composite.ms2         # may be None -> MS2 term skipped
        precursor_mz = None

    res = annotate_analyte_dia(
        ms1_peaks=ms1_peaks,
        ms2_peaks=ms2_peaks,              # may be None -> MS2 term skipped
        precursor_mz=precursor_mz,
        scorer=scorer,
        **params.search_kwargs(),
    )

    candidates = results_to_assigned_candidates(
        res.candidates, ms2_weight=params.ms2_weight, top_n=params.top_n
    )

    assignment = FormulaAssignment(
        source_uuid=ensemble.uuid,
        candidates=candidates,
        chosen_idx=0 if candidates else None,   # auto-pick the top hit
        precursor_mz=res.precursor_mz,          # precursor find-mfs actually used
        charge=res.charge,                       # resolved charge
        ms2_mode=ms2_mode,
        params=asdict(params),
        **search_provenance(res.candidates),
    )

    # Mirror the manual path: keep the ensemble's free-text formula in sync so
    # non-assignment-aware displays (sample tree, labels) show it too.
    if candidates:
        ensemble.proposed_formula = candidates[0].formula_str

    if attach_adduct_labels:
        _attach_adduct_labels(ensemble, res.grouped)

    return assignment


def annotate_ensemble_with_selected_ms2(
    ensemble: 'Ensemble',
    ms1_peaks,
    ms2_peaks,
    precursor_mz: Optional[float] = None,
    *,
    params: Optional[FindMfsParams] = None,
    model_path: Optional[str] = None,
    attach_adduct_labels: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> BatchAnnotationResult:
    """
    TEMPORARY DDA compound-annotation path.

    Runs find-mfs against caller-supplied MS1 + MS2 spectra (whatever the
    EnsembleViewer currently has on screen) plus the ensemble's precursor,
    rather than the DIA composite (which is undefined for DDA). Produces a
    single FormulaAssignment, returned as a BatchAnnotationResult to match the
    DIA completion handler.

    This is a stopgap until DDA MS2 'consensus' stitching lands and
    `composite_spectrum` works for DDA; it will be retired then. Note that
    unlike the DIA path we hand find-mfs a real precursor_mz.

    ProcessController-compatible (accepts progress_callback + cancel_event).
    """
    # Lazy import: find-mfs pulls in MistNet + scorers.
    from find_mfs import annotate_analyte_dia

    if progress_callback is not None:
        progress_callback(0.0, "Annotating (DDA, selected MS2)…")

    result = BatchAnnotationResult(n_ensembles=1)
    if cancel_event is not None and cancel_event.is_set():
        result.n_cancelled = 1
        return result

    params = params if params is not None else FindMfsParams()
    params.validate()
    scorer = _get_scorer(model_path)

    try:
        assignment = _annotate_one(
            ensemble,
            annotate_analyte_dia=annotate_analyte_dia,
            scorer=scorer,
            params=params,
            attach_adduct_labels=attach_adduct_labels,
            ms1_peaks=ms1_peaks,
            ms2_peaks=ms2_peaks,
            precursor_mz=precursor_mz,
            ms2_mode='selected_scan',
        )
    except ValueError as exc:
        logger.warning("DDA annotate skipped for %s: %s", ensemble.uuid, exc)
        if progress_callback is not None:
            progress_callback(100.0, "No resolvable envelope")
        result.n_no_envelope = 1
        return result

    if progress_callback is not None:
        progress_callback(100.0, "Annotated 1 ensemble")

    result.assignments.append(assignment)
    return result


def _attach_adduct_labels(
    ensemble: 'Ensemble',
    grouped,
) -> None:
    """
    Replace this ensemble's prior 'auto_adduct' labels with the resolved
    per-envelope adduct labels from find-mfs.

    Only groups in the resolved base component carry an `adduct_label` (others
    are None). Peak indices map 1:1 to the ensemble's MS1 cofeatures, since the
    composite MS1 is cofeature-ordered and find-mfs preserves input order.
    """
    ensemble.remove_generic_annots_by_source(ADDUCT_ANNOT_SOURCE)

    n_ms1 = len(ensemble.ms1_cofeatures)
    for gid in range(grouped.n_groups):
        label = grouped.adduct_label[gid]
        if label is None:
            continue

        cofeature_idx = int(grouped.mono_idx[gid])
        if not (0 <= cofeature_idx < n_ms1):
            logger.warning(
                "adduct label %r maps to out-of-range cofeature %d (ms1=%d); "
                "skipping", label, cofeature_idx, n_ms1
            )
            continue

        ensemble.add_generic_annot(
            cofeature_idx=cofeature_idx,
            ms_level=1,
            text=str(label),
            scan_num=None,                 # scan-agnostic: shows in composite view
            source=ADDUCT_ANNOT_SOURCE,
        )
