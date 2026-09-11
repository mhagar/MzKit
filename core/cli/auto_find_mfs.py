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
import re
import threading
from typing import Callable, Iterable, Optional, TYPE_CHECKING

from core.formula.assign_formula import (
    _get_scorer,
    results_to_assigned_candidates,
)
from core.data_structs.formula_assignment import FormulaAssignment

if TYPE_CHECKING:
    from configparser import ConfigParser
    from core.data_structs import Ensemble

logger = logging.getLogger(__name__)

# The generic-annotation source tag for adduct labels we attach, so re-runs can
# clear+replace only these (leaving user annotations alone).
ADDUCT_ANNOT_SOURCE = 'auto_adduct'

_DEFAULT_TOP_N = 50

_ELEMENT_RE = re.compile(r'[A-Z][a-z]?')


def _elements_from_counts(*count_strings: str) -> str:
    """
    Derive the decomposition element set from the count-constraint strings
    (e.g. "C*H*N*O*P1S1Br*Cl*" -> "CHNOPSBrCl").

    find-mfs's `to_bounds_dict` RAISES if `max_counts` names an element absent
    from the element set, and silently ZEROES any element in the set but absent
    from `max_counts`. Deriving the element set straight from the constraints
    keeps the two consistent by construction: exactly the elements the user
    wrote (with their bounds) get searched. Empty -> plain CHNOPS.
    """
    seen: list[str] = []
    for s in count_strings:
        for sym in _ELEMENT_RE.findall(s or ''):
            if sym not in seen:
                seen.append(sym)
    return ''.join(seen) if seen else 'CHNOPS'


def annotation_params_from_config(
    config: 'ConfigParser',
) -> dict:
    """
    Read the DIA auto-annotation parameters from the `[findmfs]` config section,
    falling back to find-mfs-friendly defaults when a key is absent. Returns a
    kwargs dict suitable for `annotate_ensembles_dia`.

    The decomposition constraints (min/max counts, RDBE window, octet rule) are
    packed into `finder_kwargs` exactly as the manual FormulaFinder dialog does,
    and the element set is derived from the count strings so they stay
    consistent (see `_elements_from_counts`).
    """
    section = 'findmfs'

    def _get(getter, key, default):
        try:
            return getter(section, key)
        except Exception:
            return default

    max_counts = (_get(config.get, 'max_counts', '') or '').strip()
    min_counts = (_get(config.get, 'min_counts', '') or '').strip()

    finder_kwargs: dict = {}
    if max_counts:
        finder_kwargs['max_counts'] = max_counts
    if min_counts:
        finder_kwargs['min_counts'] = min_counts
    min_rdbe = _get(config.getfloat, 'min_rdbe', None)
    max_rdbe = _get(config.getfloat, 'max_rdbe', None)
    if min_rdbe is not None and max_rdbe is not None:
        finder_kwargs['filter_rdbe'] = (min_rdbe, max_rdbe)
    finder_kwargs['check_octet'] = _get(config.getboolean, 'check_octet', True)

    return dict(
        elements=_elements_from_counts(max_counts, min_counts),
        error_ppm=_get(config.getfloat, 'error_ppm', 5.0),
        ms2_weight=_get(config.getfloat, 'ms2_weight', 1.0),
        instrument=_get(config.get, 'instrument', 'unknown'),
        detect_halogens=_get(config.getboolean, 'autodetect_cl_br', True),
        top_n=_get(config.getint, 'top_n', _DEFAULT_TOP_N),
        finder_kwargs=finder_kwargs,
    )


def annotate_ensembles_dia(
    ensembles: Iterable['Ensemble'],
    *,
    model_path: Optional[str] = None,
    elements: str = 'CHNOPS',
    error_ppm: float = 5.0,
    instrument: str = 'unknown',
    ms2_weight: float = 1.0,
    detect_halogens: bool = True,
    top_n: int = _DEFAULT_TOP_N,
    finder_kwargs: Optional[dict] = None,
    attach_adduct_labels: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[FormulaAssignment]:
    """
    Run find-mfs `annotate_analyte_dia` over each (DIA) ensemble.

    For each ensemble: builds a FormulaAssignment (top candidate pre-selected,
    `chosen_idx=0`), sets `ensemble.proposed_formula`, and — if
    `attach_adduct_labels` — replaces prior 'auto_adduct' generic annotations
    with the resolved per-envelope adduct labels. DDA ensembles are skipped.

    :return: one FormulaAssignment per successfully-annotated ensemble.
    """
    # Lazy import: find-mfs pulls in MistNet + scorers.
    from find_mfs import annotate_analyte_dia

    scorer = _get_scorer(model_path)  # MistNet warmed once, cached across calls

    ensembles = list(ensembles)
    n = len(ensembles)
    assignments: list[FormulaAssignment] = []

    for i, ensemble in enumerate(ensembles):
        if cancel_event is not None and cancel_event.is_set():
            logger.info("annotate_ensembles_dia cancelled at %d/%d", i, n)
            break

        if progress_callback is not None:
            progress_callback(100.0 * i / n, f"Annotating {i + 1}/{n}")

        if ensemble.is_dda:
            # Composite is DIA-only for now; nothing to do for DDA.
            continue

        try:
            assignment = _annotate_one(
                ensemble,
                annotate_analyte_dia=annotate_analyte_dia,
                scorer=scorer,
                elements=elements,
                error_ppm=error_ppm,
                instrument=instrument,
                ms2_weight=ms2_weight,
                detect_halogens=detect_halogens,
                top_n=top_n,
                finder_kwargs=finder_kwargs,
                attach_adduct_labels=attach_adduct_labels,
            )
        except ValueError as exc:
            # find-mfs raises on an empty scan / no resolvable envelope. Skip
            # this ensemble rather than aborting the whole batch.
            logger.warning(
                "Skipping ensemble %s: %s", ensemble.uuid, exc
            )
            continue

        assignments.append(assignment)

    if progress_callback is not None:
        progress_callback(100.0, f"Annotated {len(assignments)} ensemble(s)")

    return assignments


def _annotate_one(
    ensemble: 'Ensemble',
    *,
    annotate_analyte_dia,
    scorer,
    elements: str,
    error_ppm: float,
    instrument: str,
    ms2_weight: float,
    detect_halogens: bool,
    top_n: int,
    finder_kwargs: Optional[dict],
    attach_adduct_labels: bool,
) -> FormulaAssignment:
    composite = ensemble.composite_spectrum

    res = annotate_analyte_dia(
        ms1_peaks=composite.ms1,
        ms2_peaks=composite.ms2,          # may be None -> MS2 term skipped
        precursor_mz=None,                # DIA default: base = tallest envelope
        scorer=scorer,
        elements=elements,
        error_ppm=error_ppm,
        detect_halogens=detect_halogens,
        instrument=instrument,
        ms2_weight=ms2_weight,
        finder_kwargs=finder_kwargs or None,
    )

    candidates = results_to_assigned_candidates(
        res.candidates, ms2_weight=ms2_weight, top_n=top_n
    )

    assignment = FormulaAssignment(
        source_uuid=ensemble.uuid,
        candidates=candidates,
        chosen_idx=0 if candidates else None,   # auto-pick the top hit
        precursor_mz=res.precursor_mz,          # precursor find-mfs actually used
        charge=res.charge,                       # resolved charge
        elements=elements,
        autodetect_cl_br=detect_halogens,
        ms2_mode='tallest',                      # DIA composite MS2 source
        error_ppm=error_ppm,
        instrument=instrument,
        ms2_weight=ms2_weight,
    )

    # Mirror the manual path: keep the ensemble's free-text formula in sync so
    # non-assignment-aware displays (sample tree, labels) show it too.
    if candidates:
        ensemble.proposed_formula = candidates[0].formula_str

    if attach_adduct_labels:
        _attach_adduct_labels(ensemble, res.grouped)

    return assignment


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
