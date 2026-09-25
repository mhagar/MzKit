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
from dataclasses import asdict
from typing import Callable, Iterable, Optional, TYPE_CHECKING

from core.formula.assign_formula import (
    _get_scorer,
    results_to_assigned_candidates,
    search_provenance,
)
from core.formula.params import FindMfsParams
from core.data_structs.formula_assignment import FormulaAssignment

if TYPE_CHECKING:
    from core.data_structs import Ensemble

logger = logging.getLogger(__name__)

# The generic-annotation source tag for adduct labels we attach, so re-runs can
# clear+replace only these (leaving user annotations alone).
ADDUCT_ANNOT_SOURCE = 'auto_adduct'


def annotate_ensembles_dia(
    ensembles: Iterable['Ensemble'],
    *,
    params: Optional[FindMfsParams] = None,
    model_path: Optional[str] = None,
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

    :param params: find-mfs constraints/scoring; defaults to FindMfsParams().
    :return: one FormulaAssignment per successfully-annotated ensemble.
    :raises ValueError: up front, if the count constraints are invalid (rather
        than once per ensemble, where it would be indistinguishable from an
        ensemble with no resolvable envelope).
    """
    # Lazy import: find-mfs pulls in MistNet + scorers.
    from find_mfs import annotate_analyte_dia

    params = params if params is not None else FindMfsParams()
    params.validate()

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
                params=params,
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
) -> list[FormulaAssignment]:
    """
    TEMPORARY DDA compound-annotation path.

    Runs find-mfs against caller-supplied MS1 + MS2 spectra (whatever the
    EnsembleViewer currently has on screen) plus the ensemble's precursor,
    rather than the DIA composite (which is undefined for DDA). Produces a
    single FormulaAssignment, returned in a list to match the DIA completion
    handler.

    This is a stopgap until DDA MS2 'consensus' stitching lands and
    `composite_spectrum` works for DDA; it will be retired then. Note that
    unlike the DIA path we hand find-mfs a real precursor_mz.

    ProcessController-compatible (accepts progress_callback + cancel_event).
    """
    # Lazy import: find-mfs pulls in MistNet + scorers.
    from find_mfs import annotate_analyte_dia

    if progress_callback is not None:
        progress_callback(0.0, "Annotating (DDA, selected MS2)…")

    if cancel_event is not None and cancel_event.is_set():
        return []

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
        return []

    if progress_callback is not None:
        progress_callback(100.0, "Annotated 1 ensemble")

    return [assignment]


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
