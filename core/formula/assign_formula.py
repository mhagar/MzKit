"""
MS2-aware formula assignment

Compatible w/ ProcessController

Loads the MistNet reranker once and runs each FormulaQuery through find-mfs
`annotate_precursor`

This currently submits a single query.
# TODO: "assign all ensembles in this sample" batch (without re-warming MistNet).
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from find_mfs import FormulaScorer
from core.formula.query import FormulaQuery
from core.data_structs.formula_assignment import (
    AssignedCandidate,
    FormulaAssignment,
)

logger = logging.getLogger(__name__)

# Ranked candidates kept per assignment
_DEFAULT_TOP_N = 50

# MistNet scorers are read-only once loaded; cache by model path so repeated
# (synchronous GUI) calls don't pay the load/warm-up cost each time.
_SCORER_CACHE: dict = {}


def _get_scorer(
        model_path: Optional[str]
):
    key = model_path or "__bundled__"
    scorer = _SCORER_CACHE.get(key)
    if scorer is None:
        scorer = FormulaScorer().with_ms2(model_path)
        _SCORER_CACHE[key] = scorer
    return scorer


def assign_formula(
    queries: list[FormulaQuery],
    top_n: int = _DEFAULT_TOP_N,
    model_path: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[FormulaAssignment]:
    """
    Assign molecular formulae for each query.

    :param queries: one FormulaQuery per source (MVP: length 1).
    :param top_n: number of ranked candidates to retain per assignment.
    :param model_path: MistNet .npz path; None uses the weights bundled with
        find-mfs.
    :return: one FormulaAssignment per query, in the same order.
    """
    # Lazy import: find-mfs pulls in MistNet + scorers
    from find_mfs import annotate_precursor

    # Load/cache MistNet across calls
    scorer = _get_scorer(model_path)

    assignments: list[FormulaAssignment] = []
    n = len(queries)
    for i, query in enumerate(queries):
        if cancel_event is not None and cancel_event.is_set():
            logger.info("assign_formula cancelled at %d/%d", i, n)
            break

        if progress_callback is not None:
            progress_callback(100.0 * i / n, f"Assigning formula {i + 1}/{n}")

        assignments.append(
            _assign_one(query, scorer, top_n, annotate_precursor)
        )

    if progress_callback is not None:
        progress_callback(100.0, f"Assigned {len(assignments)} formula(e)")

    return assignments


def _assign_one(
    query: FormulaQuery,
    scorer,
    top_n: int,
    annotate_precursor,
) -> FormulaAssignment:
    kwargs = dict(
        elements=query.elements,
        error_ppm=query.error_ppm,
        scorer=scorer,
        autodetect_cl_br=query.autodetect_cl_br,
        ms1_peaks=query.ms1_spec,
        ms2_peaks=query.ms2_spec,
        instrument=query.instrument,
        ms2_weight=query.ms2_weight,
        finder_kwargs=query.finder_kwargs or None,
    )
    # adducts=None to find-mfs means "one adductless search";
    # to get the joint DEFAULT_ADDUCTS ranking we must omit the arg entirely
    if query.adducts:
        kwargs['adducts'] = query.adducts

    hits = annotate_precursor(query.precursor_mz, **kwargs)

    # Full posterior (incl. the set-normalized MS2 term) and the MS2 term itself,
    # both aligned to the sorted candidate order.
    posterior = hits.log_posterior(ms2_weight=query.ms2_weight)
    ms2_loglik = hits.ms2_loglik()

    candidates = [
        AssignedCandidate(
            formula_str=c.formula.formula,
            adduct=c.adduct,
            error_ppm=c.error_ppm,
            error_da=c.error_da,
            rdbe=c.rdbe,
            mass_loglik=c.mass_loglik,
            iso_loglik=c.iso_loglik,
            chem_logprior=c.chem_logprior,
            ms2_loglik=float(ms2_loglik[j]),
            log_posterior=float(posterior[j]),
        )
        for j, c in enumerate(hits.candidates[: int(top_n)])
    ]

    return FormulaAssignment(
        source_uuid=query.source_uuid,
        candidates=candidates,
        precursor_mz=query.precursor_mz,
        charge=query.charge,
        adducts=query.adducts,
        elements=query.elements,
        autodetect_cl_br=query.autodetect_cl_br,
        ms2_mode=query.ms2_mode,
        error_ppm=query.error_ppm,
        instrument=query.instrument,
        ms2_weight=query.ms2_weight,
    )
