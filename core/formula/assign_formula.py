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

from dataclasses import asdict

from find_mfs import FormulaScorer
from core.formula.params import counts_to_str
from core.formula.query import FormulaQuery
from core.data_structs.formula_assignment import (
    AssignedCandidate,
    FormulaAssignment,
)

logger = logging.getLogger(__name__)

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


def results_to_assigned_candidates(
    hits,
    ms2_weight: float,
    top_n: int,
) -> list[AssignedCandidate]:
    """
    Convert a find-mfs FormulaSearchResults into MzKit's primitive-only
    AssignedCandidate list (top-N, ranked order preserved).

    Shared by both find-mfs entry points: `_assign_one` (annotate_precursor)
    and `core.cli.auto_find_mfs` (annotate_analyte_dia) hand us the same
    FormulaSearchResults type.

    `log_posterior`/`ms2_loglik` are the SET-level arrays
    (aligned to the sorted candidate order), not the per-candidate
     `ms2_logit` attribute.
    """
    posterior = hits.log_posterior(ms2_weight=ms2_weight)
    ms2_loglik = hits.ms2_loglik()

    return [
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


def search_provenance(hits) -> dict:
    """
    FormulaAssignment provenance for the search find-mfs actually ran, read off
    its results: element set and bounds after any halogen widening, and whether
    Cl/Br was detected.
    """
    qp = getattr(hits, 'query_params', None) or {}
    return dict(
        elements=qp.get('elements'),
        max_counts=counts_to_str(qp.get('max_counts')),
        min_counts=counts_to_str(
            {k: v for k, v in (qp.get('min_counts') or {}).items() if v}
        ),
        halogen_detected=qp.get('halogen_detected'),
    )


def assign_formula(
    queries: list[FormulaQuery],
    model_path: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[FormulaAssignment]:
    """
    Assign molecular formulae for each query.

    :param queries: one FormulaQuery per source (MVP: length 1).
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
            _assign_one(query, scorer, annotate_precursor)
        )

    if progress_callback is not None:
        progress_callback(100.0, f"Assigned {len(assignments)} formula(e)")

    return assignments


def _assign_one(
    query: FormulaQuery,
    scorer,
    annotate_precursor,
) -> FormulaAssignment:
    kwargs = dict(
        scorer=scorer,
        ms1_peaks=query.ms1_spec,
        ms2_peaks=query.ms2_spec,
        **query.params.search_kwargs(),
    )
    # adducts=None to find-mfs means "one adductless search";
    # to get the joint DEFAULT_ADDUCTS ranking we must omit the arg entirely
    if query.adducts:
        kwargs['adducts'] = query.adducts

    hits = annotate_precursor(query.precursor_mz, **kwargs)

    candidates = results_to_assigned_candidates(
        hits, ms2_weight=query.params.ms2_weight, top_n=query.params.top_n
    )

    return FormulaAssignment(
        source_uuid=query.source_uuid,
        candidates=candidates,
        precursor_mz=query.precursor_mz,
        charge=query.charge,
        adducts=query.adducts,
        ms2_mode=query.ms2_mode,
        params=asdict(query.params),
        **search_provenance(hits),
    )
