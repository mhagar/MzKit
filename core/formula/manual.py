"""
Manually entered formulae.

A formula the user types in (e.g. known from a standard) becomes part of the
ensemble's FormulaAssignment, chosen by 'user', so it has the same single
source of truth as a searched one, and automatic runs won't replace it:

  - no assignment yet     -> a new assignment holding just this candidate
  - it's already a candidate of the assignment -> that candidate is chosen
  - otherwise             -> it's added to the assignment's candidates, chosen

Its mass error / adduct / scores come from find-mfs itself
(an exact-composition search at the ensemble's precursor, over the default adducts)

Qt-free.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from molmass import Formula, FormulaError

from core.data_structs.formula_assignment import (
    AssignedCandidate,
    FormulaAssignment,
)

if TYPE_CHECKING:
    from core.data_structs import Ensemble
    from core.interfaces.data_sources import AssignmentDataSource

# How far off the precursor a typed formula may be and still get an adduct +
# mass error; beyond this it's recorded without them.
MANUAL_MATCH_PPM = 50.0


def canonical_formula(formula_str: str) -> str:
    """
    Hill-order string for comparing formulae.

    :raises ValueError: if it doesn't parse as a formula.
    """
    try:
        return Formula(formula_str.strip()).formula
    except (FormulaError, ValueError, KeyError) as exc:
        raise ValueError(f"Not a valid formula: {formula_str!r}") from exc


def manual_candidate(
    formula_str: str,
    precursor_mz: Optional[float],
    charge: int = 1,
) -> AssignedCandidate:
    """
    An AssignedCandidate for a typed formula (neutral, no adduct). The adduct
    that best explains `precursor_mz` (within MANUAL_MATCH_PPM) supplies the
    mass error, RDBE and scores.

    :raises ValueError: if `formula_str` isn't a valid formula.
    """
    from find_mfs import annotate_precursor
    from find_mfs.annotate import DEFAULT_ADDUCTS
    from find_mfs.utils.filtering import get_rdbe

    formula = canonical_formula(formula_str)

    best = None
    adducts = [a for a in DEFAULT_ADDUCTS if a[1] == charge]
    if precursor_mz is not None and adducts:
        hits = annotate_precursor(
            precursor_mz,
            adducts=adducts,
            max_counts=formula,     # exactly this composition
            min_counts=formula,
            error_ppm=MANUAL_MATCH_PPM,
        )
        if len(hits):
            best = min(hits.candidates, key=lambda c: abs(c.error_ppm))

    if best is None:
        return AssignedCandidate(
            formula_str=formula, adduct=None, error_ppm=None, error_da=None,
            rdbe=get_rdbe(Formula(formula)), mass_loglik=None, iso_loglik=None,
            chem_logprior=None, ms2_loglik=None, log_posterior=None,
            manual=True,
        )

    return AssignedCandidate(
        formula_str=best.formula.formula,
        adduct=best.adduct,
        error_ppm=best.error_ppm,
        error_da=best.error_da,
        rdbe=best.rdbe,
        mass_loglik=best.mass_loglik,
        iso_loglik=best.iso_loglik,
        chem_logprior=best.chem_logprior,
        ms2_loglik=None,
        log_posterior=best.log_posterior,
        manual=True,
    )


def assign_manual_formula(
    data_source: 'AssignmentDataSource',
    ensemble: 'Ensemble',
    formula_str: Optional[str],
) -> None:
    """
    Make `formula_str` the ensemble's user-chosen formula (see module doc).
    Empty / None clears the choice, and drops an assignment that only ever
    held manual entries.

    :raises ValueError: if `formula_str` isn't a valid formula.
    """
    assignment = data_source.get_assignment_for_source(ensemble.uuid)

    if not formula_str or not formula_str.strip():
        if assignment is None:
            return
        if all(c.manual for c in assignment.candidates):
            data_source.remove_assignment(ensemble.uuid)
        else:
            data_source.set_chosen_candidate(ensemble.uuid, None)
        return

    formula = canonical_formula(formula_str)

    if assignment is not None:
        for i, c in enumerate(assignment.candidates):
            if canonical_formula(c.formula_str) == formula:
                data_source.set_chosen_candidate(ensemble.uuid, i)
                return

    candidate = manual_candidate(
        formula, ensemble.resolved_precursor_mz, ensemble.resolved_charge,
    )

    if assignment is None:
        data_source.register_assignment(FormulaAssignment(
            source_uuid=ensemble.uuid,
            candidates=[candidate],
            chosen_idx=0,
            chosen_by='user',
            precursor_mz=ensemble.resolved_precursor_mz,
            charge=ensemble.resolved_charge,
        ))
        return

    assignment.candidates.append(candidate)
    data_source.set_chosen_candidate(ensemble.uuid, len(assignment.candidates) - 1)
