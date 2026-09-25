"""
Tests for core/formula/manual.py: typing a formula in makes it the
ensemble's user-chosen FormulaAssignment (the replacement for the old
free-text Ensemble.proposed_formula).
"""
from types import SimpleNamespace

import pytest

from core.data_structs import DataRegistry
from core.data_structs.formula_assignment import AssignedCandidate, FormulaAssignment
from core.formula.manual import assign_manual_formula, manual_candidate

CAFFEINE_MH = 195.0877      # [M+H]+ of C8H10N4O2
ENS = SimpleNamespace(uuid=77, resolved_precursor_mz=CAFFEINE_MH, resolved_charge=1)


def _searched(*formulas):
    return FormulaAssignment(
        source_uuid=ENS.uuid,
        candidates=[
            AssignedCandidate(
                formula_str=f, adduct='H', error_ppm=1.0, error_da=0.0002,
                rdbe=4.0, mass_loglik=-1.0, iso_loglik=None, chem_logprior=-1.0,
                ms2_loglik=None, log_posterior=-2.0,
            ) for f in formulas
        ],
        chosen_idx=0, chosen_by='auto',
    )


def test_manual_candidate_matches_an_adduct():
    c = manual_candidate("C8H10N4O2", CAFFEINE_MH)
    assert c.manual and c.adduct == 'H'
    assert abs(c.error_ppm) < 1 and c.rdbe == 6.0
    assert c.log_posterior is not None


def test_manual_candidate_far_from_precursor_has_no_mass_terms():
    c = manual_candidate("C6H12O6", CAFFEINE_MH)
    assert c.manual and c.adduct is None and c.error_ppm is None
    assert c.rdbe == 1.0


def test_manual_formula_without_assignment_creates_one():
    reg = DataRegistry()
    assign_manual_formula(reg, ENS, "C8H10N4O2")
    a = reg.get_assignment_for_source(ENS.uuid)
    assert a.chosen.formula_str == "C8H10N4O2" and a.user_chosen
    assert a.candidates[0].manual

    # Clearing a manual-only assignment drops it entirely
    assign_manual_formula(reg, ENS, "")
    assert reg.get_assignment_for_source(ENS.uuid) is None


def test_manual_formula_already_a_candidate_is_just_chosen():
    reg = DataRegistry()
    reg.register_assignment(_searched("C6H12O6", "C8H10N4O2"))
    assign_manual_formula(reg, ENS, "c8h10n4o2".upper())
    a = reg.get_assignment_for_source(ENS.uuid)
    assert len(a.candidates) == 2 and a.chosen_idx == 1 and a.user_chosen


def test_new_manual_formula_is_added_to_searched_candidates():
    reg = DataRegistry()
    reg.register_assignment(_searched("C6H12O6"))
    assign_manual_formula(reg, ENS, "C8H10N4O2")
    a = reg.get_assignment_for_source(ENS.uuid)
    assert [c.formula_str for c in a.candidates] == ["C6H12O6", "C8H10N4O2"]
    assert a.chosen_idx == 1 and a.candidates[1].manual and a.user_chosen

    # Clearing keeps the searched candidates, just withdraws the choice
    assign_manual_formula(reg, ENS, None)
    a = reg.get_assignment_for_source(ENS.uuid)
    assert a is not None and a.chosen is None


def test_invalid_formula_is_rejected_without_changes():
    reg = DataRegistry()
    reg.register_assignment(_searched("C6H12O6"))
    with pytest.raises(ValueError):
        assign_manual_formula(reg, ENS, "Xy2")
    a = reg.get_assignment_for_source(ENS.uuid)
    assert a.chosen_idx == 0 and a.chosen_by == 'auto' and len(a.candidates) == 1
