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
ENS_UUID = 77


def _registry():
    """A registry holding one bare Ensemble (DDA-style precursor = caffeine [M+H]+)."""
    from core.data_structs import Ensemble
    ens = Ensemble([], [], uuid=ENS_UUID, precursor_mz=CAFFEINE_MH)
    reg = DataRegistry()
    reg.register_sample(SimpleNamespace(
        uuid=1, name='s', fingerprint=None,
        injection=SimpleNamespace(ensembles={ens.uuid: ens}),
    ))
    return reg, ens


def _searched(*formulas):
    return FormulaAssignment(
        source_uuid=ENS_UUID,
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
    reg, ens = _registry()
    assign_manual_formula(reg, ens, "C8H10N4O2")
    a = ens.formula_assignment
    assert a.chosen.formula_str == "C8H10N4O2" and a.user_chosen
    assert a.candidates[0].manual

    # Clearing a manual-only assignment drops it entirely
    assign_manual_formula(reg, ens, "")
    assert ens.formula_assignment is None


def test_manual_formula_already_a_candidate_is_just_chosen():
    reg, ens = _registry()
    ens.formula_assignment = (_searched("C6H12O6", "C8H10N4O2"))
    assign_manual_formula(reg, ens, "c8h10n4o2".upper())
    a = ens.formula_assignment
    assert len(a.candidates) == 2 and a.chosen_idx == 1 and a.user_chosen


def test_new_manual_formula_is_added_to_searched_candidates():
    reg, ens = _registry()
    ens.formula_assignment = (_searched("C6H12O6"))
    assign_manual_formula(reg, ens, "C8H10N4O2")
    a = ens.formula_assignment
    assert [c.formula_str for c in a.candidates] == ["C6H12O6", "C8H10N4O2"]
    assert a.chosen_idx == 1 and a.candidates[1].manual and a.user_chosen

    # Clearing keeps the searched candidates, just withdraws the choice
    assign_manual_formula(reg, ens, None)
    a = ens.formula_assignment
    assert a is not None and a.chosen is None


def test_invalid_formula_is_rejected_without_changes():
    reg, ens = _registry()
    ens.formula_assignment = (_searched("C6H12O6"))
    with pytest.raises(ValueError):
        assign_manual_formula(reg, ens, "Xy2")
    a = ens.formula_assignment
    assert a.chosen_idx == 0 and a.chosen_by == 'auto' and len(a.candidates) == 1
