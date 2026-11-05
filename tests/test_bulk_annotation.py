"""Session-3 bulk actions: envelope matching and the shared annotation edits."""
from core.data_structs.ensemble import EnsembleAnnotation, Hypothesis
from core.formula.bulk import match_envelope
from core.utils.array_types import to_spec_arr


def test_match_envelope_within_tolerance():
    spec = to_spec_arr([200.0, 301.003, 302.006, 303.01], [5.0, 100.0, 20.0, 3.0])
    assert match_envelope([301.0, 302.0], spec, 0.01) == [1, 2]


def test_match_envelope_drops_unmatched_isotope_but_not_mono():
    spec = to_spec_arr([301.003, 302.5], [100.0, 20.0])
    assert match_envelope([301.0, 302.0], spec, 0.01) == [0]
    assert match_envelope([300.9, 302.5], spec, 0.01) is None


def test_match_envelope_ignores_empty_peaks():
    spec = to_spec_arr([301.0, 301.004], [0.0, 50.0])
    assert match_envelope([301.0], spec, 0.01) == [1]


def test_with_hypothesis_dedupes_and_chooses():
    h = Hypothesis("CCO", "ethanol", origin="COCONUT", db_id="1")
    ann = EnsembleAnnotation(hypotheses=(Hypothesis("C"), h), chosen_idx=0)
    again = ann.with_hypothesis(Hypothesis("CCO", origin="COCONUT", db_id="1"))
    assert len(again.hypotheses) == 2 and again.chosen_idx == 1
    added = EnsembleAnnotation().with_hypothesis(h)
    assert added.hypotheses == (h,) and added.chosen is h


def test_with_scores_fills_only_unscored():
    scored = Hypothesis("CCO", origin="COCONUT", scores={"cosine": 0.9})
    ann = EnsembleAnnotation(hypotheses=(Hypothesis("CCO"), scored, Hypothesis("C")))
    out = ann.with_scores("CCO", {"cosine": 0.5})
    assert [h.scores for h in out.hypotheses] == [{"cosine": 0.5}, {"cosine": 0.9}, {}]


def test_apply_formula_to_members_finds_source_formula(ensemble):
    """A member identical to the source re-finds the source's (non-top) pick, as 'user'."""
    import copy
    from find_mfs.spectra.ions import ION_VOCAB
    from core.formula.bulk import apply_formula_to_members
    from core.formula.compound import search_compound
    from tests.test_compound_search import _envelope

    if ensemble.is_dda:
        import pytest
        pytest.skip("fixture ensemble is DDA")
    signals, idxs, scan = _envelope(ensemble)
    ion = next(i for i in ION_VOCAB if i.label == '[M+H]+')
    assignment, network = search_compound(ensemble, signals, idxs, scan, ion)
    assignment.chosen_idx = min(1, len(assignment.candidates) - 1)
    ensemble.formula_assignment, ensemble.adduct_network = assignment, network

    member = copy.copy(ensemble)
    member.uuid = 12345
    res = apply_formula_to_members(ensemble, [ensemble, member], 0.01)

    assert res.n_members == 1 and not res.no_feature and not res.disagreed
    (uuid, got, net), = res.updates
    assert uuid == 12345 and got.chosen_by == 'user'
    assert got.chosen.formula_str == assignment.chosen.formula_str
    assert net.base_ion == "[M+H]+" and got.envelope == assignment.envelope
