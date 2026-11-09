"""
Scoring ignores the grass (min_rel_intensity) and the precursor region on both
spectra; hovers still pair every peak; library hits get GLACIER predictions.
"""
import numpy as np
import pytest
from matchms.similarity import CosineGreedy

from core.cli.align_ensembles import _cosine_with_matches, nearest_peak, peak_pairing
from core.cli.rank_candidates import RankParams
from core.cli.search_spectra import LibraryHit, predict_hits
from core.utils import glacier
from core.utils.array_types import to_spec_arr
from core.utils.spectra import scored_mask


def spec(*peaks):
    mz, intsy = zip(*peaks)
    return to_spec_arr(np.array(mz, float), np.array(intsy, float))


EXP = spec((50.0, 1.0), (80.0, 0.5), (190.0, 0.8))     # 190: precursor region of 200
PRED = spec((50.0, 1.0), (80.0, 0.001), (120.0, 0.5))  # 80: GLACIER grass


def test_scored_mask_drops_precursor_region_and_grass():
    assert scored_mask(EXP['mz'], EXP['intsy'], 200.0, 0.01).tolist() == [True, True, False]
    assert scored_mask(PRED['mz'], PRED['intsy'], 200.0, 0.01).tolist() == [True, False, True]
    assert scored_mask(PRED['mz'], PRED['intsy'], 200.0).tolist() == [True, True, True]


def test_cosine_ignores_grass_but_hover_pairing_keeps_it():
    cosine = CosineGreedy(tolerance=0.01)
    _, matches = _cosine_with_matches(cosine, EXP, PRED, 200.0, 200.0, 0.01)
    assert matches == [(0, 0)]
    _, unfiltered = _cosine_with_matches(cosine, EXP, PRED, 200.0, 200.0)
    assert sorted(unfiltered) == [(0, 0), (1, 1)]
    assert peak_pairing(EXP, PRED, 200.0, 200.0, 0.01) == {0: 0, 1: 1}


def test_nearest_peak_within_tolerance():
    assert nearest_peak(PRED, 80.004, 0.01) == 1
    assert nearest_peak(PRED, 81.0, 0.01) is None
    assert nearest_peak(spec((1.0, 1.0))[:0], 1.0, 0.01) is None


def _hit(smiles):
    return LibraryHit(1, 'GNPS', 'X', 'x', smiles, '', '[M+H]+', 47.049, spec((29.04, 1.0), (31.02, 0.5)),
                      0.9, 0.8, 2, 0.0)


def test_predict_hits_skips_quietly_without_model(monkeypatch):
    monkeypatch.setattr('core.utils.glacier.model_available', lambda: False)
    hit = _hit('CCO')
    assert predict_hits([hit], RankParams()) and hit.glacier is None


@pytest.mark.skipif(not glacier.model_available(), reason="GLACIER model not in models/glacier/")
def test_predict_hits_scores_against_the_library_spectrum():
    hit, bare = _hit('CCO'), _hit('')
    assert predict_hits([hit, bare], RankParams())
    assert bare.glacier is None
    g = hit.glacier
    assert g.candidate.smiles == 'CCO' and len(g.prediction) == len(g.frag_atoms) == len(g.ions)
