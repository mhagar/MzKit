"""MS2 similarity and plot scaling ignore the precursor region (m/z >= precursor - 20 Da)."""
import numpy as np
from matchms.similarity import CosineGreedy

from core.cli.align_ensembles import _cosine_with_matches
from core.utils.array_types import to_spec_arr
from core.utils.spectra import PRECURSOR_MARGIN, informative_mask, informative_max, normalize_spectrum

PREC = 300.0


def _spec(mz, intsy):
    return to_spec_arr(np.array(mz, float), np.array(intsy, float))


def test_mask_edges():
    mz = np.array([100.0, PREC - PRECURSOR_MARGIN, PREC - PRECURSOR_MARGIN - 1e-6, PREC])
    assert informative_mask(mz, None).all()
    assert informative_mask(mz, PREC).tolist() == [True, False, True, False]


def test_shared_precursor_alone_does_not_score():
    cosine = CosineGreedy(tolerance=0.01, mz_power=0.0, intensity_power=1.0)
    a = _spec([100.0, 282.0, 300.0], [10.0, 50.0, 100.0])   # 282 = -H2O
    b = _spec([150.0, 282.0, 300.0], [10.0, 50.0, 100.0])
    full, _ = _cosine_with_matches(cosine, a, b)
    assert full > 0.9
    score, matches = _cosine_with_matches(cosine, a, b, PREC, PREC)
    assert score == 0.0 and matches == []


def test_matches_index_the_full_input():
    cosine = CosineGreedy(tolerance=0.01, mz_power=0.0, intensity_power=1.0)
    a = _spec([300.0, 100.0], [100.0, 10.0])   # unsorted, precursor first
    b = _spec([100.0, 300.0], [10.0, 100.0])
    score, matches = _cosine_with_matches(cosine, a, b, PREC, PREC)
    assert score == 1.0 and matches == [(1, 0)]


def test_plot_scaling_ignores_precursor():
    s = _spec([100.0, 300.0], [20.0, 100.0])
    assert informative_max(s, PREC) == 20.0
    assert normalize_spectrum(s, precursor_mz=PREC)['intsy'].tolist() == [1.0, 5.0]
    # Nothing below the window: fall back to the global max
    assert informative_max(_spec([295.0], [7.0]), PREC) == 7.0
