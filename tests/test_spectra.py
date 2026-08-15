"""
Tests for consensus spectrum generation (core.utils.spectra.merge_spectra).
"""
import numpy as np
import pytest

from core.utils.array_types import to_spec_arr
from core.utils.spectra import (
    merge_spectra,
    entropy_similarity,
    threshold_consensus,
)


def test_merge_averages_mz_and_intsy_per_bin():
    # Two spectra with peaks that share bins at ~100 and ~200.
    # Intensities are normalized (0 -> 1) before merging, so the tallest
    # peak in each spectrum becomes 1.0.
    a = to_spec_arr(np.array([100.2, 200.4]), np.array([10.0, 5.0]))
    b = to_spec_arr(np.array([100.4, 200.6]), np.array([8.0, 4.0]))

    out = merge_spectra([a, b], bin_width=1.0)

    assert out.dtype.names == ('mz', 'intsy', 'freq')
    assert len(out) == 2

    out = np.sort(out, order='mz')
    # Bin @100: mz averaged (100.2, 100.4), both normalized to 1.0.
    # freq is a relative frequency: present in 2/2 spectra -> 1.0.
    assert out[0]['mz'] == pytest.approx(100.3)
    assert out[0]['intsy'] == pytest.approx(1.0)
    assert out[0]['freq'] == pytest.approx(1.0)
    # Bin @200: mz averaged (200.4, 200.6); intsy averaged (0.5, 0.5)
    assert out[1]['mz'] == pytest.approx(200.5)
    assert out[1]['intsy'] == pytest.approx(0.5)
    assert out[1]['freq'] == pytest.approx(1.0)


def test_tallest_peak_within_bin_wins():
    # Two peaks fall in the same bin [100, 101); the tallest should be kept.
    spectrum = to_spec_arr(
        np.array([100.1, 100.9]),
        np.array([1.0, 20.0]),
    )
    out = merge_spectra([spectrum], bin_width=1.0)

    assert len(out) == 1
    # The tallest peak (mz=100.9) wins; normalized to 1.0.
    assert out[0]['mz'] == pytest.approx(100.9)
    assert out[0]['intsy'] == pytest.approx(1.0)
    assert out[0]['freq'] == 1


def test_freq_reflects_number_of_contributing_spectra():
    # Peak @100 present in all three; peak @300 present in only one.
    a = to_spec_arr(np.array([100.0, 300.0]), np.array([1.0, 1.0]))
    b = to_spec_arr(np.array([100.5]), np.array([1.0]))
    c = to_spec_arr(np.array([100.2]), np.array([1.0]))

    out = merge_spectra([a, b, c], bin_width=1.0)
    out = np.sort(out, order='mz')

    # freq is relative: @100 is in all 3 spectra (1.0), @300 in only 1 (1/3).
    assert out[0]['mz'] == pytest.approx((100.0 + 100.5 + 100.2) / 3)
    assert out[0]['freq'] == pytest.approx(1.0)
    assert out[-1]['mz'] == pytest.approx(300.0)
    assert out[-1]['freq'] == pytest.approx(1 / 3)


def test_intensities_are_normalized_per_spectrum():
    # Same shape, different absolute scale -> identical consensus.
    small = to_spec_arr(np.array([100.0, 200.0]), np.array([1.0, 2.0]))
    large = to_spec_arr(np.array([100.0, 200.0]), np.array([1000.0, 2000.0]))

    out = merge_spectra([small, large], bin_width=1.0)
    out = np.sort(out, order='mz')

    assert out[0]['intsy'] == pytest.approx(0.5)  # 1/2 in both
    assert out[1]['intsy'] == pytest.approx(1.0)  # tallest in both


def test_max_mz_inferred_from_input():
    a = to_spec_arr(np.array([50.0]), np.array([1.0]))
    b = to_spec_arr(np.array([250.0]), np.array([1.0]))

    out = merge_spectra([a, b], bin_width=1.0)
    mzs = np.sort(out['mz'])
    assert mzs == pytest.approx([50.0, 250.0])


def test_explicit_max_mz_discards_out_of_range_peaks():
    # A peak beyond max_mz is dropped, not clipped into the last bin.
    spectrum = to_spec_arr(np.array([100.0, 500.0]), np.array([0.5, 1.0]))
    out = merge_spectra([spectrum], bin_width=1.0, max_mz=200.0)

    # Only the @100 peak survives; it is the tallest in-range peak -> 1.0.
    assert len(out) == 1
    assert out[0]['mz'] == pytest.approx(100.0)
    assert out[0]['intsy'] == pytest.approx(1.0)
    assert out[0]['freq'] == 1


def test_spectrum_entirely_out_of_range_is_ignored():
    keep = to_spec_arr(np.array([100.0]), np.array([1.0]))
    drop = to_spec_arr(np.array([500.0, 600.0]), np.array([1.0, 1.0]))

    out = merge_spectra([keep, drop], bin_width=1.0, max_mz=200.0)
    assert len(out) == 1
    assert out[0]['freq'] == 1


def test_bin_width_too_large_raises():
    # bin_width must not exceed 10% of max_mz.
    spectrum = to_spec_arr(np.array([100.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        merge_spectra([spectrum], bin_width=25.0, max_mz=200.0)


def test_empty_input_returns_empty_consensus():
    out = merge_spectra([], bin_width=1.0)
    assert len(out) == 0
    assert out.dtype.names == ('mz', 'intsy', 'freq')


def test_empty_spectra_are_ignored():
    empty = to_spec_arr(np.array([]), np.array([]))
    real = to_spec_arr(np.array([100.0]), np.array([1.0]))

    out = merge_spectra([empty, real], bin_width=1.0)
    assert len(out) == 1
    assert out[0]['freq'] == 1


def test_generator_input_is_consumed_once():
    # Passing a one-shot generator must still work (two internal passes).
    def gen():
        yield to_spec_arr(np.array([100.0]), np.array([1.0]))
        yield to_spec_arr(np.array([200.0]), np.array([1.0]))

    out = merge_spectra(gen(), bin_width=1.0)
    assert len(out) == 2


def test_invalid_bin_width_raises():
    spectrum = to_spec_arr(np.array([100.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        merge_spectra([spectrum], bin_width=0.0)


# ---------------------------------------------------------------------------
# threshold_consensus
# ---------------------------------------------------------------------------

def test_threshold_consensus_keeps_frequent_bins():
    # @100 in all 3 (freq 1.0), @200 in 1 of 3 (freq 1/3).
    a = to_spec_arr(np.array([100.0, 200.0]), np.array([1.0, 1.0]))
    b = to_spec_arr(np.array([100.0]), np.array([1.0]))
    c = to_spec_arr(np.array([100.0]), np.array([1.0]))
    consensus = merge_spectra([a, b, c], bin_width=1.0)

    # 25% keeps both; 50% drops the 1/3 bin.
    lenient = threshold_consensus(consensus, min_freq=0.25)
    strict = threshold_consensus(consensus, min_freq=0.5)

    assert lenient.dtype.names == ('mz', 'intsy')  # flattened for display
    assert sorted(np.round(lenient['mz']).tolist()) == [100.0, 200.0]
    assert np.round(strict['mz']).tolist() == [100.0]


def test_threshold_consensus_passes_plain_spectrum_through():
    # A plain (mz, intsy) spectrum has no freq field: returned intact.
    plain = to_spec_arr(np.array([100.0, 200.0]), np.array([5.0, 3.0]))
    out = threshold_consensus(plain, min_freq=0.9)
    assert out.dtype.names == ('mz', 'intsy')
    assert out['mz'].tolist() == [100.0, 200.0]
    assert out['intsy'].tolist() == [5.0, 3.0]


# ---------------------------------------------------------------------------
# entropy_similarity
# ---------------------------------------------------------------------------

def test_identical_spectra_have_similarity_one():
    spectrum = to_spec_arr(
        np.array([100.0, 150.0, 200.0]),
        np.array([1.0, 0.5, 0.2]),
    )
    assert entropy_similarity(spectrum, spectrum) == pytest.approx(1.0)


def test_disjoint_spectra_have_similarity_zero():
    a = to_spec_arr(np.array([100.0, 200.0]), np.array([1.0, 1.0]))
    b = to_spec_arr(np.array([300.0, 400.0]), np.array([1.0, 1.0]))
    assert entropy_similarity(a, b) == pytest.approx(0.0)


def test_similarity_is_symmetric():
    a = to_spec_arr(np.array([100.0, 150.0, 200.0]), np.array([1.0, 0.6, 0.3]))
    b = to_spec_arr(np.array([100.0, 150.0, 250.0]), np.array([1.0, 0.4, 0.8]))
    assert entropy_similarity(a, b) == pytest.approx(entropy_similarity(b, a))


def test_partial_overlap_between_zero_and_one():
    a = to_spec_arr(np.array([100.0, 150.0, 200.0]), np.array([1.0, 0.6, 0.3]))
    b = to_spec_arr(np.array([100.0, 150.0, 250.0]), np.array([1.0, 0.4, 0.8]))
    sim = entropy_similarity(a, b)
    assert 0.0 < sim < 1.0


def test_tolerance_controls_peak_matching():
    # Peaks offset by 0.05 Da: matched under a loose tolerance, not a tight one.
    a = to_spec_arr(np.array([100.00, 200.00]), np.array([1.0, 1.0]))
    b = to_spec_arr(np.array([100.05, 200.05]), np.array([1.0, 1.0]))

    loose = entropy_similarity(a, b, mz_tol_da=0.1)
    tight = entropy_similarity(a, b, mz_tol_da=0.01)

    assert loose == pytest.approx(1.0)
    assert tight == pytest.approx(0.0)


def test_empty_spectrum_returns_zero():
    empty = to_spec_arr(np.array([]), np.array([]))
    real = to_spec_arr(np.array([100.0]), np.array([1.0]))
    assert entropy_similarity(empty, real) == 0.0
    assert entropy_similarity(real, empty) == 0.0
    assert entropy_similarity(empty, empty) == 0.0
