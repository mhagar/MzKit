"""
Tests for consensus spectrum generation (core.utils.spectra.merge_spectra).
"""
import numpy as np
import pytest

from core.utils.array_types import to_spec_arr
from core.utils.spectra import merge_spectra


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
    # Bin @100: mz averaged (100.2, 100.4), both normalized to 1.0
    assert out[0]['mz'] == pytest.approx(100.3)
    assert out[0]['intsy'] == pytest.approx(1.0)
    assert out[0]['freq'] == 2
    # Bin @200: mz averaged (200.4, 200.6); intsy averaged (0.5, 0.5)
    assert out[1]['mz'] == pytest.approx(200.5)
    assert out[1]['intsy'] == pytest.approx(0.5)
    assert out[1]['freq'] == 2


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

    assert out[0]['mz'] == pytest.approx((100.0 + 100.5 + 100.2) / 3)
    assert out[0]['freq'] == 3
    assert out[-1]['mz'] == pytest.approx(300.0)
    assert out[-1]['freq'] == 1


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
