"""
Unit tests for the pure MS2-spectrum reduction logic
(core.data_structs.ensemble.reduce_ms2_spectra), which underpins
Ensemble.get_ms2_spectra without needing a full Injection.
"""
import numpy as np
import pytest

from core.utils.array_types import to_spec_arr
from core.data_structs.ensemble import (
    MS2Spectrum,
    reduce_ms2_spectra,
)


def _ps(mzs, intsys, precursor_mz=100.0, charge=1, rt=0.0) -> MS2Spectrum:
    return MS2Spectrum(
        spectrum=to_spec_arr(np.array(mzs, 'f8'), np.array(intsys, 'f8')),
        precursor_mz=precursor_mz,
        charge=charge,
        rt=rt,
    )


def test_empty_input_returns_empty():
    assert reduce_ms2_spectra([], 'consensus', group_by_precursor=True) == []
    assert reduce_ms2_spectra([], 'tallest', group_by_precursor=False) == []


def test_all_mode_returns_spectra_unchanged():
    specs = [_ps([50, 60], [1, 2]), _ps([70], [3])]
    out = reduce_ms2_spectra(specs, 'all', group_by_precursor=True)
    assert out == specs


def test_tallest_picks_most_intense_scan():
    small = _ps([50, 60], [1, 1], rt=1.0)
    big = _ps([50, 60], [10, 20], rt=2.0)
    out = reduce_ms2_spectra([small, big], 'tallest', group_by_precursor=False)
    assert out == [big]
    assert out[0].rt == 2.0


def test_dia_consensus_merges_all_into_one():
    # Two scans, same precursor, peaks in shared bins -> single consensus.
    a = _ps([100.0, 200.0], [10, 5], precursor_mz=300.0)
    b = _ps([100.0, 200.0], [8, 4], precursor_mz=300.0)
    out = reduce_ms2_spectra([a, b], 'consensus', group_by_precursor=False)

    assert len(out) == 1
    merged = out[0]
    # Consensus retains per-bin frequency (not flattened until display/print).
    assert merged.spectrum.dtype.names == ('mz', 'intsy', 'freq')
    mzs = np.sort(merged.spectrum['mz'])
    assert mzs == pytest.approx([100.0, 200.0])
    # Both bins appear in both scans -> relative frequency 1.0.
    assert merged.spectrum['freq'] == pytest.approx([1.0, 1.0])
    assert merged.precursor_mz == pytest.approx(300.0)


def test_dda_consensus_groups_by_precursor():
    # Two distinct precursors (200 and 500) -> two consensus spectra.
    p1a = _ps([100.0], [10], precursor_mz=200.0, rt=1.0)
    p1b = _ps([100.0], [8], precursor_mz=200.05, rt=2.0)
    p2 = _ps([150.0], [5], precursor_mz=500.0, rt=3.0)

    out = reduce_ms2_spectra(
        [p1a, p1b, p2], 'consensus',
        group_by_precursor=True, precursor_tol=0.5,
    )
    out = sorted(out, key=lambda s: s.precursor_mz)

    assert len(out) == 2
    # First group: the two ~200 precursors merged (median m/z).
    assert out[0].precursor_mz == pytest.approx(np.median([200.0, 200.05]))
    # Second group: the lone 500 precursor.
    assert out[1].precursor_mz == pytest.approx(500.0)


def test_dda_consensus_without_grouping_merges_everything():
    # Same inputs, but group_by_precursor=False -> a single consensus.
    p1 = _ps([100.0], [10], precursor_mz=200.0)
    p2 = _ps([150.0], [5], precursor_mz=500.0)
    out = reduce_ms2_spectra([p1, p2], 'consensus', group_by_precursor=False)
    assert len(out) == 1


def test_consensus_charge_is_modal():
    a = _ps([100.0], [10], precursor_mz=200.0, charge=2)
    b = _ps([100.0], [8], precursor_mz=200.1, charge=2)
    c = _ps([100.0], [8], precursor_mz=200.2, charge=1)
    out = reduce_ms2_spectra(
        [a, b, c], 'consensus', group_by_precursor=True, precursor_tol=0.5,
    )
    assert len(out) == 1
    assert out[0].charge == 2  # 2 appears twice, 1 once


def test_consensus_rt_is_tallest_contributor():
    quiet = _ps([100.0], [1], precursor_mz=200.0, rt=1.0)
    loud = _ps([100.0], [50], precursor_mz=200.1, rt=9.0)
    out = reduce_ms2_spectra(
        [quiet, loud], 'consensus', group_by_precursor=True,
    )
    assert len(out) == 1
    assert out[0].rt == 9.0


def test_unknown_mode_raises():
    specs = [_ps([100.0], [1])]
    with pytest.raises(ValueError):
        reduce_ms2_spectra(specs, 'bogus', group_by_precursor=False)
