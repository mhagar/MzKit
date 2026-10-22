"""
Lanes are ordered by apex m/z (the m/z at each lane's most intense scan),
identically by the numba and pure-Python feature kernels.
"""
import numpy as np

from core.data_structs import scan_array as sa


class _Spectrum:
    def __init__(self, mz, intsy, rt):
        self._mz = np.asarray(mz, dtype=np.float64)
        self._intsy = np.asarray(intsy, dtype=np.float64)
        self._rt = rt

    def get_peaks(self):
        return self._mz, self._intsy

    def getRT(self):
        return self._rt


def _spectra():
    # Lane A drifts 300.00 -> 300.045 and is tallest at 300.045; lane B sits
    # at 300.02, shorter, so A matches first and keeps its apex. Mean m/z
    # (A 300.018) would put A first; apex m/z puts B first. The 500 peak
    # keeps every scan >= 2 peaks (the kernel skips sparser scans).
    scans = [
        ([300.00, 300.02, 500.0], [1000, 900, 10]),
        ([300.01, 300.02, 500.0], [1000, 900, 10]),
        ([300.045, 500.0], [5000, 10]),
    ]
    return [_Spectrum(mz, i, rt=float(n)) for n, (mz, i) in enumerate(scans)]


def _build(monkeypatch, numba: bool):
    if not numba:
        monkeypatch.setenv('MZKIT_DISABLE_NUMBA', '1')
    mz, intsy, _ = sa.build_features(
        _spectra(), mz_tolerance=0.05, scan_gap_tolerance=1, min_intsy=0,
    )
    return mz.toarray(), intsy.toarray()


def test_lanes_sorted_by_apex_mz(monkeypatch):
    mz, intsy = _build(monkeypatch, numba=sa._NUMBA_KERNEL_AVAILABLE)
    assert mz.shape[0] == 3                     # A stayed one lane
    apex = mz[np.arange(3), intsy.argmax(axis=1)]
    np.testing.assert_allclose(apex, [300.02, 300.045, 500.0])


def test_kernels_agree(monkeypatch):
    if not sa._NUMBA_KERNEL_AVAILABLE:
        return
    fast = _build(monkeypatch, numba=True)
    slow = _build(monkeypatch, numba=False)
    for a, b in zip(fast, slow):
        np.testing.assert_array_equal(a, b)
