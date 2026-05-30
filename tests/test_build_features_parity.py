"""
Parity test for the refactored ``build_features``.

Compares the new dense parallel-array implementation against the preserved
legacy implementation (``_build_features_legacy``) on synthetic
spectra. No external files required.
"""
import numpy as np
import pytest

from core.data_structs.scan_array import (
    build_features,
    _build_features_legacy,
)


class MockSpectrum:
    """
    Duck-typed minimal stand-in for ``pyopenms.MSSpectrum``.

    Only implements the two methods the feature builder calls
    (``get_peaks()`` and ``getRT()``).
    """
    def __init__(self, mz, intsy, rt):
        self._mz = np.ascontiguousarray(mz, dtype=np.float64)
        self._intsy = np.ascontiguousarray(intsy, dtype=np.float64)
        self._rt = float(rt)

    def get_peaks(self):
        return self._mz, self._intsy

    def getRT(self) -> float:
        return self._rt


def _legacy_to_dense(features, n_scans):
    """Stack a list of legacy ``Feature`` objects into (n_features, n_scans) arrays."""
    if not features:
        return (np.zeros((0, n_scans), dtype=np.float64),
                np.zeros((0, n_scans), dtype=np.float64))
    out_mz = np.zeros((len(features), n_scans), dtype=np.float64)
    out_intsy = np.zeros((len(features), n_scans), dtype=np.float64)
    for i, ftr in enumerate(features):
        out_mz[i] = ftr.array['mz']
        out_intsy[i] = ftr.array['intsy']
    return out_mz, out_intsy


def _make_synthetic_spectra(
    n_scans: int = 60,
    seed: int = 42,
    n_true_features: int = 8,
    n_noise_per_scan: tuple[int, int] = (3, 12),
):
    """
    Build a deterministic stack of synthetic spectra.

    Each "true feature" is a Gaussian RT envelope centered at a random scan,
    with small per-scan m/z jitter and Gaussian intensity noise. Plus
    per-scan random noise peaks. mz values are spaced widely enough to avoid
    accidental exact ties (which would expose the only known divergence
    between legacy and the new impl).
    """
    rng = np.random.default_rng(seed)

    # True feature centers, well-separated
    true_mzs = np.sort(rng.uniform(100.0, 800.0, size=n_true_features))
    true_centers = rng.integers(5, n_scans - 5, size=n_true_features)
    true_widths = rng.uniform(2.0, 8.0, size=n_true_features)
    true_peak_intsy = rng.uniform(3000, 20000, size=n_true_features)

    spectra = []
    for s in range(n_scans):
        mzs = []
        intsys = []

        # True features
        for fmz, fc, fw, fpi in zip(true_mzs, true_centers, true_widths, true_peak_intsy):
            envelope = np.exp(-0.5 * ((s - fc) / fw) ** 2)
            intsy = envelope * fpi + rng.normal(0, 20)
            if intsy > 100:
                mz_jitter = rng.normal(0, 0.005)
                mzs.append(float(fmz + mz_jitter))
                intsys.append(float(intsy))

        # Noise peaks — well-separated from true features and from each other.
        n_noise = int(rng.integers(*n_noise_per_scan))
        # Sample noise mzs from regions away from true_mzs
        for _ in range(n_noise):
            # Reject samples near existing peaks in this scan to avoid
            # adjacency that would stress tie-breaking corner cases.
            for _try in range(20):
                cand_mz = float(rng.uniform(80.0, 850.0))
                cand_intsy = float(rng.uniform(50, 500))
                if all(abs(cand_mz - m) > 0.2 for m in mzs):
                    mzs.append(cand_mz)
                    intsys.append(cand_intsy)
                    break

        spectra.append(MockSpectrum(mzs, intsys, rt=s * 0.5))

    return spectra


def _normalize_for_compare(out_mz, out_intsy):
    """
    Sort features (rows) by (mean nonzero mz, total intensity) so that the
    two implementations — which can produce features in slightly different
    orders when mean-mz values tie — compare cleanly.
    """
    mask = out_intsy > 0
    counts = mask.sum(axis=1)
    sums = np.where(mask, out_mz, 0.0).sum(axis=1)
    # Avoid div by zero for any all-zero rows (shouldn't happen, but safe).
    mean_mz = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    total_intsy = out_intsy.sum(axis=1)
    # Lexsort: primary key = mean_mz, secondary = total_intsy
    order = np.lexsort((total_intsy, mean_mz))
    return out_mz[order], out_intsy[order]


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_build_features_parity(seed):
    """New implementation matches legacy on synthetic spectra across seeds."""
    spectra = _make_synthetic_spectra(n_scans=60, seed=seed)
    mz_tolerance = 0.05
    scan_gap_tolerance = 3
    min_intsy = 200.0

    # Legacy
    legacy_features = _build_features_legacy(
        spectra=spectra,
        mz_tolerance=mz_tolerance,
        scan_gap_tolerance=scan_gap_tolerance,
        min_intsy=min_intsy,
    )
    legacy_mz, legacy_intsy = _legacy_to_dense(legacy_features, len(spectra))

    # New
    new_mz, new_intsy, rt_per_scan = build_features(
        spectra=spectra,
        mz_tolerance=mz_tolerance,
        scan_gap_tolerance=scan_gap_tolerance,
        min_intsy=min_intsy,
    )

    # Same number of features
    assert new_mz.shape == legacy_mz.shape, (
        f"feature count mismatch: legacy={legacy_mz.shape[0]}, "
        f"new={new_mz.shape[0]}"
    )

    # Same per-scan RT
    assert rt_per_scan.shape == (len(spectra),)
    assert np.allclose(rt_per_scan, [sp.getRT() for sp in spectra])

    # Compare feature contents after normalizing row order.
    legacy_mz_n, legacy_intsy_n = _normalize_for_compare(legacy_mz, legacy_intsy)
    new_mz_n, new_intsy_n = _normalize_for_compare(new_mz, new_intsy)

    assert np.allclose(legacy_mz_n, new_mz_n), (
        "mz arrays differ"
    )
    assert np.allclose(legacy_intsy_n, new_intsy_n), (
        "intensity arrays differ"
    )


def test_build_features_smoke_empty_scans():
    """A scan with <2 peaks-above-threshold should be silently skipped."""
    spectra = [
        MockSpectrum([100.0, 200.0, 300.0], [5000.0, 5000.0, 5000.0], rt=0.0),
        MockSpectrum([], [], rt=0.5),  # empty
        MockSpectrum([100.01], [5000.0], rt=1.0),  # only one peak → skipped
        MockSpectrum([100.02, 200.02, 300.02], [4500.0, 4500.0, 4500.0], rt=1.5),
    ]
    new_mz, new_intsy, rt_per_scan = build_features(
        spectra=spectra,
        mz_tolerance=0.05,
        scan_gap_tolerance=3,
        min_intsy=100.0,
    )
    # 3 features expected, each persisting across scans 0 and 3.
    assert new_mz.shape == (3, 4)
    assert np.allclose(rt_per_scan, [0.0, 0.5, 1.0, 1.5])
    # Scans 1 and 2 should have all zeros (skipped, no signal placed).
    assert np.all(new_intsy[:, 1] == 0)
    assert np.all(new_intsy[:, 2] == 0)
    # Scans 0 and 3 should have nonzero intensity for all 3 features.
    assert np.all(new_intsy[:, 0] > 0)
    assert np.all(new_intsy[:, 3] > 0)
