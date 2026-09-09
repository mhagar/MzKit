"""
Tests for the cross-sample ensemble aligner (core/cli/align_ensembles.py).

The pure-function tests (clustering, spectrum construction) need no MS data.
The integration test runs the full aligner on a multi-sample .mzk fixture and
is skipped when that (gitignored) file is absent.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from core.data_structs.alignment import AlignmentParams, EnsembleAlignment
import core.cli.align_ensembles as A

TEST_ALIGNMENT_MZK = Path(
    '/home/mh/Dropbox/MzKit/tests/test_files/test_alignment.mzk'
)


def _bundle(sample_idxs):
    """Minimal _SpectraBundle with just the fields _cluster reads."""
    n = len(sample_idxs)
    b = A._SpectraBundle()
    b.sample_uuids = [1000 + k for k in range(n)]      # unique per node
    b.ens_uuids = [2000 + k for k in range(n)]
    b.sample_idxs = np.asarray(sample_idxs, dtype=int)
    b.rts = np.arange(n, dtype=float)
    b.mzs = np.arange(n, dtype=float) + 100.0
    return b


# --- _cluster ---------------------------------------------------------------

def test_cluster_no_edges_yields_singletons():
    b = _bundle([0, 1, 2])
    empty = np.empty(0, dtype=int)
    analytes = A._cluster(b, empty, empty, np.empty(0, dtype=float))
    assert len(analytes) == 3
    assert all(len(a.ensemble_map) == 1 for a in analytes)


def test_cluster_merges_across_samples():
    # 3 ensembles in 3 different samples, all mutually linked.
    b = _bundle([0, 1, 2])
    ei = np.array([0, 1, 0])
    ej = np.array([1, 2, 2])
    es = np.array([0.9, 0.8, 0.7])
    analytes = A._cluster(b, ei, ej, es)
    assert len(analytes) == 1
    assert len(analytes[0].ensemble_map) == 3


def test_cluster_respects_sample_uniqueness():
    # Nodes 0,1 are the same sample (idx 0); a group must not hold both.
    b = _bundle([0, 0, 1])
    ei = np.array([0, 1, 0])
    ej = np.array([2, 2, 1])
    es = np.array([0.95, 0.90, 0.99])  # 0-1 edge is strongest but illegal
    analytes = A._cluster(b, ei, ej, es)
    for a in analytes:
        idxs = [b.sample_idxs[b.sample_uuids.index(su)]
                for su in a.ensemble_map]
        assert len(idxs) == len(set(idxs)), "a sample appears twice in a group"
    # Node 2 pairs with whichever of 0/1 it linked to first (strongest legal).
    sizes = sorted(len(a.ensemble_map) for a in analytes)
    assert sizes == [1, 2]


def test_cluster_consensus_values():
    b = _bundle([0, 1])
    analytes = A._cluster(
        b, np.array([0]), np.array([1]), np.array([0.8]),
    )
    assert len(analytes) == 1
    a = analytes[0]
    assert a.consensus_rt == pytest.approx(np.mean(b.rts))
    assert a.consensus_mz == pytest.approx(np.mean(b.mzs))


# --- _spectrum_from ---------------------------------------------------------

def test_spectrum_from_sorts_and_drops_zeros():
    mz = np.array([200.0, 100.0, 150.0])
    intsy = np.array([1.0, 0.0, 0.5])  # the 100.0 peak has zero intensity
    spec = A._spectrum_from(mz, intsy)
    assert list(spec.peaks.mz) == [150.0, 200.0]        # sorted, zero dropped
    assert list(spec.peaks.intensities) == [0.5, 1.0]


# --- integration ------------------------------------------------------------

@pytest.fixture
def aligned():
    if not TEST_ALIGNMENT_MZK.exists():
        pytest.skip(
            f"test data not present: {TEST_ALIGNMENT_MZK} "
            "(MS data is gitignored; place it locally to run this test)"
        )
    from core.utils.persistence import load_project
    samples, _, _ = load_project(TEST_ALIGNMENT_MZK)
    params = AlignmentParams(
        rt_tolerance=10.0, mz_tolerance=0.01,
        ms1_similarity_threshold=0.7, ms2_similarity_threshold=0.6,
    )
    return samples, A.align_ensembles(samples, params)


def test_align_ensembles_integration(aligned):
    samples, alignment = aligned
    assert isinstance(alignment, EnsembleAlignment)
    assert alignment.sample_uuids == tuple(s.uuid for s in samples)
    assert alignment.analyte_count > 0

    sample_uuids = set(s.uuid for s in samples)
    n_multi = 0
    for a in alignment.analytes:
        # Every mapped sample is real and appears at most once.
        assert set(a.ensemble_map).issubset(sample_uuids)
        assert len(a.ensemble_map) == len(set(a.ensemble_map))
        assert np.isfinite(a.consensus_rt) and np.isfinite(a.consensus_mz)
        if len(a.ensemble_map) >= 2:
            n_multi += 1

    # A real 5-sample alignment should find plenty of multi-sample analytes,
    # not collapse to one giant group or explode into all singletons.
    assert n_multi > 0
    sizes = Counter(len(a.ensemble_map) for a in alignment.analytes)
    assert max(sizes) <= len(samples)
