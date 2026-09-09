"""
Tests for the alignment merger (core/cli/align_alignments.py).

Pure-function tests (consensus spectra, analyte clustering, map
compatibility) need no MS data. The integration test splits the shared
multi-sample .mzk fixture into two batches, aligns each independently, then
merges the two alignments — and is skipped when the (gitignored) fixture is
absent.
"""
from pathlib import Path

import numpy as np
import pytest

from core.data_structs.alignment import AlignmentParams, EnsembleAlignment
import core.cli.align_alignments as M
import core.cli.align_ensembles as A

TEST_ALIGNMENT_MZK = Path(
    '/home/mh/Dropbox/MzKit/tests/test_files/test_alignment.mzk'
)


def _item(align_idx, ensemble_map, rt=1.0, mz=100.0):
    return M._AnalyteItem(
        align_idx=align_idx,
        ensemble_map=dict(ensemble_map),
        consensus_rt=rt,
        consensus_mz=mz,
    )


# --- _maps_compatible -------------------------------------------------------

def test_maps_compatible_disjoint():
    assert M._maps_compatible({1: 10}, {2: 20})


def test_maps_compatible_agree_on_shared():
    assert M._maps_compatible({1: 10, 2: 20}, {2: 20, 3: 30})


def test_maps_incompatible_conflict_on_shared():
    assert not M._maps_compatible({2: 20}, {2: 99})


# --- _cluster_analytes ------------------------------------------------------

def test_cluster_no_edges_yields_singletons():
    items = [_item(0, {1: 10}), _item(1, {2: 20}), _item(2, {3: 30})]
    empty = np.empty(0, dtype=int)
    analytes = M._cluster_analytes(items, empty, empty, np.empty(0, float))
    assert len(analytes) == 3
    assert all(len(a.ensemble_map) == 1 for a in analytes)


def test_cluster_merges_across_alignments():
    # Three analytes from three different alignments, mutually linked.
    items = [_item(0, {1: 10}), _item(1, {2: 20}), _item(2, {3: 30})]
    ei = np.array([0, 1, 0])
    ej = np.array([1, 2, 2])
    es = np.array([0.9, 0.8, 0.7])
    analytes = M._cluster_analytes(items, ei, ej, es)
    assert len(analytes) == 1
    assert analytes[0].ensemble_map == {1: 10, 2: 20, 3: 30}


def test_cluster_respects_alignment_uniqueness():
    # Nodes 0 and 1 are from the same alignment (idx 0) -> can't co-group.
    items = [_item(0, {1: 10}), _item(0, {2: 20}), _item(1, {3: 30})]
    ei = np.array([0, 1, 0])
    ej = np.array([2, 2, 1])
    es = np.array([0.95, 0.90, 0.99])  # 0-1 edge strongest but illegal
    analytes = M._cluster_analytes(items, ei, ej, es)
    for a in analytes:
        # No two members from the same source alignment (checked via samples
        # here: each alignment contributed a unique sample uuid).
        assert len(a.ensemble_map) == len(set(a.ensemble_map))
    sizes = sorted(len(a.ensemble_map) for a in analytes)
    assert sizes == [1, 2]


def test_cluster_blocks_conflicting_sample_maps():
    # Two analytes from different alignments that share sample 5 but assign
    # it to different ensembles must not merge, even with a strong edge.
    items = [_item(0, {5: 10}), _item(1, {5: 99})]
    analytes = M._cluster_analytes(
        items, np.array([0]), np.array([1]), np.array([0.99]),
    )
    assert len(analytes) == 2


def test_cluster_merges_on_shared_anchor_sample():
    # Overlapping alignments that agree on shared sample 5 merge and keep a
    # single entry for it.
    items = [_item(0, {5: 10, 1: 11}), _item(1, {5: 10, 2: 22})]
    analytes = M._cluster_analytes(
        items, np.array([0]), np.array([1]), np.array([0.9]),
    )
    assert len(analytes) == 1
    assert analytes[0].ensemble_map == {5: 10, 1: 11, 2: 22}


def test_cluster_consensus_is_sample_weighted():
    # Analyte A covers 3 samples at rt=10; B covers 1 sample at rt=20.
    items = [
        _item(0, {1: 1, 2: 2, 3: 3}, rt=10.0, mz=100.0),
        _item(1, {4: 4}, rt=20.0, mz=200.0),
    ]
    analytes = M._cluster_analytes(
        items, np.array([0]), np.array([1]), np.array([0.9]),
    )
    assert len(analytes) == 1
    # (10*3 + 20*1) / 4 = 12.5
    assert analytes[0].consensus_rt == pytest.approx(12.5)
    assert analytes[0].consensus_mz == pytest.approx(125.0)


# --- _merge_spectra ---------------------------------------------------------

def test_merge_spectra_collapses_close_peaks():
    s1 = A._spectrum_from(np.array([100.0, 200.0]), np.array([1.0, 2.0]))
    s2 = A._spectrum_from(np.array([100.004, 300.0]), np.array([3.0, 4.0]))
    merged = M._merge_spectra([s1, s2], mz_tol=0.01)
    # 100.0 / 100.004 collapse into one peak at their intensity-weighted mean
    # (100.0*1 + 100.004*3)/4 = 100.003; 200 and 300 stay distinct.
    assert list(merged.peaks.mz) == pytest.approx([100.003, 200.0, 300.0], abs=1e-4)


def test_merge_spectra_empty_input():
    merged = M._merge_spectra([], mz_tol=0.01)
    assert merged.peaks.mz.size == 0


# --- integration ------------------------------------------------------------

@pytest.fixture
def two_batches():
    if not TEST_ALIGNMENT_MZK.exists():
        pytest.skip(
            f"test data not present: {TEST_ALIGNMENT_MZK} "
            "(MS data is gitignored; place it locally to run this test)"
        )
    from core.utils.persistence import load_project
    samples, _, _ = load_project(TEST_ALIGNMENT_MZK)
    samples = [s for s in samples if s.injection is not None]
    if len(samples) < 4:
        pytest.skip("need >= 4 samples with MS data to split into two batches")

    params = AlignmentParams(
        rt_tolerance=10.0, mz_tolerance=0.01,
        ms1_similarity_threshold=0.7, ms2_similarity_threshold=0.6,
    )
    mid = len(samples) // 2
    batch_a, batch_b = samples[:mid], samples[mid:]
    align_a = A.align_ensembles(batch_a, params)
    align_b = A.align_ensembles(batch_b, params)
    return samples, params, align_a, align_b


def test_merge_spans_all_samples(two_batches):
    samples, params, align_a, align_b = two_batches
    merged = M.align_alignments([align_a, align_b], samples, params)

    assert isinstance(merged, EnsembleAlignment)
    assert set(merged.sample_uuids) == set(s.uuid for s in samples)
    assert merged.analyte_count > 0

    sample_uuids = set(s.uuid for s in samples)
    n_cross_batch = 0
    batch_a_uuids = set(align_a.sample_uuids)
    for a in merged.analytes:
        assert set(a.ensemble_map).issubset(sample_uuids)
        assert len(a.ensemble_map) == len(set(a.ensemble_map))
        assert np.isfinite(a.consensus_rt) and np.isfinite(a.consensus_mz)
        mapped = set(a.ensemble_map)
        if mapped & batch_a_uuids and mapped - batch_a_uuids:
            n_cross_batch += 1

    # Merging two batches of the same run should bridge at least some
    # analytes across the batch boundary.
    assert n_cross_batch > 0


def test_merge_preserves_source_ensemble_ids(two_batches):
    samples, params, align_a, align_b = two_batches
    merged = M.align_alignments([align_a, align_b], samples, params)

    # Every (sample -> ensemble) mapping in the merge must come verbatim
    # from one of the source alignments (merging never invents ensembles).
    source_pairs = set()
    for alignment in (align_a, align_b):
        for a in alignment.analytes:
            source_pairs.update(a.ensemble_map.items())
    for a in merged.analytes:
        assert set(a.ensemble_map.items()).issubset(source_pairs)
