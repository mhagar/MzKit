"""
Tests for compositional (Jaccard) clustering of samples
(core/cli/cluster_samples.py).
"""
import numpy as np

from core.cli.cluster_samples import cluster_samples
from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment


def _build(presence: dict[int, list[int]]):
    """presence: sample uuid -> indices of the analytes it contains."""
    n = max(i for idxs in presence.values() for i in idxs) + 1
    analytes = [
        AlignedAnalyte(
            ensemble_map={s: 100 * s + i for s, idxs in presence.items() if i in idxs},
            consensus_rt=float(i),
        )
        for i in range(n)
    ]
    alignment = EnsembleAlignment(sample_uuids=tuple(presence), analytes=analytes)
    names = {s: f"S{s}" for s in presence}
    return alignment, names


def test_similar_samples_are_contiguous():
    # 1 & 3 share analytes 0-2, 2 & 4 share 3-5: names interleave, clusters don't
    alignment, names = _build({
        1: [0, 1, 2], 2: [3, 4, 5], 3: [0, 1, 2, 6], 4: [3, 4, 5],
    })
    result = cluster_samples(alignment, names)
    order = result.sample_uuids
    assert {order[0], order[1]} in ({1, 3}, {2, 4})
    assert {order[2], order[3]} in ({1, 3}, {2, 4})
    i, j = order.index(1), order.index(3)
    assert np.isclose(result.similarity[i, j], 3 / 4)   # |{0,1,2}| / |{0,1,2,6}|
    assert len(result.nodes) == 2   # 1-3 and 2-4; the two pairs are unrelated


def test_unrelated_samples_fall_back_to_name_order():
    alignment, names = _build({10: [0], 2: [1], 1: [2]})
    result = cluster_samples(alignment, names)
    assert result.sample_uuids == [1, 2, 10]
    assert len(result.nodes) == 0
