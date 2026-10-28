"""
Tests for compositional (Jaccard) clustering of samples
(core/cli/cluster_samples.py).
"""
import numpy as np

from core.cli.cluster_analytes import NODE_DTYPE, AnalyteClustering
from configparser import ConfigParser

from core.cli.cluster_samples import (
    LINEAGE, SampleClusterParams, cluster_samples, cluster_samples_by,
    cluster_samples_by_ranks, ms2_groups, sample_cluster_params_from_config,
    sample_cluster_params_to_config,
)
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


def test_ranks_reproduce_hierarchy():
    alignment, names = _build({1: [0], 2: [0], 3: [0], 4: [0], 5: [0]})
    ranks = {
        'S1': ('Fam', 'GenA', 'org1'),
        'S2': ('Fam', 'GenB', 'org2'),
        'S3': ('Fam', 'GenA', 'org1'),   # second fraction of org1
        'S4': ('Fam', 'GenA', 'org4'),
        'S5': ('', '', 'org5'),          # unknown taxonomy: unrelated
    }
    result = cluster_samples_by_ranks(alignment, names, ranks)
    order = result.sample_uuids
    assert order.index(5) == 4
    # fractions are adjacent, then genus, then family
    assert abs(order.index(1) - order.index(3)) == 1
    genus = sorted(order.index(u) for u in (1, 3, 4))
    assert genus[2] - genus[0] == 2 and set(order[:4]) == {1, 2, 3, 4}
    assert np.allclose(sorted(result.nodes['height']), [0, 1 / 3, 2 / 3])


def _analyte_clustering(uuids, nodes):
    """nodes: (lo, hi, height) merges, in leaf slots of `uuids`."""
    arr = np.zeros(len(nodes), dtype=NODE_DTYPE)
    for k, (lo, hi, h) in enumerate(nodes):
        arr[k]['lo'], arr[k]['hi'], arr[k]['height'] = lo, hi, h
    return AnalyteClustering(
        analyte_uuids=list(uuids), similarity=np.eye(len(uuids)), nodes=arr,
        n_dropped=0, params=None,
    )


def test_ms2_groups_cut():
    # leaves 0,1 merge at similarity 0.9; with leaf 2 at 0.5
    clustering = _analyte_clustering([10, 11, 12], [(0, 1, 0.1), (0, 2, 0.5)])
    assert ms2_groups(clustering, 0.7) == {10: 0, 11: 0, 12: 2}
    assert len(set(ms2_groups(clustering, 0.4).values())) == 1
    assert len(set(ms2_groups(clustering, 0.95).values())) == 3


def test_ms2_grouped_jaccard():
    # A has analytes 0, 1; B has 2. Analytes 0 and 2 are MS2 analogues; 1 has no MS2
    alignment, names = _build({1: [0, 1], 2: [2]})
    uuid_of = [a.uuid for a in alignment.analytes]
    clustering = _analyte_clustering([uuid_of[0], uuid_of[2]], [(0, 1, 0.15)])
    labels = ms2_groups(clustering, 0.7)

    def jaccard(result):
        i, j = result.sample_uuids.index(1), result.sample_uuids.index(2)
        return result.similarity[i, j]

    assert np.isclose(jaccard(cluster_samples(alignment, names)), 0.0)
    assert np.isclose(jaccard(cluster_samples(alignment, names, analyte_labels=labels)), 0.5)


def test_cluster_samples_by_dispatches():
    alignment, names = _build({1: [0, 1], 2: [2]})
    uuid_of = [a.uuid for a in alignment.analytes]
    clustering = _analyte_clustering([uuid_of[0], uuid_of[2]], [(0, 1, 0.15)])

    def similarity(params, **kwargs):
        result = cluster_samples_by(alignment, names, params, **kwargs)
        return result.similarity[0, 1]

    grouped = SampleClusterParams()
    assert np.isclose(similarity(grouped, analytes=clustering), 0.5)
    assert np.isclose(similarity(grouped), 0.0)   # no analyte clustering: plain
    assert np.isclose(
        similarity(grouped._replace(ms2_grouping=False), analytes=clustering), 0.0,
    )
    lineage = SampleClusterParams(method=LINEAGE)
    ranks = {'S1': ('Fam', 'GenA'), 'S2': ('Fam', 'GenB')}
    assert np.isclose(similarity(lineage, ranks=ranks), 0.5)


def test_sample_config_round_trip_omits_method():
    config = ConfigParser()
    params = SampleClusterParams(
        method=LINEAGE, ms2_grouping=False, ms2_threshold=0.55, linkage_method='complete',
    )
    sample_cluster_params_to_config(config, params)
    # A lineage is per session: the method always comes back as the default
    assert sample_cluster_params_from_config(config) == params._replace(
        method=SampleClusterParams().method,
    )
