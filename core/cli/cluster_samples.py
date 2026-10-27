"""
Hierarchical clustering of an EnsembleAlignment's samples by composition.

Each sample's fingerprint is its binary analyte presence (`sample in
analyte.ensemble_map`); pairs are scored by Jaccard (Tanimoto) similarity
and clustered like analytes are (`cluster_analytes._order_and_nodes`), with
the natural-sort rank of sample names standing in for RT as the leaf-order
tiebreak - so unrelated samples fall back to name order.

`SampleClustering` is also the shape any other sample ordering (e.g. an
imported taxonomy tree) should produce: leaf order + NODE_DTYPE merges with
heights in [0, 1].
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import pdist, squareform

from core.cli.cluster_analytes import _order_and_nodes
from core.utils.natural_sort import natural_sort_key

if TYPE_CHECKING:
    from core.data_structs import SampleUUID
    from core.data_structs.alignment import EnsembleAlignment


@dataclass(frozen=True)
class SampleClustering:
    """
    sample_uuids: samples in leaf order (lane index = index).
    similarity: NxN similarity, in leaf order.
    nodes: dendrogram merges below distance 1.0 (cluster_analytes.NODE_DTYPE).
    """
    sample_uuids: list['SampleUUID']
    similarity: np.ndarray
    nodes: np.ndarray


def presence_matrix(alignment: 'EnsembleAlignment') -> np.ndarray:
    """Samples x analytes boolean presence, in `alignment.sample_uuids` order."""
    return np.array([
        [uuid in a.ensemble_map for a in alignment.analytes]
        for uuid in alignment.sample_uuids
    ], dtype=bool).reshape(alignment.sample_count, alignment.analyte_count)


def cluster_samples(
    alignment: 'EnsembleAlignment',
    names: dict['SampleUUID', str],
    linkage_method: str = 'average',
) -> SampleClustering:
    """Cluster `alignment`'s samples by Jaccard similarity of analyte presence."""
    uuids = list(alignment.sample_uuids)
    presence = presence_matrix(alignment)
    if len(uuids) < 2:
        similarity = np.ones((len(uuids), len(uuids)))
    else:
        # Two samples with no analytes at all: scipy gives distance 0; call them unrelated
        distance = squareform(pdist(presence, 'jaccard'))
        empty = ~presence.any(axis=1)
        distance[np.ix_(empty, empty)] = 1.0
        similarity = 1.0 - distance
        np.fill_diagonal(similarity, 1.0)

    by_name = sorted(range(len(uuids)), key=lambda i: natural_sort_key(names[uuids[i]]))
    rank = np.empty(len(uuids))
    rank[by_name] = np.arange(len(uuids))

    order, nodes = _order_and_nodes(similarity, rank, linkage_method)
    return SampleClustering(
        sample_uuids=[uuids[i] for i in order],
        similarity=similarity[np.ix_(order, order)],
        nodes=nodes,
    )
