"""
Hierarchical clustering of an EnsembleAlignment's samples by composition.

Each sample's fingerprint is its binary analyte presence (`sample in
analyte.ensemble_map`); pairs are scored by Jaccard (Tanimoto) similarity
and clustered like analytes are (`cluster_analytes._order_and_nodes`), with
the natural-sort rank of sample names standing in for RT as the leaf-order
tiebreak - so unrelated samples fall back to name order.

Given an analyte clustering, analytes can first be pooled into MS2 groups
(`ms2_groups`: subtrees of its dendrogram under a similarity cut), so samples
holding close analogues rather than identical analytes still score as
similar. The groups only feed this score; they are never shown as analyte
groups.

Samples can instead be clustered by an imported hierarchy (e.g. a taxonomy,
`cluster_samples_by_ranks`). `SampleClusterParams` picks the method;
`cluster_samples_by` dispatches on it (Alignment Viewer and export alike).

`SampleClustering` is also the shape any other sample ordering should
produce: leaf order + NODE_DTYPE merges with heights in [0, 1].
"""
from dataclasses import dataclass
from typing import NamedTuple, Optional, TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import pdist, squareform

from core.cli.cluster_analytes import _order_and_nodes
from core.utils.natural_sort import natural_sort_key

if TYPE_CHECKING:
    from configparser import ConfigParser
    from core.cli.cluster_analytes import AnalyteClustering
    from core.data_structs import AnalyteUUID, SampleUUID
    from core.data_structs.alignment import EnsembleAlignment


SAMPLE_CLUSTERING_SECTION = "sample_clustering"

COMPOSITION = 'composition'
LINEAGE = 'lineage'


class SampleClusterParams(NamedTuple):
    """
    method: COMPOSITION (Jaccard of analyte presence) or LINEAGE (an
        imported hierarchy, see `cluster_samples_by_ranks`).
    ms2_grouping: (composition) pool analytes into MS2 groups first, given
        an analyte clustering (`ms2_groups`).
    ms2_threshold: the similarity those groups are cut at.
    linkage_method: (composition) linkage; lineage distances are
        ultrametric, so any linkage gives the same tree.
    """
    method: str = COMPOSITION
    ms2_grouping: bool = True
    ms2_threshold: float = 0.7
    linkage_method: str = 'average'


def sample_cluster_params_from_config(config: 'ConfigParser') -> SampleClusterParams:
    """
    The ``[sample_clustering]`` section, with defaults for missing keys.
    `method` isn't stored: a lineage lives only for a session.
    """
    s = SAMPLE_CLUSTERING_SECTION
    d = SampleClusterParams()
    return SampleClusterParams(
        ms2_grouping=config.getboolean(s, "ms2_grouping", fallback=d.ms2_grouping),
        ms2_threshold=config.getfloat(s, "ms2_threshold", fallback=d.ms2_threshold),
        linkage_method=config.get(s, "linkage_method", fallback=d.linkage_method),
    )


def sample_cluster_params_to_config(
    config: 'ConfigParser', params: SampleClusterParams,
) -> None:
    """In memory; the caller persists via ``save_config``."""
    s = SAMPLE_CLUSTERING_SECTION
    if not config.has_section(s):
        config.add_section(s)
    for key, value in params._asdict().items():
        if key != 'method':
            config.set(s, key, str(value))


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


def ms2_groups(
    clustering: 'AnalyteClustering',
    threshold: float,
) -> dict['AnalyteUUID', int]:
    """
    Group label per clustered analyte: the largest subtree of
    `clustering`'s dendrogram merging at similarity >= `threshold`.
    """
    labels = np.arange(len(clustering.analyte_uuids))
    nodes = np.sort(clustering.nodes, order='height')
    # Leaf ranges are nested and parents are higher: the last write is the largest subtree
    for node in nodes[nodes['height'] <= 1.0 - threshold]:
        labels[node['lo']:node['hi'] + 1] = node['lo']
    return dict(zip(clustering.analyte_uuids, labels.tolist()))


def cluster_samples_by(
    alignment: 'EnsembleAlignment',
    names: dict['SampleUUID', str],
    params: SampleClusterParams,
    ranks: Optional[dict[str, tuple[str, ...]]] = None,
    analytes: Optional['AnalyteClustering'] = None,
) -> SampleClustering:
    """
    Cluster samples as `params` says. LINEAGE needs `ranks`; MS2 grouping
    applies only when `analytes` is given (else plain Jaccard).
    """
    if params.method == LINEAGE:
        if ranks is None:
            raise ValueError("Lineage clustering needs sample ranks")
        return cluster_samples_by_ranks(alignment, names, ranks)
    labels = (
        ms2_groups(analytes, params.ms2_threshold)
        if params.ms2_grouping and analytes is not None else None
    )
    return cluster_samples(alignment, names, params.linkage_method, labels)


def cluster_samples(
    alignment: 'EnsembleAlignment',
    names: dict['SampleUUID', str],
    linkage_method: str = 'average',
    analyte_labels: Optional[dict['AnalyteUUID', int]] = None,
) -> SampleClustering:
    """
    Cluster `alignment`'s samples by Jaccard similarity of analyte presence;
    with `analyte_labels` (`ms2_groups`), of group presence instead.
    Unlabelled analytes (e.g. without MS2) stay their own group.
    """
    uuids = list(alignment.sample_uuids)
    presence = presence_matrix(alignment)
    if analyte_labels is not None:
        # Unlabelled analytes get codes past every label, so never share one
        offset = max(analyte_labels.values(), default=-1) + 1
        codes = [
            analyte_labels.get(a.uuid, offset + i) for i, a in enumerate(alignment.analytes)
        ]
        _, column = np.unique(codes, return_inverse=True)
        onehot = np.zeros((len(codes), column.max(initial=-1) + 1), dtype=int)
        onehot[np.arange(len(codes)), column] = 1
        presence = (presence.astype(int) @ onehot) > 0
    if len(uuids) < 2:
        similarity = np.ones((len(uuids), len(uuids)))
    else:
        # Two samples with no analytes at all: scipy gives distance 0; call them unrelated
        distance = squareform(pdist(presence, 'jaccard'))
        empty = ~presence.any(axis=1)
        distance[np.ix_(empty, empty)] = 1.0
        similarity = 1.0 - distance
        np.fill_diagonal(similarity, 1.0)

    return _cluster(uuids, similarity, names, linkage_method)


def cluster_samples_by_ranks(
    alignment: 'EnsembleAlignment',
    names: dict['SampleUUID', str],
    ranks: dict[str, tuple[str, ...]],
    linkage_method: str = 'average',
) -> SampleClustering:
    """
    Cluster `alignment`'s samples by an external hierarchy (e.g. a taxonomy):
    `ranks` maps sample name -> labels coarse to fine. Similarity is the
    fraction of leading ranks two samples share; an empty label ends the
    shared run, and samples missing from `ranks` are unrelated to all.
    Distances are ultrametric, so any linkage reproduces the hierarchy.
    """
    uuids = list(alignment.sample_uuids)
    depth = max((len(r) for r in ranks.values()), default=0)
    labels = [ranks.get(names[u], ()) for u in uuids]
    similarity = np.eye(len(uuids))
    for i in range(len(uuids)):
        for j in range(i):
            shared = 0
            for a, b in zip(labels[i], labels[j]):
                if not a or a != b:
                    break
                shared += 1
            similarity[i, j] = similarity[j, i] = shared / depth
    return _cluster(uuids, similarity, names, linkage_method)


def _cluster(uuids, similarity, names, linkage_method) -> SampleClustering:
    """Leaf-order `similarity`, with natural-sorted names as the tiebreak."""
    by_name = sorted(range(len(uuids)), key=lambda i: natural_sort_key(names[uuids[i]]))
    rank = np.empty(len(uuids))
    rank[by_name] = np.arange(len(uuids))

    order, nodes = _order_and_nodes(similarity, rank, linkage_method)
    return SampleClustering(
        sample_uuids=[uuids[i] for i in order],
        similarity=similarity[np.ix_(order, order)],
        nodes=nodes,
    )
