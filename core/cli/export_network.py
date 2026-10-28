"""
Export an EnsembleAlignment as a molecular network (.graphml, e.g. for
Cytoscape), GNPS-FBMN style.

Nodes are analytes (all of them; those not in the matrix, e.g. without a
representative MS2, are singletons). Edges are pairwise similarity from a
`SimilarityMatrix` - computed (MS2DeepScore by default, or modified cosine;
`core.cli.cluster_analytes.compute_similarity_matrix`) or imported
(`core.cli.import_similarity`) - under the same transform the Alignment
Viewer clusters by. Only pairs scoring at least `min_edge_score` get an edge.

Node ids are the analyte row ids used by the feature tables / MGF
(`analyte_id` / `FEATURE_ID`).
"""
from typing import Mapping, Optional, TYPE_CHECKING
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from core.cli.import_similarity import transform

if TYPE_CHECKING:
    from core.data_structs.similarity_matrix import SimilarityMatrix
    from core.data_structs import Sample
    from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment
    from core.data_structs.uuid_types import AnalyteUUID, EnsembleUUID


# (attribute name, GraphML attr.type), declared as <key>s
_NODE_KEYS = [
    ('analyte_id', 'int'),
    ('consensus_mz', 'double'),
    ('consensus_rt', 'double'),
    ('consensus_formula', 'string'),
    ('formula_count', 'int'),
    ('formula_total', 'int'),
    ('n_samples', 'int'),
    ('group', 'string'),
]
_EDGE_KEYS = [
    ('similarity', 'double'),
    ('mz_delta', 'double'),
]


def build_matrix_network_graphml(
    alignment: 'EnsembleAlignment',
    samples: list['Sample'],
    matrix: 'SimilarityMatrix',
    formulas: Optional[Mapping['EnsembleUUID', str]] = None,
    groups: Optional[Mapping['AnalyteUUID', str]] = None,
    min_edge_score: float = 0.7,
) -> str:
    """
    GraphML text for `alignment`'s network by `matrix`, transformed under
    its `default_transform` (as clustered in the Alignment Viewer).

    :param formulas: accepted formula per ensemble, for the
        consensus_formula node attributes.
    :param groups: group titles per analyte, for the group node attribute.
    :param min_edge_score: drop edges scoring below this (GNPS default
        0.7), or below the transform's floor if that is higher.
    """
    analytes, similarity, _report = transform(
        matrix, alignment, matrix.default_transform,
    )
    edges = [
        (analytes[a], analytes[b], {'similarity': f"{similarity[a, b]:.4f}"})
        for a, b in _pairs_above(similarity, min_edge_score)
    ]
    return _graphml(
        alignment, samples, formulas, groups,
        'in_matrix' if matrix.is_imported else 'has_ms2',
        {a.uuid for a in analytes}, _EDGE_KEYS, edges,
    )


def _pairs_above(similarity: np.ndarray, threshold: float):
    """Upper-triangle (a, b) index pairs scoring > 0 and >= `threshold`."""
    a_idx, b_idx = np.triu_indices(len(similarity), k=1)
    scores = similarity[a_idx, b_idx]
    keep = (scores > 0) & (scores >= threshold)
    return zip(a_idx[keep].tolist(), b_idx[keep].tolist())


def _graphml(
    alignment: 'EnsembleAlignment',
    samples: list['Sample'],
    formulas: Optional[Mapping['EnsembleUUID', str]],
    groups: Optional[Mapping['AnalyteUUID', str]],
    flag_key: str,
    flagged: set,
    edge_keys: list[tuple[str, str]],
    edges: list[tuple['AlignedAnalyte', 'AlignedAnalyte', dict]],
) -> str:
    """
    GraphML with a node per analyte (`flag_key` true for those in
    `flagged`) and the given (source, target, data) edges; each edge also
    gets `mz_delta`.
    """
    formulas = formulas or {}
    groups = groups or {}
    sample_of = {s.uuid: s for s in samples}
    row_of = {a.uuid: i for i, a in enumerate(alignment.analytes)}

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
    ]
    node_keys = _NODE_KEYS + [(flag_key, 'boolean')]
    for owner, keys in (('node', node_keys), ('edge', edge_keys)):
        for name, attr_type in keys:
            lines.append(
                f'  <key id="{name}" for="{owner}" '
                f'attr.name="{name}" attr.type="{attr_type}"/>'
            )
    lines.append('  <graph id="G" edgedefault="undirected">')

    for i, analyte in enumerate(alignment.analytes):
        members = analyte.resolve_members(sample_of.get)
        consensus = analyte.consensus_formula(members, formulas)
        data = {
            'analyte_id': i,
            'consensus_mz': f"{analyte.consensus_mz:.5f}",
            'consensus_rt': f"{analyte.consensus_rt:.1f}",
            'consensus_formula': consensus.formula,
            'formula_count': consensus.count,
            'formula_total': consensus.total,
            'n_samples': len(analyte.ensemble_map),
            'group': groups.get(analyte.uuid),
            flag_key: 'true' if analyte.uuid in flagged else 'false',
        }
        lines.append(f'    <node id="{i}">')
        lines.extend(_data_lines(data))
        lines.append('    </node>')

    for src, dst, data in edges:
        data = {
            **data,
            'mz_delta': f"{abs(src.consensus_mz - dst.consensus_mz):.5f}",
        }
        lines.append(
            f'    <edge source="{row_of[src.uuid]}" '
            f'target="{row_of[dst.uuid]}">'
        )
        lines.extend(_data_lines(data))
        lines.append('    </edge>')

    lines += ['  </graph>', '</graphml>']
    return '\n'.join(lines) + '\n'


def _data_lines(data: dict) -> list[str]:
    """<data> elements for the non-None values of `data`."""
    return [
        f'      <data key={quoteattr(key)}>{escape(str(value))}</data>'
        for key, value in data.items()
        if value is not None
    ]
