"""
Export an EnsembleAlignment as a molecular network (.graphml, e.g. for
Cytoscape), GNPS-FBMN style.

Nodes are analytes (all of them; those without a representative MS2 are
singletons). Edges are pairwise modified cosine between the analytes'
representative MS2 spectra, scored exactly as in Alignment Viewer cluster
mode (`core.cli.cluster_analytes`, same ClusterParams / config section).
Only pairs scoring at least `min_edge_cosine` get an edge (pairs under
`min_matched_peaks` score 0).

Node ids are the analyte row ids used by the feature tables / MGF
(`analyte_id` / `FEATURE_ID`).
"""
import threading
from typing import Callable, Mapping, Optional, TYPE_CHECKING
from xml.sax.saxutils import escape, quoteattr

from core.cli.cluster_analytes import (
    ClusterParams, analyte_ms2_spectra, similarity_matrix,
)

if TYPE_CHECKING:
    from core.data_structs import Sample
    from core.data_structs.alignment import EnsembleAlignment
    from core.data_structs.uuid_types import EnsembleUUID


# (attribute name, GraphML attr.type), declared as <key>s
_NODE_KEYS = [
    ('analyte_id', 'int'),
    ('consensus_mz', 'double'),
    ('consensus_rt', 'double'),
    ('consensus_formula', 'string'),
    ('formula_count', 'int'),
    ('formula_total', 'int'),
    ('n_samples', 'int'),
    ('has_ms2', 'boolean'),
]
_EDGE_KEYS = [
    ('cosine', 'double'),
    ('matched_peaks', 'int'),
    ('mz_delta', 'double'),
]


def build_network_graphml(
    alignment: 'EnsembleAlignment',
    samples: list['Sample'],
    params: ClusterParams = ClusterParams(),
    formulas: Optional[Mapping['EnsembleUUID', str]] = None,
    min_edge_cosine: float = 0.7,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Optional[str]:
    """
    GraphML text for `alignment`'s MS2 similarity network.

    :param params: scoring parameters (see `cluster_params_from_config`);
        `linkage_method` is unused.
    :param formulas: accepted formula per ensemble, for the
        consensus_formula node attributes.
    :param min_edge_cosine: drop edges scoring below this (GNPS default
        0.7). Must be > 0; unscored pairs are 0.
    :return: None if cancelled.
    """
    formulas = formulas or {}
    sample_of = {s.uuid: s for s in samples}

    ms2_analytes, spectra = analyte_ms2_spectra(alignment, samples)
    scored = similarity_matrix(
        spectra, params, progress_callback, cancel_event,
    )
    if scored is None:
        return None
    similarity, matches = scored

    row_of = {a.uuid: i for i, a in enumerate(alignment.analytes)}
    has_ms2 = {a.uuid for a in ms2_analytes}

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
    ]
    for owner, keys in (('node', _NODE_KEYS), ('edge', _EDGE_KEYS)):
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
            'has_ms2': 'true' if analyte.uuid in has_ms2 else 'false',
        }
        lines.append(f'    <node id="{i}">')
        lines.extend(_data_lines(data))
        lines.append('    </node>')

    n = len(ms2_analytes)
    for a in range(n):
        for b in range(a + 1, n):
            score = similarity[a, b]
            if score <= 0 or score < min_edge_cosine:
                continue
            src, dst = ms2_analytes[a], ms2_analytes[b]
            data = {
                'cosine': f"{score:.4f}",
                'matched_peaks': int(matches[a, b]),
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
