"""
Hierarchical clustering of an EnsembleAlignment's AlignedAnalytes by MS2
similarity ("which analytes are structurally related").

1. Each analyte's representative MS2 (`AlignedAnalyte.representative_spectrum`)
   is converted to a matchms Spectrum; analytes without MS2 are dropped.
2. All pairs are scored by modified cosine (`matchms.similarity.ModifiedCosine`,
   the greedy implementation).
3. Average-linkage (by default) clustering on distance = 1 - similarity.

Leaf order
----------
Merges at distance 1.0 join analytes with *no* similarity, so their order
is arbitrary. The tree is therefore split into a forest of "real" subtrees
(merges below 1.0); subtrees are ordered by the mean consensus RT of their
leaves, as are the two children of every merge. Unrelated analytes thus
form an RT-sorted comb, and every subtree's leaves are contiguous.
"""
from dataclasses import dataclass
import threading
from typing import Callable, NamedTuple, Optional, TYPE_CHECKING

import numpy as np
from matchms import Spectrum
from matchms.similarity import ModifiedCosine
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from core.cli.align_ensembles import _spectrum_from

if TYPE_CHECKING:
    from configparser import ConfigParser
    from core.data_structs import AnalyteUUID, Sample
    from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment


# Merges at/above this distance join analytes with no similarity at all
_UNRELATED_DISTANCE = 1.0 - 1e-9

CLUSTERING_SECTION = "analyte_clustering"


class ClusterParams(NamedTuple):
    """
    tolerance: m/z tolerance for matching peaks (ModifiedCosine).
    mz_power / intensity_power: peak weighting in the cosine.
    min_matched_peaks: pairs matching fewer peaks than this score 0.
    linkage_method: scipy linkage method ('average', 'complete', ...).
    """
    tolerance: float = 0.02
    mz_power: float = 0.0
    intensity_power: float = 1.0
    min_matched_peaks: int = 3
    linkage_method: str = 'average'


# One row per (sub-1.0) merge, for drawing / hit-testing the dendrogram.
# x_left / x_right: x of the two children (leaf slot, or a merge's midpoint);
# h_left / h_right: their heights (0 for leaves); height: the merge distance;
# lo / hi: the (contiguous, inclusive) leaf-slot range under the merge.
NODE_DTYPE = np.dtype([
    ('x_left', 'f8'), ('x_right', 'f8'),
    ('h_left', 'f8'), ('h_right', 'f8'),
    ('height', 'f8'),
    ('lo', 'i8'), ('hi', 'i8'),
])


@dataclass(frozen=True)
class AnalyteClustering:
    """
    analyte_uuids: clustered analytes in leaf order (leaf slot = index).
    similarity: NxN modified-cosine similarity, in leaf order.
    nodes: dendrogram merges below distance 1.0 (see NODE_DTYPE).
    n_dropped_no_ms2: analytes left out for lack of a representative MS2.
    """
    analyte_uuids: list['AnalyteUUID']
    similarity: np.ndarray
    nodes: np.ndarray
    n_dropped_no_ms2: int
    params: ClusterParams


def cluster_params_from_config(config: 'ConfigParser') -> ClusterParams:
    """
    Build ClusterParams from the ``[analyte_clustering]`` config section,
    falling back to ClusterParams' defaults for any missing key.
    """
    s = CLUSTERING_SECTION
    d = ClusterParams()
    return ClusterParams(
        tolerance=config.getfloat(s, "tolerance", fallback=d.tolerance),
        mz_power=config.getfloat(s, "mz_power", fallback=d.mz_power),
        intensity_power=config.getfloat(
            s, "intensity_power", fallback=d.intensity_power),
        min_matched_peaks=config.getint(
            s, "min_matched_peaks", fallback=d.min_matched_peaks),
        linkage_method=config.get(
            s, "linkage_method", fallback=d.linkage_method),
    )


def cluster_params_to_config(
    config: 'ConfigParser',
    params: ClusterParams,
) -> None:
    """
    Write ClusterParams into the ``[analyte_clustering]`` section (in memory;
    caller persists via ``save_config``).
    """
    s = CLUSTERING_SECTION
    if not config.has_section(s):
        config.add_section(s)
    for key, value in params._asdict().items():
        config.set(s, key, str(value))


def cluster_analytes(
    alignment: 'EnsembleAlignment',
    samples: list['Sample'],
    params: ClusterParams = ClusterParams(),
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Optional[AnalyteClustering]:
    """
    Cluster `alignment`'s analytes by modified cosine between their
    representative MS2 spectra.

    ProcessController-compatible. Returns None if cancelled.
    """
    analytes, spectra = analyte_ms2_spectra(alignment, samples)
    n_dropped = len(alignment.analytes) - len(analytes)

    scored = similarity_matrix(
        spectra, params, progress_callback, cancel_event,
    )
    if scored is None:
        return None
    similarity, _matches = scored

    rts = np.array([a.consensus_rt for a in analytes], dtype=float)
    order, nodes = _order_and_nodes(similarity, rts, params.linkage_method)

    return AnalyteClustering(
        analyte_uuids=[analytes[i].uuid for i in order],
        similarity=similarity[np.ix_(order, order)],
        nodes=nodes,
        n_dropped_no_ms2=n_dropped,
        params=params,
    )


def analyte_ms2_spectra(
    alignment: 'EnsembleAlignment',
    samples: list['Sample'],
) -> tuple[list['AlignedAnalyte'], list[Spectrum]]:
    """
    `alignment`'s analytes that have a representative MS2, and those MS2s
    as matchms Spectra (parallel lists, in alignment order).
    """
    sample_of = {s.uuid: s for s in samples}
    analytes: list['AlignedAnalyte'] = []
    spectra: list[Spectrum] = []
    for analyte in alignment.analytes:
        spectrum = _ms2_spectrum(analyte, sample_of.get)
        if spectrum is not None:
            analytes.append(analyte)
            spectra.append(spectrum)
    return analytes, spectra


def _ms2_spectrum(
    analyte: 'AlignedAnalyte',
    get_sample: Callable,
) -> Optional[Spectrum]:
    """The analyte's representative MS2 as a matchms Spectrum, or None."""
    rep = analyte.representative_spectrum(analyte.resolve_members(get_sample))
    if rep is None:
        return None
    ms2 = rep.composite.ms2
    if ms2 is None or not np.any(ms2['intsy'] > 0):
        return None
    spectrum = _spectrum_from(ms2['mz'], ms2['intsy'])
    spectrum.set('precursor_mz', float(rep.composite.precursor_mz))
    return spectrum


def similarity_matrix(
    spectra: list[Spectrum],
    params: ClusterParams,
    progress_callback=None,
    cancel_event=None,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Symmetric modified-cosine matrix (diagonal 1), and the matched-peak
    count of each pair. Scores from fewer than `min_matched_peaks` matched
    peaks are zeroed. None if cancelled.
    """
    n = len(spectra)
    similarity = np.eye(n)
    matches = np.zeros((n, n), dtype=int)
    cosine = ModifiedCosine(
        tolerance=params.tolerance,
        mz_power=params.mz_power,
        intensity_power=params.intensity_power,
    )

    n_pairs = n * (n - 1) // 2
    done = 0
    for i in range(n):
        if cancel_event is not None and cancel_event.is_set():
            return None
        for j in range(i + 1, n):
            result = cosine.pair(spectra[i], spectra[j])
            n_matched = int(result['matches'])
            matches[i, j] = matches[j, i] = n_matched
            if n_matched >= params.min_matched_peaks:
                similarity[i, j] = similarity[j, i] = float(result['score'])
        done += n - 1 - i
        if progress_callback is not None and (i % 20 == 0 or i == n - 1):
            progress_callback(
                100.0 * done / max(n_pairs, 1),
                f"Scored {done}/{n_pairs} analyte pairs",
            )

    return similarity, matches


def _order_and_nodes(
    similarity: np.ndarray,
    rts: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Leaf order (indices into `similarity`) and the dendrogram's sub-1.0
    merges (NODE_DTYPE, in leaf-slot coordinates). See module docstring.
    """
    n = similarity.shape[0]
    if n < 2:
        return np.arange(n), np.zeros(0, dtype=NODE_DTYPE)

    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    Z = linkage(squareform(distance, checks=False), method=method)

    # Node ids: leaves 0..n-1, merge k is node n + k (children precede it)
    n_nodes = 2 * n - 1
    left = np.full(n_nodes, -1, dtype=int)
    right = np.full(n_nodes, -1, dtype=int)
    height = np.zeros(n_nodes)
    parent = np.full(n_nodes, -1, dtype=int)
    rt_sum = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    rt_sum[:n] = rts
    count[:n] = 1

    for k, (a, b, h, _) in enumerate(Z):
        node, a, b = n + k, int(a), int(b)
        left[node], right[node], height[node] = a, b, h
        parent[a] = parent[b] = node
        rt_sum[node] = rt_sum[a] + rt_sum[b]
        count[node] = count[a] + count[b]
    mean_rt = rt_sum / count

    def is_real(node: int) -> bool:
        return node < n or height[node] < _UNRELATED_DISTANCE

    roots = [
        node for node in range(n_nodes)
        if is_real(node)
        and (parent[node] < 0 or not is_real(parent[node]))
    ]
    roots.sort(key=lambda node: mean_rt[node])

    # Iterative DFS (chained trees can be deeper than the recursion limit),
    # visiting each merge's children in RT order
    order: list[int] = []
    for root in roots:
        stack = [root]
        while stack:
            node = stack.pop()
            if node < n:
                order.append(node)
                continue
            first, second = sorted(
                (left[node], right[node]), key=lambda c: mean_rt[c],
            )
            stack.append(second)
            stack.append(first)
    order_arr = np.asarray(order, dtype=int)

    # Leaf slots, then each real merge's x / leaf range (children first)
    x = np.zeros(n_nodes)
    lo = np.zeros(n_nodes, dtype=int)
    hi = np.zeros(n_nodes, dtype=int)
    x[order_arr] = np.arange(n)
    lo[order_arr] = hi[order_arr] = np.arange(n)

    nodes = []
    for k in range(n - 1):
        node = n + k
        if not is_real(node):
            continue
        a, b = left[node], right[node]
        if x[a] > x[b]:
            a, b = b, a
        x[node] = 0.5 * (x[a] + x[b])
        lo[node], hi[node] = lo[a], hi[b]
        nodes.append((
            x[a], x[b], height[a], height[b], height[node], lo[node], hi[node],
        ))

    return order_arr, np.array(nodes, dtype=NODE_DTYPE)
