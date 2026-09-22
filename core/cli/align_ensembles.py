"""
Cross-sample ensemble alignment algorithm

Matches ensembles across samples by spectral similarity
(MS1 and optionally MS2), producing AlignedAnalytes.

Design
------
The work is split into three stages so each can be tuned or
swapped independently:

1. `_build_spectra`:
    Extract every ensemble's MS1 (and MS2) spectrum *once* into
    `matchms.Spectrum` objects.

2. `_score_pairs`: RT-windowed, cross-sample similarity scoring
   using matchms' `CosineGreedy`. Only pairs within `rt_tolerance`
    are scored.
   TODO: Implement other scoring systems i.e. `ModifiedCosine`, `NeutralLosses`

3. `_cluster`: group ensembles from the precomputed score graph
   via score-descending single-linkage with a per-sample-uniqueness
   constraint (i.e. each group holds up to one ensemble per sample).
"""
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
from numba import njit
from matchms import Spectrum
from matchms.similarity import CosineGreedy

from core.data_structs.alignment import (
    AlignmentParams,
    AlignedAnalyte,
    EnsembleAlignment,
)

if TYPE_CHECKING:
    from configparser import ConfigParser
    from core.data_structs import Sample, SampleUUID, EnsembleUUID
    from core.data_structs.feature_pointer import FeaturePointer
    from core.data_structs.scan_array import ScanArray


# Config section holding the persisted alignment defaults. Single source of
# truth for both the CLI (`mzkit align`) and the GUI alignment dialog.
ALIGNMENT_SECTION = "alignment"


def alignment_params_from_config(config: 'ConfigParser') -> AlignmentParams:
    """
    Build an AlignmentParams from the ``[alignment]`` config section, falling
    back to AlignmentParams' own defaults for any missing key.
    """
    s = ALIGNMENT_SECTION
    d = AlignmentParams()
    return AlignmentParams(
        rt_tolerance=config.getfloat(
            s, "rt_tolerance", fallback=d.rt_tolerance),
        mz_tolerance=config.getfloat(
            s, "mz_tolerance", fallback=d.mz_tolerance),
        ms1_similarity_threshold=config.getfloat(
            s, "ms1_similarity_threshold",
            fallback=d.ms1_similarity_threshold),
        ms2_similarity_threshold=config.getfloat(
            s, "ms2_similarity_threshold",
            fallback=d.ms2_similarity_threshold),
        ms1_weight=config.getfloat(s, "ms1_weight", fallback=d.ms1_weight),
        ms2_weight=config.getfloat(s, "ms2_weight", fallback=d.ms2_weight),
    )


def alignment_params_to_config(
    config: 'ConfigParser',
    params: AlignmentParams,
) -> None:
    """
    Write an AlignmentParams back into the ``[alignment]`` config section (in
    memory; caller persists via ``save_config``). Inverse of
    ``alignment_params_from_config``.
    """
    s = ALIGNMENT_SECTION
    if not config.has_section(s):
        config.add_section(s)

    for key, value in params._asdict().items():
        config.set(s, key, str(value))


@dataclass
class _SpectraBundle:
    """
    Flat, index-aligned view of every ensemble across all samples.

    All list/array attributes are parallel. `k` refers to the
    k-th ensemble in the pooled order.
    """
    sample_uuids: list['SampleUUID'] = field(default_factory=list)
    ens_uuids: list['EnsembleUUID'] = field(default_factory=list)
    sample_idxs: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    rts: np.ndarray = field(default_factory=lambda: np.empty(0, float))
    mzs: np.ndarray = field(default_factory=lambda: np.empty(0, float))
    ms1_specs: list[Spectrum] = field(default_factory=list)
    ms2_specs: list[Optional[Spectrum]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ens_uuids)


def align_ensembles(
    samples: list['Sample'],
    params: AlignmentParams,
    progress_callback=None,  # progress_callback(percent: float, msg: str)
    cancel_event=None,       # threading.Event-like; .is_set() to abort
) -> EnsembleAlignment:
    """
    Align ensembles across multiple samples by spectral
    similarity.

    :param samples: List of Samples with Injections containing
        ensembles
    :param params: AlignmentParams controlling tolerances
    :return: EnsembleAlignment result
    """
    bundle = _build_spectra(samples)

    edges_i, edges_j, edges_s = _score_pairs(
        bundle, params, progress_callback, cancel_event,
    )

    analytes = _cluster(bundle, edges_i, edges_j, edges_s)

    if progress_callback is not None:
        progress_callback(100.0, f"Aligned {len(analytes)} analytes")

    return EnsembleAlignment(
        sample_uuids=tuple(s.uuid for s in samples),
        analytes=analytes,
        parameters=params,
    )


@njit(cache=True)
def _extract_peaks_kernel(
    mz_data, intsy_data, indices, indptr,
    lanes, starts, ends, out_mz, out_intsy,
):
    """
    Per-cofeature windowed max m/z and max intensity, over shared CSR.

    `mz_data`/`intsy_data` share the CSR sparsity pattern
    (`indices`/`indptr`), so one pass yields both.

    For cofeature `k` the peak is the max m/z and max intensity
     over columns (scans) in `[starts[k], ends[k])` of row (m/z lane)
      `lanes[k]`. CSR column indices are assumed sorted ascending
        within a row.
    """
    for k in range(lanes.shape[0]):
        r = lanes[k]
        s = starts[k]
        e = ends[k]
        p0 = indptr[r]
        p1 = indptr[r + 1]

        # Binary search for the first stored column >= s.
        lo = p0
        hi = p1
        while lo < hi:
            mid = (lo + hi) // 2
            if indices[mid] < s:
                lo = mid + 1
            else:
                hi = mid
        best_mz = 0.0
        best_intsy = 0.0
        p = lo
        while p < p1:
            c = indices[p]
            if c >= e:
                break
            m = mz_data[p]
            if m > best_mz:
                best_mz = m
            iv = intsy_data[p]
            if iv > best_intsy:
                best_intsy = iv
            p += 1
        out_mz[k] = best_mz
        out_intsy[k] = best_intsy


def _extract_peaks(
    scan_array: 'ScanArray',
    feature_lists: list[list['FeaturePointer']],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Extract (mz, intsy) peak arrays for many ensembles in
    one numba pass.

    `feature_lists[i]` is ensemble `i`'s cofeature pointers;
     the result at `i` is that ensemble's `(mz, intsy)` peak
      arrays (one peak per cofeature).
    """
    lanes: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    seg_lens: list[int] = []
    for feats in feature_lists:
        seg_lens.append(len(feats))
        for fp in feats:
            scan_idxs = fp.scan_idxs
            lanes.append(fp.mz_lane_idx)
            starts.append(int(scan_idxs[0]))
            ends.append(int(scan_idxs[-1]) + 1)  # inclusive of the apex scan

    total = len(lanes)
    out_mz = np.empty(total, dtype=np.float64)
    out_intsy = np.empty(total, dtype=np.float64)
    if total:
        mz_arr = scan_array.mz_arr
        intsy_arr = scan_array.intsy_arr
        # Kernel relies on ascending column indices within each row.
        mz_arr.sort_indices()
        intsy_arr.sort_indices()
        _extract_peaks_kernel(
            mz_arr.data, intsy_arr.data,
            mz_arr.indices.astype(np.int64),
            mz_arr.indptr.astype(np.int64),
            np.asarray(lanes, dtype=np.int64),
            np.asarray(starts, dtype=np.int64),
            np.asarray(ends, dtype=np.int64),
            out_mz, out_intsy,
        )

    result: list[tuple[np.ndarray, np.ndarray]] = []
    pos = 0
    for length in seg_lens:
        result.append((out_mz[pos:pos + length], out_intsy[pos:pos + length]))
        pos += length
    return result


def _spectrum_from(mz: np.ndarray, intsy: np.ndarray) -> Spectrum:
    """
    Build an m/z-sorted matchms Spectrum,
     dropping zero-intensity peaks
    """
    keep = intsy > 0.0
    mz = mz[keep]
    intsy = intsy[keep]
    order = np.argsort(mz, kind='stable')  # matchms requires ascending m/z
    return Spectrum(
        mz=np.ascontiguousarray(mz[order], dtype=float),
        intensities=np.ascontiguousarray(intsy[order], dtype=float),
        metadata_harmonization=False,
    )


def _build_spectra(samples: list['Sample']) -> _SpectraBundle:
    """
    Pool every ensemble across samples, extracting each spectrum once

    Spectrum peaks are pulled straight from the injection's
     CSR scan arrays via a single batched numba pass per
      sample per MS level (more efficient here)
    """
    sample_idx_of: dict['SampleUUID', int] = {
        s.uuid: i for i, s in enumerate(samples)
    }

    bundle = _SpectraBundle()
    sample_idxs: list[int] = []
    rts: list[float] = []
    mzs: list[float] = []

    for sample in samples:
        injection = sample.injection
        if not injection:
            continue
        si = sample_idx_of[sample.uuid]
        ens_items = list(injection.ensembles.items())

        ms1_peaks = _extract_peaks(
            injection.scan_array_ms1,
            [ens.ms1_cofeatures for _, ens in ens_items],
        )

        has_ms2 = [bool(ens.ms2_cofeatures) for _, ens in ens_items]
        if any(has_ms2) and injection.scan_array_ms2 is not None:
            ms2_peaks = _extract_peaks(
                injection.scan_array_ms2,
                [ens.ms2_cofeatures if flag else []
                 for (_, ens), flag in zip(ens_items, has_ms2)],
            )
        else:
            ms2_peaks = [None] * len(ens_items)

        for idx, (ens_uuid, ensemble) in enumerate(ens_items):
            mz1, intsy1 = ms1_peaks[idx]
            ms2_spec = None
            if has_ms2[idx] and ms2_peaks[idx] is not None:
                mz2, intsy2 = ms2_peaks[idx]
                ms2_spec = _spectrum_from(mz2, intsy2)

            bundle.sample_uuids.append(sample.uuid)
            bundle.ens_uuids.append(ens_uuid)
            bundle.ms1_specs.append(_spectrum_from(mz1, intsy1))
            bundle.ms2_specs.append(ms2_spec)
            sample_idxs.append(si)
            rts.append(ensemble.peak_rt)
            mzs.append(ensemble.base_mz)

    bundle.sample_idxs = np.asarray(sample_idxs, dtype=int)
    bundle.rts = np.asarray(rts, dtype=float)
    bundle.mzs = np.asarray(mzs, dtype=float)
    return bundle


def _score_pairs(
    bundle: _SpectraBundle,
    params: AlignmentParams,
    progress_callback=None,
    cancel_event=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Score all cross-sample ensemble pairs within RT tolerance

    Returns three parallel arrays `(i, j, score)` describing
     the surviving edges: `i`/`j` are ensemble indices into
      `bundle` and `score` is the combined MS1(/MS2) similarity

    RT blocking: ensembles are visited in ascending-RT order and each
    is only compared against the forward run of ensembles still within
    `rt_tolerance` (found via `searchsorted`).
    """
    n = len(bundle)
    cosine = CosineGreedy(
        tolerance=params.mz_tolerance,
        mz_power=0.0,
        intensity_power=1.0,
    )
    ms1_thr = params.ms1_similarity_threshold
    ms2_thr = params.ms2_similarity_threshold
    w1 = params.ms1_weight
    w2 = params.ms2_weight
    rt_tol = params.rt_tolerance

    ms1 = bundle.ms1_specs
    ms2 = bundle.ms2_specs
    sample_idxs = bundle.sample_idxs

    # Visit ensembles in ascending-RT order.
    rt_order = np.argsort(bundle.rts, kind='stable')
    rts_sorted = bundle.rts[rt_order]

    edges_i: list[int] = []
    edges_j: list[int] = []
    edges_s: list[float] = []

    for a in range(n):
        if cancel_event is not None and cancel_event.is_set():
            break

        i = int(rt_order[a])
        hi = int(np.searchsorted(rts_sorted, rts_sorted[a] + rt_tol, side='right'))

        for b in range(a + 1, hi):
            j = int(rt_order[b])
            if sample_idxs[i] == sample_idxs[j]:
                continue

            score = 0.0
            total_weight = 0.0

            # MS1 only gates/contributes when it carries weight; w1 == 0
            # disables MS1 comparison entirely (its threshold included).
            if w1 > 0.0:
                ms1_sim = float(cosine.pair(ms1[i], ms1[j])['score'])
                if ms1_sim < ms1_thr:
                    continue
                score += ms1_sim * w1
                total_weight += w1

            # Likewise MS2: w2 == 0 disables MS2 comparison entirely, so a
            # poor (or DDA-ambiguous) MS2 match can't veto a strong MS1 one.
            if w2 > 0.0:
                spec2_i, spec2_j = ms2[i], ms2[j]
                if spec2_i is not None and spec2_j is not None:
                    ms2_sim = float(cosine.pair(spec2_i, spec2_j)['score'])
                    if ms2_sim < ms2_thr:
                        continue
                    score += ms2_sim * w2
                    total_weight += w2

            # Both weights zero (or no usable spectra) => nothing to score on.
            if total_weight == 0.0:
                continue

            edges_i.append(i)
            edges_j.append(j)
            edges_s.append(score / total_weight)

        if progress_callback is not None and (a % 100 == 0 or a == n - 1):
            progress_callback(
                100.0 * (a + 1) / max(n, 1),
                f"Scored {a + 1}/{n} ensembles, {len(edges_s)} candidate pairs",
            )

    return (
        np.asarray(edges_i, dtype=int),
        np.asarray(edges_j, dtype=int),
        np.asarray(edges_s, dtype=float),
    )


def _cluster(
    bundle: _SpectraBundle,
    edges_i: np.ndarray,
    edges_j: np.ndarray,
    edges_s: np.ndarray,
) -> list[AlignedAnalyte]:
    """
    Group ensembles from the score graph into AlignedAnalytes

    Single-linkage clustering that walks edges in descending score and
    merges the two endpoints' groups only when the merged group would
    still hold at most one ensemble per sample. Ensembles that never
    merge remain singleton analytes
    """
    n = len(bundle)

    # Per-node group bookkeeping. We can't use plain union-find because
    # the merge is conditional on sample-disjointness, so track members
    # and covered samples per group explicitly.
    group_of = list(range(n))
    group_members: list[list[int]] = [[k] for k in range(n)]
    group_samples: list[set[int]] = [
        {int(bundle.sample_idxs[k])} for k in range(n)
    ]

    # Walk edges strongest-first.
    for e in np.argsort(edges_s, kind='stable')[::-1]:
        i = int(edges_i[e])
        j = int(edges_j[e])
        gi = group_of[i]
        gj = group_of[j]
        if gi == gj:
            continue
        if group_samples[gi] & group_samples[gj]:
            continue  # sample collision - cannot merge

        # Merge the smaller group into the larger.
        if len(group_members[gi]) < len(group_members[gj]):
            gi, gj = gj, gi
        for node in group_members[gj]:
            group_of[node] = gi
        group_members[gi].extend(group_members[gj])
        group_samples[gi] |= group_samples[gj]
        group_members[gj] = []
        group_samples[gj] = set()

    analytes: list[AlignedAnalyte] = []
    for g, members in enumerate(group_members):
        if not members:
            continue
        ensemble_map: dict['SampleUUID', 'EnsembleUUID'] = {}
        for node in members:
            ensemble_map[bundle.sample_uuids[node]] = bundle.ens_uuids[node]
        rt_vals = bundle.rts[members]
        mz_vals = bundle.mzs[members]
        analytes.append(AlignedAnalyte(
            ensemble_map=ensemble_map,
            consensus_rt=float(np.mean(rt_vals)),
            consensus_mz=float(np.mean(mz_vals)),
        ))

    return analytes
