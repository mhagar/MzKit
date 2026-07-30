"""
Merge (align) existing EnsembleAlignments into a single alignment.

This reuses the `align_ensembles` machinery almost verbatim, with the
analyte playing the role that an ensemble plays in a from-scratch align,
and the source alignment playing the role of the sample:

1. Build one *consensus* MS1 (and MS2) spectrum per analyte by pooling the
   spectra of its member ensembles (extracted once via `_build_spectra`) and
    merging co-located peaks.

2. Score cross-alignment analyte pairs within `rt_tolerance` using the
   same `_score_pairs` cosine scorer

3. Single-linkage cluster with a per-source-alignment uniqueness
   constraint (each merged group holds at most one analyte per input
   alignment) *and* a sample-consistency constraint (when input
   alignments share samples, two analytes may only merge if they agree on
   the ensemble for every shared sample). For disjoint input alignments —
   i.e. "align batches, then merge", the second constraint is always
   satisfied.
"""
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
from matchms import Spectrum

from core.data_structs.alignment import (
    AlignmentParams,
    AlignedAnalyte,
    EnsembleAlignment,
)
from core.cli.align_ensembles import (
    _SpectraBundle,
    _build_spectra,
    _spectrum_from,
    _score_pairs,
)

if TYPE_CHECKING:
    from core.data_structs import Sample, SampleUUID, EnsembleUUID


@dataclass
class _AnalyteItem:
    """
    One analyte pulled from a source alignment, ready to re-align
    """
    align_idx: int
    ensemble_map: dict['SampleUUID', 'EnsembleUUID']
    consensus_rt: float
    consensus_mz: float


def align_alignments(
    alignments: list[EnsembleAlignment],
    samples: list['Sample'],
    params: AlignmentParams,
    progress_callback=None,  # progress_callback(percent: float, msg: str)
    cancel_event=None,       # threading.Event-like; .is_set() to abort
) -> EnsembleAlignment:
    """
    Merge several EnsembleAlignments into one by matching analytes.

    :param alignments: Source alignments to merge. Their sample sets may
        overlap; a merged analyte keeps at most one ensemble per sample.
    :param samples: Samples referenced by the alignments (needed to build
        each analyte's consensus spectrum). Extra samples are harmless.
    :param params: AlignmentParams controlling tolerances / thresholds.
    :return: A new EnsembleAlignment spanning the union of all samples.
    """
    if progress_callback is not None:
        progress_callback(
            0.0,
            "Building analyte consensus spectra"
        )

    items, bundle = _build_analyte_spectra(alignments, samples)

    edges_i, edges_j, edges_s = _score_pairs(
        bundle, params, progress_callback, cancel_event,
    )

    analytes = _cluster_analytes(items, edges_i, edges_j, edges_s)

    # Union of every source alignment's samples, order-preserving
    seen: set['SampleUUID'] = set()
    sample_uuids: list['SampleUUID'] = []
    for alignment in alignments:
        for su in alignment.sample_uuids:
            if su not in seen:
                seen.add(su)
                sample_uuids.append(su)

    if progress_callback is not None:
        progress_callback(
            100.0,
            f"Merged {len(alignments)} alignments into {len(analytes)} analytes",
        )

    return EnsembleAlignment(
        sample_uuids=tuple(sample_uuids),
        analytes=analytes,
        parameters=params,
    )


def _merge_spectra(
    specs: list[Spectrum],
    mz_tol: float,
) -> Spectrum:
    """
    Pool several spectra into one consensus spectrum.

    Peaks within `mz_tol` of a bin's first peak are collapsed into a
    single peak at their intensity-weighted mean m/z, with the mean of the
    contributing intensities. Absolute scale is irrelevant (CosineGreedy
    normalizes internally), so no renormalization is done.
    """
    mz_parts = [s.peaks.mz for s in specs if s.peaks.mz.size]
    intsy_parts = [s.peaks.intensities for s in specs if s.peaks.mz.size]
    if not mz_parts:
        return _spectrum_from(np.empty(0), np.empty(0))

    mz = np.concatenate(mz_parts)
    intsy = np.concatenate(intsy_parts)
    order = np.argsort(mz, kind='stable')
    mz = mz[order]
    intsy = intsy[order]

    out_mz: list[float] = []
    out_intsy: list[float] = []
    bin_mz = [float(mz[0])]
    bin_intsy = [float(intsy[0])]
    ref = float(mz[0])

    def _flush() -> None:
        w = np.asarray(bin_intsy)
        if w.sum() > 0:
            out_mz.append(float(np.average(bin_mz, weights=w)))
        else:
            out_mz.append(float(np.mean(bin_mz)))
        out_intsy.append(float(np.mean(bin_intsy)))

    for k in range(1, mz.size):
        if mz[k] - ref <= mz_tol:
            bin_mz.append(float(mz[k]))
            bin_intsy.append(float(intsy[k]))
        else:
            _flush()
            bin_mz = [float(mz[k])]
            bin_intsy = [float(intsy[k])]
            ref = float(mz[k])
    _flush()

    return _spectrum_from(
        np.asarray(out_mz),
        np.asarray(out_intsy),
    )


def _build_analyte_spectra(
    alignments: list[EnsembleAlignment],
    samples: list['Sample'],
) -> tuple[list[_AnalyteItem], _SpectraBundle]:
    """
    Flatten every analyte across all alignments into re-alignable items

    Each analyte's consensus MS1/MS2 spectrum is merged from its member
    ensembles' spectra, which are extracted once by `_build_spectra` and
    looked up by (sample_uuid, ensemble_uuid)
    """
    ens_bundle = _build_spectra(samples)
    spec_of: dict[tuple['SampleUUID', 'EnsembleUUID'], int] = {
        (ens_bundle.sample_uuids[k], ens_bundle.ens_uuids[k]): k
        for k in range(len(ens_bundle))
    }
    # A shared m/z tolerance for collapsing consensus peaks. Alignments
    # may carry differing parameters, so use the merge params instead.
    mz_tol = alignments[0].parameters.mz_tolerance if alignments else 0.01

    items: list[_AnalyteItem] = []
    align_idxs: list[int] = []
    rts: list[float] = []
    mzs: list[float] = []
    ms1_specs: list[Spectrum] = []
    ms2_specs: list[Optional[Spectrum]] = []

    for align_idx, alignment in enumerate(alignments):
        mz_tol = alignment.parameters.mz_tolerance
        for analyte in alignment.analytes:
            member_ms1: list[Spectrum] = []
            member_ms2: list[Spectrum] = []
            for su, eu in analyte.ensemble_map.items():
                k = spec_of.get((su, eu))
                if k is None:
                    continue
                member_ms1.append(ens_bundle.ms1_specs[k])
                if ens_bundle.ms2_specs[k] is not None:
                    member_ms2.append(ens_bundle.ms2_specs[k])

            ms1_spec = _merge_spectra(member_ms1, mz_tol)
            ms2_spec = _merge_spectra(member_ms2, mz_tol) if member_ms2 else None

            items.append(_AnalyteItem(
                align_idx=align_idx,
                ensemble_map=dict(analyte.ensemble_map),
                consensus_rt=analyte.consensus_rt,
                consensus_mz=analyte.consensus_mz,
            ))
            align_idxs.append(align_idx)
            rts.append(analyte.consensus_rt)
            mzs.append(analyte.consensus_mz)
            ms1_specs.append(ms1_spec)
            ms2_specs.append(ms2_spec)

    # Pack into a _SpectraBundle so _score_pairs can be reused as-is. Its
    # "sample index" is the source-alignment index (so same-alignment
    # analyte pairs are skipped); ens_uuids only needs a length.
    bundle = _SpectraBundle()
    n = len(items)
    bundle.ens_uuids = list(range(n))
    bundle.sample_idxs = np.asarray(align_idxs, dtype=int)
    bundle.rts = np.asarray(rts, dtype=float)
    bundle.mzs = np.asarray(mzs, dtype=float)
    bundle.ms1_specs = ms1_specs
    bundle.ms2_specs = ms2_specs
    return items, bundle


def _maps_compatible(
    a: dict['SampleUUID', 'EnsembleUUID'],
    b: dict['SampleUUID', 'EnsembleUUID'],
) -> bool:
    """True if two ensemble maps agree on every sample they share."""
    smaller, larger = (a, b) if len(a) <= len(b) else (b, a)
    for su, eu in smaller.items():
        other = larger.get(su)
        if other is not None and other != eu:
            return False
    return True


def _cluster_analytes(
    items: list[_AnalyteItem],
    edges_i: np.ndarray,
    edges_j: np.ndarray,
    edges_s: np.ndarray,
) -> list[AlignedAnalyte]:
    """
    Group analytes from the score graph into merged AlignedAnalytes.

    Score-descending single-linkage, merging two groups only when the
    result would hold at most one analyte per source alignment *and* the
    two groups' ensemble maps stay consistent on any shared sample.
    """
    n = len(items)
    group_of = list(range(n))
    group_members: list[list[int]] = [[k] for k in range(n)]
    group_aligns: list[set[int]] = [{items[k].align_idx} for k in range(n)]
    group_maps: list[dict['SampleUUID', 'EnsembleUUID']] = [
        dict(items[k].ensemble_map) for k in range(n)
    ]

    for e in np.argsort(edges_s, kind='stable')[::-1]:
        i = int(edges_i[e])
        j = int(edges_j[e])
        gi = group_of[i]
        gj = group_of[j]
        if gi == gj:
            continue
        if group_aligns[gi] & group_aligns[gj]:
            continue  # two analytes from the same source alignment
        if not _maps_compatible(group_maps[gi], group_maps[gj]):
            continue  # shared sample assigned to conflicting ensembles

        # Merge the smaller group into the larger.
        if len(group_members[gi]) < len(group_members[gj]):
            gi, gj = gj, gi
        for node in group_members[gj]:
            group_of[node] = gi
        group_members[gi].extend(group_members[gj])
        group_aligns[gi] |= group_aligns[gj]
        group_maps[gi].update(group_maps[gj])  # safe: checked compatible
        group_members[gj] = []
        group_aligns[gj] = set()
        group_maps[gj] = {}

    analytes: list[AlignedAnalyte] = []
    for g, members in enumerate(group_members):
        if not members:
            continue
        rt_acc = 0.0
        mz_acc = 0.0
        weight = 0.0
        for node in members:
            it = items[node]
            w = max(len(it.ensemble_map), 1)
            rt_acc += it.consensus_rt * w
            mz_acc += it.consensus_mz * w
            weight += w
        analytes.append(AlignedAnalyte(
            ensemble_map=group_maps[g],
            consensus_rt=rt_acc / weight,
            consensus_mz=mz_acc / weight,
        ))

    return analytes
