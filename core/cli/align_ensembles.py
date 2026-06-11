"""
Cross-sample ensemble alignment *algorithm*.

Matches ensembles across samples by spectral similarity
(MS1 and optionally MS2), producing AlignedAnalytes.

The result types (``EnsembleAlignment``, ``AlignedAnalyte``,
``AlignmentParams``) are domain objects and live in
``core.data_structs.alignment``; they are re-exported here for
backwards compatibility with existing ``from core.cli.align_ensembles
import EnsembleAlignment`` call sites.
"""
from typing import Optional, TYPE_CHECKING

import numpy as np

from core.data_structs.alignment import (
    AlignmentParams,
    AlignedAnalyte,
    EnsembleAlignment,
)

if TYPE_CHECKING:
    from core.data_structs import (
        Ensemble, Injection, Sample,
        SampleUUID, EnsembleUUID,
    )
    from core.utils.array_types import SpectrumArray


def align_ensembles(
    samples: list['Sample'],
    params: AlignmentParams,
    progress_callback=None,  # injected by ProcessRunner; unused here
    cancel_event=None,       # injected by ProcessRunner; unused here
) -> EnsembleAlignment:
    """
    Align ensembles across multiple samples by spectral
    similarity.

    Algorithm:
        1. Pool all ensembles from all samples, sorted by
           intensity (most intense first)
        2. Take the top unassigned ensemble as a seed
        3. Find the best matching unassigned ensemble in each
           other sample (within RT tolerance, above similarity
           threshold)
        4. Group matches into an AlignedAnalyte
        5. Repeat until all ensembles have been processed

    :param samples: List of Samples with Injections containing
        ensembles
    :param params: AlignmentParams controlling tolerances
    :return: EnsembleAlignment result
    """
    # Build the global pool: (intensity, sample_uuid, ensemble_uuid, ensemble)
    pool: list[tuple[float, 'SampleUUID', 'EnsembleUUID', 'Ensemble']] = []

    for sample in samples:
        if not sample.injection:
            continue
        for ens_uuid, ensemble in sample.injection.ensembles.items():
            pool.append((
                ensemble.base_intsy,
                sample.uuid,
                ens_uuid,
                ensemble,
            ))

    # Sort by intensity, descending
    pool.sort(key=lambda x: x[0], reverse=True)

    # Track which ensembles have been assigned
    assigned: set[tuple['SampleUUID', 'EnsembleUUID']] = set()

    # Build a lookup: sample_uuid -> list of (ens_uuid, ensemble)
    # for efficient per-sample searching
    sample_ensembles: dict['SampleUUID', list[tuple['EnsembleUUID', 'Ensemble']]] = {}
    for sample in samples:
        if not sample.injection:
            continue
        sample_ensembles[sample.uuid] = [
            (ens_uuid, ens)
            for ens_uuid, ens in sample.injection.ensembles.items()
        ]

    analytes: list[AlignedAnalyte] = []

    for seed_intsy, seed_sample_uuid, seed_ens_uuid, seed_ensemble in pool:
        if (seed_sample_uuid, seed_ens_uuid) in assigned:
            continue

        # Start a new AlignedAnalyte with this seed
        ensemble_map: dict['SampleUUID', 'EnsembleUUID'] = {
            seed_sample_uuid: seed_ens_uuid,
        }
        assigned.add((seed_sample_uuid, seed_ens_uuid))

        # Precompute seed spectra
        seed_ms1_spec = seed_ensemble._generate_spectrum(ms_level=1)
        seed_ms2_spec = None
        if seed_ensemble.ms2_cofeatures:
            seed_ms2_spec = seed_ensemble._generate_spectrum(ms_level=2)

        # Search each other sample for the best match
        for other_sample_uuid, other_ensembles in sample_ensembles.items():
            if other_sample_uuid == seed_sample_uuid:
                continue

            best_score = -1.0
            best_ens_uuid: Optional['EnsembleUUID'] = None

            for other_ens_uuid, other_ensemble in other_ensembles:
                if (other_sample_uuid, other_ens_uuid) in assigned:
                    continue

                # RT filter
                rt_diff = abs(seed_ensemble.peak_rt - other_ensemble.peak_rt)
                if rt_diff > params.rt_tolerance:
                    continue

                # MS1 similarity
                other_ms1_spec = other_ensemble._generate_spectrum(ms_level=1)
                ms1_sim = cosine_similarity(
                    seed_ms1_spec, other_ms1_spec,
                    mz_tolerance=params.mz_tolerance,
                )

                if ms1_sim < params.ms1_similarity_threshold:
                    continue

                # MS2 similarity (if both have it)
                score = ms1_sim * params.ms1_weight
                total_weight = params.ms1_weight

                if (seed_ms2_spec is not None
                    and other_ensemble.ms2_cofeatures):
                    other_ms2_spec = other_ensemble._generate_spectrum(
                        ms_level=2
                    )
                    ms2_sim = cosine_similarity(
                        seed_ms2_spec, other_ms2_spec,
                        mz_tolerance=params.mz_tolerance,
                    )
                    if ms2_sim < params.ms2_similarity_threshold:
                        continue

                    score += ms2_sim * params.ms2_weight
                    total_weight += params.ms2_weight

                score /= total_weight

                if score > best_score:
                    best_score = score
                    best_ens_uuid = other_ens_uuid

            if best_ens_uuid is not None:
                ensemble_map[other_sample_uuid] = best_ens_uuid
                assigned.add((other_sample_uuid, best_ens_uuid))

        # Compute consensus RT and m/z from all matched ensembles
        matched_ensembles = []
        for s_uuid in ensemble_map:
            ens_uuid = ensemble_map[s_uuid]
            for sample in samples:
                if sample.uuid == s_uuid and sample.injection:
                    matched_ensembles.append(
                        sample.injection.ensembles[ens_uuid]
                    )

        analyte = AlignedAnalyte(
            ensemble_map=ensemble_map,
            consensus_rt=float(np.mean([e.peak_rt for e in matched_ensembles])),
            consensus_mz=float(np.mean([e.base_mz for e in matched_ensembles])),
        )
        analytes.append(analyte)

    return EnsembleAlignment(
        sample_uuids=tuple(s.uuid for s in samples),
        analytes=analytes,
        parameters=params,
    )


def cosine_similarity(
    spec_a: 'SpectrumArray',
    spec_b: 'SpectrumArray',
    mz_tolerance: float,
) -> float:
    """
    Cosine similarity between two spectra.

    Pairs peaks from spec_a and spec_b by nearest m/z within
    tolerance, then computes cosine similarity on the paired
    intensity vectors.

    Unpaired peaks contribute zero to the dot product but
    still contribute to the magnitude (they penalize the
    score for having unmatched peaks).

    :param spec_a: First spectrum (SpectrumArray with mz, intsy)
    :param spec_b: Second spectrum
    :param mz_tolerance: Maximum m/z difference for peak pairing
    :return: Cosine similarity in [0, 1]
    """
    if spec_a.size == 0 or spec_b.size == 0:
        return 0.0

    mz_a, intsy_a = spec_a['mz'], spec_a['intsy']
    mz_b, intsy_b = spec_b['mz'], spec_b['intsy']

    # Normalize intensities
    norm_a = intsy_a / np.max(intsy_a) if np.max(intsy_a) > 0 else intsy_a
    norm_b = intsy_b / np.max(intsy_b) if np.max(intsy_b) > 0 else intsy_b

    # Build paired intensity vectors
    # For each peak in A, find best match in B
    used_b: set[int] = set()
    dot_product = 0.0

    for i in range(len(mz_a)):
        best_j = -1
        best_diff = mz_tolerance + 1

        for j in range(len(mz_b)):
            if j in used_b:
                continue
            diff = abs(mz_a[i] - mz_b[j])
            if diff < best_diff and diff <= mz_tolerance:
                best_diff = diff
                best_j = j

        if best_j >= 0:
            dot_product += norm_a[i] * norm_b[best_j]
            used_b.add(best_j)

    mag_a = np.sqrt(np.sum(norm_a ** 2))
    mag_b = np.sqrt(np.sum(norm_b ** 2))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_product / (mag_a * mag_b)
