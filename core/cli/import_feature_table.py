"""
Import a feature table (m/z + RT coordinates) and generate
Ensembles by locating signals in the raw LC/MS data.

The idea: the user runs feature-finding elsewhere (MZmine,
XCMS, MS-DIAL, etc.), then imports the resulting coordinates
here to reconnect them to the actual MS data via MzKit's
ensemble system.

For each coordinate × each sample, we:
    1. Find the nearest m/z lane in the ScanArray
    2. Extract a FeaturePointer around the target RT
    3. Run cofeature-finding to build a full Ensemble
    4. Group results into an EnsembleAlignment
"""
import logging
import threading
from typing import Callable, NamedTuple, Optional, TYPE_CHECKING

import numpy as np

from core.data_structs import Ensemble
from core.cli.generate_ensemble import get_cofeature_ensemble

if TYPE_CHECKING:
    from core.data_structs import (
        Sample, SampleUUID, EnsembleUUID,
        FeaturePointer, Injection,
    )
    from core.data_structs.alignment import EnsembleAlignment, AlignedAnalyte

logger = logging.getLogger(__name__)


class FeatureCoordinate(NamedTuple):
    mz: float
    rt: float
    analyte_id: str = ""


class FeatureTableImportParams(NamedTuple):
    """
    rt_window: Half-width in seconds for locating the parent
        feature around the target RT.
    mz_window: Tolerance in Da for finding the m/z lane.
    ms1_corr_threshold: Pearson correlation threshold for
        MS1 cofeature grouping.
    ms2_corr_threshold: Pearson correlation threshold for
        MS2 cofeature grouping.
    min_intsy: Minimum intensity for a cofeature signal.
    use_rel_intsy: Normalize chromatograms before correlation.
    pregroup: Whether to pre-group redundant features by peak
        shape before importing.
    pregroup_rt_tolerance: Maximum RT difference (seconds)
        between features to consider for grouping.
    pregroup_corr_threshold: Minimum Pearson correlation to
        group two features together.
    """
    rt_window: float = 4.0
    mz_window: float = 0.01
    ms1_corr_threshold: float = 0.8
    ms2_corr_threshold: float = 0.7
    min_intsy: float = 1000.0
    use_rel_intsy: bool = True
    pregroup: bool = False
    pregroup_rt_tolerance: float = 5.0
    pregroup_corr_threshold: float = 0.8


def import_feature_table(
    features: list[FeatureCoordinate],
    samples: list['Sample'],
    params: FeatureTableImportParams,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> 'EnsembleAlignment':
    """
    Given a list of feature coordinates and samples, generate
    Ensembles by locating signals in the raw data, then build
    an EnsembleAlignment.

    Each feature coordinate becomes one AlignedAnalyte row.
    For each sample, if signal is found at that coordinate,
    the resulting Ensemble is linked in the ensemble_map.

    :param features: List of (mz, rt, analyte_id) coordinates
    :param samples: Samples with Injections to search
    :param params: Import parameters
    :return: EnsembleAlignment with one analyte per feature
    """
    from core.data_structs.alignment import (
        AlignedAnalyte, EnsembleAlignment, AlignmentParams,
    )

    # Filter to samples that have injections
    valid_samples = [s for s in samples if s.injection]
    if not valid_samples:
        raise ValueError("No samples with injection data provided")

    if params.pregroup:
        features = pregroup_features(features, valid_samples, params)

    analytes: list[AlignedAnalyte] = []
    n_features = len(features)
    cancelled = False

    for i, feat in enumerate(features):
        if cancel_event is not None and cancel_event.is_set():
            logger.info(
                f"Feature table import cancelled after "
                f"{i}/{n_features} features"
            )
            cancelled = True
            break

        if progress_callback is not None:
            progress_callback(
                100.0 * i / max(n_features, 1),
                f"Feature {i + 1}/{n_features} "
                f"(mz={feat.mz:.4f} rt={feat.rt:.1f})",
            )

        ensemble_map: dict['SampleUUID', 'EnsembleUUID'] = {}

        for sample in valid_samples:
            ensemble = _extract_ensemble_at_coordinate(
                injection=sample.injection,
                target_mz=feat.mz,
                target_rt=feat.rt,
                params=params,
            )

            if ensemble is not None:
                ensemble_map[sample.uuid] = ensemble.uuid

        analyte = AlignedAnalyte(
            ensemble_map=ensemble_map,
            consensus_mz=feat.mz,
            consensus_rt=feat.rt,
        )
        analytes.append(analyte)

        matched = len(ensemble_map)
        total = len(valid_samples)
        logger.info(
            f"Feature mz={feat.mz:.4f} rt={feat.rt:.1f}: "
            f"found in {matched}/{total} samples"
        )

    alignment = EnsembleAlignment(
        sample_uuids=tuple(s.uuid for s in valid_samples),
        analytes=analytes,
        parameters=AlignmentParams(),
    )

    if progress_callback is not None and not cancelled:
        progress_callback(100.0, "Done")

    logger.info(
        f"Feature table import complete: "
        f"{len(features)} features, "
        f"{len(valid_samples)} samples, "
        f"{len(analytes)} analytes"
    )

    return alignment


def _extract_ensemble_at_coordinate(
    injection: 'Injection',
    target_mz: float,
    target_rt: float,
    params: FeatureTableImportParams,
) -> Optional[Ensemble]:
    """
    Attempt to extract an Ensemble at a given (mz, rt) coordinate
    within an Injection's MS1 ScanArray.

    Returns None if no signal is found at the coordinate.
    """
    scan_array = injection.scan_array_ms1

    ftr_ptr = scan_array.extract_feature_pointer(
        target_mz=target_mz,
        mz_window=params.mz_window,
        target_rt=target_rt,
        rt_window=params.rt_window,
    )

    if ftr_ptr is None:
        return None

    # Check that there's actually meaningful signal
    intsy_values = ftr_ptr.get_intensity_values(scan_array)
    if intsy_values.size == 0:
        return None
    max_intsy = intsy_values.max()
    if max_intsy < params.min_intsy:
        return None

    ensemble = get_cofeature_ensemble(
        injection=injection,
        search_ftr_ptr=ftr_ptr,
        ms1_corr_threshold=params.ms1_corr_threshold,
        ms2_corr_threshold=params.ms2_corr_threshold,
        min_intsy=params.min_intsy,
        use_rel_intsy=params.use_rel_intsy,
    )

    return ensemble


def pregroup_features(
    features: list[FeatureCoordinate],
    samples: list['Sample'],
    params: FeatureTableImportParams,
) -> list[FeatureCoordinate]:
    """
    Group redundant features by peak shape correlation, returning
    one representative per group (the most intense).

    Algorithm:
        1. For each feature, find the sample with strongest signal
           and extract a FeaturePointer + max intensity
        2. Sort features by max intensity descending
        3. Greedy grouping: seed from the most intense ungrouped
           feature. For each candidate within RT tolerance,
           extract both chromatograms from the seed's best sample
           over the seed's scan range, correlate using non-zero
           overlap (same logic as cofeature finding). Absorb if
           above threshold.
        4. Return the seed of each group

    Features with no detectable signal in any sample are dropped.
    """
    from core.cli.find_cofeatures import _calculate_pearson_correlations

    # Step 1: for each feature, find the best sample and extract
    # a FeaturePointer from it
    feat_info: list[tuple[
        FeatureCoordinate,
        float,              # max intensity
        'FeaturePointer',   # feature pointer in best sample
        'ScanArray',        # scan array of best sample
    ]] = []

    for feat in features:
        best_intsy = 0.0
        best_ftr_ptr = None
        best_scan_array = None

        for sample in samples:
            scan_array = sample.injection.scan_array_ms1
            ftr_ptr = scan_array.extract_feature_pointer(
                target_mz=feat.mz,
                mz_window=params.mz_window,
                target_rt=feat.rt,
                rt_window=params.rt_window,
            )
            if ftr_ptr is None:
                continue

            intsy_values = ftr_ptr.get_intensity_values(scan_array)
            if intsy_values.size == 0:
                continue

            max_val = float(intsy_values.max())
            if max_val > best_intsy:
                best_intsy = max_val
                best_ftr_ptr = ftr_ptr
                best_scan_array = scan_array

        if best_ftr_ptr is not None and best_intsy >= params.min_intsy:
            feat_info.append((feat, best_intsy, best_ftr_ptr, best_scan_array))

    if not feat_info:
        return []

    # Step 2: sort by intensity descending
    feat_info.sort(key=lambda x: x[1], reverse=True)

    # Step 3: greedy grouping
    n = len(feat_info)
    assigned = [False] * n
    groups: list[FeatureCoordinate] = []

    for i in range(n):
        if assigned[i]:
            continue

        assigned[i] = True
        seed_feat, _, seed_ftr_ptr, seed_scan_array = feat_info[i]
        groups.append(seed_feat)

        # Extract seed chromatogram (normalized)
        seed_xic = seed_ftr_ptr.get_intensity_values(seed_scan_array)
        if seed_xic.max() > 0:
            seed_xic_norm = seed_xic / seed_xic.max()
        else:
            continue

        # Collect candidates within RT tolerance
        candidates = []
        candidate_indices = []
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            other_feat = feat_info[j][0]
            if abs(seed_feat.rt - other_feat.rt) > params.pregroup_rt_tolerance:
                continue
            candidates.append(j)
            candidate_indices.append(j)

        if not candidates:
            continue

        # Extract candidate chromatograms from the seed's sample
        # over the seed's scan range — same pattern as cofeature finding
        candidate_xics = []
        for j in candidates:
            other_feat = feat_info[j][0]
            other_ftr_ptr = seed_scan_array.extract_feature_pointer(
                target_mz=other_feat.mz,
                mz_window=params.mz_window,
                target_rt=seed_feat.rt,
                rt_window=params.rt_window,
            )
            if other_ftr_ptr is None:
                candidate_xics.append(np.zeros_like(seed_xic))
                continue

            other_xic = other_ftr_ptr.get_intensity_values(seed_scan_array)

            # Align to seed length
            if other_xic.size == seed_xic.size:
                xic = other_xic
            elif other_xic.size > seed_xic.size:
                xic = other_xic[:seed_xic.size]
            else:
                xic = np.zeros_like(seed_xic)
                xic[:other_xic.size] = other_xic

            if xic.max() > 0:
                xic = xic / xic.max()

            candidate_xics.append(xic)

        if not candidate_xics:
            continue

        candidate_xic_arr = np.array(candidate_xics)

        # Use the same correlation function as cofeature finding
        correlations = _calculate_pearson_correlations(
            candidate_xics=candidate_xic_arr,
            search_xic=seed_xic_norm,
        )

        for k, j in enumerate(candidates):
            if (not np.isnan(correlations[k])
                    and correlations[k] >= params.pregroup_corr_threshold):
                assigned[j] = True

    logger.info(
        f"Pre-grouping: {len(features)} features -> "
        f"{len(groups)} groups "
        f"({len(features) - len(feat_info)} had no signal)"
    )

    return groups
