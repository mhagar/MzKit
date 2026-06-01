"""
Script for extracting a co-feature ensemble
from a pair of MS1 and MS2 scan arrays given a search
feature pointer (from the MS1 array).

These must be set to an Injection to be viable
"""
from collections import defaultdict
from typing import NamedTuple, TYPE_CHECKING

import numpy as np

from core.data_structs import Ensemble
from core.cli.find_cofeatures import (
    find_cofeatures_within_scan_array,
    find_cofeatures_across_scan_array,
    get_all_features_in_scan_array, # for testing
)
from core.cli.segment_chromatogram import find_peak_boundaries, validate_peak

if TYPE_CHECKING:
    from core.data_structs import (
        FeaturePointer, ScanArray,
        Injection
    )


class EnsembleExtractionParams(NamedTuple):
    search_ftr_ptr: 'FeaturePointer'
    injection: 'Injection'
    ms1_corr_threshold: float
    ms2_corr_threshold: float
    min_intsy: float
    use_rel_intsy: bool
    # Only used when injection.acquisition_mode == 'dda'. Tuneable from
    # the settings menu post-ASMS; for now a wide default that comfortably
    # covers typical DDA isolation widths even when not explicitly encoded.
    precursor_mz_tolerance: float = 0.5


def get_cofeature_ensembles(
    search_ftr_ptrs: list[ 'FeaturePointer' ],
    injection: 'Injection',
    ms1_corr_threshold: float,
    ms2_corr_threshold: float,
    min_intsy: float,
    use_rel_intsy: bool,
    precursor_mz_tolerance: float = 0.5,
    progress_callback=None,  # injected by ProcessRunner; unused here
    cancel_event=None,       # injected by ProcessRunner; unused here
) -> list[ Ensemble ]:
    """
    Given a list of feature pointers, generates a
    list of Ensembles
    :param injection:
        Injection object to parse
    :param search_ftr_ptrs:
        FeaturePointers to use as references
    :param min_intsy:
        Minimum intensity that a signal must have to be considered for analysis
    :param ms1_corr_threshold:
        Pearson correlation threshold to be considered cofeature
    :param ms2_corr_threshold:
        Pearson correlation threshold to be considered cofeature
    :param use_rel_intsy:
        Whether to use absolute or relative intensities when calculating
        Pearson correlation
    :return:
    """

    ensembles: list[Ensemble] = []
    for search_ftr_ptr in search_ftr_ptrs:
        ensemble = get_cofeature_ensemble(
            injection=injection,
            min_intsy=min_intsy,
            ms1_corr_threshold=ms1_corr_threshold,
            ms2_corr_threshold=ms2_corr_threshold,
            search_ftr_ptr=search_ftr_ptr,
            use_rel_intsy=use_rel_intsy,
            precursor_mz_tolerance=precursor_mz_tolerance,
        )

        ensembles.append(
            ensemble
        )

    return ensembles


def get_cofeature_ensemble(
    injection: 'Injection',
    search_ftr_ptr: 'FeaturePointer',
    ms1_corr_threshold: float,
    ms2_corr_threshold: float,
    min_intsy: float,
    use_rel_intsy: bool,
    precursor_mz_tolerance: float = 0.5,
) -> Ensemble:
    ms1_cofeatures = find_cofeatures_within_scan_array(
        scan_array=injection.scan_array_ms1,
        search_target=search_ftr_ptr,
        min_correlation=ms1_corr_threshold,
        min_intsy=min_intsy,
        use_rel_intsy=use_rel_intsy,
    )

    ms2_cofeatures: list['FeaturePointer'] = []
    precursor_mz: 'float | None' = None
    precursor_charge: 'int | None' = None

    if injection.acquisition_mode == 'dda':
        # DDA: link MS2 by precursor m/z + RT window, not by correlation.
        ms2_cofeatures, precursor_mz, precursor_charge = (
            _dda_link_ms2_cofeatures(
                injection=injection,
                ms1_cofeatures=ms1_cofeatures,
                search_ftr_ptr=search_ftr_ptr,
                min_intsy=min_intsy,
                precursor_mz_tolerance=precursor_mz_tolerance,
            )
        )
    elif injection.scan_array_ms2 is not None:
        # DIA / MS1_only-but-MS2-present: original correlation path.
        ms2_cofeatures = find_cofeatures_across_scan_array(
            source_scan_array=injection.scan_array_ms1,
            target_scan_array=injection.scan_array_ms2,
            search_target=search_ftr_ptr,
            min_correlation=ms2_corr_threshold,
            min_intsy=min_intsy,
            use_rel_intsy=use_rel_intsy,
        )

    ensemble = Ensemble(
        ms1_cofeatures=ms1_cofeatures,
        ms2_cofeatures=ms2_cofeatures,
        precursor_mz=precursor_mz,
        precursor_charge=precursor_charge,
    )

    injection.add_ensemble(ensemble)
    return ensemble


def _dda_link_ms2_cofeatures(
    injection: 'Injection',
    ms1_cofeatures: list['FeaturePointer'],
    search_ftr_ptr: 'FeaturePointer',
    min_intsy: float,
    precursor_mz_tolerance: float,
) -> tuple[list['FeaturePointer'], 'float | None', 'int | None']:
    """
    DDA MS2 linkage: take every MS2 scan whose precursor m/z matches any
    MS1 cofeature (within `precursor_mz_tolerance`) AND whose RT falls
    inside the search feature's RT window. The resulting MS2 cofeatures
    are FeaturePointers, one per MS2 mass lane that carries signal in
    those scans, with `scan_idxs` restricted to the matched MS2 scans.

    Returns the cofeatures, plus the precursor m/z (median of matched
    scans) and precursor charge (mode of matched scans) — both `None`
    if nothing matched.
    """
    ms1_arr = injection.scan_array_ms1
    ms2_arr = injection.scan_array_ms2
    if ms2_arr is None or ms2_arr.precursor_mz_arr is None:
        return [], None, None

    # RT window from the search feature.
    search_rts = search_ftr_ptr.get_retention_times(ms1_arr)
    rt_lo, rt_hi = float(search_rts.min()), float(search_rts.max())

    # m/z of every MS1 cofeature (lane-label mean is good enough for matching).
    cofeature_mzs = ms1_arr.mz_lane_label[
        [cf.mz_lane_idx for cf in ms1_cofeatures]
    ]

    # Mask MS2 scans by RT window.
    rt_mask = (ms2_arr.rt_arr >= rt_lo) & (ms2_arr.rt_arr <= rt_hi)
    if not rt_mask.any():
        return [], None, None

    # Mask MS2 scans by precursor-mz proximity to any cofeature m/z.
    # Outer-difference matrix is fine for the cardinalities involved
    # (cofeatures ~ tens, MS2 scans ~ thousands).
    diff = np.abs(
        ms2_arr.precursor_mz_arr[:, None]
        - np.asarray(cofeature_mzs)[None, :]
    )
    mz_mask = (diff < precursor_mz_tolerance).any(axis=1)

    matched_ms2_idxs = np.where(rt_mask & mz_mask)[0]
    if matched_ms2_idxs.size == 0:
        return [], None, None

    # Find MS2 mass lanes that carry signal in the matched scans.
    intsy_slice = ms2_arr.intsy_arr[:, matched_ms2_idxs]
    lane_max = intsy_slice.max(axis=1).toarray().flatten()
    active_lane_idxs = np.where(lane_max >= min_intsy)[0]

    ms2_cofeatures: list['FeaturePointer'] = [
        ms2_arr.make_feature_pointer(
            mass_lane_idx=int(lane_idx),
            scan_idxs=matched_ms2_idxs,
        )
        for lane_idx in active_lane_idxs
    ]

    # Precursor metadata: median m/z and modal charge across matched scans.
    precursor_mz = float(np.median(ms2_arr.precursor_mz_arr[matched_ms2_idxs]))
    charges = ms2_arr.precursor_charge_arr[matched_ms2_idxs]
    nonzero_charges = charges[charges != 0]
    if nonzero_charges.size:
        vals, counts = np.unique(nonzero_charges, return_counts=True)
        precursor_charge = int(vals[counts.argmax()])
    else:
        precursor_charge = None

    return ms2_cofeatures, precursor_mz, precursor_charge


def get_ungrouped_ensemble(
    injection: 'Injection',
    search_ftr_ptr: 'FeaturePointer',
    min_intsy: float,
) -> Ensemble:
    """
    For *testing*; retrieves an 'ensemble' that's just a window
    into an Injection's ScanArrays
    """
    rts = search_ftr_ptr.get_retention_times(
        injection.get_scan_array(ms_level=1)
    )

    rt_start, rt_end = rts[0], rts[-1]
    print(
        f"rt start: {rt_start}\n"
        f"rt end: {rt_end}"
    )

    ms1_cofeatures: list['FeaturePointer'] = []
    ms2_cofeatures: list['FeaturePointer'] = []
    for ms_level, featurelist in [
            (1, ms1_cofeatures),
            (2, ms2_cofeatures),
    ]:
        featurelist +=  get_all_features_in_scan_array(
            scan_array=injection.get_scan_array(ms_level),
            rt_start=rt_start,# type:ignore
            rt_end=rt_end,    # type:ignore
            min_intsy=min_intsy,
        )

    ensemble = Ensemble(
        ms1_cofeatures=ms1_cofeatures,
        ms2_cofeatures=ms2_cofeatures,
    )
    ensemble.set_injection(injection)

    return ensemble


class AutoEnsembleParams(NamedTuple):
    """
    Parameters for automated ensemble generation.

    parent_threshold: Minimum peak height to seed a new ensemble.
        Only the tallest signal in an ensemble needs to exceed this.
    cofeature_threshold: Minimum intensity for a signal to be
        considered as a cofeature. Can be much lower than
        parent_threshold.
    ms1_corr_threshold: Pearson correlation threshold for MS1
        cofeature grouping.
    ms2_corr_threshold: Pearson correlation threshold for MS2
        cofeature grouping.
    use_rel_intsy: Whether to normalize chromatograms before
        computing Pearson correlation.
    extraction_half_width: Number of scans on each side of the
        apex to use for cofeature correlation. The extraction
        window is always 2 * extraction_half_width + 1 scans,
        regardless of peak shape.
    edge_fraction: For consumption boundaries — stop descending
        when intensity drops below this fraction of apex.
    min_rise_ratio: Peak apex must be at least this many times
        the edge intensity to be considered valid.
    min_peak_width: Peak must span at least this many scans to
        be considered valid.
    """
    parent_threshold: float
    cofeature_threshold: float
    ms1_corr_threshold: float
    ms2_corr_threshold: float
    use_rel_intsy: bool = True
    extraction_half_width: int = 10
    edge_fraction: float = 0.1
    min_rise_ratio: float = 2.0
    min_peak_width: int = 5
    rt_range: tuple[float, float] | None = None


def auto_generate_ensembles(
    injection: 'Injection',
    params: AutoEnsembleParams,
    progress_callback=None,  # injected by ProcessRunner; unused here
    cancel_event=None,       # injected by ProcessRunner; unused here
) -> list[Ensemble]:
    """
    Automatically discover and extract all ensembles in an
    Injection's MS1 ScanArray.

    Algorithm:
        1. Find the most intense unassigned signal across all
           m/z lanes
        2. Find consumption boundaries (valley/edge-aware)
        3. Validate the peak (rise ratio, width)
        4. If invalid: zero out consumption region, skip
        5. If valid: extract ensemble using a fixed window
           around the apex (±extraction_half_width scans)
        6. Zero out consumption region for the seed lane;
           mark cofeature regions as assigned
        7. Repeat until no unassigned signal exceeds parent_threshold

    :param injection: Injection with assembled ScanArrays
    :param params: AutoEnsembleParams controlling thresholds
    :return: List of generated Ensembles
    """
    scan_array: 'ScanArray' = injection.scan_array_ms1
    n_scans = scan_array.intsy_arr.shape[1]

    # 1D array tracking the current max intensity per m/z lane.
    # Updated as regions get consumed.
    lane_max_intsy: np.ndarray = (
        scan_array.intsy_arr
        .max(axis=1)
        .toarray()
        .flatten()
    )

    # Track which scan ranges have been assigned per lane
    assigned_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)

    # If RT range specified, build a mask of scan indices outside it
    rt_mask: np.ndarray | None = None
    if params.rt_range is not None:
        rt_start, rt_end = params.rt_range
        rt_mask = (
            (scan_array.rt_arr < rt_start) |
            (scan_array.rt_arr > rt_end)
        )
        # Zero out lane maxima outside the RT range
        allowed = ~rt_mask
        if allowed.any():
            allowed_idxs = np.where(allowed)[0]
            lane_max_intsy = (
                scan_array.intsy_arr[:, allowed_idxs[0]:allowed_idxs[-1] + 1]
                .max(axis=1)
                .toarray()
                .flatten()
            )

    ensembles: list[Ensemble] = []

    while True:
        seed_lane_idx: int = int(np.argmax(lane_max_intsy))

        if lane_max_intsy[seed_lane_idx] < params.parent_threshold:
            break

        # Extract this lane's chromatogram, zeroing assigned regions
        chromatogram: np.ndarray = (
            scan_array.intsy_arr[seed_lane_idx]
            .toarray()
            .flatten()
        )
        if rt_mask is not None:
            chromatogram[rt_mask] = 0
        for start, end in assigned_ranges.get(seed_lane_idx, []):
            chromatogram[start:end] = 0

        max_scan_idx: int = int(np.argmax(chromatogram))

        if chromatogram[max_scan_idx] < params.parent_threshold:
            lane_max_intsy[seed_lane_idx] = 0
            continue

        # Find consumption boundaries (valley/edge-aware)
        seg_start, seg_end = find_peak_boundaries(
            chromatogram,
            max_scan_idx,
            edge_fraction=params.edge_fraction,
        )

        # Validate: is this a real peak worth extracting?
        if not validate_peak(
            chromatogram,
            max_scan_idx,
            seg_start,
            seg_end,
            min_rise_ratio=params.min_rise_ratio,
            min_peak_width=params.min_peak_width,
        ):
            # Bad peak: zero out and skip, no cofeatures consumed
            _zero_out_lane_region(
                seed_lane_idx, seg_start, seg_end,
                assigned_ranges, lane_max_intsy, scan_array,
            )
            continue

        # Build extraction window: +-N scans around apex,
        # but clamped to peak boundaries for narrow peaks
        ext_start = max(seg_start, max_scan_idx - params.extraction_half_width)
        ext_end = min(seg_end, max_scan_idx + params.extraction_half_width + 1)
        extraction_scan_idxs = np.arange(ext_start, ext_end)

        seed_ftr_ptr = scan_array.make_feature_pointer(
            mass_lane_idx=seed_lane_idx,
            scan_idxs=extraction_scan_idxs,
        )

        # Find MS1 cofeatures
        ms1_cofeatures = find_cofeatures_within_scan_array(
            scan_array=scan_array,
            search_target=seed_ftr_ptr,
            min_correlation=params.ms1_corr_threshold,
            min_intsy=params.cofeature_threshold,
            use_rel_intsy=params.use_rel_intsy,
        )

        # Find MS2 cofeatures (if MS2 data exists)
        ms2_cofeatures: list['FeaturePointer'] = []
        if injection.scan_array_ms2 is not None:
            ms2_cofeatures = find_cofeatures_across_scan_array(
                source_scan_array=scan_array,
                target_scan_array=injection.scan_array_ms2,
                search_target=seed_ftr_ptr,
                min_correlation=params.ms2_corr_threshold,
                min_intsy=params.cofeature_threshold,
                use_rel_intsy=params.use_rel_intsy,
            )

        ensemble = Ensemble(
            ms1_cofeatures=ms1_cofeatures,
            ms2_cofeatures=ms2_cofeatures,
        )
        injection.add_ensemble(ensemble)
        ensembles.append(ensemble)

        # Zero out the seed lane's consumption region (full peak,
        # not just extraction window)
        _zero_out_lane_region(
            seed_lane_idx, seg_start, seg_end,
            assigned_ranges, lane_max_intsy, scan_array,
        )

        # Mark cofeature extraction regions as assigned
        _mark_assigned(
            ms1_cofeatures,
            assigned_ranges,
            lane_max_intsy,
            scan_array,
        )

    return ensembles


def _zero_out_lane_region(
    lane_idx: int,
    seg_start: int,
    seg_end: int,
    assigned_ranges: dict[int, list[tuple[int, int]]],
    lane_max_intsy: np.ndarray,
    scan_array: 'ScanArray',
):
    """
    Zero out a single lane's region and update its max intensity.
    Used for both valid peaks (consumption) and invalid peaks (skip).
    """
    assigned_ranges[lane_idx].append((seg_start, seg_end))
    lane_chrom = (
        scan_array.intsy_arr[lane_idx]
        .toarray()
        .flatten()
    )
    for start, end in assigned_ranges[lane_idx]:
        lane_chrom[start:end] = 0
    lane_max_intsy[lane_idx] = lane_chrom.max()


def _mark_assigned(
    cofeatures: list['FeaturePointer'],
    assigned_ranges: dict[int, list[tuple[int, int]]],
    lane_max_intsy: np.ndarray,
    scan_array: 'ScanArray',
):
    """
    After forming an ensemble, mark all cofeature scan regions
    as assigned and update the lane max intensity tracker.
    """
    affected_lanes: set[int] = set()

    for ftr in cofeatures:
        lane_idx = ftr.mz_lane_idx
        assigned_ranges[lane_idx].append(
            (ftr.scan_start, ftr.scan_end)
        )
        affected_lanes.add(lane_idx)

    # Recompute max intensity for each affected lane
    for lane_idx in affected_lanes:
        lane_chrom = (
            scan_array.intsy_arr[lane_idx]
            .toarray()
            .flatten()
        )
        for start, end in assigned_ranges[lane_idx]:
            lane_chrom[start:end] = 0

        lane_max_intsy[lane_idx] = lane_chrom.max()










