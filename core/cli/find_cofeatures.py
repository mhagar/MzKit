"""
Given a ScanArray, a scan number, and the index of a signal in
the scan number's corresponding ScanArray, returns a list of
FeaturePointers which specify features in the ScanArray that
have matching peak-shapes (within some correlation R)
"""
import numpy as np

from core.data_structs import FeaturePointer, ScanArray


def find_cofeatures_within_scan_array(
    scan_array: 'ScanArray',
    search_target: 'FeaturePointer',
    min_correlation: float,
    min_intsy: float,
    use_rel_intsy: bool,
) -> list['FeaturePointer']:
    """
    Given a *SOURCE* ScanArray, and a target FeaturePointer,
    returns a list of FeaturePointers with the same scan window duration
    that give a Pearson correlation > min_correlation
    :param scan_array:
        ScanArray to search for cofeatures
    :param search_target:
        FeaturePointer to use as reference
    :param min_intsy:
        Minimum intensity that a signal must have to be considered for analysis
    :param min_correlation:
        Pearson correlation threshold to be considered cofeature
    :param use_rel_intsy:
        Whether to use absolute or relative intensities when calculating
        Pearson correlation
    :return:
    """
    # Find mass lanes within the search scan range that actually have signals
    nonzero_mass_lane_idxs = _find_nonzero_mass_lanes(
        scan_array=scan_array,
        scan_idxs=search_target.scan_idxs,
        min_intsy=min_intsy,
    )

    # Generate XICs for all cofeatures and the search cofeature
    candidate_xics: np.ndarray[..., ...] = _get_xic_grid(
        mass_lane_idxs=nonzero_mass_lane_idxs,
        scan_array=scan_array,
        scan_start=search_target.scan_start,
        scan_end=search_target.scan_end,
        use_rel_intsy=use_rel_intsy,
    )
    search_xic = search_target.get_intensity_values(
        scan_array
    )

    if use_rel_intsy:
        search_xic /= search_xic.max()

    # Calculate their Pearson correlation coeffs against search_target xic
    correlations = _calculate_pearson_correlations(
        candidate_xics,
        search_xic,
    )

    # For testing:
    print("Pearson correlations:")
    for xic_idx, mass_lane_idx in enumerate(nonzero_mass_lane_idxs):
        mz = scan_array.mz_lane_label[mass_lane_idx]
        corr = correlations[xic_idx]
        print(f"mz: {mz} \t corr: {corr}")

    # Get the ones that surpass min corr. threshold
    matching_mass_lane_idxs = _filter_candidates_by_correlation(
        correlations,
        min_correlation,
        nonzero_mass_lane_idxs,
    )

    matching_cofeatures: list['FeaturePointer'] = [search_target]
    for mass_lane_idx in matching_mass_lane_idxs:
        if mass_lane_idx == search_target.mz_lane_idx:
            continue
            
        matching_cofeatures.append(
            scan_array.make_feature_pointer(
                mass_lane_idx=mass_lane_idx,
                scan_idxs=search_target.scan_idxs,
            ),
        )

    return matching_cofeatures


def _get_xic_grid(
    mass_lane_idxs: np.ndarray,
    scan_array: 'ScanArray',
    scan_start: int,
    scan_end: int,
    use_rel_intsy: bool,
) -> np.ndarray:
    """
    Generates a 2D np.ndarray representing XICs.

    :param scan_array: ScanArray to parse
    :param mass_lane_idxs: Indices of ScanArray to parse
    :param scan_start: Scan to start with
    :param scan_end: Scan to end with
    :return:
    """
    candidate_xics = scan_array.intsy_arr[
                     mass_lane_idxs,
                     scan_start:scan_end,
                     ].toarray()


    # If requested, normalize (use_rel_intsy)
    if use_rel_intsy:
        candidate_xics = (  # Normalize against max intsy in each ms lane
            candidate_xics
            /
            np.max(
                candidate_xics,
                axis=1,
            ).reshape(-1, 1)
        )

    return candidate_xics


def _find_nonzero_mass_lanes(
    scan_array: 'ScanArray',
    scan_idxs: np.ndarray,
    min_intsy: float,
):
    mass_lane_intsy_max = scan_array.intsy_arr.toarray()[
                           :, scan_idxs
                           ].max(axis=1)
    nonzero_mass_lane_idxs: np.ndarray = np.where(
        mass_lane_intsy_max > min_intsy,
    )[0]
    return nonzero_mass_lane_idxs


def _filter_candidates_by_correlation(
    correlations: np.ndarray,
    min_correlation: float,
    mass_lane_idxs: np.ndarray,
) -> np.ndarray:
    matching_mass_lane_idxs = mass_lane_idxs[
        np.where(
            correlations > min_correlation
        )[0]
    ]
    return matching_mass_lane_idxs


def _calculate_pearson_correlations(
    candidate_xics: np.ndarray[..., ...],
    search_xic: np.ndarray,
    min_nonzero_overlap: int = 4,  # TODO: Expose to user
) -> np.ndarray:
    """
    Calculate Pearson correlation considering only non-zero elements.

    :param candidate_xics: 2D array of candidate XICs (n_candidates x n_timepoints)
    :param search_xic: 1D array of the search XIC (n_timepoints)
    :param min_nonzero_overlap: Minimum number of overlapping non-zero elements
                                 required for a valid correlation (default: 3)
    :return: Array of correlation coefficients (NaN for insufficient overlap)
    """
    n_candidates = candidate_xics.shape[0]
    correlations = np.full(n_candidates, np.nan)

    for i in range(n_candidates):
        candidate_xic = candidate_xics[i]

        # Find positions where both arrays are non-zero
        nonzero_mask = (candidate_xic != 0) & (search_xic != 0)
        n_overlap = np.sum(nonzero_mask)

        # If insufficient overlap, leave as NaN
        if n_overlap < min_nonzero_overlap:
            continue

        # Extract non-zero overlapping values
        candidate_nonzero = candidate_xic[nonzero_mask]
        search_nonzero = search_xic[nonzero_mask]

        # Calculate means of non-zero elements
        candidate_mean = candidate_nonzero.mean()
        search_mean = search_nonzero.mean()

        # Center the values
        candidate_centered = candidate_nonzero - candidate_mean
        search_centered = search_nonzero - search_mean

        # Calculate correlation
        numerator = np.dot(candidate_centered, search_centered)
        denominator = np.sqrt(
            np.sum(candidate_centered ** 2) * np.sum(search_centered ** 2)
        )

        # Avoid division by zero
        if denominator == 0:
            correlations[i] = np.nan
        else:
            correlations[i] = numerator / denominator

    return correlations


def find_cofeatures_across_scan_array(
    source_scan_array: 'ScanArray',
    target_scan_array: 'ScanArray',
    search_target: 'FeaturePointer',
    min_correlation: float,
    min_intsy: float,
    use_rel_intsy: bool,
) -> list['FeaturePointer']:
    """
    Given a *TARGET* ScanArray, and a search_target FeaturePointer,
    returns a list of FeaturePointers with the same *retention time* window
    duration that give a Pearson correlation > min_correlation

    This is different from find_cofeatures_within_scan_array() because
    it's used to match features from a different scan set (i.e. grouping
    fragments in a DIA experiment)
    :param source_scan_array:
        ScanArray where the search_target points (i.e. MS1)
    :param target_scan_array:
        ScanArray to parse for matches (i.e. MS2)
    :param search_target:
        FeaturePointer to use as reference
    :param min_intsy:
        Minimum intensity that a signal must have to be considered for analysis
    :param min_correlation:
        Pearson correlation threshold to be considered cofeature
    :param use_rel_intsy:
        Whether to use absolute or relative intensities when calculating
        Pearson correlation
    :return:
    """
    # Find the search scan range that corresponds to the rt of search_target
    rts_search: np.ndarray = search_target.get_retention_times(
        source_scan_array
    )
    rt_start = rts_search.min()
    rt_end = rts_search.max()

    target_scan_idxs: np.ndarray = np.where(
        (rt_start < target_scan_array.rt_arr) &
        (target_scan_array.rt_arr < rt_end)
    )[0]
    target_scan_start = target_scan_idxs.min()
    target_scan_end = target_scan_idxs.max()

    # Find mass lanes within the search scan range that actually have signals
    nonzero_mass_lane_idxs = _find_nonzero_mass_lanes(
        scan_array=target_scan_array,
        scan_idxs=target_scan_idxs,
        min_intsy=min_intsy,
    )

    # Get a grid of XIC/intensity values
    candidate_xics: np.ndarray[..., ...] = _get_xic_grid(
        mass_lane_idxs=nonzero_mass_lane_idxs,
        scan_array=target_scan_array,
        scan_start=target_scan_start,
        scan_end=target_scan_end,
        use_rel_intsy=use_rel_intsy,
    )

    # Interpolate search_xic based on the retention times of target_scan_array
    rt_source: np.ndarray = source_scan_array.rt_arr[
        search_target.scan_start: search_target.scan_end
    ]

    rt_target: np.ndarray = target_scan_array.rt_arr[
        target_scan_start:target_scan_end,
    ]

    search_xic: np.ndarray[...,] = search_target.get_intensity_values(
        source_scan_array,
    )
    if use_rel_intsy:
        search_xic /= search_xic.max()

    interp_search_xic: np.ndarray = np.interp(
        x=rt_target,
        xp=rt_source,
        fp=search_xic,
    )

    # Calculate XIC grid Pearson correlation coeffs against *interpolated*
    #   search_target xic
    correlations = _calculate_pearson_correlations(
        candidate_xics=candidate_xics,
        search_xic=interp_search_xic,  # type: ignore
    )

    # Get the ones that surpass min threshold
    matching_mass_lane_idxs = _filter_candidates_by_correlation(
        correlations=correlations,
        min_correlation=min_correlation,
        mass_lane_idxs=nonzero_mass_lane_idxs,
    )

    matching_cofeatures: list['FeaturePointer'] = []
    for idx in matching_mass_lane_idxs:
        matching_cofeatures.append(
            target_scan_array.make_feature_pointer(
                mass_lane_idx=idx,
                scan_idxs=target_scan_idxs, # type: ignore
            ),
        )

    if len(matching_cofeatures) == 0:
        print('hi')

    return matching_cofeatures


def get_all_features_in_scan_array(
    scan_array: 'ScanArray',
    rt_start: float,
    rt_end: float,
    min_intsy: float,
) -> list['FeaturePointer']:
    """
    Used for testing. Generates an 'ensemble' without
    doing any feature grouping, i.e. just a window into
    the ScanArray
    """
    # Convert rt_start and rt_end into scan numbers
    scan_start = scan_array.rt_to_scan_num(rt_start)
    scan_end = scan_array.rt_to_scan_num(rt_end)
    scan_idxs = np.arange(scan_start, scan_end + 1)

    # Find mass-lanes that are non-zero within that range
    mass_lane_idxs = _find_nonzero_mass_lanes(
        scan_array=scan_array,
        scan_idxs=scan_idxs,
        min_intsy=min_intsy,
    )

    features: list['FeaturePointer'] = []
    for idx in mass_lane_idxs:
        ftr_ptr = scan_array.make_feature_pointer(
            mass_lane_idx=idx,
            scan_idxs=scan_idxs,  # type: ignore
        )

        features.append(
            ftr_ptr
        )

    return features


def backfill_isotope_envelope(
    search_ftr_ptrs: list['FeaturePointer'],
    candidate_ftr_ptrs: list['FeaturePointer'],
    scan_array: 'ScanArray',
    min_corr_threshold: float,
    isotopologue_offset_window: tuple[float, float],
) -> list['FeaturePointer']:
    """
    Given a list of FeaturePointers, checks their originating ScanArray
    to see if any of them are actually *isotopologues*.

    In order to qualify, candidate feature pointers must:
        - Have a lower intensity than the search_ftr_ptr

        - Have a mass offset within isotopologue_offset_window

        - Have an intensity that correlates with the search_ftr_ptr within
            min_corr_threshold
    """
    pass


