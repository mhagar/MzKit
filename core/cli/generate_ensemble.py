"""
Script for extracting a co-feature ensemble
from a pair of MS1 and MS2 scan arrays given a search
feature pointer (from the MS1 array).

These must be set to an Injection to be viable
"""

from typing import NamedTuple, TYPE_CHECKING

from core.data_structs import Ensemble
from core.cli.find_cofeatures import (
    find_cofeatures_within_scan_array,
    find_cofeatures_across_scan_array,
    get_all_features_in_scan_array, # for testing
)

if TYPE_CHECKING:
    from core.data_structs import (
        FeaturePointer, ScanArray,
        Injection
    )


class InputParams(NamedTuple):
    search_ftr_ptr: 'FeaturePointer'
    injection: 'Injection'
    ms1_corr_threshold: float
    ms2_corr_threshold: float
    min_intsy: float
    use_rel_intsy: bool


def get_cofeature_ensembles(
    search_ftr_ptrs: list[ 'FeaturePointer' ],
    injection: 'Injection',
    ms1_corr_threshold: float,
    ms2_corr_threshold: float,
    min_intsy: float,
    use_rel_intsy: bool,
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
        # This conditiona lis just for testing:
        # if use_rel_intsy:
        #     print("UNGROUPED ENSEMBLE")
        #     ensemble = get_ungrouped_ensemble(
        #         injection=injection,
        #         search_ftr_ptr=search_ftr_ptr,
        #         min_intsy=min_intsy,
        #     )
        #
        # else:
        #     print("GROUPED ENSEMBLE")
        #     ensemble = get_cofeature_ensemble(
        #         injection=injection,
        #         min_intsy=min_intsy,
        #         ms1_corr_threshold=ms1_corr_threshold,
        #         ms2_corr_threshold=ms2_corr_threshold,
        #         search_ftr_ptr=search_ftr_ptr,
        #         use_rel_intsy=use_rel_intsy,
        #     )

        ensemble = get_cofeature_ensemble(
            injection=injection,
            min_intsy=min_intsy,
            ms1_corr_threshold=ms1_corr_threshold,
            ms2_corr_threshold=ms2_corr_threshold,
            search_ftr_ptr=search_ftr_ptr,
            use_rel_intsy=use_rel_intsy,
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
) -> Ensemble:
    ms1_cofeatures = find_cofeatures_within_scan_array(
        scan_array=injection.scan_array_ms1,
        search_target=search_ftr_ptr,
        min_correlation=ms1_corr_threshold,
        min_intsy=min_intsy,
        use_rel_intsy=use_rel_intsy,
    )
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
    )
    ensemble.set_injection(
        injection
    )
    return ensemble


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










