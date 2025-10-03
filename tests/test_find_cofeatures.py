from typing import TYPE_CHECKING
from pathlib import Path

from core.data_structs import DataRegistry, FeaturePointer
from core.cli.find_cofeatures import (
    find_cofeatures_within_scan_array,
    find_cofeatures_across_scan_array,
)
from core.utils.persistence import load_project

import numpy as np
import pyqtgraph as pg

if TYPE_CHECKING:
    from core.data_structs import Sample, ScanArray

TARGET_MZ: float = 387.1803

def get_data_registry() -> 'DataRegistry':

    samples: list['Sample'] = load_project(
        filepath=Path('test_project.mzk')
    )

    data_registry = DataRegistry()
    data_registry.register_samples(samples)

    return data_registry


def test_find_cofeatures_within_scan_array(
    scan_array: 'ScanArray',
    show_plot: bool = False,
):
    search_ftr_ptr: FeaturePointer = get_test_featurepointer(scan_array)
    co_ftr_ptrs = find_cofeatures_within_scan_array(
        scan_array=scan_array,
        min_correlation=0.8,
        search_target=search_ftr_ptr,
    )

    if show_plot:
        test_plot(
            search_ftr=search_ftr_ptr,
            co_ftrs=co_ftr_ptrs,
            search_scan_array=scan_array,
            cofeature_scan_array=scan_array,
        )


def test_find_cofeatures_across_scan_array(
    source_scan_array: 'ScanArray',
    target_scan_array: 'ScanArray',
    show_plot: bool = False,
):
    search_ftr = get_test_featurepointer(source_scan_array)
    cofeatures = find_cofeatures_across_scan_array(
        source_scan_array, target_scan_array,
        search_ftr,
        0.8,
    )

    if show_plot:
        test_plot(
            search_ftr=search_ftr,
            co_ftrs=cofeatures,
            search_scan_array=source_scan_array,
            cofeature_scan_array=target_scan_array,
        )

def test_plot(
    search_ftr: 'FeaturePointer',
    co_ftrs: list['FeaturePointer'],
    search_scan_array: 'ScanArray',
    cofeature_scan_array: 'ScanArray',
):
    pw = pg.plot(
        search_ftr.get_retention_times(search_scan_array),
        search_ftr.get_intensity_values(search_scan_array),
        pen='r',
    )

    for ftr in co_ftrs:
        pw.plot(
            ftr.get_retention_times(cofeature_scan_array),
            ftr.get_intensity_values(cofeature_scan_array),
            pen='g',
        )

    pg.exec()


def get_test_featurepointer(
    scan_array: 'ScanArray',
):
    # Get mz lane idx
    mz_lane_idx: int = np.where(
        np.abs(
            TARGET_MZ - scan_array.mz_lane_label
        ) < 0.01
    )[0].max()

    # Get scan idxs
    scan_idxs: np.ndarray[...,] = scan_array.intsy_arr[mz_lane_idx].nonzero()[0]

    ftr_ptr = scan_array.make_feature_pointer(
        mass_lane_idx=mz_lane_idx,
        scan_idxs=scan_idxs,
    )

    return ftr_ptr


if __name__ == "__main__":
    data_registry: 'DataRegistry' = get_data_registry()
    sample: 'Sample' = data_registry.get_all_samples()[0]

    ms1_scan_array: 'ScanArray' = sample.injection.scan_array_ms1
    ms2_scan_array: 'ScanArray' = sample.injection.scan_array_ms2

    test_find_cofeatures_within_scan_array(
        scan_array=ms1_scan_array,
        show_plot=True,
    )
    test_find_cofeatures_across_scan_array(
        source_scan_array=ms1_scan_array,
        target_scan_array=ms2_scan_array,
        show_plot=True,
    )









