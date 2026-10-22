"""
FeaturePointer accessors read exactly `scan_idxs`: the last scan included,
and only the listed scans when they're sparse (DDA MS2 cofeatures).
"""
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_array

from core.data_structs import FeaturePointer


def _scan_array():
    # 2 lanes x 6 scans; lane 1 intensity == 10 * scan index
    intsy = np.zeros((2, 6))
    intsy[1] = np.arange(6) * 10.0
    return SimpleNamespace(
        uuid=1,
        intsy_arr=csr_array(intsy),
        mz_arr=csr_array(np.where(intsy > 0, 100.0, 0.0)),
        rt_arr=np.arange(6) * 0.5,
    )


def _ptr(scan_idxs):
    return FeaturePointer(
        mz_lane_idx=1, scan_idxs=np.array(scan_idxs),
        source_array_uuid=1, source_array_shape=(2, 6),
    )


def test_contiguous_includes_last_scan():
    arr = _scan_array()
    ptr = _ptr([2, 3, 4, 5])
    np.testing.assert_array_equal(ptr.get_intensity_values(arr), [20, 30, 40, 50])
    np.testing.assert_array_equal(ptr.get_retention_times(arr), [1.0, 1.5, 2.0, 2.5])
    assert ptr.get_mz_values(arr).size == 4
    assert ptr.get_max_intsy(arr) == 50


def test_sparse_and_single_scan():
    arr = _scan_array()
    np.testing.assert_array_equal(
        _ptr([5, 1, 3]).get_intensity_values(arr), [10, 30, 50]  # sorted, gaps skipped
    )
    assert _ptr([4]).get_max_intsy(arr) == 40


def test_apex_mz_is_one_measured_value():
    arr = _scan_array()
    arr.mz_arr = csr_array([[0] * 6, [0, 100.0, 100.2, 100.4, 100.1, 0]])
    arr.intsy_arr = csr_array([[0] * 6, [0, 5.0, 9.0, 7.0, 1.0, 0]])
    assert _ptr([0, 1, 2, 3, 4, 5]).get_apex_mz(arr) == 100.2
