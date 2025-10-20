"""
Utility functions for ensemble viewer operations
"""
import numpy as np
from numpy.typing import NDArray
from typing import Optional


def match_chrom_arrs(
    reference_chrom: np.ndarray[float],
    target_chrom: np.ndarray[float],
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a reference chrom_array and a target chrom_array,
    returns a slice where the two chroms overlap in time.

    :param reference_chrom:
    :param target_chrom:
    :param normalize: If true, normalizes intensities of both chroms
    such that their maximum intsy = 1
    :return:
    """
    # Slice two arrays into just the overlapping regions
    (ref_start, ref_end), (tgt_start, tgt_end) = find_overlap_region(
        reference_chrom['rt'],
        target_chrom['rt'],
    )

    if not any([ref_start, ref_end, tgt_start, tgt_end]):
        # find_overlap_region() found no overlap
        return np.array([]), np.array([])

    ref_arr = reference_chrom[ref_start: ref_end].copy()
    tgt_arr = target_chrom[tgt_start: tgt_end].copy()

    # Occasionally, the length still mismatches by 1 scan. Fix:
    while len(ref_arr) != len(tgt_arr):
        if len(ref_arr) > len(tgt_arr):
            ref_arr = ref_arr[:-1]
        else:
            tgt_arr = tgt_arr[:-1]

    # Normalize both of them to 1
    if normalize:
        ref_arr['intsy'] = ref_arr['intsy'] / max(ref_arr['intsy'])
        tgt_arr['intsy'] = tgt_arr['intsy'] / max(tgt_arr['intsy'])

    return ref_arr, tgt_arr


def find_overlap_region(
    arr_a: np.ndarray[float],
    arr_b: np.ndarray[float],
) -> Optional[tuple[tuple, tuple]]:
    """
    Returns indices where arr_a and arr_b overlap in values,
    assuming that both arrays contain monotonically increasing elements

    (i.e. represent successive retention time values)
    """
    start = max(arr_a[0], arr_b[0])
    end = min(arr_a[-1], arr_b[-1])

    if start > end:
        return (None, None), (None, None)  # No overlap

    # Find indices for overlapping region
    a_start_idx = np.searchsorted(
        arr_a, start, side='left',
    )

    a_end_idx = np.searchsorted(
        arr_a, end, side='right',
    )

    b_start_idx = np.searchsorted(
        arr_b, start, side='left',
    )

    b_end_idx = np.searchsorted(
        arr_b, end, side='right',
    )

    return (a_start_idx, a_end_idx), (b_start_idx, b_end_idx)


def normalize_chrom_arr(
    chrom: np.ndarray
) -> np.ndarray:
    """
    Returns a chrom array that's been normalized such that
    maximum intensity = 1.0
    """
    arr = chrom.copy()
    arr['intsy'] = arr['intsy']/max(arr['intsy'])

    return arr


def diff_chrom_arr(
    chrom: np.ndarray
) -> np.ndarray:
    """
    Returns a chrom array that's been 'differentiated'
    by subtracting each intsy value from the next
    """
    arr = chrom.copy()
    arr['intsy'] = np.diff(arr['intsy'], append=0.0)

    return arr


def get_pearson_coeff(
    arr_a: NDArray,
    arr_b: NDArray,
) -> float:
    """
    FOR TESTING
    """
    corr_matrix: NDArray[np.float64, np.float64] = np.corrcoef(arr_a, arr_b)
    return corr_matrix[0, 1]
