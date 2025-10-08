"""
Given an Injection object:
    - Finds features
    - Groups them based on chromatographic peak-shape
"""
import pyopenms as oms
import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d
from scipy import sparse
from scipy.sparse.linalg import spsolve

from core.data_structs.scan_array import ScanArray
from core.data_structs.feature_pointer import FeaturePointer
from core.utils.array_types import SpectrumArray, to_spec_arr

from dataclasses import dataclass
from typing import Literal, Optional, TYPE_CHECKING
import argparse
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)


def build_features(
    scan_array: ScanArray,
    min_peak_length_in_scans: int,
    min_peak_height: float,
    prominence: float,
    min_num_scans_between_peaks: int,
) -> list[FeaturePointer]:
    """
    Given a ScanArray, returns a list of FeaturePointer objects
    that can be used as an index to retrieve MS signals.
    """
    feature_pointers: list[FeaturePointer] = []

    # Iterate over the ScanArray's mass lanes and define FeaturePointers based
    #   on chromatographic peaks

    rt_arr: np.ndarray = scan_array.rt_arr
    for lane_idx in range(0, scan_array.mz_arr.shape[0]):
        mz_arr: np.ndarray = scan_array.mz_arr[lane_idx].toarray()
        intsy_arr: np.ndarray = scan_array.intsy_arr[lane_idx].toarray()

        # Check tallest signal before comitting to peak finding
        if not _array_passes_threshold(
            intsy_arr,
            min_peak_height,
        ):
            continue

        # _, baseline_corr_intsy_arr, _ = adaptive_tophat(
        #    intsy_arr
        # )
        # baseline_corr_intsy_arr = baseline_correction(
        #    intsy_arr,
        #     lambda_value=100,
        #     porder=1,
        #     iterations=10,
        # )

        # Identify islands of 'non-zero' elements
        intsy_arr_chunks, chunk_idxs = _split_nonzero_islands(
            intsy_arr,
            return_island_idxs=True,
        )

        # Iterate over these islands and find peaks
        peak_windows: list[tuple[int, int]] = []
        for chunk, chunk_idxs in zip(
            intsy_arr_chunks,
            chunk_idxs,
        ):
            chunk: np.ndarray[float]        # Intensity values
            chunk_idxs: np.ndarray[int]     # Idxs in pre-split intsy_arr

            # Smooth the chunk (more effective peak finding)
            smoothed_chunk: np.ndarray = gaussian_filter1d(
                chunk,
                sigma=1.1,
            )


            peaks, properties = find_peaks(
                x=smoothed_chunk,
                prominence=prominence,
                width=min_peak_length_in_scans,
            )

            if not peaks.any():
                continue

            for peak_start, peak_end in zip(
                properties['left_bases'],
                properties['right_bases'],
            ):
                peak_start: int
                peak_end: int

                peak_windows.append(
                    (
                        chunk_idxs[peak_start],  # convert to pre-split idx
                        chunk_idxs[peak_end],
                    )
                )

        if len(peak_windows) == 0:
            continue

        # Now create FeaturePointers
        for peak_window in peak_windows:
            if intsy_arr[peak_window[0]:peak_window[1]].max() < min_peak_height:
                continue

            if peak_window[1] - peak_window[0] > 400:
                continue

            feature_pointer = FeaturePointer(
                mz_lane_idx=lane_idx,
                scan_start_idx=peak_window[0],
                scan_end_idx=peak_window[1],
            )

            feature_pointers.append(
                feature_pointer
            )

        print(
            f"Finished lane {lane_idx}"
        )
    return feature_pointers


def _split_nonzero_islands(
    arr: np.ndarray,
    return_island_idxs: bool = False,
) -> tuple[list[np.ndarray], Optional[list[np.ndarray]]]:
    """
    Given a 1D array, returns a list of arrays corresponding to
    'islands' of non-zero elements
    """
    nonzero_idxs = np.nonzero(arr)[0]

    if len(nonzero_idxs) == 0:
        return []  # Array is all 0's

    breaks = np.where(
        np.diff(nonzero_idxs) > 1
    )[0] + 1

    # Get split indices
    island_idxs: list[np.ndarray] = np.split(
        nonzero_idxs,
        breaks,
    )

    islands = [
        arr[idxs] for idxs in island_idxs
    ]

    if return_island_idxs:
        return islands, island_idxs

    return islands, None


def _array_passes_threshold(
    arr: np.ndarray,
    threshold: float,
) -> bool:
    return arr.max() > threshold


# def baseline_correction(
#     intsy_arr: np.ndarray,
#     lambda_value: int,
#     porder: int,
#     iterations: int,
# ) -> np.ndarray:
#     # m = len(intsy_arr)
#     # D = sparse.diags(
#     #     diagonals=[1, -2, 1],
#     #     offsets=[0, -1, -2],
#     #     shape=(m, m-2),
#     # )
#     #
#     # w = np.ones(m)
#     #
#     # z = None
#     # for i in range(iterations):
#     #     W = sparse.spdiags(w, 0, m, m)
#     #     Z = W + lambda_value * D.dot(D.transpose())
#     #     z = spsolve(Z, w * intsy_arr)
#     #     w = porder * (intsy_arr > z) + (1 - porder ) * (intsy_arr <= z)
#     #
#     # return intsy_arr - z
#     return intsy_arr
#
#
# def baseline_als(y, lam=1e5, p=0.001, niter=10):
#     """
#     Asymmetric Least Squares Baseline Correction
#
#     Parameters:
#         y : array_like
#             Signal to be baseline-corrected
#         lam : float, optional
#             Smoothness parameter. Higher values make the baseline more rigid.
#             Typically 1e5-1e8 for chromatographic data.
#         p : float, optional
#             Asymmetry parameter. Smaller values make the baseline stick
#             more closely to noise than to peaks. Typically 0.001-0.1.
#         niter : int, optional
#             Number of iterations. Usually 10 is sufficient.
#
#     Returns:
#         baseline : ndarray
#             The calculated baseline
#         corrected_signal : ndarray
#             The baseline-corrected signal
#     """
#     L = len(y)
#     # Create the sparsity pattern for the diagonal matrix of weights
#     D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
#     # Initialize weights
#     w = np.ones(L)
#
#     z = None
#     for i in range(niter):
#         # Create diagonal weight matrix using current weights
#         W = sparse.diags(w)
#         # Solve the linear system to find the baseline
#         Z = W + lam * D.dot(D.transpose())
#         z = spsolve(Z, w * y)
#         # Update weights based on the difference of original signal and baseline
#         w = p * (y > z) + (1 - p) * (y <= z)
#
#     # Return both the baseline and the corrected signal
#     return z, y - z
#
#
# def adaptive_threshold_baseline(y, window_size=50, quantile=0.05):
#     """
#     Simple baseline correction by using a rolling minimum/quantile
#
#     Parameters:
#         y : array_like
#             Signal to be baseline-corrected
#         window_size : int, optional
#             Size of the rolling window
#         quantile : float, optional
#             Quantile to use for baseline estimation (0-1)
#
#     Returns:
#         baseline : ndarray
#             The calculated baseline
#         corrected_signal : ndarray
#             The baseline-corrected signal
#     """
#     from scipy.ndimage import minimum_filter1d
#
#     # For small arrays, adjust window size
#     if len(y) < window_size * 2:
#         window_size = max(3, len(y) // 5)
#
#     if quantile == 0:
#         # Pure minimum filter
#         baseline = minimum_filter1d(y, size=window_size)
#     else:
#         # Use percentile within rolling window
#         baseline = np.zeros_like(y)
#         for i in range(len(y)):
#             start = max(0, i - window_size // 2)
#             end = min(len(y), i + window_size // 2)
#             baseline[i] = np.percentile(y[start:end], quantile * 100)
#
#     # Return both the baseline and the corrected signal
#     return baseline, y - baseline
#
#
# def tophat_filter(signal, window_size):
#     """
#     Apply TopHat filter for baseline correction
#
#     Parameters:
#         signal : ndarray
#             Input signal (1D array)
#         window_size : int
#             Size of the structuring element (window)
#             Should be larger than the width of peaks you want to detect
#             but smaller than the baseline variations
#
#     Returns:
#         baseline : ndarray
#             The estimated baseline
#         corrected_signal : ndarray
#             The baseline-corrected signal (TopHat result)
#     """
#     # Morphological erosion (minimum filter)
#     eroded = minimum_filter1d(signal, size=window_size)
#
#     # Morphological dilation (maximum filter)
#     opened = maximum_filter1d(eroded, size=window_size)
#
#     # The opening operation is the estimated baseline
#     baseline = opened
#
#     # TopHat transform is the original signal minus the opening
#     tophat = signal - baseline
#
#     return baseline, tophat
#
# def adaptive_tophat(signal, min_window=5, max_window=101, snr_threshold=3.0):
#     """
#     Adaptive TopHat filter that automatically selects the best window size
#     based on signal characteristics
#
#     Parameters:
#         signal : ndarray
#             Input signal (1D array)
#         min_window : int, optional
#             Minimum window size to try
#         max_window : int, optional
#             Maximum window size to try
#         snr_threshold : float, optional
#             Signal-to-noise ratio threshold for selecting the window size
#
#     Returns:
#         baseline : ndarray
#             The estimated baseline
#         corrected_signal : ndarray
#             The baseline-corrected signal
#         optimal_window : int
#             The optimal window size selected
#     """
#     from scipy.stats import median_abs_deviation
#
#     # Try different window sizes and evaluate the results
#     best_snr = -1
#     best_window = min_window
#     best_baseline = None
#     best_corrected = None
#
#     # Only test odd window sizes
#     windows = range(min_window, max_window + 1, 2)
#
#     for window in windows:
#         # Apply TopHat filter
#         baseline, corrected = tophat_filter(signal, window)
#
#         # Calculate signal-to-noise ratio
#         # Use median absolute deviation as a robust noise estimator
#         noise = median_abs_deviation(corrected)
#         if noise == 0:  # Avoid division by zero
#             noise = 1e-10
#
#         # Signal is the maximum value in the corrected signal
#         signal_max = np.max(corrected)
#
#         # Calculate SNR
#         snr = signal_max / noise
#
#         if snr > best_snr:
#             best_snr = snr
#             best_window = window
#             best_baseline = baseline
#             best_corrected = corrected
#
#         # Stop if we've reached a good enough SNR
#         if snr > snr_threshold:
#             break
#
#     return best_baseline, best_corrected, best_window

import numpy as np


def tophat_filter(signal, window_size):
    """
    Apply TopHat filter for baseline correction

    Parameters:
        signal : ndarray
            Input signal (1D array)
        window_size : int
            Size of the structuring element (window)
            Should be larger than the width of peaks you want to detect
            but smaller than the baseline variations

    Returns:
        baseline : ndarray
            The estimated baseline
        corrected_signal : ndarray
            The baseline-corrected signal (TopHat result)
    """
    # Morphological erosion (minimum filter)
    eroded = minimum_filter1d(signal, size=window_size)

    # Morphological dilation (maximum filter)
    opened = maximum_filter1d(eroded, size=window_size)

    # The opening operation is the estimated baseline
    baseline = opened

    # TopHat transform is the original signal minus the opening
    tophat = signal - baseline

    return baseline, tophat


def circular_structuring_element_tophat(signal, radius):
    """
    TopHat filter with a circular/parabolic structuring element.
    This can provide smoother results than the flat structuring element.

    Parameters:
        signal : ndarray
            Input signal (1D array)
        radius : int
            Radius of the circular structuring element

    Returns:
        baseline : ndarray
            The estimated baseline
        corrected_signal : ndarray
            The baseline-corrected signal
    """
    from scipy import ndimage

    # Create a parabolic structuring element
    x = np.arange(-radius, radius + 1)
    parabola = -(x ** 2) / radius  # Parabolic shape, inverted

    # Perform grayscale opening (erosion followed by dilation)
    # Using a 1D filter via correlation
    opened = ndimage.grey_opening(signal, size=2 * radius + 1)

    # The opening is the baseline estimate
    baseline = opened

    # TopHat is the original minus the opened signal
    tophat = signal - baseline

    return baseline, tophat


def rolling_ball_baseline(signal, radius, iterations=1):
    """
    Rolling ball baseline correction - conceptually similar to TopHat
    but uses a "rolling ball" algorithm that can better follow the contours
    of the baseline

    Parameters:
        signal : ndarray
            Input signal (1D array)
        radius : int
            Radius of the rolling ball
        iterations : int, optional
            Number of iterations for the algorithm

    Returns:
        baseline : ndarray
            The estimated baseline
        corrected_signal : ndarray
            The baseline-corrected signal
    """
    # Make a copy to avoid modifying the input
    working = np.copy(signal)

    # Repeat the smoothing process if requested
    for _ in range(iterations):
        # Apply minimum filter (erosion)
        eroded = minimum_filter1d(working, size=2 * radius + 1)

        # Apply maximum filter (dilation) to the eroded signal
        opened = maximum_filter1d(eroded, size=2 * radius + 1)

        # Update the working copy for next iteration
        working = opened

    # The opened signal is our baseline estimate
    baseline = working

    # Subtract the baseline to get the corrected signal
    corrected = signal - baseline

    return baseline, corrected


def adaptive_tophat(signal, min_window=5, max_window=101, snr_threshold=3.0):
    """
    Adaptive TopHat filter that automatically selects the best window size
    based on signal characteristics

    Parameters:
        signal : ndarray
            Input signal (1D array)
        min_window : int, optional
            Minimum window size to try
        max_window : int, optional
            Maximum window size to try
        snr_threshold : float, optional
            Signal-to-noise ratio threshold for selecting the window size

    Returns:
        baseline : ndarray
            The estimated baseline
        corrected_signal : ndarray
            The baseline-corrected signal
        optimal_window : int
            The optimal window size selected
    """
    from scipy.stats import median_abs_deviation

    # Try different window sizes and evaluate the results
    best_snr = -1
    best_window = min_window
    best_baseline = None
    best_corrected = None

    # Only test odd window sizes
    windows = range(min_window, max_window + 1, 2)

    for window in windows:
        # Apply TopHat filter
        baseline, corrected = tophat_filter(signal, window)

        # Calculate signal-to-noise ratio
        # Use median absolute deviation as a robust noise estimator
        noise = median_abs_deviation(corrected)
        if noise == 0:  # Avoid division by zero
            noise = 1e-10

        # Signal is the maximum value in the corrected signal
        signal_max = np.max(corrected)

        # Calculate SNR
        snr = signal_max / noise

        if snr > best_snr:
            best_snr = snr
            best_window = window
            best_baseline = baseline
            best_corrected = corrected

        # Stop if we've reached a good enough SNR
        if snr > snr_threshold:
            break

    return best_baseline, best_corrected, best_window


# Example usage in your mass spec code
def improve_build_features_with_tophat(
        scan_array,
        min_peak_length_in_scans: int,
        min_peak_height: float,
        prominence: float,
        min_num_scans_between_peaks: int,
        tophat_window_size: int = 51,  # Window size for TopHat filter
) -> list:
    """
    Enhanced version of build_features with TopHat baseline correction

    Parameters:
        scan_array: ScanArray
            The scan array containing mass spec data
        min_peak_length_in_scans: int
            Minimum number of scans a peak must span
        min_peak_height: float
            Minimum intensity for a peak
        prominence: float
            Minimum prominence for a peak
        min_num_scans_between_peaks: int
            Minimum number of scans between adjacent peaks
        tophat_window_size: int, optional
            Window size for the TopHat filter

    Returns:
        list of FeaturePointer
            Feature pointers for the detected peaks
    """
    feature_pointers = []
    rt_arr = scan_array.rt_arr

    for lane_idx in range(scan_array.mz_arr.shape[0]):
        mz_arr = scan_array.mz_arr[lane_idx].toarray()
        intsy_arr = scan_array.intsy_arr[lane_idx].toarray()

        # Check if the maximum intensity is above the threshold
        if np.max(intsy_arr) < min_peak_height:
            continue

        # Apply TopHat filter for baseline correction
        _, intsy_arr_corrected = tophat_filter(intsy_arr, tophat_window_size)

        # Ensure non-negative values
        intsy_arr_corrected = np.maximum(intsy_arr_corrected, 0)

        # Identify non-zero regions in the corrected data
        intsy_arr_chunks, chunk_idxs = _split_nonzero_islands(
            intsy_arr_corrected,
            return_island_idxs=True,
        )

        # Iterate over these islands and find peaks
        peak_windows = []
        for chunk, chunk_idxs in zip(intsy_arr_chunks, chunk_idxs):
            # Skip very small chunks
            if len(chunk) < min_peak_length_in_scans:
                continue

            # Smooth the chunk
            smoothed_chunk = gaussian_filter1d(chunk, sigma=1.0)

            # Find peaks
            peaks, properties = find_peaks(
                x=smoothed_chunk,
                prominence=prominence,
                width=min_peak_length_in_scans,
            )

            if not peaks.any():
                continue

            # Filter peaks by height after baseline correction
            valid_peaks = []
            for i, peak_idx in enumerate(peaks):
                if smoothed_chunk[peak_idx] >= min_peak_height:
                    valid_peaks.append(i)

            if not valid_peaks:
                continue

            # Extract valid peak windows
            for i in valid_peaks:
                peak_start = int(properties['left_bases'][i])
                peak_end = int(properties['right_bases'][i])

                peak_windows.append((
                    chunk_idxs[peak_start],  # convert to pre-split idx
                    chunk_idxs[peak_end],
                ))

        # Sort peak windows by start position
        peak_windows.sort(key=lambda x: x[0])

        # Filter overlapping windows
        filtered_peak_windows = []
        for i, window in enumerate(peak_windows):
            # Skip if this window overlaps with a previously accepted window
            if i > 0 and window[0] <= filtered_peak_windows[-1][1] + min_num_scans_between_peaks:
                continue
            filtered_peak_windows.append(window)

        # Create FeaturePointers
        for peak_window in filtered_peak_windows:
            feature_pointer = FeaturePointer(
                mz_lane_idx=lane_idx,
                scan_start_idx=peak_window[0],
                scan_end_idx=peak_window[1],
            )
            feature_pointers.append(feature_pointer)

    return feature_pointers


def build_features_masscube(
    spectra: list[oms.MSSpectrum],
    mz_tolerance: float,
    scan_gap_tolerance: int,
    min_intsy: float,
):
    total_num_scans = len(spectra)
    final_features = []
    wip_features = []

    # Build initial features from first scan
    first_spectrum: tuple[np.ndarray, np.ndarray] = spectra[0].get_peaks()

    for spec_mz, spec_intsy in zip(*first_spectrum):
        ftr = Feature(
            total_num_scans=total_num_scans,
        )
        ftr.array['mz'][0] = spec_mz
        ftr.array['intsy'][0] = spec_intsy
        ftr.array['rt'][0] = spectra[0].getRT()

        wip_features.append(ftr)

    # Iterate over subsequent scans:
    for scan_num in range(1, total_num_scans):
        spec_mz, spec_intsy = _get_peaks_higher_than_intsy(
                spectra[scan_num],
                intsy_threshold=min_intsy,
        )

        if spec_mz.size < 2:
            continue

        avlb_signals = np.ones(len(spec_mz), dtype=bool)
        avlb_ftrs = np.ones(len(wip_features), dtype=bool)
        to_be_moved = []

        for i, ftr in enumerate(wip_features):
            min_idx = _find_closest_idx(
                arr=spec_mz,
                target=ftr.latest_scan['mz'],
                tolerance=mz_tolerance,
            )

            if min_idx != -1 and avlb_signals[min_idx]:
                ftr.array['mz'][scan_num] = spec_mz[min_idx]
                ftr.array['intsy'][scan_num] = spec_intsy[min_idx]
                ftr.array['rt'][scan_num] = spectra[scan_num].getRT()
                ftr.gap_counter = 0
                avlb_signals[min_idx] = False  # Mark signal as unavailable for binning
                avlb_ftrs[i] = False  # Mark feature as unavailable for binning
            else:
                # Feature didn't match anything in this scan
                ftr.gap_counter += 1
                if ftr.gap_counter > scan_gap_tolerance:
                    to_be_moved.append(i)

        # Move all features not extended recently to `final features`
        for i in to_be_moved[::-1]:
            final_features.append(
                wip_features.pop(i)
            )

        # Create new rois for remaining signals
        for i, (spec_mz, spec_intsy) in enumerate(
                zip(spec_mz, spec_intsy)
        ):
            if not avlb_signals[i]:
                continue

            ftr = Feature(
                total_num_scans=total_num_scans,
            )
            ftr.array['mz'][scan_num] = spec_mz
            ftr.array['intsy'][scan_num] = spec_intsy
            ftr.array['rt'][scan_num] = spectra[scan_num].getRT()

            wip_features.append(ftr)

        wip_features.sort(
            key=lambda x: x.latest_scan['intsy'],
            reverse=True
        )

    # Move all remaining wip_features to final_features
    for ftr in wip_features:
        final_features.append(ftr)

    # Sort by mz
    final_features.sort(
        key=lambda x: np.mean(x.nonzero_scans['mz'])
    )

    return final_features


def _find_closest_idx(
        arr: np.ndarray,
        target: float,
        tolerance: float,
) -> int:
    """
    Returns the index of the element in `arr` that's closest to `target`.
    If no elements are found within `tolerance`, returns -1.
    :param arr: Array to match against `target`
    :param target: Target value to match against `array`
    :param tolerance: Window for acceptable match
    :return: Index of `arr` corresponding to the best match, or -1
    """
    diff = np.abs(arr - target)
    min_idx = np.argmin(diff)
    if diff[min_idx] < tolerance:
        return min_idx
    return -1


def _get_peaks_higher_than_intsy(
        spectrum: oms.MSSpectrum,
        intsy_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    spec_mz, spec_intsy = spectrum.get_peaks()
    spec_mz: np.ndarray
    spec_intsy: np.ndarray

    idxs = np.where(
        spec_intsy > intsy_threshold
    )[0]

    return spec_mz[idxs], spec_intsy[idxs]


@dataclass
class Feature:
    total_num_scans: int
    gap_counter: int = 0

    def __post_init__(self):
        self.array = np.zeros(
            self.total_num_scans,
            dtype= [
                ('mz', 'f8'),
                ('intsy', 'f8'),
                ('rt', 'f8'),
            ]
        )

    @property
    def nonzero_scans(self) -> np.ndarray:
        """
        :return: returns all non-zero elements
        """
        return self.array[
            np.where(self.array['intsy'] > 0)[0]
        ]

    @property
    def latest_scan(self) -> np.ndarray:
        """
        :return: Returns the latest non-zero element
        """
        return self.array[
            np.where(self.array['intsy'] > 0)[0].max()
        ]
