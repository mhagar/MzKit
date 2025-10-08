"""
This is a data structure for creating 'mass lanes' then storing
LCMS data as an array. Here, I use the algorithm from MassCube

This implementation uses sparse arrays to preserve memory.

This representation is appropriate for the algorithms used later on
"""
from dataclasses import dataclass, field
import uuid
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.sparse import csr_array, csc_array
import pyopenms as oms

from core.utils.array_types import to_spec_arr, SpectrumArray
from core.data_structs.feature_pointer import FeaturePointer

if TYPE_CHECKING:
    from core.data_structs import ScanArrayUUID


@dataclass
class ScanArray:
    """
    A structured container for MS data with slicing methods (i.e. BPC/XIC)

    This class represents a collection of mass spectrometry scans in a sparse
    matrix format, where each row represents an m/z trace and each column
    represents a time point.

    Attributes:
        mz_arr (csr_array): Sparse matrix of m/z values.
            Shape (n_mz_traces, n_scans).

        intsy_arr (csr_array): Sparse matrix of intensity values corresponding
            to mz_arr.
            Shape (n_mz_traces, n_scans).

        rt_arr (np.ndarray): 1D array of retention times for each scan.
            Shape (n_scans,).

        scan_num_arr (np.ndarray): 1D array of scan numbers for each scan.
            Shape (n_scans,).

        mz_lane_label (np.ndarray): Array containing the m/z value of the most
            intense signal in each row. Automatically computed in __post_init__.
            Shape (n_mz_traces,).

    Methods:
        get_bpc: Extracts Base Peak Chromatogram within specified m/z and
        RT ranges.

        get_xic: Extracts Extracted Ion Chromatogram within specified m/z and
        RT ranges.

        get_spectrum: Extracts a spectrum at either a specified scan index,
        or at the nearest retention time

    Examples:
        >>> from core.data_structs.injection import Injection
        ... injection: Injection
        ... scan_array = build_scan_array(injection)
        ...
        ... # Get BPC for m/z range 400-401
        ... bpc = scan_array.get_bpc(mz_range=(400, 401))
        ...
        ... # Get XIC for specific time range
        ... xic = scan_array.get_xic(rt_range=(10.5, 12.5))

    Notes:
        - mz_lane_label is automatically computed during initialization and
          represents the primary m/z value for each row in the matrix

        - mz_arr_csc and intsy_arr_csc are generated during initializaiton,
            unless already provided (i.e. if loading from disk)
    """
    mz_arr: csr_array
    intsy_arr: csr_array
    rt_arr: np.ndarray[float]
    scan_num_arr: np.ndarray[int]
    uuid: 'ScanArrayUUID' = field(default_factory=lambda: uuid.uuid4().int)
    mz_lane_label: Optional[np.ndarray[float]] = None
    mz_arr_csc: Optional[csc_array] = None
    intsy_arr_csc: Optional[csc_array] = None


    def __post_init__(self):
        # Get m/z value of the tallest signal in each row
        # This is used as a measure of the 'm/z lane' represented by row
        max_col_idxs = self.intsy_arr.argmax(axis=1)
        row_idxs = np.arange(self.intsy_arr.shape[0])

        self.mz_lane_label = self.mz_arr[
            row_idxs,
            max_col_idxs,
        ]

        # Build CSC versions of mz_arr and intsy_arr
        # (useful for fast spectrum slices)
        if type(self.mz_arr_csc) is type(None):
            self.mz_arr_csc = self.mz_arr.tocsc(copy=True)

        if type(self.intsy_arr_csc) is type(None):
            self.intsy_arr_csc = self.intsy_arr.tocsc(copy=True)

    def get_bpc(
            self,
            mz_range:Optional[tuple[float, float]] = None,
            rt_range: Optional[tuple[float, float]] = None,
    ):
        """
        Extract Base Peak Chromatogram for specified m/z and retention time range

        Args:
            mz_range (tuple[float, float], optional): Min and max m/z values to include.
                If None, uses entire m/z range
            rt_range (tuple[float, float], optional): Min and max retention times to include.
                If None, uses entire time range.

        Returns:
            np.ndarray: Structured array with fields:
                - 'rt' (float32): Retention time in minutes
                - 'intsy' (float32): Intensity value

        Examples:
            bpc = scan_array.get_bpc(mz_range=(400, 401))
            plt.plot(bpc['rt'], bpc['intsy'])

        """
        if mz_range:
            # Get indices that fall within the specified range
            mz_start, mz_end = mz_range
            match_arr = np.where(
                (self.mz_lane_label >= mz_start) &
                (self.mz_lane_label <= mz_end)
            )[0]

            # If this subset exists:
            if match_arr.any():
                # Now get the MAX of that subset
                bpc_arr = self.intsy_arr[match_arr].max(0).toarray()

            else:
                # Otherwise, return empty array
                bpc_arr = np.zeros(len(self.rt_arr))


        else:
            # Just get the sum of the whole column
            bpc_arr = self.intsy_arr.max(0).toarray()

        start, end = 0, len(bpc_arr)
        if rt_range:
            start, end = argrange(
                self.rt_arr,
                rt_range[0],
                rt_range[1]
            )

        return np.rec.fromarrays(
            arrayList=[
                bpc_arr[start:end],
                self.rt_arr[start:end],
            ],
            dtype=[
                ('intsy', 'f4'),
                ('rt', 'f4')
            ]
        )

    def get_xic(
            self,
            mz_range: Optional[tuple[float, float]] = None,
            rt_range: Optional[tuple[float, float]] = None,
    ):
        """
        Extract Ion Chromatogram for specified m/z and retention time ranges.

        Computes the sum of intensities at each time point within the specified m/z range.
        Also tracks the m/z value of the most intense peak at each time point.

        Args:
            mz_range (tuple[float, float], optional): Min and max m/z values to include.
                If None, uses entire m/z range (i.e. effectively a TIC)
            rt_range (tuple[float, float], optional): Min and max retention times to include.
                If None, uses entire time range.

        Returns:
            np.ndarray: Structured array with fields:
                - 'mz' (float32): M/z value of most intense peak
                - 'rt' (float32): Retention time in minutes
                - 'intsy' (float32): Summed intensity value

        Examples:
            xic = scan_array.get_xic(mz_range=(400, 401))
            # plt.plot(xic['rt'], xic['intsy'])
        """
        if mz_range:
            # Get indices that fall within the specified range
            mz_start, mz_end = mz_range
            match_arr = np.where(
                (self.mz_lane_label >= mz_start) &
                (self.mz_lane_label <= mz_end)
            )[0]

            # Now get the sum of that subset
            xic_arr = self.intsy_arr[match_arr].sum(0)

            if match_arr.any():
                # Get the m/z value of the tallest peak in each scan subset
                max_row_idxs = self.intsy_arr[match_arr].argmax(axis=0)

                tallest_mzs = self.mz_arr[match_arr][
                    max_row_idxs,
                    np.arange(self.mz_arr.shape[1])
                ]
            else:
                # Returns empty array
                tallest_mzs = xic_arr

        else:
            # Just get the sum of the whole column
            xic_arr: np.ndarray = self.intsy_arr.sum(0)

            # Get the m/z value of the tallest peak in each scan
            max_row_idxs = self.intsy_arr.argmax(axis=0)
            tallest_mzs = self.mz_arr[
                max_row_idxs,
                np.arange(self.mz_arr.shape[1])
            ]


        start, end = 0, len(xic_arr)
        if rt_range:
            start, end = argrange(
                self.rt_arr,
                rt_range[0],
                rt_range[1]
            )


        return np.rec.fromarrays(
            arrayList=[
                tallest_mzs[start:end],
                xic_arr[start:end],
                self.rt_arr[start:end],
            ],
            dtype=[
                ('mz', 'f4'),
                ('intsy', 'f4'),
                ('rt', 'f4'),
            ]
        )

    def get_spectrum(
            self,
            scan_num: Optional[int] = None,
    ) -> SpectrumArray:
        return to_spec_arr(
            mz_arr=self.mz_arr_csc._getcol(scan_num).toarray().flatten(),
            intsy_arr=self.intsy_arr_csc._getcol(scan_num).toarray().flatten(),
        )

    def rt_to_scan_num(
            self,
            rt: float,
    ) -> int:
        """
        Finds the scan_num that is closest to the given rt

        :param rt: Retention time
        :return: scan number (integer)
        """
        rt_diff = np.abs(self.rt_arr - rt)
        return np.argmin(rt_diff)

    def make_feature_pointer(
        self,
        mass_lane_idx: int,
        scan_idxs: Optional[np.ndarray[...,]] = None,
    ) -> 'FeaturePointer':
        if scan_idxs is None:
            scan_idxs = np.arange(
                0,
                self.scan_num_arr.size
            )

        return FeaturePointer(
            mz_lane_idx=mass_lane_idx,
            scan_idxs=scan_idxs,
            source_array_uuid=self.uuid,
            source_array_shape=self.mz_arr.shape,
        )

    def __repr__(self):
        return f"ScanArray{self.mz_arr.shape}"


@dataclass
class ScanArrayParameters:
    ms_level: int
    mz_tolerance: float
    scan_gap_tolerance: int
    min_intsy: float
    scan_nums: Optional[list[int]]


def build_scan_array(
    spectra: list[oms.MSSpectrum],
    mz_tolerance: float,
    scan_gap_tolerance: int,
    min_intsy: float,
    scan_nums: Optional[list[int]],
) -> ScanArray:
    """
    Converts mz, intensity, rt arrays into a ScanArray object.

    This function processes a stack of MSSpectrum objects, applying intensity
    cutoffs and using the MassCube algorithm to create structured bundle of sparse
    arrays where each column correspond to a 'm/z lane' and each row corresponds to a
    mass spectrum (i.e. a scan).

    Args:
        spectra (list[oms.MSSpectrum]): List of sequential spectra
        mz_tolerance (float): Maximum m/z difference to consider a 'mass lane'
        scan_gap_tolerance (int): Maximum number of empty scans before starting a new lane
        min_intsy (float): Minimum intensity to consider (i.e. noise threshold)
        scan_nums (list[int]): A list of scan indices to label each scan.
                    If not given, will generate a monotonic series (i.e. 0, 1, 2 ..)

    Returns:
        ScanArray: A an object con containing processed mass spectrometry data
            with the following attributes:
            - mz: mass-to-charge ratios
            - intensity: peak intensities
            - scan_indices: indices mapping peaks to original scans

    Raises:
        ValueError: If intsy_cutoff is negative

    Examples:
        >>> from core.data_structs.injection import Injection
        ... injection: Injection
        ...
        ... scan_array = build_scan_array(
        ...     injection,
        ...     intsy_cutoff=1000,
        ... )

    Notes:
        - REMEMBER TO ONLY PASS IN SPECTRA OF THE SAME MS LEVEL!!
        - Large files may require significant memory for processing
    """
    if min_intsy < 0:
        raise ValueError(
            f"min_intsy must be greater than 0.0 (min_ints={min_intsy})"
        )

    if not scan_nums:
        scan_nums = list(range(len(spectra)))

    features: list[Feature] = build_features_masscube(
        spectra=spectra,
        mz_tolerance=mz_tolerance,
        scan_gap_tolerance=scan_gap_tolerance,
        min_intsy=min_intsy,
    )

    if not features:
        raise ValueError(
            "No signals found in .mzML file"
        )

    ms_data = np.array([x.array for x in features])

    # Get rt array
    rts = list(ms_data[:]['rt'].max(axis=0))

    scan_array = ScanArray(
        mz_arr=csr_array(ms_data['mz']),
        intsy_arr=csr_array(ms_data['intsy']),
        scan_num_arr=np.array(scan_nums, dtype='u4'),
        rt_arr=np.array(rts, dtype='f4'),
    )

    return scan_array


def argrange(
        arr: np.ndarray,
        start: float | int,
        end: float | int,
) -> tuple[int, int]:
    """
    Given a sorted 1D array, returns the indices that
    correspond to the slice 'start -> end'.

    i.e. an array of retention times
    """
    start_idx = np.searchsorted(
        arr,
        start,
        side='left',
    )

    end_idx = np.searchsorted(
        arr,
        end,
        side='right'
    )

    return int(start_idx), int(end_idx)


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

def build_features_masscube(
    spectra: list[oms.MSSpectrum],
    mz_tolerance: float,
    scan_gap_tolerance: int,
    min_intsy: float,
) -> list[Feature]:
    """
    An adaptation of the algorithm used by MassCube.
    Given a stack of MS spectra, builds 'mass lanes'
    (such that each lane roughly corresponds to a feature)
    :param spectra:
    :param mz_tolerance:
    :param scan_gap_tolerance:
    :param min_intsy:
    :return:
    """
    total_num_scans: int = len(spectra)
    final_features: list[Feature] = []
    wip_features: list[Feature] = []

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


