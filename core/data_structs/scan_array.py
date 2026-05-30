"""
This is a data structure for creating 'mass lanes' then storing
LCMS data as an array. Here, I use the algorithm from MassCube

This implementation uses sparse arrays to preserve memory.

This representation is appropriate for the algorithms used later on
"""
from dataclasses import dataclass, field
import uuid
from typing import TYPE_CHECKING, Optional
from hashlib import sha256

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
        """
        Given a scan number, retrieves the spectrum corresponding to that scan
        """
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
        """
        'Low level API' - makes a feature pointer using mass_lane_idx
        and scan_idx.
        """
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

    def extract_feature_pointer(
        self,
        target_mz: float,
        mz_window: float,
        target_rt: float,
        rt_window: float,
    ) -> Optional['FeaturePointer']:
        """
        Given a target mz/window and target rt/window, generates a
        feature pointer
        """
        # Get mz lane idxs
        mz_lane_idxs: np.ndarray = np.where(
            np.abs(
                target_mz - self.mz_lane_label
            ) < mz_window
        )[0]

        if mz_lane_idxs.size == 0:
            return None

        # Get the one with highest intsy
        mz_lane_idx: int = mz_lane_idxs[
            (
                self.intsy_arr[mz_lane_idxs]
                .max(1)     # Get max of each mz lane
                .argmax()   # Get the mz lane with highest max
            )
        ]

        # Get scan idxs
        rts: np.ndarray = self.rt_arr[
            np.abs(self.rt_arr - target_rt) < rt_window
        ]

        if rts.size == 0:
            return None

        scan_idxs: np.ndarray[int] = np.array(
            [self.rt_to_scan_num(rt) for rt in rts],
        )

        return self.make_feature_pointer(
            mass_lane_idx=mz_lane_idx,
            scan_idxs=scan_idxs,
        )

    def get_hash(self) -> str:
        h = sha256()
        for arr in (
            self.mz_arr.data, self.mz_arr.indices, self.mz_arr.indptr,
            self.intsy_arr.data, self.intsy_arr.indices, self.intsy_arr.indptr,
            self.rt_arr
        ):
            a = np.ascontiguousarray(arr)
            h.update(a.dtype.str.encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())

        return h.hexdigest()

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

    out_mz, out_intsy, rt_per_scan = build_features(
        spectra=spectra,
        mz_tolerance=mz_tolerance,
        scan_gap_tolerance=scan_gap_tolerance,
        min_intsy=min_intsy,
    )

    if out_mz.shape[0] == 0:
        raise ValueError(
            "No signals found in .mzML file"
        )

    scan_array = ScanArray(
        mz_arr=csr_array(out_mz),
        intsy_arr=csr_array(out_intsy),
        scan_num_arr=np.array(scan_nums, dtype='u4'),
        rt_arr=rt_per_scan.astype('f4'),
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

def _build_features_legacy(
    spectra: list[oms.MSSpectrum],
    mz_tolerance: float,
    scan_gap_tolerance: int,
    min_intsy: float,
) -> list[Feature]:
    """
    Legacy reference implementation of the MassCube-style feature builder.

    Preserved for parity testing against the new dense/parallel-array
    implementation in ``build_features``. Not used in production.

    Original docstring follows:
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

        if spec_intsy == 0:
            continue

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


def build_features(
    spectra: list[oms.MSSpectrum],
    mz_tolerance: float,
    scan_gap_tolerance: int,
    min_intsy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel-array reimplementation of the MassCube-style feature builder.

    Replaces the previous per-feature ``Feature`` dataclass representation
    with dense 2D output buffers. Algorithmically equivalent to
    ``_build_features_legacy`` (modulo tie-breaking on identical
    m/z values, which is irrelevant in real LC/MS data). Structured to be
    a clean target for a numba-JITed inner kernel in a follow-up step.

    Args:
        spectra: List of sequential ``oms.MSSpectrum`` (one MS level only).
        mz_tolerance: Maximum m/z difference to assign a peak to a feature.
        scan_gap_tolerance: Max consecutive unmatched scans before a feature
            is retired.
        min_intsy: Intensity threshold applied to scans 2..N (the first scan
            is intentionally unfiltered, matching legacy behavior).

    Returns:
        Tuple of:
          - ``out_mz``: ``(n_features, n_scans)`` float64 — m/z per feature
            per scan; zero where the feature had no signal in that scan.
          - ``out_intsy``: ``(n_features, n_scans)`` float64 — intensities,
            aligned with ``out_mz``.
          - ``rt_per_scan``: ``(n_scans,)`` float64 — retention time of
            each scan (taken from ``spectrum.getRT()``).

        Features are sorted by mean nonzero m/z.

    Notes:
        - The first scan does not apply ``min_intsy`` (legacy behavior).
        - Scans with fewer than 2 above-threshold peaks are skipped
          entirely — the gap counter is not incremented for them
          (legacy behavior).
    """
    n_scans = len(spectra)

    # Pre-extract: RT per scan, and per-scan filtered + mz-sorted peaks.
    # Sorting by m/z lets us use np.searchsorted in the inner matching loop.
    rt_per_scan = np.empty(n_scans, dtype=np.float64)
    peaks_mz: list[np.ndarray] = []
    peaks_intsy: list[np.ndarray] = []
    for i, sp in enumerate(spectra):
        rt_per_scan[i] = sp.getRT()
        if i == 0:
            mz_raw, intsy_raw = sp.get_peaks()
        else:
            mz_raw, intsy_raw = _get_peaks_higher_than_intsy(
                sp, intsy_threshold=min_intsy,
            )
        mz_arr = np.ascontiguousarray(mz_raw, dtype=np.float64)
        intsy_arr = np.ascontiguousarray(intsy_raw, dtype=np.float64)
        order = np.argsort(mz_arr, kind='stable')
        peaks_mz.append(mz_arr[order])
        peaks_intsy.append(intsy_arr[order])

    # Run the matching kernel. Prefer the numba-JIT implementation when
    # available; fall back transparently to the pure-Python kernel.
    # Set ``MZKIT_DISABLE_NUMBA=1`` in the environment to force the pure-
    # Python kernel (useful for A/B benchmarking on real data).
    import os
    use_numba = (
        _NUMBA_KERNEL_AVAILABLE
        and not os.environ.get("MZKIT_DISABLE_NUMBA")
    )
    if use_numba:
        peaks_mz_flat, peaks_intsy_flat, peak_offsets = _flatten_peaks(
            peaks_mz, peaks_intsy
        )
        out_mz, out_intsy = _run_feature_kernel_numba(
            peaks_mz_flat=peaks_mz_flat,
            peaks_intsy_flat=peaks_intsy_flat,
            peak_offsets=peak_offsets,
            n_scans=n_scans,
            mz_tolerance=float(mz_tolerance),
            scan_gap_tolerance=int(scan_gap_tolerance),
        )
    else:
        out_mz, out_intsy = _run_feature_kernel(
            peaks_mz=peaks_mz,
            peaks_intsy=peaks_intsy,
            n_scans=n_scans,
            mz_tolerance=mz_tolerance,
            scan_gap_tolerance=scan_gap_tolerance,
        )

    if out_mz.shape[0] == 0:
        return out_mz, out_intsy, rt_per_scan

    # Sort features by mean nonzero m/z (matches legacy final sort).
    mean_mz = _row_mean_nonzero(out_mz, out_intsy)
    order = np.argsort(mean_mz, kind='stable')
    out_mz = np.ascontiguousarray(out_mz[order])
    out_intsy = np.ascontiguousarray(out_intsy[order])

    return out_mz, out_intsy, rt_per_scan


def _row_mean_nonzero(
    out_mz: np.ndarray,
    out_intsy: np.ndarray,
) -> np.ndarray:
    """
    Per-row mean of ``out_mz`` over columns where ``out_intsy > 0``.
    Matches ``np.mean(feature.nonzero_scans['mz'])`` for each legacy feature.
    """
    mask = out_intsy > 0
    counts = mask.sum(axis=1)
    # All features have at least one nonzero scan (we never allocate
    # a feature without writing a peak), so counts > 0 everywhere.
    sums = np.where(mask, out_mz, 0.0).sum(axis=1)
    return sums / counts


def _run_feature_kernel(
    peaks_mz: list[np.ndarray],
    peaks_intsy: list[np.ndarray],
    n_scans: int,
    mz_tolerance: float,
    scan_gap_tolerance: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Core matching loop. Returns dense ``(n_features, n_scans)`` arrays.

    Each ``peaks_mz[i]`` must be sorted ascending; ``peaks_intsy[i]`` is
    aligned with it.

    Designed so its hot inner loop can later be lifted into a ``@njit``
    function with minimal further refactoring.
    """
    INITIAL_CAP = 4096
    buf_cap = INITIAL_CAP
    out_mz = np.zeros((buf_cap, n_scans), dtype=np.float64)
    out_intsy = np.zeros((buf_cap, n_scans), dtype=np.float64)
    n_alloc = 0

    # WIP feature state as parallel lists (cheap dynamic resize in Python).
    # Each entry is one currently-active feature.
    wip_latest_mz: list[float] = []
    wip_latest_intsy: list[float] = []
    wip_gap: list[int] = []
    wip_row: list[int] = []  # row index into out_mz / out_intsy

    def _ensure_capacity(extra: int) -> None:
        """Grow out_mz / out_intsy if appending ``extra`` rows would overflow."""
        nonlocal buf_cap, out_mz, out_intsy
        needed = n_alloc + extra
        if needed <= buf_cap:
            return
        new_cap = buf_cap
        while new_cap < needed:
            new_cap *= 2
        # Allocate empty + slice-copy used portion + zero tail. Avoids the
        # full-buffer memset that np.zeros would do — matters at gigabyte scale.
        new_mz = np.empty((new_cap, n_scans), dtype=np.float64)
        new_intsy = np.empty((new_cap, n_scans), dtype=np.float64)
        new_mz[:n_alloc] = out_mz[:n_alloc]
        new_intsy[:n_alloc] = out_intsy[:n_alloc]
        new_mz[n_alloc:] = 0.0
        new_intsy[n_alloc:] = 0.0
        out_mz = new_mz
        out_intsy = new_intsy
        buf_cap = new_cap

    # === First scan ===
    spec_mz_0 = peaks_mz[0]
    spec_intsy_0 = peaks_intsy[0]
    # Pre-grow once for the first scan's worth of features.
    _ensure_capacity(len(spec_mz_0))
    for k in range(len(spec_mz_0)):
        # Legacy creates a Feature even for intsy==0 then `continue`s; the
        # resulting object is never used. We just skip directly.
        if spec_intsy_0[k] == 0:
            continue
        row = n_alloc
        out_mz[row, 0] = spec_mz_0[k]
        out_intsy[row, 0] = spec_intsy_0[k]
        n_alloc += 1
        wip_latest_mz.append(float(spec_mz_0[k]))
        wip_latest_intsy.append(float(spec_intsy_0[k]))
        wip_gap.append(0)
        wip_row.append(row)

    # === Subsequent scans ===
    for scan_num in range(1, n_scans):
        spec_mz = peaks_mz[scan_num]
        spec_intsy = peaks_intsy[scan_num]

        if spec_mz.size < 2:
            # Legacy: skip the scan entirely. Gap counters are NOT
            # incremented. (Preserved verbatim, even though arguably a quirk.)
            continue

        n_wip = len(wip_latest_mz)
        n_peaks = spec_mz.shape[0]
        avlb_signals = np.ones(n_peaks, dtype=bool)
        # -1 = no match; otherwise index into spec_mz/spec_intsy.
        matched_signal_idx = np.full(n_wip, -1, dtype=np.int64)

        # Matching pass: for each WIP feature (in current intensity-sorted
        # order), find the globally-closest peak in mz. If that peak is
        # already claimed, treat as unmatched (does NOT fall back to
        # next-closest — preserves legacy semantics).
        for i in range(n_wip):
            target = wip_latest_mz[i]
            # Binary search to find candidate neighbors in sorted spec_mz.
            j = int(np.searchsorted(spec_mz, target, side='left'))
            best_k = -1
            best_d = mz_tolerance  # legacy uses strict < tolerance
            if j > 0:
                d = target - spec_mz[j - 1]
                if d < best_d:
                    best_d = d
                    best_k = j - 1
            if j < n_peaks:
                d = spec_mz[j] - target
                # Strict `<` keeps j-1 on exact ties, matching np.argmin's
                # first-occurrence preference.
                if d < best_d:
                    best_d = d
                    best_k = j
            if best_k != -1 and avlb_signals[best_k]:
                matched_signal_idx[i] = best_k
                avlb_signals[best_k] = False

        # === Apply updates and rebuild WIP state ===
        new_wip_latest_mz: list[float] = []
        new_wip_latest_intsy: list[float] = []
        new_wip_gap: list[int] = []
        new_wip_row: list[int] = []

        for i in range(n_wip):
            row = wip_row[i]
            k = int(matched_signal_idx[i])
            if k >= 0:
                mz_val = float(spec_mz[k])
                intsy_val = float(spec_intsy[k])
                out_mz[row, scan_num] = mz_val
                out_intsy[row, scan_num] = intsy_val
                new_wip_latest_mz.append(mz_val)
                new_wip_latest_intsy.append(intsy_val)
                new_wip_gap.append(0)
                new_wip_row.append(row)
            else:
                gc = wip_gap[i] + 1
                if gc > scan_gap_tolerance:
                    # Feature retired — its row in out_* stays as-is.
                    continue
                new_wip_latest_mz.append(wip_latest_mz[i])
                new_wip_latest_intsy.append(wip_latest_intsy[i])
                new_wip_gap.append(gc)
                new_wip_row.append(row)

        # New features from unclaimed signals.
        unmatched = np.where(avlb_signals)[0]
        if unmatched.size:
            _ensure_capacity(unmatched.size)
            for k in unmatched:
                k = int(k)
                row = n_alloc
                mz_val = float(spec_mz[k])
                intsy_val = float(spec_intsy[k])
                out_mz[row, scan_num] = mz_val
                out_intsy[row, scan_num] = intsy_val
                n_alloc += 1
                new_wip_latest_mz.append(mz_val)
                new_wip_latest_intsy.append(intsy_val)
                new_wip_gap.append(0)
                new_wip_row.append(row)

        # Sort WIP by latest intensity descending, stably (matches legacy).
        if new_wip_latest_intsy:
            intsy_arr = np.asarray(new_wip_latest_intsy, dtype=np.float64)
            order = np.argsort(-intsy_arr, kind='stable')
            wip_latest_mz = [new_wip_latest_mz[j] for j in order]
            wip_latest_intsy = [new_wip_latest_intsy[j] for j in order]
            wip_gap = [new_wip_gap[j] for j in order]
            wip_row = [new_wip_row[j] for j in order]
        else:
            wip_latest_mz = []
            wip_latest_intsy = []
            wip_gap = []
            wip_row = []

    # Trim to actual feature count.
    out_mz = np.ascontiguousarray(out_mz[:n_alloc])
    out_intsy = np.ascontiguousarray(out_intsy[:n_alloc])
    return out_mz, out_intsy


def _flatten_peaks(
    peaks_mz: list[np.ndarray],
    peaks_intsy: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate per-scan peak arrays into flat float64 arrays with a CSR-
    style ``peak_offsets`` index. Required for the numba kernel, which
    can't take a Python list of variable-length arrays cheaply.

    ``peak_offsets[i]:peak_offsets[i+1]`` is the slice of flat arrays
    belonging to scan ``i``. Each per-scan slice is already sorted by m/z
    (sorting is done upstream in ``build_features``).
    """
    n_scans = len(peaks_mz)
    peak_offsets = np.zeros(n_scans + 1, dtype=np.int64)
    for i in range(n_scans):
        peak_offsets[i + 1] = peak_offsets[i] + peaks_mz[i].shape[0]
    total = int(peak_offsets[-1])
    peaks_mz_flat = np.empty(total, dtype=np.float64)
    peaks_intsy_flat = np.empty(total, dtype=np.float64)
    for i in range(n_scans):
        a, b = int(peak_offsets[i]), int(peak_offsets[i + 1])
        if b > a:
            peaks_mz_flat[a:b] = peaks_mz[i]
            peaks_intsy_flat[a:b] = peaks_intsy[i]
    return peaks_mz_flat, peaks_intsy_flat, peak_offsets


# ---------------------------------------------------------------------------
# numba-JIT kernel
# ---------------------------------------------------------------------------
# Lifted from ``_run_feature_kernel``. Same algorithm and semantics; differs
# only in mechanical detail (flat inputs, preallocated scratch buffers,
# manual binary search). Detected and dispatched to opportunistically by
# ``build_features``. Falls back to the pure-Python kernel if the numba
# import fails, so this stays optional from a packaging perspective.

try:
    from numba import njit
    _NUMBA_KERNEL_AVAILABLE = True
except ImportError:
    _NUMBA_KERNEL_AVAILABLE = False


if _NUMBA_KERNEL_AVAILABLE:

    @njit(cache=True)
    def _grow_2d(buf: np.ndarray, n_used: int, new_cap: int) -> np.ndarray:
        """Copy a 2D buffer into one with ``new_cap`` rows, preserving the first
        ``n_used`` rows. Uses slice assignment so numba emits a single memcpy
        rather than a per-element loop. We only zero the *newly-allocated*
        tail rather than the whole buffer — saves a full memset on each grow,
        which matters when the output reaches gigabyte scale."""
        n_cols = buf.shape[1]
        new_buf = np.empty((new_cap, n_cols), dtype=np.float64)
        new_buf[:n_used] = buf[:n_used]
        new_buf[n_used:] = 0.0
        return new_buf

    @njit(cache=True)
    def _grow_1d_f64(buf: np.ndarray, n_used: int, new_cap: int) -> np.ndarray:
        new_buf = np.empty(new_cap, dtype=np.float64)
        new_buf[:n_used] = buf[:n_used]
        return new_buf

    @njit(cache=True)
    def _grow_1d_i64(buf: np.ndarray, n_used: int, new_cap: int) -> np.ndarray:
        new_buf = np.empty(new_cap, dtype=np.int64)
        new_buf[:n_used] = buf[:n_used]
        return new_buf

    @njit(cache=True)
    def _run_feature_kernel_numba(
        peaks_mz_flat: np.ndarray,    # float64[total_peaks]
        peaks_intsy_flat: np.ndarray,  # float64[total_peaks]
        peak_offsets: np.ndarray,      # int64[n_scans + 1]
        n_scans: int,
        mz_tolerance: float,
        scan_gap_tolerance: int,
    ):
        """
        Numba-JIT version of ``_run_feature_kernel``. Algorithmically identical;
        operates on flat input arrays and preallocated scratch buffers.

        Notes on parity:
          - Uses ``np.argsort(-x)`` for the per-scan WIP intensity sort. This
            is not guaranteed stable, but on real data ties are vanishingly
            rare and the parity tests confirm equivalence on synthetic data.
          - Matching semantics (closest peak in m/z, claim-once, no fallback
            to next-closest) preserved exactly via manual binary search.
        """
        # === Output buffers (grow on demand) ===
        BUF_INITIAL = 4096
        buf_cap = BUF_INITIAL
        out_mz = np.zeros((buf_cap, n_scans), dtype=np.float64)
        out_intsy = np.zeros((buf_cap, n_scans), dtype=np.float64)
        n_alloc = 0

        # === WIP feature scratch (grow on demand) ===
        WIP_INITIAL = 4096
        wip_cap = WIP_INITIAL
        wip_latest_mz = np.empty(wip_cap, dtype=np.float64)
        wip_latest_intsy = np.empty(wip_cap, dtype=np.float64)
        wip_gap = np.empty(wip_cap, dtype=np.int64)
        wip_row = np.empty(wip_cap, dtype=np.int64)
        n_wip = 0

        # Per-scan scratch buffers, sized to the largest scan we'll see.
        max_peaks = 0
        for s in range(n_scans):
            n = int(peak_offsets[s + 1] - peak_offsets[s])
            if n > max_peaks:
                max_peaks = n
        if max_peaks < 1:
            max_peaks = 1
        avlb_signals = np.empty(max_peaks, dtype=np.bool_)
        matched_signal_idx = np.empty(wip_cap, dtype=np.int64)

        # 'Next-iteration' WIP scratch — same capacity as wip_*.
        new_wip_latest_mz = np.empty(wip_cap, dtype=np.float64)
        new_wip_latest_intsy = np.empty(wip_cap, dtype=np.float64)
        new_wip_gap = np.empty(wip_cap, dtype=np.int64)
        new_wip_row = np.empty(wip_cap, dtype=np.int64)

        # === First scan ===
        s0 = int(peak_offsets[0])
        s1 = int(peak_offsets[1])
        n_peaks_0 = s1 - s0

        # Ensure out_* and wip_* fit.
        if n_alloc + n_peaks_0 > buf_cap:
            new_cap = buf_cap
            while new_cap < n_alloc + n_peaks_0:
                new_cap *= 2
            out_mz = _grow_2d(out_mz, n_alloc, new_cap)
            out_intsy = _grow_2d(out_intsy, n_alloc, new_cap)
            buf_cap = new_cap
        if n_wip + n_peaks_0 > wip_cap:
            new_cap = wip_cap
            while new_cap < n_wip + n_peaks_0:
                new_cap *= 2
            wip_latest_mz = _grow_1d_f64(wip_latest_mz, n_wip, new_cap)
            wip_latest_intsy = _grow_1d_f64(wip_latest_intsy, n_wip, new_cap)
            wip_gap = _grow_1d_i64(wip_gap, n_wip, new_cap)
            wip_row = _grow_1d_i64(wip_row, n_wip, new_cap)
            matched_signal_idx = np.empty(new_cap, dtype=np.int64)
            new_wip_latest_mz = np.empty(new_cap, dtype=np.float64)
            new_wip_latest_intsy = np.empty(new_cap, dtype=np.float64)
            new_wip_gap = np.empty(new_cap, dtype=np.int64)
            new_wip_row = np.empty(new_cap, dtype=np.int64)
            wip_cap = new_cap

        for k in range(s0, s1):
            if peaks_intsy_flat[k] == 0.0:
                continue
            row = n_alloc
            out_mz[row, 0] = peaks_mz_flat[k]
            out_intsy[row, 0] = peaks_intsy_flat[k]
            n_alloc += 1
            wip_latest_mz[n_wip] = peaks_mz_flat[k]
            wip_latest_intsy[n_wip] = peaks_intsy_flat[k]
            wip_gap[n_wip] = 0
            wip_row[n_wip] = row
            n_wip += 1

        # === Subsequent scans ===
        for scan_num in range(1, n_scans):
            ps = int(peak_offsets[scan_num])
            pe = int(peak_offsets[scan_num + 1])
            n_peaks = pe - ps

            if n_peaks < 2:
                continue

            # Reset avlb_signals
            for k in range(n_peaks):
                avlb_signals[k] = True

            # Ensure scratch arrays sized for current n_wip and prospective new wip.
            max_new_wip = n_wip + n_peaks
            if max_new_wip > wip_cap:
                new_cap = wip_cap
                while new_cap < max_new_wip:
                    new_cap *= 2
                wip_latest_mz = _grow_1d_f64(wip_latest_mz, n_wip, new_cap)
                wip_latest_intsy = _grow_1d_f64(wip_latest_intsy, n_wip, new_cap)
                wip_gap = _grow_1d_i64(wip_gap, n_wip, new_cap)
                wip_row = _grow_1d_i64(wip_row, n_wip, new_cap)
                matched_signal_idx = np.empty(new_cap, dtype=np.int64)
                new_wip_latest_mz = np.empty(new_cap, dtype=np.float64)
                new_wip_latest_intsy = np.empty(new_cap, dtype=np.float64)
                new_wip_gap = np.empty(new_cap, dtype=np.int64)
                new_wip_row = np.empty(new_cap, dtype=np.int64)
                wip_cap = new_cap

            # === Matching pass ===
            # For each WIP feature, manual binary search for the insertion
            # point of its latest m/z in the (sorted) current scan's peaks.
            # Then pick the closer of the two neighbors. Preserves legacy
            # claim-once semantics (no fallback to next-closest).
            for i in range(n_wip):
                target = wip_latest_mz[i]
                lo = 0
                hi = n_peaks
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if peaks_mz_flat[ps + mid] < target:
                        lo = mid + 1
                    else:
                        hi = mid
                j = lo

                best_k = -1
                best_d = mz_tolerance
                if j > 0:
                    d = target - peaks_mz_flat[ps + j - 1]
                    if d < best_d:
                        best_d = d
                        best_k = j - 1
                if j < n_peaks:
                    d = peaks_mz_flat[ps + j] - target
                    # Strict `<` keeps j-1 on exact ties (matches np.argmin).
                    if d < best_d:
                        best_d = d
                        best_k = j

                if best_k != -1 and avlb_signals[best_k]:
                    matched_signal_idx[i] = best_k
                    avlb_signals[best_k] = False
                else:
                    matched_signal_idx[i] = -1

            # === Apply updates and rebuild WIP ===
            n_new = 0
            for i in range(n_wip):
                row = wip_row[i]
                k = matched_signal_idx[i]
                if k >= 0:
                    mz_val = peaks_mz_flat[ps + k]
                    intsy_val = peaks_intsy_flat[ps + k]
                    out_mz[row, scan_num] = mz_val
                    out_intsy[row, scan_num] = intsy_val
                    new_wip_latest_mz[n_new] = mz_val
                    new_wip_latest_intsy[n_new] = intsy_val
                    new_wip_gap[n_new] = 0
                    new_wip_row[n_new] = row
                    n_new += 1
                else:
                    gc = wip_gap[i] + 1
                    if gc > scan_gap_tolerance:
                        continue
                    new_wip_latest_mz[n_new] = wip_latest_mz[i]
                    new_wip_latest_intsy[n_new] = wip_latest_intsy[i]
                    new_wip_gap[n_new] = gc
                    new_wip_row[n_new] = row
                    n_new += 1

            # === New features from unclaimed signals ===
            # First, count and grow output buffer if needed.
            n_unmatched = 0
            for k in range(n_peaks):
                if avlb_signals[k]:
                    n_unmatched += 1

            if n_alloc + n_unmatched > buf_cap:
                new_cap = buf_cap
                while new_cap < n_alloc + n_unmatched:
                    new_cap *= 2
                out_mz = _grow_2d(out_mz, n_alloc, new_cap)
                out_intsy = _grow_2d(out_intsy, n_alloc, new_cap)
                buf_cap = new_cap

            for k in range(n_peaks):
                if not avlb_signals[k]:
                    continue
                row = n_alloc
                mz_val = peaks_mz_flat[ps + k]
                intsy_val = peaks_intsy_flat[ps + k]
                out_mz[row, scan_num] = mz_val
                out_intsy[row, scan_num] = intsy_val
                n_alloc += 1
                new_wip_latest_mz[n_new] = mz_val
                new_wip_latest_intsy[n_new] = intsy_val
                new_wip_gap[n_new] = 0
                new_wip_row[n_new] = row
                n_new += 1

            # === Sort WIP by latest intensity descending ===
            # Negate as key; np.argsort default ('quicksort') is not stable
            # but ties in floating-point intensity are astronomically rare on
            # real data.
            if n_new > 0:
                key = -new_wip_latest_intsy[:n_new]
                order = np.argsort(key)
                for i in range(n_new):
                    j = order[i]
                    wip_latest_mz[i] = new_wip_latest_mz[j]
                    wip_latest_intsy[i] = new_wip_latest_intsy[j]
                    wip_gap[i] = new_wip_gap[j]
                    wip_row[i] = new_wip_row[j]
            n_wip = n_new

        # Trim to actual feature count.
        return out_mz[:n_alloc].copy(), out_intsy[:n_alloc].copy()


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


