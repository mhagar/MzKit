"""
This is a data structure that indexes ScanArrays.
Can be used to retrieve a particular set of scans of one
m/z lane (i.e. a feature).
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from core.utils.array_types import to_chrom_arr

if TYPE_CHECKING:
    from core.data_structs.scan_array import ScanArray
    from core.utils.array_types import ChromArray

@dataclass
class FeaturePointer:
    """
    A reference to an m/z signal within a ScanArray, at `scan_idxs`.

    `scan_idxs` is contiguous for MS1 / DIA features, but sparse for DDA MS2
    cofeatures (only the scans that fragmented the ensemble's precursors).
    Every accessor reads exactly these scans, last one included.
    """
    mz_lane_idx: int
    scan_idxs: np.ndarray[...,]
    source_array_uuid: int
    source_array_shape: tuple

    def __post_init__(self):
        self.scan_idxs.sort()

    def get_mz_values(
            self,
            scan_array: 'ScanArray',
    ) -> np.ndarray:
        """
        Get the m/z values for this feature.

        Parameters:
            scan_array (ScanArray): The ScanArray containing the data.

        Returns:
            numpy.ndarray: Array of m/z values for this feature.
        """
        return self._take_lane(scan_array, scan_array.mz_arr)

    def get_intensity_values(
            self,
            scan_array: 'ScanArray',
    ) -> np.ndarray[...,]:
        """
        Get the intensity values for this feature.

        Parameters:
            scan_array (ScanArray): The ScanArray containing the data.

        Returns:
            numpy.ndarray: Array of intensity values for this feature.
        """
        return self._take_lane(scan_array, scan_array.intsy_arr)

    def get_retention_times(
            self,
            scan_array: 'ScanArray',
    ) -> np.ndarray:
        """
        Get the retention times for this feature.

        Parameters:
            scan_array (ScanArray): The ScanArray containing the data.

        Returns:
            numpy.ndarray: Array of retention times for this feature.
        """
        self.validate_source(scan_array)

        return scan_array.rt_arr[self.scan_idxs]

    def _take_lane(
            self,
            scan_array: 'ScanArray',
            lane_arr,
    ) -> np.ndarray:
        """
        This feature's values from `lane_arr` (one of `scan_array`'s sparse
        lane x scan arrays): its lane, at its `scan_idxs`.
        """
        self.validate_source(scan_array)

        # Densify only the spanned range, then pick the (possibly sparse) scans
        s0, s1 = self.scan_start, self.scan_end
        row = lane_arr[self.mz_lane_idx, s0:s1 + 1].toarray().flatten()
        return row[self.scan_idxs - s0]

    def get_chrom_array(
        self,
        scan_array: 'ScanArray',
    ) -> 'ChromArray':
        rts = self.get_retention_times(scan_array)
        intsys = self.get_intensity_values(scan_array)
        return to_chrom_arr(
            rts, intsys  # type: ignore
        )

    @property
    def scan_start(self) -> int:
        return self.scan_idxs[0]

    @property
    def scan_end(self) -> int:
        return self.scan_idxs[-1]

    @property
    def n_scans(self):
        """Number of scans this feature spans."""
        return len(self.scan_idxs)

    def get_max_intsy(
        self,
        scan_array: 'ScanArray',
    ) -> float:
        """
        Return the maximum intensity given by this pointer
        :return:
        """
        return self.get_intensity_values(scan_array).max()

    def get_apex_mz(
        self,
        scan_array: 'ScanArray',
    ) -> float:
        """
        This feature's m/z at its most intense scan (a measured value, never
        an average across scans)
        """
        idx = self.get_intensity_values(scan_array).argmax()
        return float(self.get_mz_values(scan_array)[idx])

    def get_max_intsy_scan_num(
        self,
        scan_array: 'ScanArray',
    ):
        """
        Return the scan number containing the maximum intensity
        given by this pointer
        """
        idx = self.get_intensity_values(scan_array).argmax()
        rt = self.get_retention_times(scan_array)[idx]
        return scan_array.rt_to_scan_num(rt)

    def validate_source(
        self,
        scan_array: 'ScanArray',
    ):
        """
        Given a ScanArray, confirms that it is indeed the
        source of this FeaturePointer
        :param scan_array:
        :return:
        """
        if scan_array.uuid != self.source_array_uuid:
            raise ValueError(
                f"FeaturePointer was used to access a ScanArray with "
                f"a non-matching UUID. \n"
                f"FeaturePointer.source_array_uuid: {self.source_array_uuid} \n"
                f"scan_array.uuid: {scan_array.uuid}"
            )

    def __repr__(self):
        return f"FeaturePointer(n_scans={self.n_scans})"