"""
Dataclass containing data from an LC/MS injection
"""
import pyopenms as oms
# import numpy as np

from core.data_structs.scan_array import (
    ScanArray, build_scan_array, ScanArrayParameters)
# from core.utils.array_types import to_spec_arr, SpectrumArray

from dataclasses import dataclass, field
import uuid
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structs import (
        InjectionUUID,
        EnsembleUUID,
        Ensemble,
    )

@dataclass
class Injection:
    """
    Container for raw LC/MS data
    """
    filename: str
    scan_array_parameters: tuple[ScanArrayParameters, ...]
    exp: Optional[oms.MSExperiment] = None
    uuid: 'InjectionUUID' = field(default_factory=lambda: uuid.uuid4().int)
    scan_array_ms1: Optional[ScanArray] = None
    scan_array_ms2: Optional[ScanArray] = None
    ensembles: dict['EnsembleUUID', 'Ensemble'] = field(
        default_factory=dict
    )

    def __post_init__(self):
        """
        Constructs ScanArrays if initialized using `exp` argument
        (i.e. given raw .mzML data)
        :return:
        """
        if not self.exp:
            return

        # Calling this loads the thing in memory
        # (some functions don't work otherwise)
        self.exp.get_df()

        # Build MS1 and MS2 scan arrays
        available_ms_levels = self.exp.getMSLevels()

        for level, params in enumerate(self.scan_array_parameters):
            level = level + 1
            if level not in available_ms_levels:
                raise ValueError(
                    f"Requested MS level {level}, but file {self.filename} only"
                    f" contains MS levels {available_ms_levels}"
                )

            self.assemble_scan_array(
                ms_level=level,
                mz_tolerance=params.mz_tolerance,
                scan_gap_tolerance=params.scan_gap_tolerance,
                min_intsy=params.min_intsy,
            )

    def get_scan_array(
        self,
        ms_level: int,
    ):
        match ms_level:
            case 1:
                return self.scan_array_ms1

            case 2:
                return self.scan_array_ms2

            case _:
                raise ValueError(
                    f"Invalid ms_level specified: {ms_level}"
                )


    def assemble_scan_array(
        self,
        ms_level: int,
        mz_tolerance: float,
        scan_gap_tolerance: int,
        min_intsy: float,
    ) -> None:
        """
        Constructs a ScanArray and fills the self.scan_array property.
        This is used for rapid spectrum/chromatogram retrieval.

        Will immediately return None if self.scan_array already exists

        :param ms_level: Can be 1 or 2
        :param mz_tolerance: Maximum m/z difference to consider a 'mass lane'
        :param scan_gap_tolerance: Maximum number of empty scans before
                    starting a new mass lane
        :param min_intsy: Minimum intensity to consider (i.e. noise threshold)
        :return:
        """
        if ms_level not in self.exp.getMSLevels():
            raise ValueError(
                f"Invalid ms_level specified: {ms_level}. "
                f"Experiment only contains {self.exp.getMSLevels()}"
            )

        match ms_level:
            case 1:
                if self.scan_array_ms1:
                    # Skip construction if scan_array already exists
                    return
            case 2:
                if self.scan_array_ms2:
                    return

        # Iterate through experiment and retrieve appropriate spectra
        spectra: list[oms.MSSpectrum] = []
        scan_nums: list[int] = []

        for num, spectrum in enumerate(self.exp.getSpectra()):
            spectrum: oms.MSSpectrum

            if spectrum.getMSLevel() != ms_level:
                continue

            spectra.append(spectrum)
            scan_nums.append(num)

        scan_array: ScanArray = build_scan_array(
            spectra=spectra,
            mz_tolerance=mz_tolerance,
            scan_gap_tolerance=scan_gap_tolerance,
            min_intsy=min_intsy,
            scan_nums=scan_nums,
        )

        match ms_level:
            case 1:
                self.scan_array_ms1 = scan_array
            case 2:
                self.scan_array_ms2 = scan_array
            case _:
                print(
                    f"Warning: requested MS{ms_level} ScanArray, but currently "
                    f"only MS1 and MS2 are supported."
                )


    def __repr__(self):
        return (f"Injection("
                f"{self.filename}, "
                f"uuid={self.uuid}"
                f")")