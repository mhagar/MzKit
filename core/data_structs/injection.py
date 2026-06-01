"""
Dataclass containing data from an LC/MS injection
"""
import pyopenms as oms
# import numpy as np

from core.data_structs.scan_array import (
    ScanArray, build_scan_array, ScanArrayParameters)
# from core.utils.array_types import to_spec_arr, SpectrumArray

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
import numpy as np
from typing import Literal, Optional, TYPE_CHECKING

AcquisitionMode = Literal['ms1_only', 'dda', 'dia']

if TYPE_CHECKING:
    from core.data_structs import (
        SampleUUID,
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
    uuid: 'InjectionUUID' = field(default_factory=lambda: uuid4().int)
    sample_uuid: Optional['SampleUUID'] = None
    scan_array_ms1: Optional[ScanArray] = None
    scan_array_ms2: Optional[ScanArray] = None
    acquisition_mode: AcquisitionMode = 'ms1_only'
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

        # DDA at MS2: replicate MS2 scans of one precursor are interleaved
        # with MS2s of other precursors, so any finite gap tolerance would
        # fragment mass lanes incorrectly. The Ensemble layer does the
        # compound-level filtering downstream.
        is_dda_ms2 = (ms_level == 2 and self.acquisition_mode == 'dda')

        # Iterate through experiment and retrieve appropriate spectra,
        # tracking the most recent MS1 scan_num as we go (used to populate
        # `triggering_ms1_scan_arr` for DDA MS2 ScanArrays).
        spectra: list[oms.MSSpectrum] = []
        scan_nums: list[int] = []
        precursor_mzs: list[float] = []
        precursor_charges: list[int] = []
        isolation_los: list[float] = []
        isolation_his: list[float] = []
        triggering_ms1_scans: list[int] = []
        last_ms1_scan_num: int = -1

        for num, spectrum in enumerate(self.exp.getSpectra()):
            spectrum: oms.MSSpectrum

            current_level = spectrum.getMSLevel()
            if current_level == 1:
                last_ms1_scan_num = num

            if current_level != ms_level:
                continue

            spectra.append(spectrum)
            scan_nums.append(num)

            if is_dda_ms2:
                precursors = spectrum.getPrecursors()
                if precursors:
                    prec = precursors[0]
                    prec_mz = prec.getMZ()
                    lo_off = prec.getIsolationWindowLowerOffset()
                    hi_off = prec.getIsolationWindowUpperOffset()
                    precursor_mzs.append(prec_mz)
                    precursor_charges.append(prec.getCharge())
                    isolation_los.append(prec_mz - lo_off)
                    isolation_his.append(prec_mz + hi_off)
                else:
                    # MS2 scan with no precursor metadata — shouldn't happen
                    # in real DDA data but stay defensive
                    precursor_mzs.append(np.nan)
                    precursor_charges.append(0)
                    isolation_los.append(np.nan)
                    isolation_his.append(np.nan)
                triggering_ms1_scans.append(last_ms1_scan_num)

        effective_gap = scan_gap_tolerance
        if is_dda_ms2:
            # Disable gap-counting entirely. `build_features` compares
            # gap_counter > tolerance, so any value >= len(spectra) suffices.
            effective_gap = len(spectra) + 1

        scan_array: ScanArray = build_scan_array(
            spectra=spectra,
            mz_tolerance=mz_tolerance,
            scan_gap_tolerance=effective_gap,
            min_intsy=min_intsy,
            scan_nums=scan_nums,
        )

        if is_dda_ms2:
            scan_array.precursor_mz_arr = np.array(precursor_mzs, dtype='f4')
            scan_array.precursor_charge_arr = np.array(precursor_charges, dtype='i4')
            scan_array.isolation_lo_arr = np.array(isolation_los, dtype='f4')
            scan_array.isolation_hi_arr = np.array(isolation_his, dtype='f4')
            scan_array.triggering_ms1_scan_arr = np.array(triggering_ms1_scans, dtype='i4')

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

    def add_ensemble(
        self,
        ensemble: 'Ensemble',
    ) -> None:
        """
        Add and register an Ensemble to this injection
        """
        if ensemble.uuid in self.ensembles:
            raise ValueError(
                f"Ensemble already exists: {ensemble}"
            )

        ensemble.set_injection(self)
        self.ensembles[ensemble.uuid] = ensemble

    def remove_ensemble(
        self,
        uuid: 'EnsembleUUID',
    ) -> None:
        """
        Remove an ensemble from this injection.
        """
        if uuid in self.ensembles:
            del self.ensembles[uuid]

    @property
    def name(self) -> str:
        """
        Returns the filename stripped of any suffixes
        """
        return Path(self.filename).stem


    def __repr__(self):
        return (f"Injection("
                f"{self.filename}, "
                f"uuid={self.uuid}"
                f")")