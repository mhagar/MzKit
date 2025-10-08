"""
Data structue for organizing co-feature ensembles
"""
from dataclasses import dataclass, field
import uuid
from typing import Literal, Optional, TYPE_CHECKING

import numpy as np

from core.utils.array_types import to_spec_arr, to_chrom_arr

if TYPE_CHECKING:
    from core.data_structs import(
        Injection,
        ScanArray,
        FeaturePointer,
        EnsembleUUID
    )

    from core.utils.array_types import SpectrumArray

@dataclass
class Ensemble:
    ms1_cofeatures: list['FeaturePointer']
    ms2_cofeatures: list['FeaturePointer']

    uuid: 'EnsembleUUID' = field(default_factory=lambda: uuid.uuid4().int)
    injection: Optional[ 'Injection' ] = None

    # Calculated on initialization
    peak_rt: float = field(init=False, repr=False)
    base_mz: float = field(init=False, repr=False)
    base_intsy: float = field(init=False, repr=False)
    base_ms1_cofeature_idx: int = field(init=False, repr=False)
    base_scan_num: int = field(init=False, repr=False)

    # Calculated and cached on demand
    _ms1_spectrum: 'SpectrumArray' = field(
        default=None, init=False, repr=False
    )
    _ms2_spectrum: 'SpectrumArray' = field(
        default=None, init=False, repr=False
    )

    def __repr__(self):
        return (f"Ensemble({len(self.ms1_cofeatures)} ms1, "
                f"{len(self.ms2_cofeatures)} ms2 cofeatures)")

    def _populate_attrs(self):
        # Find base co-feature
        ms1_scan_array: 'ScanArray' = self.injection.get_scan_array(ms_level=1)
        ftr_ptr_intsys: np.ndarray = np.array(
            [
                x.get_max_intsy(
                    scan_array=ms1_scan_array,
                ) for x in self.ms1_cofeatures
            ]
        )

        self.base_ms1_cofeature_idx: int = np.argmax(ftr_ptr_intsys) # type: ignore
        base_ftr_ptr = self.ms1_cofeatures[self.base_ms1_cofeature_idx]

        self.base_scan_num = base_ftr_ptr.get_max_intsy_scan_num(
            scan_array=self.injection.scan_array_ms1
        )

        self.base_mz = base_ftr_ptr.get_mz_values(
            self.injection.scan_array_ms1
        ).mean()

        bpc = base_ftr_ptr.get_chrom_array(ms1_scan_array)
        self.base_intsy = np.max(bpc['intsy'])
        self.peak_rt = bpc['rt'][np.argmax(bpc['intsy'])]

    def set_injection(
        self,
        injection: 'Injection',
    ):
        self.injection = injection
        self._populate_attrs()

    def get_spectrum(
        self,
        ms_level: Literal[1, 2],
    ) -> 'SpectrumArray':
        spec: 'SpectrumArray' = {
            1: self._ms1_spectrum,
            2: self._ms2_spectrum,
        }[ms_level]

        if not spec:
            spec: 'SpectrumArray' = self._generate_spectrum(
                ms_level=ms_level,
            )

        return spec

    def get_chromatograms(
        self,
        ms_level: Literal[1, 2],
        idxs: slice = slice(None),
    ) -> list[np.ndarray]:
        """
        Returns a list of chrom arrays for each of the
        cofeatures

        :param ms_level:
        :param idxs: Which chromatograms to return. If none, returns all of
                    them. For example, passing `slice(5, 23)` is the same as
                    doing `chromatograms[5:23]`.
        :return:
        """
        if not self.injection:
            raise ValueError(
                "Ensemble has not been assigned to an Injection yet. "
                "Use set_injection()"
            )

        cofeatures: list['FeaturePointer'] = self._get_cofeatures(
            ms_level,
            idxs,
        )
        scan_array: 'ScanArray' = self._get_scan_array(ms_level)

        chroms: list[np.ndarray] = []
        for ftr_ptr in cofeatures:
            chroms.append(
                ftr_ptr.get_chrom_array(scan_array)
            )

        return chroms

    def get_base_chromatogram(
        self,
        ms_level: Literal[1, 2],
    ) -> np.ndarray:
        """
        Return the chromatogram of the base feature
        :param ms_level:
        :return:
        """
        scan_array: 'ScanArray' = self._get_scan_array(ms_level)
        return self.base_cofeature.get_chrom_array(scan_array)

    @property
    def base_cofeature(self) -> 'FeaturePointer':
        return self.ms1_cofeatures[self.base_ms1_cofeature_idx]

    def _get_scan_array(
        self,
        ms_level: Literal[1, 2],
    ) -> 'ScanArray':
        """
        Returns the ScanArray defined by this Ensemble

        If this Ensemble was
         loaded from disk, make sure `set_injection()` was
         called at some point

        :param ms_level:
        :return:
        """
        if not self.injection:
            raise ValueError(
                "Ensemble has not been assigned to an Injection yet. "
                "Use set_injection()"
            )

        match ms_level:
            case 1:
                return self.injection.scan_array_ms1

            case 2:
                return self.injection.scan_array_ms2

            case _:
                raise ValueError(
                    f"Invalid ms_level specified: {ms_level}"
                )

    def _get_cofeatures(
        self,
        ms_level: Literal[1, 2],
        idxs: slice = slice(None),
    ) -> list['FeaturePointer']:
        match ms_level:
            case 1:
                return self.ms1_cofeatures[idxs]

            case 2:
                return self.ms2_cofeatures[idxs]

            case _:
                raise ValueError(
                    f"Invalid ms_level specified: {ms_level}"
                )

    def _generate_spectrum(
        self,
        ms_level: Literal[1, 2],
    ) -> 'SpectrumArray':
        scan_array = self._get_scan_array(ms_level=ms_level)
        ftr_ptrs = self._get_cofeatures(ms_level=ms_level)

        mz_values: list[float] = []
        intsy_values: list[float] = []
        for ftr_ptr in ftr_ptrs:
            mz_values.append(
                ftr_ptr.get_mz_values(scan_array).max()
                # ftr_ptr.get_mz_values(scan_array).mean()
            )
            intsy_values.append(
                ftr_ptr.get_max_intsy(scan_array)
            )

        return to_spec_arr(
            mz_arr=np.array(mz_values),
            intsy_arr=np.array(intsy_values),
        )

        # print(
        #     f"base_scan_num: {self.base_scan_num}"
        # )

        # return scan_array.get_spectrum(self.base_scan_num)
