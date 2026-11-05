"""
Data structure for organizing co-feature ensembles.

MS2 production lives in `ms2.py`, annotation types and adders in
`annotations.py`; both are mixed into `Ensemble`.
"""
from dataclasses import dataclass, field
import logging
import uuid
from typing import Literal, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from core.utils.array_types import to_spec_arr
from core.data_structs.composite_spectrum import CompositeSpectrum
from core.utils.config import get_saturation_threshold
from core.utils.spectra import normalize_spectrum

from .annotations import EnsembleAnnotationMixin
from .ms2 import EnsembleMS2Mixin

if TYPE_CHECKING:
    from core.data_structs import(
        Injection,
        ScanArray,
        FeaturePointer,
        EnsembleUUID
    )

    from core.utils.array_types import ChromArray, EnsembleArray, SpectrumArray
    from core.data_structs.formula_assignment import FormulaAssignment
    from .annotations import (
        EnsembleAnnotation,
        MzDiffAnnotation, IonAnnotation, IonPairAnnotation, GenericAnnotation,
    )

logger = logging.getLogger(__name__)


@dataclass
class Ensemble(EnsembleMS2Mixin, EnsembleAnnotationMixin):
    ms1_cofeatures: list['FeaturePointer']
    ms2_cofeatures: list['FeaturePointer']

    uuid: 'EnsembleUUID' = field(default_factory=lambda: uuid.uuid4().int)
    injection: Optional[ 'Injection' ] = None

    # Calculated on initialization
    # TODO: Calculate scan range
    peak_rt: float = field(init=False, repr=False)
    base_mz: float = field(init=False, repr=False)
    base_intsy: float = field(init=False, repr=False)
    base_ms1_cofeature_idx: int = field(init=False, repr=False)
    base_scan_num: int = field(init=False, repr=False)

    # Calculated and cached on demand
    _ms1_cofeature_mz_lane_idxs: np.ndarray[int, ...] = field(
        default=None, init=False, repr=False,
    )
    _ms2_cofeature_mz_lane_idxs: np.ndarray[int, ...] = field(
        default=None, init=False, repr=False,
    )
    # Representative (MS1, MS2) pair
    _composite: Optional['CompositeSpectrum'] = field(
        default=None, init=False, repr=False,
    )

    # Spectrum Annotations
    mz_diffs: list['MzDiffAnnotation'] = field(
        default_factory=list, repr=False
    )

    ion_annots: dict[int, 'IonAnnotation'] = field(
        default_factory=dict, repr=False
    )

    ion_pair_annots: list['IonPairAnnotation'] = field(
        default_factory=list, repr=False
    )

    generic_annots: dict[int, 'GenericAnnotation'] = field(
        default_factory=dict, repr=False
    )

    # Ranked formula candidates + the accepted one. Set through
    # DataRegistry.update_ensemble so views hear about it
    formula_assignment: Optional['FormulaAssignment'] = field(
        default=None, repr=False,
    )

    # Structure hypotheses, chosen structure, name, notes. Set through
    # DataRegistry.update_ensemble so views hear about it
    annotation: Optional['EnsembleAnnotation'] = field(default=None, repr=False)

    # User-editable properties
    identity: Optional[str] = None
    user_metadata: dict[str, str] = field(
        default_factory=dict, repr=False
    )

    # DDA precursor info. Populated at construction time for DDA-mode
    # ensembles; None for MS1-only / DIA.
    precursor_mz: Optional[float] = None
    precursor_charge: Optional[int] = None

    # Detector saturation intensity; scans at/above it are avoided when
    # picking the composite MS1. Taken from the config when the Ensemble is
    # formed (`[instrument] saturation_threshold`)
    saturation_threshold: float = field(
        default_factory=get_saturation_threshold, repr=False,
    )

    def __repr__(self):
        return (f"Ensemble({len(self.ms1_cofeatures)} ms1, "
                f"{len(self.ms2_cofeatures)} ms2 cofeatures. "
                f"UUID: {self.uuid})")

    @property
    def format_string(self) -> str:
        """
        Returns a short string with injection name, retention time,
        that kind of stuff. Useful for file naming
        """
        inj_name: str = self.injection.name
        return f"{inj_name}_{self.peak_rt:.1f}s_{self.base_mz:.5f}mz"

    def _populate_attrs(self):
        # Find the ensemble's MS1 apex and get the scan at that position
        ms1_scan_array: 'ScanArray' = self.injection.get_scan_array(ms_level=1)

        best_cofeature_idx = 0
        best_scan_num = 0
        best_intsy = -np.inf
        best_rt = 0.0

        for idx, cofeature in enumerate(self.ms1_cofeatures):
            scan_idxs = cofeature.scan_idxs
            if scan_idxs.size == 0:
                continue

            s0 = int(scan_idxs[0])
            s1 = int(scan_idxs[-1])
            intsys = ms1_scan_array.intsy_arr[
                cofeature.mz_lane_idx, s0:s1 + 1
            ].toarray().flatten()
            if intsys.size == 0:
                continue

            i = int(intsys.argmax())
            if intsys[i] > best_intsy:
                best_intsy = float(intsys[i])
                best_cofeature_idx = idx
                best_scan_num = s0 + i
                best_rt = float(ms1_scan_array.rt_arr[s0 + i])

        self.base_ms1_cofeature_idx = best_cofeature_idx
        self.base_scan_num = best_scan_num
        self.base_intsy = best_intsy if best_intsy != -np.inf else 0.0
        self.peak_rt = best_rt

        base_ftr_ptr = self.ms1_cofeatures[self.base_ms1_cofeature_idx]
        self.base_mz = base_ftr_ptr.get_apex_mz(self.injection.scan_array_ms1)

    def set_injection(
        self,
        injection: 'Injection',
    ):
        self.injection = injection
        self._populate_attrs()

    def get_spectrum(
        self,
        ms_level: Literal[1, 2],
        scan_num: Optional[int] = None,
        scan_rt: Optional[float] = None,
        normalized: bool = False,
    ) -> NDArray:
        scan_array = self._get_scan_array(ms_level)
        if scan_num is None:
            if scan_rt is None:
                raise ValueError(
                    "Neither scan_num nor scan_rt arguments given"
                )

            scan_num = scan_array.rt_to_scan_num(
                scan_rt
            )

        # Get mz lane idxs corresponding to the ftr_ptrs in this ensemble
        mz_lane_idxs: NDArray[int] = self._get_mz_lane_idxs(ms_level)
        spec = scan_array.get_spectrum(scan_num)

        if mz_lane_idxs.size == 0:
            # TODO: Sometimes an ensemble has no MS2 features..?
            print("EMPTY SPEC!!")
            return spec

        spec_arr = spec[mz_lane_idxs]
        if normalized:
            spec_arr['intsy'] = spec_arr['intsy'] / spec_arr['intsy'].max()

        return spec_arr

    # ------------------------------------------------------------------
    # Identity / precursor resolution
    # ------------------------------------------------------------------

    @property
    def is_dda(self) -> bool:
        """
        True if this ensemble's MS2 comes from DDA acquisition
        """
        return bool(
            self.injection is not None
            and getattr(self.injection, 'acquisition_mode', None) == 'dda'
        )

    def get_meta(
            self,
            key: str,
    ) -> Optional[str]:
        """
        Case-insensitive lookup into user_metadata
        """
        for k, v in self.user_metadata.items():
            if k.lower() == key.lower():
                return v
        return None

    @property
    def resolved_charge(self) -> int:
        """
        Precedence:
        `charge` in user_metadata > DDA precursor charge > (else) 1
        """
        raw = self.get_meta('charge')
        if raw:
            try:
                return int(str(raw).strip().rstrip('+-') or '1')
            except ValueError:
                logger.warning(
                    "Could not parse charge %r; defaulting to 1", raw
                )

        if self.precursor_charge:
            return int(self.precursor_charge)

        return 1

    @property
    def resolved_precursor_mz(self) -> float:
        """
        Precursor m/z if known, otherwise the ensemble's base_mz
        """
        if self.precursor_mz is not None:
            return float(self.precursor_mz)

        return float(self.base_mz)

    # ------------------------------------------------------------------
    # Composite spectrum
    # ------------------------------------------------------------------

    @property
    def composite_spectrum(self) -> CompositeSpectrum:
        """
        The ensemble's representative (MS1, MS2) pair
        i.e. for use with find-mfs or showing by default in EnsembleViewer

        MS1: the most intense unsaturated scan (see `composite_ms1_scan_num`)
        MS2: DIA / MS1-only uses the tallest MS2 scan; DDA uses the
        (normalized) MS2 of the tallest precursor, and takes its precursor
        m/z + charge from that precursor (see `_dda_tallest_precursor_ms2`)

        Computed lazily
        """
        if self._composite is None:
            precursor_mz = self.resolved_precursor_mz
            charge = self.resolved_charge
            if self.is_dda:
                dda_ms2 = self._dda_tallest_precursor_ms2()
                ms2 = None
                if dda_ms2 is not None:
                    # Single scans are raw, consensus spectra 0-1: same scale
                    ms2 = normalize_spectrum(dda_ms2.spectrum)
                    precursor_mz = dda_ms2.precursor_mz
                    charge = dda_ms2.charge
            else:
                ms2_spectra = self.get_ms2_spectra()
                ms2 = ms2_spectra[0].spectrum if ms2_spectra else None

            self._composite = CompositeSpectrum(
                ms1=self.get_spectrum(
                    ms_level=1, scan_num=self.composite_ms1_scan_num()
                ),
                ms2=ms2,
                precursor_mz=precursor_mz,
                charge=charge,
            )

        return self._composite

    def composite_ms1_scan_num(self) -> int:
        """
        The scan the composite MS1 is taken from: within the base
        cofeature's scan range, the scan whose tallest ensemble ion is most
        intense while still below `saturation_threshold`.

        Falls back to the apex (`base_scan_num`) if every scan is saturated.
        """
        scan_idxs = self.base_cofeature.scan_idxs
        lane_idxs = self._get_mz_lane_idxs(1)
        if scan_idxs.size == 0 or lane_idxs.size == 0:
            return self.base_scan_num

        s0, s1 = int(scan_idxs[0]), int(scan_idxs[-1])
        intsys = self.injection.scan_array_ms1.intsy_arr[
            lane_idxs, s0:s1 + 1
        ].toarray().max(axis=0)

        intsys[intsys >= self.saturation_threshold] = -np.inf
        i = int(intsys.argmax())
        if intsys[i] == -np.inf:
            return self.base_scan_num
        return s0 + i

    def _get_mz_lane_idxs(
        self,
        ms_level: Literal[1, 2],
        force_refresh: bool = False,
    ) -> NDArray[int]:
        """
        Returns the mz_lane idxs of the ftr_ptrs comprising
        this ensemble. This retrieval is done only once, then
        cached for later use, unless 'force_refresh' is True
        """
        # Select based on MS1 or MS2. (Use `is None` checks rather than
        # truthiness — these are ndarrays, whose truth value is ambiguous.)
        if ms_level == 1:
            cached = self._ms1_cofeature_mz_lane_idxs
            cofeatures = self.ms1_cofeatures
        else:
            cached = self._ms2_cofeature_mz_lane_idxs
            cofeatures = self.ms2_cofeatures

        if not force_refresh and cached is not None:
            return cached

        mz_lane_idxs: NDArray[int] = np.array(
            [x.mz_lane_idx for x in cofeatures]
        )

        # Write back to the cache (the previous implementation never did,
        # so every call recomputed).
        if ms_level == 1:
            self._ms1_cofeature_mz_lane_idxs = mz_lane_idxs
        else:
            self._ms2_cofeature_mz_lane_idxs = mz_lane_idxs

        return mz_lane_idxs


    def get_chromatograms(
        self,
        ms_level: Literal[1, 2],
        idxs: slice = slice(None),
    ) -> list[np.ndarray]:
        """
        Returns a list of chrom arrays for each of the
        cofeatures at ms_level

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
        Return the chromatogram of the base feature at ms_level
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
        Returns the ScanArray referred to by this Ensemble

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
        """
        Generates a SpectrumArray for plotting
        """
        scan_array = self._get_scan_array(
            ms_level=ms_level
        )
        ftr_ptrs = self._get_cofeatures(
            ms_level=ms_level
        )

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
