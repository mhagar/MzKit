"""
Data structue for organizing co-feature ensembles
"""
from dataclasses import dataclass, field
import uuid
from typing import Literal, Optional, TYPE_CHECKING

import numpy as np
from find_mfs import FormulaCandidate
from molmass import Formula
from numpy.typing import NDArray

from core.utils.array_types import to_spec_arr, to_ensemble_arr
from gui.utils.formula_formatting import format_formula_obj_to_html

if TYPE_CHECKING:
    from core.data_structs import(
        Injection,
        ScanArray,
        FeaturePointer,
        EnsembleUUID
    )

    from core.utils.array_types import SpectrumArray, ChromArray, EnsembleArray

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
    _ms1_cofeature_mz_lane_idxs: np.ndarray[int, ...] = field(
        default=None, init=False, repr=False,
    )
    _ms2_cofeature_mz_lane_idxs: np.ndarray[int, ...] = field(
        default=None, init=False, repr=False,
    )

    # Annotations
    mz_diffs: list['MzDiffAnnotation'] = field(
        default_factory=list, repr=False
    )

    ion_annots: dict[int, 'IonAnnotation'] = field(
        default_factory=dict, repr=False
    )

    ion_pair_annots: list['IonPairAnnotation'] = field(
        default_factory=list, repr=False
    )

    # User-editable properties
    proposed_formula: Optional[str] = None
    identity: Optional[str] = None
    user_metadata: dict[str, str] = field(
        default_factory=dict, repr=False
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
        scan_num: Optional[int] = None,
        scan_rt: Optional[float] = None,
    ) -> NDArray:
        scan_array = self._get_scan_array(ms_level)
        if not scan_num:
            if not scan_rt:
                raise ValueError(
                    "Neither scan_idx nor scan_rt arguments given"
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

        return spec[mz_lane_idxs]


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
        # Select based on MS1 or MS2
        mz_lane_idxs, cofeatures = {
            1: (self._ms1_cofeature_mz_lane_idxs, self.ms1_cofeatures),
            2: (self._ms2_cofeature_mz_lane_idxs, self.ms2_cofeatures),
        }[ms_level]

        if not force_refresh and mz_lane_idxs:
            return mz_lane_idxs

        mz_lane_idxs: NDArray[int] = np.array(
            [x.mz_lane_idx for x in cofeatures]
        )

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

    def add_mz_diff_annot(
        self,
        cofeature_a_idx: int,
        cofeature_b_idx: int,
        ms_level: Literal[1, 2],
        label: Optional[str] = None
    ) -> 'MzDiffAnnotation':
        """
        Creates an MzDiffAnnotation, and also returns it
        """

        # Validate that idxs are real
        cofeatures = self._get_cofeatures(ms_level)
        for idx in (cofeature_a_idx, cofeature_b_idx):
            if not (0 <= idx <len(cofeatures)):
                raise ValueError(
                    f"Invalid cofeature idx: {idx}, "
                    f"Ensemble only contains {len(cofeatures)} cofeatures "
                )

        # Calc mean_delta mz
        scan_array = self._get_scan_array(ms_level)
        mz_a = cofeatures[cofeature_a_idx].get_mz_values(scan_array).mean()
        mz_b = cofeatures[cofeature_b_idx].get_mz_values(scan_array).mean()

        mean_delta_mz: float = abs(mz_a - mz_b)

        annot = MzDiffAnnotation(
            cofeature_a_idx=cofeature_a_idx,
            cofeature_b_idx=cofeature_b_idx,
            ms_level=ms_level,
            mean_delta_mz=mean_delta_mz,
            user_label=label,
        )

        self.mz_diffs.append(annot)
        return annot

    def add_ion_annot(
        self,
        cofeature_idxs: list[int],
        ms_level: Literal[1, 2],
        formula: FormulaCandidate,
        label: Optional[str],
    ) -> 'IonAnnotation':
        """
        Create, validate, and add an ion annotation
        """
        # Validate that indices are real
        cofeatures = self._get_cofeatures(ms_level)
        for idx in cofeature_idxs:
            if not (0 <= idx < len(cofeatures)):
                raise ValueError(
                    f"Invalid cofeature_idx: {idx}. Ensemble only contains "
                    f"{len(cofeatures)} cofeatures"
                )

        annot = IonAnnotation(
            cofeature_idxs=cofeature_idxs,
            ms_level=ms_level,
            formula=formula,
            user_label=label,
        )

        self.ion_annots[annot.uuid] = annot

        return annot

    def add_ion_pair_annot(
        self,
        ion_a_uuid: int,
        ion_b_uuid: int,
        relationship: Literal[
            "adduct", "neutral_loss", "charge_state"
        ],
        label: Optional[str] = None,
    ) -> 'IonPairAnnotation':
        """
        Create and add an IonPairAnnotation, while validating
        """

        # Validate that UUIDs are real:
        for ion_uuid in (ion_a_uuid, ion_b_uuid):
            if ion_a_uuid not in self.ion_annots.keys():
                raise ValueError(
                    f"Ion UUID {ion_uuid} not found in this ensemble"
                )

        # Determine formula difference:
        ## TODO: THIS DOESN'T WORK FOR ADDUCTS. MUST FIX BEFORE USE
        ion_a_formula: Formula = self.ion_annots[ion_a_uuid].formula
        ion_b_formula: Formula = self.ion_annots[ion_b_uuid].formula

        formula_diff = ion_a_formula - ion_b_formula

        annot = IonPairAnnotation(
            ion_a_uuid=ion_a_uuid,
            ion_b_uuid=ion_b_uuid,
            relationship=relationship,
            formula_diff=formula_diff,
            user_label=label,
        )

        self.ion_pair_annots.append(annot)

        return annot


####    Ensemble Annotations    ####

@dataclass
class MzDiffAnnotation:
    """
    A record of m/z difference between two co-features
    """
    cofeature_a_idx: int
    cofeature_b_idx: int
    ms_level: Literal[1, 2]
    mean_delta_mz: float
    user_label: Optional[str] = None


@dataclass
class IonAnnotation:
    """
    Claim a group of features are isotopoogues
    """
    cofeature_idxs: list[int]
    ms_level: Literal[1, 2]
    formula: FormulaCandidate
    uuid: int = field(default_factory=lambda: uuid.uuid4().int)
    user_label: Optional[str] = None

    @property
    def format_string(self) -> str:
        """
        Returns an HTML-formatted string suitable for
        display in MSPlotWidget
        """
        formula_html = format_formula_obj_to_html(self.formula.formula)
        return (f"{formula_html}<br>"
                f"{self.formula.error_ppm:.1f} ppm")


@dataclass
class IonPairAnnotation:
    """
    Claim about the relationship between two ions
    """
    ion_a_uuid: int
    ion_b_uuid: int
    relationship: Literal[
        "adduct", "neutral_loss", "charge_state"
    ]
    formula_diff: Formula
    user_label: Optional[str] = None




