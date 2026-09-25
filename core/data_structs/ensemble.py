"""
Data structue for organizing co-feature ensembles
"""
from dataclasses import dataclass, field
import logging
import uuid
from typing import Literal, Optional, TYPE_CHECKING

import numpy as np
from find_mfs import FormulaCandidate, get_isotope_envelope
from molmass import Formula
from numpy.typing import NDArray

from core.utils.array_types import (
    to_spec_arr, to_ensemble_arr,
    SpectrumArray, ConsensusSpectrumArray
)
from core.data_structs.composite_spectrum import CompositeSpectrum
from core.utils.formula_formatting import format_formula_obj_to_html
from core.utils.spectra import merge_spectra, normalize_spectrum

if TYPE_CHECKING:
    from core.data_structs import(
        Injection,
        ScanArray,
        FeaturePointer,
        EnsembleUUID
    )

    from core.utils.array_types import ChromArray, EnsembleArray

logger = logging.getLogger(__name__)

# MS2 export/reduction strategies (see Ensemble.get_ms2_spectra).
MS2Mode = Literal['tallest', 'all', 'consensus']


@dataclass
class MS2Spectrum:
    """
    A single MS2 spectrum together with the precursor it fragments.

    Produced by `Ensemble.get_ms2_spectra()`

    For DDA, corresponds to a designated precursor
        (a single scan, or a per-precursor consensus);

    For DIA / MS1-only, corresponds to
         the ensemble's resolved (virtual) precursor.
    """
    spectrum: SpectrumArray | ConsensusSpectrumArray | np.ndarray
    precursor_mz: float
    charge: int
    rt: float

@dataclass
class Ensemble:
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

    # User-editable properties. (The formula lives in the ensemble's
    # FormulaAssignment, in the DataRegistry.)
    identity: Optional[str] = None
    user_metadata: dict[str, str] = field(
        default_factory=dict, repr=False
    )

    # DDA precursor info. Populated at construction time for DDA-mode
    # ensembles; None for MS1-only / DIA.
    precursor_mz: Optional[float] = None
    precursor_charge: Optional[int] = None

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
        self.base_mz = base_ftr_ptr.get_mz_values(
            self.injection.scan_array_ms1
        ).mean()

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

        MS1 is the apex scan for all acquisition modes.
        MS2: DIA / MS1-only uses the tallest MS2 scan; DDA uses the matched
        MS2 scan with the tallest precursor (see `_dda_tallest_precursor_ms2`)

        TODO: The DDA behaviour is a placeholder until precursor stitching lands

        Computed lazily
        """
        if self._composite is None:
            if self.is_dda:
                ms2 = self._dda_tallest_precursor_ms2()
            else:
                ms2_spectra = self.get_ms2_spectra()
                ms2 = ms2_spectra[0].spectrum if ms2_spectra else None

            self._composite = CompositeSpectrum(
                ms1=self.get_spectrum(
                    ms_level=1, scan_num=self.base_scan_num
                ),
                ms2=ms2,
                precursor_mz=self.resolved_precursor_mz,
                charge=self.resolved_charge,
            )

        return self._composite

    def _dda_tallest_precursor_ms2(self) -> Optional[SpectrumArray]:
        """
        Placeholder DDA composite MS2: the matched MS2 scan whose precursor
        was most intense in the MS1 scan nearest to it.

        Each scan's precursor is matched to the closest MS1 cofeature lane
        (by m/z); falls back to the tallest MS2 scan when no MS1 cofeature
        lies within 0.5 m/z of any precursor.
        """
        scan_specs = self._iter_ms2_scan_spectra()
        if not scan_specs:
            return None

        ms1_arr = self.injection.scan_array_ms1
        lane_mzs = np.array([
            np.mean(mzs[mzs > 0]) if np.any(mzs > 0) else np.nan
            for mzs in (
                cf.get_mz_values(ms1_arr) for cf in self.ms1_cofeatures
            )
        ])

        def precursor_intsy(spec: MS2Spectrum) -> float:
            diffs = np.abs(lane_mzs - spec.precursor_mz)
            if np.all(np.isnan(diffs)) or np.nanmin(diffs) > 0.5:
                return -np.inf
            cofeature = self.ms1_cofeatures[int(np.nanargmin(diffs))]
            scan_num = ms1_arr.rt_to_scan_num(spec.rt)
            return float(ms1_arr.intsy_arr[cofeature.mz_lane_idx, scan_num])

        intsys = [precursor_intsy(s) for s in scan_specs]
        if max(intsys) == -np.inf:
            return max(scan_specs, key=_max_intsy).spectrum

        return scan_specs[int(np.argmax(intsys))].spectrum

    # ------------------------------------------------------------------
    # MS2 spectrum production
    # ------------------------------------------------------------------

    @property
    def default_ms2_mode(self) -> MS2Mode:
        """
        The default export/production mode, depending on
         this ensemble's acquisition type.

        `consensus` makes more sense for DDA,
        but DIA/MS1-only should use `tallest`
        """
        return 'consensus' if self.is_dda else 'tallest'

    def get_ms2_spectra(
        self,
        mode: Optional[MS2Mode] = None,
        normalize: bool = False,
        bin_width: float = 0.02,
        precursor_tol: float = 0.5,
    ) -> list[MS2Spectrum]:
        """
        Produce this ensemble's MS2 spectra, depending on given `mode`:

        - 'tallest':   the single most intense MS2 scan
        - 'all':       every MS2 scan
                         - DDA: one per matched scan;
                         - DIA/MS1-only:
                            co-feature 'pseudoscan' across elution
        - 'consensus': merged consensus spectrum/spectra
                         - DDA: scans are first grouped by precursor m/z
                            then merged according to Bittremieux 2022
                            (one consensus per precursor)
                         - DIA: all scans merge into a single consensus.

        When `mode` is None (the default), `default_ms2_mode` is used:
        'consensus' for DDA, 'tallest' for DIA / MS1-only.

        Consensus spectra are ConsensusSpectrumArrays that keep
        their per-bin frequency. They can be thresholded at display / print time
        with core.utils.spectra.threshold_consensus (Bittremieux et al. use 0.25).

        Returns an empty list when the ensemble carries no MS2.

        :param mode: MS2 strategy, or None to use `default_ms2_mode`.
        :param normalize: normalize each output spectrum to a peak of 1.
        :param bin_width: consensus bin width (m/z). Only used in 'consensus' mode
        :param precursor_tol: When merging DDA, precursor-grouping tolerance (m/z).
        """
        if mode is None:
            mode = self.default_ms2_mode

        reduced: list[MS2Spectrum] = reduce_ms2_spectra(
            self._iter_ms2_scan_spectra(),
            mode=mode,
            group_by_precursor=self.is_dda,
            bin_width=bin_width,
            precursor_tol=precursor_tol,
        )

        if normalize:
            reduced = [
                MS2Spectrum(
                    spectrum=normalize_spectrum(s.spectrum),
                    precursor_mz=s.precursor_mz,
                    charge=s.charge,
                    rt=s.rt,
                )
                for s in reduced
            ]

        return reduced


    def _iter_ms2_scan_spectra(self) -> list[MS2Spectrum]:
        """
        Gather this ensemble's per-scan MS2 spectra
        (zero-intensity peaks dropped),
        each tagged with its precursor m/z, charge and rt.

        DDA: every matched MS2 scan is a full precursor fragmentation,
        tagged with the instrument-designated precursor.

        DIA / MS1-only:
        the MS2 is reconstructed from the ensemble's MS2 cofeature lanes
        at each scan, tagged with the resolved precursor.
        """
        if self.injection is None:
            return []

        ms2_arr: Optional['ScanArray'] = self.injection.scan_array_ms2
        if ms2_arr is None or not self.ms2_cofeatures:
            return []

        if self.is_dda:
            return self._dda_scan_spectra(ms2_arr)
        return self._dia_scan_spectra(ms2_arr)


    def _dda_scan_spectra(
        self,
        ms2_arr: 'ScanArray',
    ) -> list[MS2Spectrum]:
        # All MS2 cofeatures share the same matched scan_idxs by construction.
        scan_idxs = np.unique(np.asarray(self.ms2_cofeatures[0].scan_idxs))
        default_charge = self.resolved_charge

        out: list[MS2Spectrum] = []
        for raw_idx in scan_idxs:
            scan_idx = int(raw_idx)
            spec = ms2_arr.get_spectrum(scan_idx)
            spec = spec[spec['intsy'] > 0]
            if spec.size == 0:
                continue

            if ms2_arr.precursor_mz_arr is not None:
                precursor_mz = float(ms2_arr.precursor_mz_arr[scan_idx])
            else:
                precursor_mz = self.resolved_precursor_mz

            charge = default_charge
            if ms2_arr.precursor_charge_arr is not None:
                scan_charge = int(ms2_arr.precursor_charge_arr[scan_idx])
                if scan_charge:
                    charge = abs(scan_charge)

            rt = (
                float(ms2_arr.rt_arr[scan_idx])
                if ms2_arr.rt_arr is not None
                else 0.0
            )
            out.append(MS2Spectrum(spec, precursor_mz, charge, rt))

        return out


    def _dia_scan_spectra(
        self,
        ms2_arr: 'ScanArray',
    ) -> list[MS2Spectrum]:
        scan_idxs = np.unique(
            np.concatenate(
                [np.asarray(cf.scan_idxs) for cf in self.ms2_cofeatures]
            )
        )
        precursor_mz = self.resolved_precursor_mz
        charge = self.resolved_charge

        out: list[MS2Spectrum] = []
        for raw_idx in scan_idxs:
            scan_idx = int(raw_idx)
            spec = self.get_spectrum(ms_level=2, scan_num=scan_idx)
            spec = spec[spec['intsy'] > 0]
            if spec.size == 0:
                continue
            rt = (
                float(ms2_arr.rt_arr[scan_idx])
                if ms2_arr.rt_arr is not None
                else 0.0
            )
            out.append(MS2Spectrum(spec, precursor_mz, charge, rt))

        return out


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

    def add_mz_diff_annot(
        self,
        cofeature_a_idx: int,
        cofeature_b_idx: int,
        ms_level: Literal[1, 2],
        delta_mz: float,
        scan_num: Optional[int] = None,
        label: Optional[str] = None,
        formula: Optional[FormulaCandidate] = None,
    ) -> 'MzDiffAnnotation':
        """
        Snapshot of a delta m/z measurement. `delta_mz` is whatever the
        user saw on screen when committing the click — we don't recompute
        from the scan array (the previous behaviour returned 0 / wrong
        values whenever the cofeature lane had no signal at peak_rt,
        which is the common case for sparse DDA MS2 lanes).

        `formula` is an optional neutral-loss formula chosen by the user
        in the formula finder. No validation against the rest of the
        spectrum — it's purely a labelling aid.
        """
        cofeatures = self._get_cofeatures(ms_level)
        for idx in (cofeature_a_idx, cofeature_b_idx):
            if not (0 <= idx <len(cofeatures)):
                raise ValueError(
                    f"Invalid cofeature idx: {idx}, "
                    f"Ensemble only contains {len(cofeatures)} cofeatures "
                )

        annot = MzDiffAnnotation(
            cofeature_a_idx=cofeature_a_idx,
            cofeature_b_idx=cofeature_b_idx,
            ms_level=ms_level,
            delta_mz=delta_mz,
            user_label=label,
            scan_num=scan_num,
            formula=formula,
        )

        self.mz_diffs.append(annot)
        return annot

    def add_ion_annot(
        self,
        cofeature_idxs: list[int],
        ms_level: Literal[1, 2],
        formula: FormulaCandidate,
        label: Optional[str],
        scan_num: Optional[int] = None,
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
            scan_num=scan_num,
        )

        self.ion_annots[annot.uuid] = annot

        return annot

    def add_generic_annot(
        self,
        cofeature_idx: int,
        ms_level: Literal[1, 2],
        text: str,
        scan_num: Optional[int] = None,
        source: str = 'user',
    ) -> 'GenericAnnotation':
        """
        Create and add a free-form annotation anchored to a peak.

        `source` tags the origin ('user', or e.g. 'auto_adduct'); see
        `remove_generic_annots_by_source`.
        """
        cofeatures = self._get_cofeatures(ms_level)
        if not (0 <= cofeature_idx < len(cofeatures)):
            raise ValueError(
                f"Invalid cofeature_idx: {cofeature_idx}. Ensemble only "
                f"contains {len(cofeatures)} cofeatures"
            )

        annot = GenericAnnotation(
            cofeature_idx=cofeature_idx,
            ms_level=ms_level,
            text=text,
            scan_num=scan_num,
            source=source,
        )

        self.generic_annots[annot.uuid] = annot
        return annot

    def remove_generic_annots_by_source(
        self,
        source: str,
    ) -> None:
        """
        Drop every generic annotation with the given `source` tag, leaving all
        others intact. Used to clear a prior automated pass's labels (e.g.
        'auto_adduct') before re-attaching, so re-runs don't stack duplicates.
        """
        self.generic_annots = {
            uuid_: annot
            for uuid_, annot in self.generic_annots.items()
            if annot.source != source
        }

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


####    MS2 spectrum reduction    ####
def reduce_ms2_spectra(
    scan_specs: list[MS2Spectrum],
    mode: MS2Mode,
    *,
    group_by_precursor: bool,
    bin_width: float = 0.02,
    precursor_tol: float = 0.5,
) -> list[MS2Spectrum]:
    """
    Collapse per-scan MS2 spectra according to `mode`

    - 'all':       return the spectra unchanged.
    - 'tallest':   the single most intense scan (by max intensity)
    - 'consensus': merge into consensus spectra (each retaining per-bin
                   frequency for deferred thresholding). When
                   `group_by_precursor` (DDA), scans are first clustered by
                   precursor m/z (within `precursor_tol`) and merged per
                   cluster; otherwise (DIA) all scans merge into a single
                   consensus.
    """
    if not scan_specs:
        return []

    match mode:
        case 'all':
            return list(scan_specs)

        case 'tallest':
            return [max(scan_specs, key=_max_intsy)]

        case 'consensus':
            if group_by_precursor:
                groups: list[list[MS2Spectrum]] = _group_by_precursor(
                    scan_specs,
                    precursor_tol,
                )
            else:
                groups: list[list[MS2Spectrum]] = [list(scan_specs)]
            return [
                _merge_by_group(group, bin_width=bin_width)
                for group in groups
            ]

        case _:
            raise ValueError(
                f"Unknown MS2 mode: {mode!r}"
            )


def _max_intsy(spec: MS2Spectrum) -> float:
    return float(spec.spectrum['intsy'].max())


def _group_by_precursor(
    scan_specs: list[MS2Spectrum],
    precursor_tol: float,
) -> list[list[MS2Spectrum]]:
    """
    Greedily cluster spectra whose precursor m/z falls within
    `precursor_tol` of the group's opening precursor. Discrete DDA
    isolation targets cluster cleanly; the input need not be sorted.
    """
    ordered = sorted(scan_specs, key=lambda s: s.precursor_mz)
    groups: list[list[MS2Spectrum]] = [[ordered[0]]]
    for spec in ordered[1:]:
        if spec.precursor_mz - groups[-1][0].precursor_mz <= precursor_tol:
            groups[-1].append(spec)
        else:
            groups.append([spec])
    return groups


def _merge_by_group(
    specs: list[MS2Spectrum],
    bin_width: float,
) -> MS2Spectrum:
    """
    Merge a group of MS2 spectra (i.e. list of MS2s) into one consensus
    MS2Spectrum.

    The merged spectrum is kept as a ConsensusSpectrumArray so its per-bin
    frequency survives for downstream thresholding at display / print time
    (see core.utils.spectra.threshold_consensus) — nothing is dropped here.
    Precursor m/z is the group median, charge the modal charge, rt that of
    the most intense contributing scan.
    """
    consensus: ConsensusSpectrumArray = merge_spectra(
        (s.spectrum for s in specs),
        bin_width=bin_width,
    )

    precursor_mz = float(
        np.median([s.precursor_mz for s in specs])
    )

    charges = [
        s.charge for s in specs if s.charge
    ]
    if charges:
        vals, counts = np.unique(charges, return_counts=True)
        charge = int(vals[counts.argmax()])
    else:
        charge = 1

    tallest: MS2Spectrum = max(specs, key=_max_intsy)

    return MS2Spectrum(
        spectrum=consensus,
        precursor_mz=precursor_mz,
        charge=charge,
        rt=tallest.rt,
    )


####    Ensemble Annotations    ####

@dataclass
class MzDiffAnnotation:
    """
    A record of m/z difference between two co-features.

    `scan_num` identifies the scan (column index into the ms_level's
    ScanArray) at which the user made the measurement. The viewer shows
    the annotation only when displaying that scan. `None` = scan-agnostic
    (back-compat for .mzk files saved before scan-tied annotations).
    """
    cofeature_a_idx: int
    cofeature_b_idx: int
    ms_level: Literal[1, 2]
    delta_mz: float
    uuid: int = field(default_factory=lambda: uuid.uuid4().int)
    user_label: Optional[str] = None
    scan_num: Optional[int] = None
    formula: Optional[FormulaCandidate] = None


@dataclass
class GenericAnnotation:
    """
    Free-form user annotation anchored to a single cofeature peak.
    `scan_num`: see `MzDiffAnnotation.scan_num`.
    `source`: who created it -- 'user' for manual annotations, or a tag like
    'auto_adduct' for programmatically-attached labels, so an automated pass can
    clear+replace only its own labels without touching the user's.
    """
    cofeature_idx: int
    ms_level: Literal[1, 2]
    text: str
    uuid: int = field(default_factory=lambda: uuid.uuid4().int)
    scan_num: Optional[int] = None
    source: str = 'user'


@dataclass
class IonAnnotation:
    """
    Claim a group of features are isotopoogues.

    `scan_num`: see `MzDiffAnnotation.scan_num`.
    """
    cofeature_idxs: list[int]
    ms_level: Literal[1, 2]
    formula: FormulaCandidate
    uuid: int = field(default_factory=lambda: uuid.uuid4().int)
    user_label: Optional[str] = None
    scan_num: Optional[int] = None

    @property
    def format_string(self) -> str:
        """
        Returns an HTML-formatted string suitable for
        display in MSPlotWidget
        """
        formula_html = format_formula_obj_to_html(self.formula.formula)
        return (f"{formula_html}<br>"
                f"{self.formula.error_ppm:.1f} ppm")

    @property
    def isotope_envelope(self) -> np.ndarray:
        """
        Assembles the neutral formula + adduct + charge and then
        returns the isotope envelope as a numpy array
        (used for plotting)
        """
        # TODO: Crude/inelegant. Can probably just fix on find-mfs side
        # Assemble formula:
        ion_formula = str(self.formula.formula)
        if self.formula.adduct:
            ion_formula += self.formula.adduct

            charge = self.formula.formula.charge
            if charge > 0:
                ion_formula += "+"*charge
            if charge < 0:
                ion_formula += "-"*charge

        envelope = get_isotope_envelope(
            formula=Formula(ion_formula),
            mz_tolerance=0.1,
            threshold=0.005,
        )

        return envelope


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




