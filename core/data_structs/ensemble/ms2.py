"""
MS2 spectrum production (Ensemble mixin) and reduction helpers
"""
from dataclasses import dataclass
from typing import Literal, Optional, TYPE_CHECKING

import numpy as np

from core.utils.array_types import SpectrumArray, ConsensusSpectrumArray
from core.utils.spectra import merge_spectra, normalize_spectrum

if TYPE_CHECKING:
    from core.data_structs import ScanArray

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


class EnsembleMS2Mixin:
    """
    Ensemble's MS2 production methods. Mixed into `Ensemble` only.
    """

    def _dda_representative_ms2(self, ms1) -> Optional[MS2Spectrum]:
        """
        DDA composite MS2: the MS2 of the fragmented precursor whose ion
        fragments best (`core.formula.network.ion_tier`: non-metal monomers
        first, metal adducts last), the most intense in MS1 among equals
        (each scan's precursor intensity is read in the MS1 scan nearest to
        it). A precursor's ion is its envelope's in `ms1` (the composite MS1),
        from the ensemble's AdductNetwork if it has one, else find-mfs's
        network (`envelope_ions`). If that precursor was fragmented several
        times, its scans are merged into one consensus (see `reduce_ms2_spectra`).

        Each scan's precursor is matched to the closest MS1 cofeature lane
        (by m/z); falls back to the tallest MS2 scan when no MS1 cofeature
        lies within 0.5 m/z of any precursor.

        The returned precursor_mz is the matched cofeature's MS1-measured
        m/z (more accurate than the instrument's isolation target), or the
        fallback scan's own precursor.
        """
        scan_specs = self._iter_ms2_scan_spectra()
        if not scan_specs:
            return None

        ms1_arr = self.injection.scan_array_ms1
        lane_mzs = np.array([
            cf.get_apex_mz(ms1_arr) or np.nan  # 0.0: lane empty in its scans
            for cf in self.ms1_cofeatures
        ])

        def precursor_lane(spec: MS2Spectrum) -> Optional[int]:
            diffs = np.abs(lane_mzs - spec.precursor_mz)
            if np.all(np.isnan(diffs)) or np.nanmin(diffs) > 0.5:
                return None
            return int(np.nanargmin(diffs))

        def precursor_intsy(spec: MS2Spectrum) -> float:
            lane = precursor_lane(spec)
            if lane is None:
                return -np.inf
            cofeature = self.ms1_cofeatures[lane]
            scan_num = ms1_arr.rt_to_scan_num(spec.rt)
            return float(ms1_arr.intsy_arr[cofeature.mz_lane_idx, scan_num])

        groups = _group_by_precursor(scan_specs, precursor_tol=0.5)
        group_tallest = [max(group, key=precursor_intsy) for group in groups]
        group_intsys = [precursor_intsy(s) for s in group_tallest]
        if max(group_intsys) == -np.inf:
            return max(scan_specs, key=_max_intsy)

        from core.formula.network import envelope_ions, ion_tier
        ions = envelope_ions(ms1, self.adduct_network)
        i = min(
            (k for k, intsy in enumerate(group_intsys) if intsy > -np.inf),
            key=lambda k: (ion_tier(ions.get(precursor_lane(group_tallest[k]))), -group_intsys[k]),
        )
        best = groups[i]
        ms2 = best[0] if len(best) == 1 else reduce_ms2_spectra(
            best, 'consensus', group_by_precursor=False,
        )[0]
        return MS2Spectrum(
            spectrum=ms2.spectrum,
            precursor_mz=float(lane_mzs[precursor_lane(group_tallest[i])]),
            charge=ms2.charge,
            rt=ms2.rt,
        )

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
    Precursor m/z and rt are those of the most intense contributing scan,
    charge the modal charge.
    """
    consensus: ConsensusSpectrumArray = merge_spectra(
        (s.spectrum for s in specs),
        bin_width=bin_width,
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
        precursor_mz=tallest.precursor_mz,
        charge=charge,
        rt=tallest.rt,
    )
