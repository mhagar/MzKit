"""
Group DDA MS2 spectra originating from the same chemical entity

The unit of grouping is the precursor feature, i.e. the isolated MS1 ion
Found by taking each MS2 scan isolation window and identify tallest MS1 peak

1)
Groups MS2 scans that share a mass lane + chromatographic peak
(i.e. repeated MS2 of one precursor across elution)

2)
Groups precursor features greedily, tallest first, with hard assignment
(centroid/star linkage) where membership is assigned by peak-shape cosine
of the precursor MS1 XICs:
  - default: a tight threshold (0.97)
  - if the seed <-> candidate delta m/z matches a known relationship
    (see core/cli/adduct_table.py), the threshold is loosened (0.85)

The correlation window is derived from the seed peak itself via `find_peak_boundaries`.
`validate_peak` rejects noise spikes.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, TYPE_CHECKING

import numpy as np

from core.data_structs import Ensemble
from core.cli.find_cofeatures import _get_xic_grid, _calculate_cosine_scores
from core.cli.segment_chromatogram import find_peak_boundaries, is_peak
from core.cli.adduct_table import relationship_label

if TYPE_CHECKING:
    from core.data_structs import FeaturePointer, ScanArray, Injection


class GroupingParams(NamedTuple):
    """
    Parameters for precursor-feature grouping.

    min_intsy: Minimum intensity for a signal to count (peak in the isolation
        window, and MS2 lane activity)
    edge_fraction: Stop descending the seed peak when intensity falls below this
        fraction of apex (defines the correlation window). Passed to
        `find_peak_boundaries`
    min_peak_width: Minimum seed-peak width in scans (rejects spikes). Passed to
        `is_peak`.
    min_prominence: Seed-peak bilateral-prominence threshold (`is_peak`) — the
        peakiness knob that rejects slopes and smeared/constant background
        signals from seeding (and thus emitting) a group. 0.5 => apex must rise
        to >= 2x the surrounding baseline on its weaker side.
    min_iso_halfwidth: isolation window used when not specified in scan (in Da)
    tight_cosine: Default membership threshold (no recognized adduct/ISF delta m/z)
    loose_cosine: Membership threshold when the delta m/z is recognized (i.e. Na-H)
    adduct_ppm_tol: ppm tolerance for the delta m/z checking
    use_rel_intsy: Normalize each XIC by its max before scoring.
    polarity: 1 positive, 0 negative. Default 1. Used by adduct table
    """
    min_intsy: float
    edge_fraction: float = 0.1
    min_peak_width: int = 5
    min_prominence: float = 0.5
    min_iso_halfwidth: float = 0.5
    tight_cosine: float = 0.97
    loose_cosine: float = 0.85
    adduct_ppm_tol: float = 20.0
    use_rel_intsy: bool = True
    polarity: int = 1


@dataclass
class PrecursorFeature:
    """
    One MS1 ion isolated for fragmentation, with the MS2 scans it produced

    mz_lane_idx: row in the MS1 ScanArray
    apex_scan: column index of this feature's MS1 chromatographic apex
    mz: representative m/z (MS1 lane label)
    apex_intsy: MS1 intensity at the apex (used to sort seeds)
    ms2_scan_idxs: column indices into the MS2 ScanArray for the scans this
        precursor produced.
    """
    mz_lane_idx: int
    apex_scan: int
    mz: float
    apex_intsy: float
    ms2_scan_idxs: np.ndarray = field(repr=False)


def group_precursor_ensembles(
    injection: 'Injection',
    params: GroupingParams,
    progress_callback=None,  # injected by ProcessRunner; unused here
    cancel_event=None,       # injected by ProcessRunner; unused here
) -> list[Ensemble]:
    """
    Group precursor features originating from same chemical entity.
    Singletons left unassigned.
    Constructs an Ensemble per grouping (as long as it's multi-membered)
     and adds to Injection
    """
    ms1: 'ScanArray' = injection.scan_array_ms1
    ms2: 'ScanArray' = injection.scan_array_ms2
    if ms1 is None or ms2 is None:
        raise ValueError(
            "Missing MS1 or MS2 data"
        )

    if ms2.precursor_mz_arr is None:
        raise ValueError(
            "MS2 data has no precursor information"
        )

    pfs: list[PrecursorFeature] = _build_precursor_features(
        injection,
        params,
    )

    if not pfs:
        return []

    n_scans = ms1.intsy_arr.shape[1]
    apex_scans = np.array([pf.apex_scan for pf in pfs])
    order = sorted(
        range(len(pfs)), key=lambda k: pfs[k].apex_intsy, reverse=True
    )
    assigned = np.zeros(len(pfs), dtype=bool)

    ensembles: list[Ensemble] = []
    for si in order:
        if assigned[si]:
            continue
        seed = pfs[si]
        assigned[si] = True

        chrom = ms1.intsy_arr[seed.mz_lane_idx].toarray().flatten()
        seg_start, seg_end = find_peak_boundaries(
            chrom, seed.apex_scan, edge_fraction=params.edge_fraction
        )

        # The seed must be a genuine chromatographic peak — not a slope or a
        # smeared/constant background signal — to anchor a group at all. This
        # gates both recruitment AND emission, so background-smear precursors
        # (even when fragmented repeatedly) are dropped rather than emitted as
        # lone groups.
        if not is_peak(
            chrom, seed.apex_scan, seg_start, seg_end,
            min_peak_width=params.min_peak_width,
            min_prominence=params.min_prominence,
        ):
            continue

        recruited = _recruit_cofeatures(
            ms1, seed, pfs, apex_scans, assigned, seg_start, seg_end, params
        )
        for k in recruited:
            assigned[k] = True

        group = [si] + recruited
        ms2_union = np.unique(
            np.concatenate([pfs[k].ms2_scan_idxs for k in group])
        )
        # Emit any grouping with >= 2 MS2 scans (incl. lone-precursor replicates
        # on a real peak); a single-MS2 precursor is a true singleton, skipped.
        if ms2_union.size < 2:
            continue

        display_idxs = _display_window(
            seg_start, seg_end, seed.apex_scan, n_scans
        )
        ensembles.append(
            _build_ensemble(injection, group, pfs, display_idxs, ms2_union, params)
        )

    return ensembles


def _recruit_cofeatures(
    ms1: 'ScanArray',
    seed: PrecursorFeature,
    pfs: list[PrecursorFeature],
    apex_scans: np.ndarray,
    assigned: np.ndarray,
    seg_start: int,
    seg_end: int,
    params: GroupingParams,
) -> list[int]:
    """
    Cosine score unassigned precursor features against the seed's MS1 XIC,
    (as long as the apex matches), and return passing cofeatures.
    Threshold is loosened when the delta m/z matches a known
    relationship
    """
    cand_mask = (~assigned) & (apex_scans >= seg_start) & (apex_scans < seg_end)
    cand_idxs = np.where(cand_mask)[0]
    if cand_idxs.size == 0:
        return []

    scan_idxs = np.arange(seg_start, seg_end)
    seed_ftr = ms1.make_feature_pointer(seed.mz_lane_idx, scan_idxs)
    cand_lanes = np.array([pfs[k].mz_lane_idx for k in cand_idxs])
    # A candidate whose only signal sits on the truncated last window scan can
    # normalize to an all-zero row (0/0 -> NaN). That's fine — NaN scores are
    # dropped below — so silence the expected divide warning.
    with np.errstate(invalid="ignore", divide="ignore"):
        cand_xics = _get_xic_grid(
            mass_lane_idxs=cand_lanes,
            scan_array=ms1,
            scan_start=seed_ftr.scan_start,
            scan_end=seed_ftr.scan_end,
            use_rel_intsy=params.use_rel_intsy,
        )
        search_xic = seed_ftr.get_intensity_values(ms1)
        if params.use_rel_intsy:
            search_xic = search_xic / search_xic.max()

        scores = _calculate_cosine_scores(cand_xics, search_xic)

    recruited: list[int] = []
    for j, k in enumerate(cand_idxs):
        score = scores[j]
        if np.isnan(score):
            continue
        label = relationship_label(
            seed.mz, pfs[k].mz, params.adduct_ppm_tol, params.polarity
        )
        threshold = params.loose_cosine if label else params.tight_cosine
        if score >= threshold:
            recruited.append(int(k))
    return recruited


def _display_window(
    seg_start: int,
    seg_end: int,
    apex: int,
    n_scans: int,
    min_half: int = 3,
) -> np.ndarray:
    """
    Scan indices for the MS1 cofeature pointers in an emitted Ensemble.

    Uses seed peak boundaries, padded to at least `min_half` scans on each side of
    the apex so that lone/spike seeds still yield non-degenerate trace
    """
    lo = max(0, min(seg_start, apex - min_half))
    hi = min(n_scans, max(seg_end, apex + min_half + 1))
    return np.arange(lo, hi)


def _build_precursor_features(
    injection: 'Injection',
    params: GroupingParams,
) -> list[PrecursorFeature]:
    """
    Stage 1: map each MS2 scan to an MS1 precursor lane + peak
    """
    ms1 = injection.scan_array_ms1
    ms2 = injection.scan_array_ms2

    # lane -> {ms1_col -> [ms2 scan idxs]}
    lane_to_cols: dict[int, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_ms2 = ms2.rt_arr.size
    for i in range(n_ms2):
        window = _isolation_window(ms2, i, params.min_iso_halfwidth)
        if window is None:
            continue
        eff_lo, eff_hi = window
        ms1_col = _ms1_col_for_ms2(injection, i)
        lane = _tallest_lane_in_window(
            ms1, ms1_col, eff_lo, eff_hi, params.min_intsy
        )
        if lane is None:
            continue
        lane_to_cols[lane][ms1_col].append(i)

    features: list[PrecursorFeature] = []
    for lane, cols_dict in lane_to_cols.items():
        chrom = ms1.intsy_arr[lane].toarray().flatten()
        features.extend(_cluster_lane_into_features(ms1, lane, chrom, cols_dict))
    return features


def _isolation_window(
    ms2: 'ScanArray',
    ms2_idx: int,
    min_halfwidth: float,
) -> Optional[tuple[float, float]]:
    """
    Effective [lo, hi] m/z window to search for
     the precursor, added to at least `min_halfwidth`
      on each side.

    Falls back to the recorded precursor m/z as the window center
    when the encoded window is missing/degenerate.
    Returns None if neither a window nor a precursor m/z is available.
    """
    lo = ms2.isolation_lo_arr[ms2_idx]
    hi = ms2.isolation_hi_arr[ms2_idx]
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        center = 0.5 * (lo + hi)
        halfwidth = 0.5 * (hi - lo)
    else:
        prec = ms2.precursor_mz_arr[ms2_idx]
        if not np.isfinite(prec):
            return None
        center = float(prec)
        halfwidth = 0.0
    halfwidth = max(halfwidth, min_halfwidth)
    return center - halfwidth, center + halfwidth


def _ms1_col_for_ms2(injection: 'Injection', ms2_idx: int) -> int:
    """
    Column index into the MS1 ScanArray to inspect for this MS2 scan.

    Prefer the triggering MS1 scan number (matched against MS1 `scan_num_arr`);
    fall back to the nearest MS1 scan by retention time.
    """
    ms1 = injection.scan_array_ms1
    ms2 = injection.scan_array_ms2
    if ms2.triggering_ms1_scan_arr is not None:
        trig = int(ms2.triggering_ms1_scan_arr[ms2_idx])
        hits = np.where(ms1.scan_num_arr == trig)[0]
        if hits.size:
            return int(hits[0])
    return int(ms1.rt_to_scan_num(float(ms2.rt_arr[ms2_idx])))


def _tallest_lane_in_window(
    ms1: 'ScanArray',
    ms1_col: int,
    lo: float,
    hi: float,
    min_intsy: float,
) -> Optional[int]:
    """
    Row (mass lane) of the tallest MS1 signal inside [lo, hi] at `ms1_col`, or
    None if the window is empty

    # TODO: I'm pretty sure scanarray already has this method implemented
    """
    mzs = ms1.mz_arr_csc._getcol(ms1_col).toarray().flatten()
    intsys = ms1.intsy_arr_csc._getcol(ms1_col).toarray().flatten()
    mask = (mzs >= lo) & (mzs <= hi) & (intsys > min_intsy)
    rows = np.where(mask)[0]
    if rows.size == 0:
        return None
    return int(rows[np.argmax(intsys[rows])])


def _climb_to_apex(chrom: np.ndarray, col: int) -> int:
    """
    Climb monotonically uphill from `col` to the local chromatographic apex
    """
    i = int(col)
    n = chrom.size
    while i + 1 < n and chrom[i + 1] > chrom[i]:
        i += 1
    while i - 1 >= 0 and chrom[i - 1] > chrom[i]:
        i -= 1
    return i


def _cluster_lane_into_features(
    ms1: 'ScanArray',
    lane: int,
    chrom: np.ndarray,
    cols_dict: dict[int, list[int]],
) -> list[PrecursorFeature]:
    """
    Split one mass lane's MS2-triggering columns into per-peak precursor features

    Greedily anchor on the tallest unassigned column, climb to its apex, take the
    peak boundaries, and absorb all triggering columns within them
    """
    mz = float(ms1.mz_lane_label[lane])
    unassigned = set(cols_dict.keys())
    features: list[PrecursorFeature] = []

    while unassigned:
        anchor = max(unassigned, key=lambda c: chrom[c])
        apex = _climb_to_apex(chrom, anchor)
        seg_start, seg_end = find_peak_boundaries(chrom, apex)

        in_peak = [c for c in unassigned if seg_start <= c < seg_end]
        if not in_peak:  # safety: anchor sat exactly on the boundary
            in_peak = [anchor]

        ms2_idxs: list[int] = []
        for c in in_peak:
            ms2_idxs.extend(cols_dict[c])
            unassigned.discard(c)

        features.append(
            PrecursorFeature(
                mz_lane_idx=lane,
                apex_scan=apex,
                mz=mz,
                apex_intsy=float(chrom[apex]),
                ms2_scan_idxs=np.array(sorted(ms2_idxs)),
            )
        )

    return features


def _build_ensemble(
    injection: 'Injection',
    group_idxs: list[int],
    pfs: list[PrecursorFeature],
    scan_idxs: np.ndarray,
    ms2_idxs: np.ndarray,
    params: GroupingParams,
) -> Ensemble:
    """
    Assemble an Ensemble from a group of precursor features.

    MS1 cofeatures are deduplicated by mass lane
    (a single elution over-split in stage 1, or a trivial-positive
     lone precursor, collapses to one MS1 trace),
      while the MS2 cofeatures span the union of all grouped
       MS2 scans.
    """
    ms1 = injection.scan_array_ms1
    ms2 = injection.scan_array_ms2

    ms1_cofeatures: list['FeaturePointer'] = []
    seen_lanes: set[int] = set()
    for k in group_idxs:
        lane = pfs[k].mz_lane_idx
        if lane in seen_lanes:
            continue
        seen_lanes.add(lane)
        ms1_cofeatures.append(ms1.make_feature_pointer(lane, scan_idxs.copy()))

    ms2_cofeatures = _make_ms2_cofeatures(ms2, ms2_idxs)

    ensemble = Ensemble(
        ms1_cofeatures=ms1_cofeatures,
        ms2_cofeatures=ms2_cofeatures,
    )
    injection.add_ensemble(ensemble)
    return ensemble


def _make_ms2_cofeatures(
    ms2: 'ScanArray',
    ms2_idxs: np.ndarray,
) -> list['FeaturePointer']:
    """
    One FeaturePointer per MS2 mass lane carrying signal in the grouped scans.

    NOT intensity-filtered: in DDA every peak in the precursor's MS2 scans is a
    real fragment of that precursor (and MS2 is never noise-filtered at build
    time), so filtering here would silently drop low-intensity fragment peaks
    from the ensemble's reconstructed spectrum.
    """
    if ms2 is None or ms2_idxs.size == 0:
        return []
    lane_max = ms2.intsy_arr[:, ms2_idxs].max(axis=1).toarray().flatten()
    active = np.where(lane_max > 0)[0]
    return [
        ms2.make_feature_pointer(int(lane), ms2_idxs.copy())
        for lane in active
    ]
