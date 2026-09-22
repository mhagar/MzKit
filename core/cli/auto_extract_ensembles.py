"""
Auto-Ensemble Extraction.

Both DIA-style (correlate every coeluting lane against a seed) and DDA-style
(group MS2-triggering precursor features) acquisition share ONE greedy
seed-and-recruit skeleton.

This module holds engine for greedy seed-and-recruit Ensemble generation algorithm;
used for both DIA and DDA.

    1. Pick the tallest unassigned *seed*
    2. Find its chromatographic peak boundaries; reject non-peaks (`is_peak`)
    3. Choose a correlation/extraction *window* (fixed half-width OR peak bounds).
    4. Recruit MS1 cofeatures by peak-shape similarity (cosine|pearson), with an
       optional adduct-aware threshold loosening when the seed<->candidate
       delta-m/z matches a known relationship (isotope / adduct / neutral loss)
    5. Attach MS2 signals
        - DIA: correlation across the MS2 array;
        - DDA: union of the group's precursor MS2 scans
    6. Emit an `Ensemble` and consume the assigned regions
    7. Repeat for the next tallest unassigned *seed*

Acquisition-specific behaviour (i.e. where seeds come from, how consumption is
tracked, which candidates are eligible, and how MS2 is attached) is defined by
`SeedProvider` and a handful of switches on `ExtractionConfig`.

The scoring core is vectorized, so only the DDA path materializes
per-precursor objects.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Optional, TYPE_CHECKING

import numpy as np

from core.data_structs import Ensemble
from core.cli.find_cofeatures import (
    _find_nonzero_mass_lanes,
    _calculate_cofeature_scores,
    find_cofeatures_across_scan_array,
    CofeatureMetric,
    DEFAULT_COFEATURE_METRIC,
)
from core.cli.segment_chromatogram import (
    find_peak_boundaries,
    is_peak,
    survives_smoothing,
)
from core.cli.adduct_table import relationship_label

if TYPE_CHECKING:
    from core.data_structs import FeaturePointer, ScanArray, Injection


# Where seeds come from
#   "all_lanes"  -> every MS1 mass lane, tallest-first (DIA / auto)
#   "precursor"  -> MS2-triggering precursor features only (DDA)
SeedStrategy = Literal["all_lanes", "precursor"]

# How the correlation/extraction window is derived from the seed peak
#   "fixed"       -> +-extraction_half_width scans around the apex
#   "peak_bounds" -> the peak boundaries from find_peak_boundaries (non-fixed)
WindowStrategy = Literal["fixed", "peak_bounds"]

# How MS2 cofeatures are attached to an emitted ensemble.
#   "none"              -> MS1-only
#   "correlation"       -> correlate MS2 lanes against the seed XIC (DIA)
#   "precursor"         -> union of the grouped precursor features' MS2 scans;
#                          seeds ARE precursors, so the seed/candidate objects carry
#                          the MS2 scan idxs directly (DDA, precursor seeding)
#   "precursor_by_lane" -> union of the triggered MS2 of any precursor feature whose
#                          lane is in the emitted ensemble and whose apex falls in the
#                          seed peak. Decoupled from the seed strategy (looked up via a
#                          prebuilt lane->precursor index), so it works under all-lanes
#                          seeding: DIA-style MS1 grouping over DDA data, keeping the
#                          real triggered MS2 for whichever members were fragmented.
MS2Strategy = Literal["none", "correlation", "precursor", "precursor_by_lane"]


@dataclass
class ExtractionConfig:
    """
    Config for one extraction run.

    The three `*_strategy` fields select the acquisition-specific behavior;
    everything else is a scalar the two strategies share.
    Preset builders `dia_config` / `dda_config` fill sensible values.

    Seeding
        seed_strategy: "all_lanes" (DIA) or "precursor" (DDA)
        parent_threshold: minimum apex intensity for a
            signal to seed a new ensemble
        cofeature_threshold: minimum intensity for a candidate
            signal/lane to be considered for recruitment
            (and, for DDA, for precursor detection)

    Window
        window_strategy: "fixed" or "peak_bounds".
        extraction_half_width: half-window in scans (used when "fixed")
        edge_fraction: passed to find_peak_boundaries.
        min_window_halfwidth: minimum padding (scans) on each side of the apex
            for "peak_bounds", so narrow/spike seeds still yield a usable trace

    Peak validation (is_peak)
        min_peak_width, min_prominence: reject spikes / slopes / smears.
        baseline_pct: percentile used to estimate each shoulder's baseline.
            Higher values judge prominence against a higher surrounding level,
            so bumps riding on elevated background are rejected more readily.

    MS1 recruitment
        method: "cosine" | "pearson" peak-shape metric.
        use_rel_intsy: normalize each XIC by its max before scoring.
        ms1_corr_threshold: the tight (default) membership threshold.
        adduct_aware: if True, loosen to ``loose_corr_threshold`` when the
            seed<->candidate delta-m/z matches a known relationship.
        loose_corr_threshold, adduct_ppm_tol, polarity: adduct-loosening params.
        require_apex_in_window: restrict candidates to those whose own apex falls
            inside the seed window (DDA precursor features; ignored for lanes).

    MS2 attachment
        ms2_strategy: "none" | "correlation" | "precursor".
        ms2_corr_threshold: correlation threshold ("correlation").
        min_iso_halfwidth: isolation-window half-width fallback ("precursor").
        min_ms2_scans: emit gate for "precursor" — drop groups with fewer than
            this many unique MS2 scans.

    Scope
        rt_range: optional (rt_start, rt_end) to restrict seeding.
    """
    # seeding
    seed_strategy: SeedStrategy
    parent_threshold: float
    cofeature_threshold: float

    # window
    window_strategy: WindowStrategy = "peak_bounds"
    extraction_half_width: int = 10
    edge_fraction: float = 0.1
    min_window_halfwidth: int = 3

    # peak validation (is_peak)
    peak_method: str = "prominence"  # "prominence" (bilateral) | "flank" (unilateral)
    min_peak_width: int = 5
    min_prominence: float = 0.5
    baseline_pct: float = 10.0
    min_turn: float = 0.05  # flank: min fractional descent on both shoulders

    # smoothing-survival gate (rejects apices carried by high-frequency noise;
    # complements min_prominence, which can't tell a coherent band from a spiky
    # bump on a persistent-background plateau). Off by default; DIA enables it.
    require_smoothing_survival: bool = False
    smoothing_sigma: float = 1.0
    min_smoothing_survival: float = 0.5

    # persistent-lane rejection (contaminant/background suppression). A mass lane
    # carrying signal in more than this fraction of the run's scans is treated as
    # persistent background and never seeds an ensemble — a real analyte elutes in
    # a narrow RT window, a contaminant (e.g. 279 from formic acid) is everywhere.
    # None disables. "Present" = intensity at/above cofeature_threshold (an
    # absolute floor; see _compute_persistent_lanes for why not a relative one).
    max_lane_persistence: Optional[float] = None

    # MS1 recruitment
    method: CofeatureMetric = DEFAULT_COFEATURE_METRIC
    use_rel_intsy: bool = True
    ms1_corr_threshold: float = 0.95
    adduct_aware: bool = False
    loose_corr_threshold: float = 0.85
    adduct_ppm_tol: float = 20.0
    polarity: int = 1
    require_apex_in_window: bool = False

    # MS2 attachment
    ms2_strategy: MS2Strategy = "none"
    ms2_corr_threshold: float = 0.9
    min_iso_halfwidth: float = 0.5
    min_ms2_scans: int = 2

    # scope
    rt_range: Optional[tuple[float, float]] = None


@dataclass
class Seed:
    """
    One candidate anchor for an ensemble: a mass lane + chromatographic apex.

    ms2_scan_idxs is only populated for the "precursor" seed strategy (DDA),
    where the seed is an isolated precursor that produced MS2 scans.
    """
    lane: int
    apex: int
    mz: float
    apex_intsy: float
    ms2_scan_idxs: Optional[np.ndarray] = None


@dataclass
class CandidateSet:
    """
    Vectorized bundle of recruitment candidates for one seed window.

    lane_idxs / mzs are always present. apex_scans and ms2_scan_idxs / pf_indices
    are only used by the precursor (DDA) provider.
    """
    lane_idxs: np.ndarray
    mzs: np.ndarray
    apex_scans: Optional[np.ndarray] = None
    pf_indices: Optional[np.ndarray] = None
    ms2_scan_idxs: Optional[list[np.ndarray]] = field(default=None, repr=False)

    @property
    def size(self) -> int:
        return int(self.lane_idxs.size)


# ---------------------------------------------------------------------------
# Preset builders
# ---------------------------------------------------------------------------


def dia_config(
    parent_threshold: float,
    cofeature_threshold: float,
    **overrides,
) -> ExtractionConfig:
    """
    DIA / auto preset: seed from all MS1 lanes, non-fixed (peak-bounds) window,
    cosine recruitment with adduct-aware loosening, MS2 by correlation.
    """
    base: dict = dict(
        seed_strategy="all_lanes",
        parent_threshold=parent_threshold,
        cofeature_threshold=cofeature_threshold,
        window_strategy="peak_bounds",
        method="cosine",
        ms1_corr_threshold=0.9,
        adduct_aware=True,
        loose_corr_threshold=0.8,
        require_apex_in_window=False,
        ms2_strategy="correlation",
        ms2_corr_threshold=0.8,
        require_smoothing_survival=True,
        max_lane_persistence=0.5,
    )
    base.update(overrides)
    return ExtractionConfig(**base)


def dda_config(
    min_intsy: float,
    **overrides,
) -> ExtractionConfig:
    """
    DDA preset: seed from precursor features, peak-bounds window, cosine
    recruitment with adduct-aware loosening (tight 0.97 / loose 0.85), MS2 by
    precursor-scan union. Mirrors the original ``group_precursor_ensembles``.
    """
    base: dict = dict(
        seed_strategy="precursor",
        parent_threshold=min_intsy,
        cofeature_threshold=min_intsy,
        window_strategy="peak_bounds",
        method="cosine",
        ms1_corr_threshold=0.97,
        adduct_aware=True,
        loose_corr_threshold=0.85,
        require_apex_in_window=True,
        ms2_strategy="precursor",
        min_ms2_scans=2,
    )
    base.update(overrides)
    return ExtractionConfig(**base)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def extract_ensembles(
    injection: 'Injection',
    config: ExtractionConfig,
    progress_callback=None,  # injected by ProcessRunner; unused here
    cancel_event=None,       # injected by ProcessRunner; unused here
) -> list[Ensemble]:
    """
    Run the greedy seed-and-recruit engine over an Injection's MS1 ScanArray and
    return (and attach) the extracted Ensembles.
    """
    ms1 = injection.scan_array_ms1
    if ms1 is None:
        raise ValueError("Injection has no MS1 scan array")

    provider = _make_seed_provider(injection, config)
    n_scans = ms1.intsy_arr.shape[1]

    # For "precursor_by_lane" MS2 attachment the seed provider (all-lanes) knows
    # nothing about precursors, so build the lane->precursor index once here. Empty
    # (never raises) when the injection lacks MS2/precursor metadata, so this mode
    # still emits MS1-only ensembles on such data.
    precursor_index = (
        _build_precursor_index(injection, config)
        if config.ms2_strategy == "precursor_by_lane"
        else {}
    )

    ensembles: list[Ensemble] = []
    while True:
        if cancel_event is not None and cancel_event.is_set():
            break

        seed = provider.next_seed()
        if seed is None:
            break

        chrom = provider.seed_chrom(seed)
        seg_start, seg_end = find_peak_boundaries(
            chrom, seed.apex, edge_fraction=config.edge_fraction
        )

        # The seed must be a genuine chromatographic peak (not a slope or a
        # smeared/constant background) to anchor a group at all. Prominence is
        # judged against the RAW lane so that regions already zeroed by earlier
        # consumption don't read as a false 0 baseline (which would make every
        # leftover sliver of a persistent contaminant look infinitely prominent).
        # The raw baseline is only needed by the "prominence" method; skip the
        # (sparse -> dense) fetch for "flank", which is neighbour-invariant.
        baseline = (
            provider.raw_seed_chrom(seed)
            if config.peak_method == "prominence"
            else None
        )
        if not is_peak(
            chrom, seed.apex, seg_start, seg_end,
            min_peak_width=config.min_peak_width,
            min_prominence=config.min_prominence,
            baseline_pct=config.baseline_pct,
            baseline_intensity=baseline,
            method=config.peak_method,
            min_turn=config.min_turn,
        ):
            provider.consume_rejected(seed, seg_start, seg_end)
            continue

        # Second gate: the apex's rise must be made of coherent (multi-scan)
        # signal, not a spike/jitter that a narrow smooth erases. Judged against
        # the same raw baseline as is_peak so consumed neighbours don't skew it.
        if config.require_smoothing_survival and not survives_smoothing(
            chrom, seed.apex, seg_start, seg_end,
            sigma=config.smoothing_sigma,
            min_survival=config.min_smoothing_survival,
            baseline_pct=config.baseline_pct,
            baseline_intensity=baseline,
        ):
            provider.consume_rejected(seed, seg_start, seg_end)
            continue

        # Two windows: score over the tight peak (discriminates shape), but emit
        # FeaturePointers over a padded window so narrow seeds stay non-degenerate.
        # For "fixed" the two coincide.
        score_idxs = _score_window_idxs(
            config, seg_start, seg_end, seed.apex, n_scans
        )
        display_idxs = _display_window_idxs(
            config, seg_start, seg_end, seed.apex, n_scans
        )

        candidates = provider.candidates_in_window(
            seed, seg_start, seg_end, score_idxs, config
        )
        matched_positions = _recruit_ms1(provider, seed, score_idxs, candidates, config)

        ms1_cofeatures = _build_ms1_cofeatures(
            ms1, seed, candidates, matched_positions, display_idxs
        )
        ms2_cofeatures, ms2_scan_count = _attach_ms2(
            injection, config, seed, candidates, matched_positions, display_idxs,
            ms1_cofeatures=ms1_cofeatures,
            seg_start=seg_start, seg_end=seg_end,
            precursor_index=precursor_index,
        )

        # DDA emit gate: a lone precursor fragmented once is a true singleton.
        if config.ms2_strategy == "precursor" and ms2_scan_count < config.min_ms2_scans:
            provider.consume_rejected(seed, seg_start, seg_end)
            continue

        ensemble = Ensemble(
            ms1_cofeatures=ms1_cofeatures,
            ms2_cofeatures=ms2_cofeatures,
        )
        injection.add_ensemble(ensemble)
        ensembles.append(ensemble)

        provider.consume_emitted(
            seed, seg_start, seg_end, candidates, matched_positions, ms1_cofeatures
        )

    return ensembles


def _score_window_idxs(
    config: ExtractionConfig,
    seg_start: int,
    seg_end: int,
    apex: int,
    n_scans: int,
) -> np.ndarray:
    """
    Window used for peak-shape scoring (and, for DIA, nonzero-lane detection).

    "fixed": +-extraction_half_width around the apex, clamped to the peak.
    "peak_bounds": exactly the peak boundaries — the tight peak discriminates
        shape best; padding it would only dilute the score with baseline scans.
    """
    if config.window_strategy == "fixed":
        hw = config.extraction_half_width
        return np.arange(max(seg_start, apex - hw), min(seg_end, apex + hw + 1))
    return np.arange(seg_start, seg_end)


def _display_window_idxs(
    config: ExtractionConfig,
    seg_start: int,
    seg_end: int,
    apex: int,
    n_scans: int,
) -> np.ndarray:
    """
    Window the emitted FeaturePointers span.

    "fixed": identical to the score window.
    "peak_bounds": the peak boundaries padded to at least min_window_halfwidth
        scans on each side of the apex, so narrow/spike seeds still yield a
        non-degenerate trace.
    """
    if config.window_strategy == "fixed":
        hw = config.extraction_half_width
        return np.arange(max(seg_start, apex - hw), min(seg_end, apex + hw + 1))
    mh = config.min_window_halfwidth
    lo = max(0, min(seg_start, apex - mh))
    hi = min(n_scans, max(seg_end, apex + mh + 1))
    return np.arange(lo, hi)


def _recruit_ms1(
    provider: 'SeedProvider',
    seed: Seed,
    window_idxs: np.ndarray,
    candidates: CandidateSet,
    config: ExtractionConfig,
) -> np.ndarray:
    """
    Score each candidate's XIC against the seed XIC over the window and return
    the positions (indices into ``candidates`` arrays) that pass threshold.

    Threshold is the tight ``ms1_corr_threshold`` by default; when
    ``adduct_aware`` and the seed<->candidate delta-m/z matches a known
    relationship, it loosens to ``loose_corr_threshold``.

    XICs come from ``provider.window_intensity``, which reflects consumption:
    signal already claimed by an earlier ensemble reads as zero, so a claimed
    lane scores ~0 here and cannot be recruited into a second ensemble.
    """
    if candidates.size == 0:
        return np.empty(0, dtype=int)

    # Match find_cofeatures' windowing: FeaturePointer.scan_end is the LAST scan
    # index (inclusive), and XIC slices are [scan_start:scan_end], so the final
    # scan of the window is excluded. Replicate that here (scan_end == last idx)
    # for both the seed and candidate XICs so they stay aligned (and DDA stays
    # byte-identical to the FeaturePointer-based path).
    scan_start = int(window_idxs[0])
    scan_end = int(window_idxs[-1])  # exclusive of the final window scan
    if scan_end <= scan_start:
        return np.empty(0, dtype=int)

    # A candidate whose only signal sits on a truncated window edge (or was fully
    # consumed) can normalize to an all-zero row (0/0 -> NaN). NaN scores are
    # dropped below, so silence the expected divide warning.
    with np.errstate(invalid="ignore", divide="ignore"):
        cand_xics = provider.window_intensity(
            candidates.lane_idxs, scan_start, scan_end
        )
        search_xic = provider.window_intensity(
            np.array([seed.lane]), scan_start, scan_end
        )[0]
        if config.use_rel_intsy:
            cand_xics = cand_xics / np.max(cand_xics, axis=1).reshape(-1, 1)
            peak = search_xic.max()
            if peak > 0:
                search_xic = search_xic / peak

        scores = _calculate_cofeature_scores(
            cand_xics, search_xic, method=config.method
        )

    matched: list[int] = []
    for j in range(candidates.size):
        score = scores[j]
        if np.isnan(score):
            continue
        threshold = config.ms1_corr_threshold
        if config.adduct_aware:
            label = relationship_label(
                seed.mz, float(candidates.mzs[j]),
                config.adduct_ppm_tol, config.polarity,
            )
            if label:
                threshold = config.loose_corr_threshold
        if score >= threshold:
            matched.append(j)

    return np.array(matched, dtype=int)


def _build_ms1_cofeatures(
    ms1: 'ScanArray',
    seed: Seed,
    candidates: CandidateSet,
    matched_positions: np.ndarray,
    window_idxs: np.ndarray,
) -> list['FeaturePointer']:
    """
    One MS1 FeaturePointer per distinct mass lane (seed first), each spanning the
    window. Deduplicated by lane so an over-split elution (or a same-lane DDA
    precursor on another peak) collapses to a single trace.
    """
    cofeatures: list['FeaturePointer'] = [
        ms1.make_feature_pointer(seed.lane, window_idxs.copy())
    ]
    seen: set[int] = {seed.lane}
    for pos in matched_positions:
        lane = int(candidates.lane_idxs[pos])
        if lane in seen:
            continue
        seen.add(lane)
        cofeatures.append(ms1.make_feature_pointer(lane, window_idxs.copy()))
    return cofeatures


def _attach_ms2(
    injection: 'Injection',
    config: ExtractionConfig,
    seed: Seed,
    candidates: CandidateSet,
    matched_positions: np.ndarray,
    window_idxs: np.ndarray,
    ms1_cofeatures: Optional[list['FeaturePointer']] = None,
    seg_start: Optional[int] = None,
    seg_end: Optional[int] = None,
    precursor_index: Optional[dict[int, list[tuple[int, np.ndarray]]]] = None,
) -> tuple[list['FeaturePointer'], int]:
    """
    Build MS2 cofeatures for an emitted ensemble.

    Returns (cofeatures, ms2_scan_count). The count is only meaningful for the
    "precursor" strategy (used for the DDA emit gate); other strategies return 0.

    ``ms1_cofeatures`` / ``seg_start`` / ``seg_end`` / ``precursor_index`` are only
    consulted by the "precursor_by_lane" strategy (ignored otherwise).
    """
    ms2 = injection.scan_array_ms2
    if config.ms2_strategy == "none" or ms2 is None:
        return [], 0

    if config.ms2_strategy == "correlation":
        seed_ftr = injection.scan_array_ms1.make_feature_pointer(
            seed.lane, window_idxs.copy()
        )
        cofeatures = find_cofeatures_across_scan_array(
            source_scan_array=injection.scan_array_ms1,
            target_scan_array=ms2,
            search_target=seed_ftr,
            min_correlation=config.ms2_corr_threshold,
            min_intsy=config.cofeature_threshold,
            use_rel_intsy=config.use_rel_intsy,
            method=config.method,
        )
        return cofeatures, len(cofeatures)

    if config.ms2_strategy == "precursor":
        scan_lists: list[np.ndarray] = []
        if seed.ms2_scan_idxs is not None:
            scan_lists.append(seed.ms2_scan_idxs)
        if candidates.ms2_scan_idxs is not None:
            for pos in matched_positions:
                idxs = candidates.ms2_scan_idxs[pos]
                if idxs is not None and idxs.size:
                    scan_lists.append(idxs)
        if not scan_lists:
            return [], 0
        ms2_union = np.unique(np.concatenate(scan_lists))
        if ms2_union.size < config.min_ms2_scans:
            return [], int(ms2_union.size)
        return _make_ms2_cofeatures(ms2, ms2_union), int(ms2_union.size)

    if config.ms2_strategy == "precursor_by_lane":
        # Lane-indexed variant of "precursor": the ensemble was grouped by MS1 peak
        # shape (all-lanes seeding), so map its member lanes back to any precursor
        # feature that fired on them within this peak, and union the real triggered
        # MS2 scans. No emit gate — an ensemble with no fragmented member is a valid
        # MS1-only group (the whole point of this mode).
        if precursor_index is None or ms1_cofeatures is None:
            return [], 0
        lanes = {ftr.mz_lane_idx for ftr in ms1_cofeatures}
        scan_lists = [
            idxs
            for lane in lanes
            for (apex, idxs) in precursor_index.get(lane, [])
            if seg_start <= apex < seg_end and idxs.size
        ]
        if not scan_lists:
            return [], 0
        ms2_union = np.unique(np.concatenate(scan_lists))
        return _make_ms2_cofeatures(ms2, ms2_union), int(ms2_union.size)

    return [], 0


def _make_ms2_cofeatures(
    ms2: 'ScanArray',
    ms2_idxs: np.ndarray,
) -> list['FeaturePointer']:
    """
    One FeaturePointer per MS2 mass lane carrying signal in the grouped scans.

    NOT intensity-filtered: in DDA every peak in a precursor's MS2 scans is a
    real fragment of that precursor (MS2 is never noise-filtered at build time),
    so filtering here would silently drop low-intensity fragment peaks.
    """
    if ms2 is None or ms2_idxs.size == 0:
        return []
    lane_max = ms2.intsy_arr[:, ms2_idxs].max(axis=1).toarray().flatten()
    active = np.where(lane_max > 0)[0]
    return [
        ms2.make_feature_pointer(int(lane), ms2_idxs.copy())
        for lane in active
    ]


# ---------------------------------------------------------------------------
# Seed providers
# ---------------------------------------------------------------------------


def _make_seed_provider(
    injection: 'Injection',
    config: ExtractionConfig,
) -> 'SeedProvider':
    if config.seed_strategy == "all_lanes":
        return LaneMaxSeedProvider(injection, config)
    if config.seed_strategy == "precursor":
        return PrecursorSeedProvider(injection, config)
    raise ValueError(f"Unknown seed strategy {config.seed_strategy!r}")


class SeedProvider:
    """
    Interface for the two acquisition-specific concerns the engine defers:
    where the next seed comes from, which candidates are eligible in a window,
    and how the search space is consumed after a seed is rejected or emitted.
    """

    def next_seed(self) -> Optional[Seed]:
        raise NotImplementedError

    def seed_chrom(self, seed: Seed) -> np.ndarray:
        """1D MS1 working chromatogram for the seed's lane, used to locate the
        apex and peak boundaries. May differ from the raw lane (DIA zeroes
        already-consumed regions before finding boundaries)."""
        raise NotImplementedError

    def raw_seed_chrom(self, seed: Seed) -> np.ndarray:
        """1D MS1 *raw* chromatogram for the seed's lane (no consumption zeroing),
        used to judge peak prominence against the true surrounding signal. For
        providers that don't zero anything this equals ``seed_chrom``."""
        raise NotImplementedError

    def window_intensity(
        self,
        lane_idxs: np.ndarray,
        scan_start: int,
        scan_end: int,
    ) -> np.ndarray:
        """
        Dense (len(lane_idxs) x (scan_end - scan_start)) MS1 intensity for the
        given lanes over ``[scan_start:scan_end)``, used to build recruitment
        XICs. Providers that track consumption zero out already-claimed regions
        here, so a signal cannot be recruited into more than one ensemble.
        """
        raise NotImplementedError

    def candidates_in_window(
        self,
        seed: Seed,
        seg_start: int,
        seg_end: int,
        window_idxs: np.ndarray,
        config: ExtractionConfig,
    ) -> CandidateSet:
        raise NotImplementedError

    def consume_rejected(self, seed: Seed, seg_start: int, seg_end: int) -> None:
        raise NotImplementedError

    def consume_emitted(
        self,
        seed: Seed,
        seg_start: int,
        seg_end: int,
        candidates: CandidateSet,
        matched_positions: np.ndarray,
        ms1_cofeatures: list['FeaturePointer'],
    ) -> None:
        raise NotImplementedError


class LaneMaxSeedProvider(SeedProvider):
    """
    DIA / auto seeding: greedily pick the tallest signal across all MS1 lanes,
    tracking consumption by zeroing assigned scan ranges per lane. Candidates are
    every nonzero lane in the window (adduct-aware recruitment decides membership),
    minus lanes flagged as persistent background, which neither seed nor recruit.
    """

    def __init__(self, injection: 'Injection', config: ExtractionConfig):
        self.scan_array: 'ScanArray' = injection.scan_array_ms1
        self.config = config
        self.n_scans = self.scan_array.intsy_arr.shape[1]

        self.assigned_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self._seed_chrom_cache: dict[int, np.ndarray] = {}

        # Optional RT restriction: mask scans outside the range everywhere.
        self.rt_mask: Optional[np.ndarray] = None
        if config.rt_range is not None:
            rt_start, rt_end = config.rt_range
            self.rt_mask = (
                (self.scan_array.rt_arr < rt_start)
                | (self.scan_array.rt_arr > rt_end)
            )

        # Lanes flagged as persistent background never seed (see next_seed /
        # _recompute_lane_max, which both hold these at zero max).
        self.persistent_lane_mask = self._compute_persistent_lanes()

        self.lane_max_intsy = self._initial_lane_max()
        self.lane_max_intsy[self.persistent_lane_mask] = 0.0

    def _compute_persistent_lanes(self) -> np.ndarray:
        """
        Boolean mask (per MS1 lane) of lanes that carry signal across too much of
        the run to be a chromatographic peak — persistent background / contaminant
        ions. A lane is persistent when it is "present" in more than
        ``max_lane_persistence`` of all scans. ``None`` disables (all-False).

        "Present" is judged against an ABSOLUTE floor — ``cofeature_threshold``,
        the same intensity that defines "real signal" everywhere else — not a
        fraction of the lane's own max. That distinction is the whole ballgame: a
        contaminant streak almost always carries the odd tall spike (a real
        co-eluting ion, or a noise burst), which inflates a self-relative floor
        above the streak's own baseline and hides the very persistence we want to
        catch. An absolute floor sees the streak for what it is.

        Computed once over the raw MS1 matrix, vectorized over the sparse entries.
        """
        frac = self.config.max_lane_persistence
        intsy = self.scan_array.intsy_arr
        n_lanes, n_scans = intsy.shape
        if frac is None or n_scans == 0:
            return np.zeros(n_lanes, dtype=bool)

        # Count, per lane, the stored entries clearing the absolute floor.
        # bincount over (lane-of-each-entry, filtered by the floor) is O(nnz).
        indptr = intsy.indptr
        lane_of_entry = np.repeat(np.arange(n_lanes), np.diff(indptr))
        present = intsy.data >= self.config.cofeature_threshold
        present_counts = np.bincount(
            lane_of_entry[present], minlength=n_lanes
        )
        return (present_counts / n_scans) > frac

    def _initial_lane_max(self) -> np.ndarray:
        if self.rt_mask is None:
            return self.scan_array.intsy_arr.max(axis=1).toarray().flatten()
        allowed = ~self.rt_mask
        if not allowed.any():
            return np.zeros(self.scan_array.intsy_arr.shape[0])
        allowed_idxs = np.where(allowed)[0]
        return (
            self.scan_array.intsy_arr[:, allowed_idxs[0]:allowed_idxs[-1] + 1]
            .max(axis=1)
            .toarray()
            .flatten()
        )

    def _lane_chrom(self, lane: int) -> np.ndarray:
        chrom = self.scan_array.intsy_arr[lane].toarray().flatten()
        if self.rt_mask is not None:
            chrom[self.rt_mask] = 0
        for start, end in self.assigned_ranges.get(lane, []):
            chrom[start:end] = 0
        return chrom

    def next_seed(self) -> Optional[Seed]:
        while True:
            lane = int(np.argmax(self.lane_max_intsy))
            if self.lane_max_intsy[lane] < self.config.parent_threshold:
                return None

            chrom = self._lane_chrom(lane)
            apex = int(np.argmax(chrom))
            if chrom[apex] < self.config.parent_threshold:
                # Stale tracker entry; retire this lane and keep looking.
                self.lane_max_intsy[lane] = 0
                continue

            self._seed_chrom_cache[lane] = chrom
            return Seed(
                lane=lane,
                apex=apex,
                mz=float(self.scan_array.mz_lane_label[lane]),
                apex_intsy=float(chrom[apex]),
            )

    def seed_chrom(self, seed: Seed) -> np.ndarray:
        chrom = self._seed_chrom_cache.get(seed.lane)
        if chrom is None:
            chrom = self._lane_chrom(seed.lane)
        return chrom

    def raw_seed_chrom(self, seed: Seed) -> np.ndarray:
        # Raw lane signal (no consumption zeroing) for a truthful prominence
        # baseline. RT masking is deliberately NOT applied — the chemical
        # background outside the RT window is still real baseline.
        return self.scan_array.intsy_arr[seed.lane].toarray().flatten()

    def window_intensity(
        self,
        lane_idxs: np.ndarray,
        scan_start: int,
        scan_end: int,
    ) -> np.ndarray:
        # Raw slice, with regions already claimed by earlier ensembles zeroed so
        # they can't be recruited again. This is what enforces "one signal
        # belongs to at most one ensemble" at recruitment time.
        grid = self.scan_array.intsy_arr[
            lane_idxs, scan_start:scan_end
        ].toarray()
        for i, lane in enumerate(lane_idxs):
            ranges = self.assigned_ranges.get(int(lane))
            if not ranges:
                continue
            for a_start, a_end in ranges:
                lo = max(a_start, scan_start)
                hi = min(a_end, scan_end)
                if hi > lo:
                    grid[i, lo - scan_start:hi - scan_start] = 0
        return grid

    def candidates_in_window(
        self, seed, seg_start, seg_end, window_idxs, config,
    ) -> CandidateSet:
        lane_idxs = _find_nonzero_mass_lanes(
            scan_array=self.scan_array,
            scan_idxs=window_idxs,
            min_intsy=config.cofeature_threshold,
        )
        # Seed lane is added to the ensemble separately; exclude it as a
        # candidate so it can't self-match at score 1.0.
        lane_idxs = lane_idxs[lane_idxs != seed.lane]
        # Persistent-background lanes are barred from recruitment too: a
        # contaminant plateau can incidentally correlate with a real peak over a
        # short window and get pulled in as a spurious cofeature.
        if self.persistent_lane_mask.any():
            lane_idxs = lane_idxs[~self.persistent_lane_mask[lane_idxs]]
        if lane_idxs.size == 0:
            return CandidateSet(lane_idxs=lane_idxs, mzs=np.empty(0))
        mzs = np.asarray(self.scan_array.mz_lane_label[lane_idxs])
        return CandidateSet(lane_idxs=lane_idxs, mzs=mzs)

    def consume_rejected(self, seed, seg_start, seg_end) -> None:
        self._zero_region(seed.lane, seg_start, seg_end)

    def consume_emitted(
        self, seed, seg_start, seg_end, candidates, matched_positions, ms1_cofeatures,
    ) -> None:
        # Consume the seed's full peak, then claim each cofeature's FULL local
        # peak (not just the extraction window). A signal broader than the
        # window would otherwise leave tails that a later ensemble could re-seed
        # or re-recruit at an offset — enforcing one-peak-one-ensemble.
        self._zero_region(seed.lane, seg_start, seg_end)
        affected: set[int] = set()
        for ftr in ms1_cofeatures:
            lane = ftr.mz_lane_idx
            if lane == seed.lane:
                continue
            c_start, c_end = self._cofeature_peak_bounds(
                lane, ftr.scan_start, ftr.scan_end
            )
            self.assigned_ranges[lane].append((c_start, c_end))
            affected.add(lane)
        self._recompute_lane_max(affected)

    def _cofeature_peak_bounds(
        self, lane: int, w_start: int, w_end: int,
    ) -> tuple[int, int]:
        """
        Full peak boundaries of a cofeature on its own lane, spanning outward
        from its apex within the extraction window ``[w_start, w_end]``.

        Measured on the *working* (already-consumed) chromatogram so the claimed
        span naturally stops at regions earlier ensembles already hold, and never
        exceeds this lane's own peak (find_peak_boundaries cuts at valleys). Falls
        back to the window itself if the lane has no signal there.
        """
        chrom = self._lane_chrom(lane)
        lo = max(0, w_start)
        hi = min(chrom.size, w_end + 1)  # w_end is an inclusive scan index
        if hi <= lo or chrom[lo:hi].max() <= 0:
            return (w_start, w_end)
        apex = lo + int(np.argmax(chrom[lo:hi]))
        apex = _climb_to_apex(chrom, apex)
        return find_peak_boundaries(
            chrom, apex, edge_fraction=self.config.edge_fraction
        )

    def _zero_region(self, lane: int, seg_start: int, seg_end: int) -> None:
        self.assigned_ranges[lane].append((seg_start, seg_end))
        self._recompute_lane_max({lane})

    def _recompute_lane_max(self, lanes: set[int]) -> None:
        for lane in lanes:
            self._seed_chrom_cache.pop(lane, None)
            # A persistent-background lane must never become seedable, even after
            # a cofeature on it is consumed and its max is recomputed.
            if self.persistent_lane_mask[lane]:
                self.lane_max_intsy[lane] = 0.0
                continue
            chrom = self._lane_chrom(lane)
            self.lane_max_intsy[lane] = chrom.max()


class PrecursorSeedProvider(SeedProvider):
    """
    DDA seeding: seeds are MS2-triggering precursor features (one isolated MS1
    ion + the MS2 scans it produced), taken tallest-first with hard assignment.
    Candidates are other unassigned precursor features whose apex falls in the
    seed window; MS2 is the union of the group's precursor scans.
    """

    def __init__(self, injection: 'Injection', config: ExtractionConfig):
        self.injection = injection
        self.ms1: 'ScanArray' = injection.scan_array_ms1
        self.ms2: 'ScanArray' = injection.scan_array_ms2
        if self.ms2 is None or self.ms2.precursor_mz_arr is None:
            raise ValueError(
                "Precursor seeding needs MS2 data with precursor information"
            )

        self.pfs: list[PrecursorFeature] = _build_precursor_features(
            injection, config
        )
        self.apex_scans = np.array(
            [pf.apex_scan for pf in self.pfs], dtype=int
        ) if self.pfs else np.empty(0, dtype=int)
        self.assigned = np.zeros(len(self.pfs), dtype=bool)
        self._order = sorted(
            range(len(self.pfs)),
            key=lambda k: self.pfs[k].apex_intsy,
            reverse=True,
        )
        self._cursor = 0

    def next_seed(self) -> Optional[Seed]:
        while self._cursor < len(self._order):
            si = self._order[self._cursor]
            self._cursor += 1
            if self.assigned[si]:
                continue
            self.assigned[si] = True
            pf = self.pfs[si]
            return Seed(
                lane=pf.mz_lane_idx,
                apex=pf.apex_scan,
                mz=pf.mz,
                apex_intsy=pf.apex_intsy,
                ms2_scan_idxs=pf.ms2_scan_idxs,
            )
        return None

    def seed_chrom(self, seed: Seed) -> np.ndarray:
        return self.ms1.intsy_arr[seed.lane].toarray().flatten()

    def raw_seed_chrom(self, seed: Seed) -> np.ndarray:
        # Precursor seeding never zeroes lanes, so raw == working here.
        return self.ms1.intsy_arr[seed.lane].toarray().flatten()

    def window_intensity(
        self,
        lane_idxs: np.ndarray,
        scan_start: int,
        scan_end: int,
    ) -> np.ndarray:
        # Precursor grouping assigns whole precursor features (via `assigned`),
        # not lane regions, and candidates are already assignment-gated, so
        # recruitment reads the raw signal.
        return self.ms1.intsy_arr[lane_idxs, scan_start:scan_end].toarray()

    def candidates_in_window(
        self, seed, seg_start, seg_end, window_idxs, config,
    ) -> CandidateSet:
        mask = (
            (~self.assigned)
            & (self.apex_scans >= seg_start)
            & (self.apex_scans < seg_end)
        )
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return CandidateSet(
                lane_idxs=np.empty(0, dtype=int),
                mzs=np.empty(0),
                apex_scans=np.empty(0, dtype=int),
                pf_indices=np.empty(0, dtype=int),
                ms2_scan_idxs=[],
            )
        return CandidateSet(
            lane_idxs=np.array([self.pfs[k].mz_lane_idx for k in idxs], dtype=int),
            mzs=np.array([self.pfs[k].mz for k in idxs]),
            apex_scans=self.apex_scans[idxs],
            pf_indices=idxs,
            ms2_scan_idxs=[self.pfs[k].ms2_scan_idxs for k in idxs],
        )

    def consume_rejected(self, seed, seg_start, seg_end) -> None:
        # The seed pf was already marked assigned in next_seed().
        pass

    def consume_emitted(
        self, seed, seg_start, seg_end, candidates, matched_positions, ms1_cofeatures,
    ) -> None:
        if candidates.pf_indices is None:
            return
        for pos in matched_positions:
            self.assigned[int(candidates.pf_indices[pos])] = True


# ---------------------------------------------------------------------------
# Precursor-feature construction (DDA stage 1)
# ---------------------------------------------------------------------------


@dataclass
class PrecursorFeature:
    """
    One MS1 ion isolated for fragmentation, with the MS2 scans it produced.

    mz_lane_idx: row in the MS1 ScanArray.
    apex_scan: column of this feature's MS1 chromatographic apex.
    mz: representative m/z (MS1 lane label).
    apex_intsy: MS1 intensity at the apex (seed sort key).
    ms2_scan_idxs: column indices into the MS2 ScanArray for this precursor.
    """
    mz_lane_idx: int
    apex_scan: int
    mz: float
    apex_intsy: float
    ms2_scan_idxs: np.ndarray = field(repr=False)


def _build_precursor_index(
    injection: 'Injection',
    config: ExtractionConfig,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """
    Map each MS1 lane to the precursor features that fired on it, as
    ``lane -> [(apex_scan, ms2_scan_idxs), ...]``, for lane-based MS2 attachment
    ("precursor_by_lane").

    Returns an empty dict — never raises — when the injection carries no MS2 with
    precursor information, so DIA-style grouping over such data still yields
    MS1-only ensembles.
    """
    ms2 = injection.scan_array_ms2
    if ms2 is None or ms2.precursor_mz_arr is None:
        return {}
    index: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for pf in _build_precursor_features(injection, config):
        index[pf.mz_lane_idx].append((pf.apex_scan, pf.ms2_scan_idxs))
    return index


def _build_precursor_features(
    injection: 'Injection',
    config: ExtractionConfig,
) -> list[PrecursorFeature]:
    """
    Map each MS2 scan to an MS1 precursor lane + peak, then split each lane's
    triggering columns into per-peak precursor features.
    """
    ms1 = injection.scan_array_ms1
    ms2 = injection.scan_array_ms2

    # lane -> {ms1_col -> [ms2 scan idxs]}
    lane_to_cols: dict[int, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_ms2 = ms2.rt_arr.size
    for i in range(n_ms2):
        window = _isolation_window(ms2, i, config.min_iso_halfwidth)
        if window is None:
            continue
        eff_lo, eff_hi = window
        ms1_col = _ms1_col_for_ms2(injection, i)
        lane = _tallest_lane_in_window(
            ms1, ms1_col, eff_lo, eff_hi, config.cofeature_threshold
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
    Effective [lo, hi] m/z window to search for the precursor, at least
    ``min_halfwidth`` on each side. Falls back to the recorded precursor m/z as
    the center when the encoded window is missing/degenerate; None if neither is
    available.
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
    Column into the MS1 ScanArray to inspect for an MS2 scan: the triggering MS1
    scan number if recorded, else the nearest MS1 scan by retention time.
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
    Row (mass lane) of the tallest MS1 signal inside [lo, hi] at ``ms1_col``, or
    None if the window is empty.
    """
    mzs = ms1.mz_arr_csc._getcol(ms1_col).toarray().flatten()
    intsys = ms1.intsy_arr_csc._getcol(ms1_col).toarray().flatten()
    mask = (mzs >= lo) & (mzs <= hi) & (intsys > min_intsy)
    rows = np.where(mask)[0]
    if rows.size == 0:
        return None
    return int(rows[np.argmax(intsys[rows])])


def _climb_to_apex(chrom: np.ndarray, col: int) -> int:
    """Climb monotonically uphill from ``col`` to the local apex."""
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
    Split one mass lane's MS2-triggering columns into per-peak precursor
    features: greedily anchor on the tallest unassigned column, climb to its
    apex, take the peak boundaries, and absorb all triggering columns within.
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
