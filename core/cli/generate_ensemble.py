"""
Script for extracting a co-feature ensemble
from a pair of MS1 and MS2 scan arrays given a search
feature pointer (from the MS1 array).

These must be set to an Injection to be viable
"""
from typing import NamedTuple, TYPE_CHECKING

import numpy as np

from core.data_structs import Ensemble
from core.cli.find_cofeatures import (
    find_cofeatures_within_scan_array,
    find_cofeatures_across_scan_array,
    get_all_features_in_scan_array, # for testing
    CofeatureMetric,
    DEFAULT_COFEATURE_METRIC,
)
from core.cli.auto_extract_ensembles import (
    extract_ensembles,
    dia_config,
    dda_config,
    WindowStrategy,
)

if TYPE_CHECKING:
    from configparser import ConfigParser
    from core.data_structs import (
        FeaturePointer, ScanArray,
        Injection
    )


class EnsembleExtractionParams(NamedTuple):
    search_ftr_ptr: 'FeaturePointer'
    injection: 'Injection'
    ms1_corr_threshold: float
    ms2_corr_threshold: float
    min_intsy: float
    use_rel_intsy: bool
    # Only used when injection.acquisition_mode == 'dda'. Tuneable from
    # the settings menu post-ASMS; for now a wide default that comfortably
    # covers typical DDA isolation widths even when not explicitly encoded.
    precursor_mz_tolerance: float = 0.5
    # Peak-shape scoring metric for co-feature grouping ('cosine' | 'pearson').
    method: 'CofeatureMetric' = DEFAULT_COFEATURE_METRIC


def get_cofeature_ensembles(
    search_ftr_ptrs: list[ 'FeaturePointer' ],
    injection: 'Injection',
    ms1_corr_threshold: float,
    ms2_corr_threshold: float,
    min_intsy: float,
    use_rel_intsy: bool,
    precursor_mz_tolerance: float = 0.5,
    method: 'CofeatureMetric' = DEFAULT_COFEATURE_METRIC,
    progress_callback=None,  # injected by ProcessRunner; unused here
    cancel_event=None,       # injected by ProcessRunner; unused here
) -> list[ Ensemble ]:
    """
    Given a list of feature pointers, generates a
    list of Ensembles
    :param injection:
        Injection object to parse
    :param search_ftr_ptrs:
        FeaturePointers to use as references
    :param min_intsy:
        Minimum intensity that a signal must have to be considered for analysis
    :param ms1_corr_threshold:
        Pearson correlation threshold to be considered cofeature
    :param ms2_corr_threshold:
        Pearson correlation threshold to be considered cofeature
    :param use_rel_intsy:
        Whether to use absolute or relative intensities when calculating
        Pearson correlation
    :return:
    """

    ensembles: list[Ensemble] = []
    for search_ftr_ptr in search_ftr_ptrs:
        ensemble = get_cofeature_ensemble(
            injection=injection,
            min_intsy=min_intsy,
            ms1_corr_threshold=ms1_corr_threshold,
            ms2_corr_threshold=ms2_corr_threshold,
            search_ftr_ptr=search_ftr_ptr,
            use_rel_intsy=use_rel_intsy,
            precursor_mz_tolerance=precursor_mz_tolerance,
            method=method,
        )

        ensembles.append(
            ensemble
        )

    return ensembles


def get_cofeature_ensemble(
    injection: 'Injection',
    search_ftr_ptr: 'FeaturePointer',
    ms1_corr_threshold: float,
    ms2_corr_threshold: float,
    min_intsy: float,
    use_rel_intsy: bool,
    precursor_mz_tolerance: float = 0.5,
    method: 'CofeatureMetric' = DEFAULT_COFEATURE_METRIC,
) -> Ensemble:
    ms1_cofeatures = find_cofeatures_within_scan_array(
        scan_array=injection.scan_array_ms1,
        search_target=search_ftr_ptr,
        min_correlation=ms1_corr_threshold,
        min_intsy=min_intsy,
        use_rel_intsy=use_rel_intsy,
        method=method,
    )

    ms2_cofeatures: list['FeaturePointer'] = []
    precursor_mz: 'float | None' = None
    precursor_charge: 'int | None' = None

    if injection.acquisition_mode == 'dda':
        # DDA: link MS2 by precursor m/z + RT window, not by correlation.
        ms2_cofeatures, precursor_mz, precursor_charge = (
            _dda_link_ms2_cofeatures(
                injection=injection,
                ms1_cofeatures=ms1_cofeatures,
                search_ftr_ptr=search_ftr_ptr,
                min_intsy=min_intsy,
                precursor_mz_tolerance=precursor_mz_tolerance,
            )
        )
    elif injection.scan_array_ms2 is not None:
        # DIA / MS1_only-but-MS2-present: original correlation path.
        ms2_cofeatures = find_cofeatures_across_scan_array(
            source_scan_array=injection.scan_array_ms1,
            target_scan_array=injection.scan_array_ms2,
            search_target=search_ftr_ptr,
            min_correlation=ms2_corr_threshold,
            min_intsy=min_intsy,
            use_rel_intsy=use_rel_intsy,
            method=method,
        )

    ensemble = Ensemble(
        ms1_cofeatures=ms1_cofeatures,
        ms2_cofeatures=ms2_cofeatures,
        precursor_mz=precursor_mz,
        precursor_charge=precursor_charge,
    )

    injection.add_ensemble(ensemble)
    return ensemble


def _dda_link_ms2_cofeatures(
    injection: 'Injection',
    ms1_cofeatures: list['FeaturePointer'],
    search_ftr_ptr: 'FeaturePointer',
    min_intsy: float,
    precursor_mz_tolerance: float,
) -> tuple[list['FeaturePointer'], 'float | None', 'int | None']:
    """
    DDA MS2 linkage: take every MS2 scan whose precursor m/z matches any
    MS1 cofeature (within `precursor_mz_tolerance`) AND whose RT falls
    inside the search feature's RT window. The resulting MS2 cofeatures
    are FeaturePointers, one per MS2 mass lane that carries signal in
    those scans, with `scan_idxs` restricted to the matched MS2 scans.

    Returns the cofeatures, plus the precursor m/z (median of matched
    scans) and precursor charge (mode of matched scans) — both `None`
    if nothing matched.
    """
    ms1_arr = injection.scan_array_ms1
    ms2_arr = injection.scan_array_ms2
    if ms2_arr is None or ms2_arr.precursor_mz_arr is None:
        return [], None, None

    # RT window from the search feature.
    search_rts = search_ftr_ptr.get_retention_times(ms1_arr)
    rt_lo, rt_hi = float(search_rts.min()), float(search_rts.max())

    # m/z of every MS1 cofeature (lane-label mean is good enough for matching).
    cofeature_mzs = ms1_arr.mz_lane_label[
        [cf.mz_lane_idx for cf in ms1_cofeatures]
    ]

    # Mask MS2 scans by RT window.
    rt_mask = (ms2_arr.rt_arr >= rt_lo) & (ms2_arr.rt_arr <= rt_hi)
    if not rt_mask.any():
        return [], None, None

    # Mask MS2 scans by precursor-mz proximity to any cofeature m/z.
    # Outer-difference matrix is fine for the cardinalities involved
    # (cofeatures ~ tens, MS2 scans ~ thousands).
    diff = np.abs(
        ms2_arr.precursor_mz_arr[:, None]
        - np.asarray(cofeature_mzs)[None, :]
    )
    mz_mask = (diff < precursor_mz_tolerance).any(axis=1)

    matched_ms2_idxs = np.where(rt_mask & mz_mask)[0]
    if matched_ms2_idxs.size == 0:
        return [], None, None

    # Find MS2 mass lanes that carry signal in the matched scans. NOT
    # intensity-filtered: in DDA every peak in the precursor's MS2 scans is a
    # real fragment of that precursor (MS2 is never noise-filtered at build
    # time), so applying `min_intsy` here would silently drop low-intensity
    # fragment peaks from the ensemble's reconstructed spectrum.
    intsy_slice = ms2_arr.intsy_arr[:, matched_ms2_idxs]
    lane_max = intsy_slice.max(axis=1).toarray().flatten()
    active_lane_idxs = np.where(lane_max > 0)[0]

    ms2_cofeatures: list['FeaturePointer'] = [
        ms2_arr.make_feature_pointer(
            mass_lane_idx=int(lane_idx),
            scan_idxs=matched_ms2_idxs,
        )
        for lane_idx in active_lane_idxs
    ]

    # Precursor metadata: median m/z and modal charge across matched scans.
    precursor_mz = float(np.median(ms2_arr.precursor_mz_arr[matched_ms2_idxs]))
    charges = ms2_arr.precursor_charge_arr[matched_ms2_idxs]
    nonzero_charges = charges[charges != 0]
    if nonzero_charges.size:
        vals, counts = np.unique(nonzero_charges, return_counts=True)
        precursor_charge = int(vals[counts.argmax()])
    else:
        precursor_charge = None

    return ms2_cofeatures, precursor_mz, precursor_charge


def get_ungrouped_ensemble(
    injection: 'Injection',
    search_ftr_ptr: 'FeaturePointer',
    min_intsy: float,
) -> Ensemble:
    """
    For *testing*; retrieves an 'ensemble' that's just a window
    into an Injection's ScanArrays
    """
    rts = search_ftr_ptr.get_retention_times(
        injection.get_scan_array(ms_level=1)
    )

    rt_start, rt_end = rts[0], rts[-1]
    print(
        f"rt start: {rt_start}\n"
        f"rt end: {rt_end}"
    )

    ms1_cofeatures: list['FeaturePointer'] = []
    ms2_cofeatures: list['FeaturePointer'] = []
    for ms_level, featurelist in [
            (1, ms1_cofeatures),
            (2, ms2_cofeatures),
    ]:
        featurelist +=  get_all_features_in_scan_array(
            scan_array=injection.get_scan_array(ms_level),
            rt_start=rt_start,# type:ignore
            rt_end=rt_end,    # type:ignore
            min_intsy=min_intsy,
        )

    ensemble = Ensemble(
        ms1_cofeatures=ms1_cofeatures,
        ms2_cofeatures=ms2_cofeatures,
    )
    ensemble.set_injection(injection)

    return ensemble


class AutoEnsembleParams(NamedTuple):
    """
    Parameters for automated (DIA/auto) ensemble generation. Mapped onto the
    unified engine's ExtractionConfig via ``dia_config`` (or ``dda_config`` when
    ``injection.acquisition_mode == 'dda'``; DIA-only fields like
    ``ms2_corr_threshold`` are then unused).

    parent_threshold: Minimum peak height to seed a new ensemble.
        Only the tallest signal in an ensemble needs to exceed this.
    cofeature_threshold: Minimum intensity for a signal to be
        considered as a cofeature. Can be much lower than
        parent_threshold.
    ms1_corr_threshold: (tight) peak-shape threshold for MS1 cofeature grouping.
    ms2_corr_threshold: correlation threshold for MS2 cofeature grouping.
    use_rel_intsy: Whether to normalize chromatograms before scoring
        (i.e. if using Pearson).
    window_strategy: 'peak_bounds' (non-fixed, from the peak) or 'fixed'
        (+-extraction_half_width around the apex).
    extraction_half_width: Half-window in scans when window_strategy=='fixed'.
    min_window_halfwidth: Minimum apex padding (scans) for 'peak_bounds'.
    edge_fraction: For consumption boundaries — stop descending
        when intensity drops below this fraction of apex.
    min_prominence: Bilateral peak prominence (`is_peak`) — the apex
        must rise to at least this fraction of its height above the
        surrounding baseline on its weaker side. Rejects slopes and
        smeared/constant background signals. 0.5 => apex >= 2x baseline.
    min_peak_width: Peak must span at least this many scans to be valid.
    peak_method: is_peak shape test — "prominence" (bilateral; strict, best for
        isolated peaks) or "flank" (unilateral, neighbour-invariant; keeps peaks
        fused to a taller neighbour while still rejecting whine).
    baseline_pct: Percentile for the is_peak shoulder-baseline estimate; higher
        rejects bumps sitting on elevated background more readily. (prominence)
    min_turn: flank method — min fractional descent on both shoulders for the
        apex to count as a turned-over peak (rejects ramps).
    require_smoothing_survival: Second peak gate — after is_peak, require the
        apex's rise to survive a narrow Gaussian smooth, so spikes/plateau jitter
        that pass a loose min_prominence are dropped while coherent bands aren't.
    smoothing_sigma: Gaussian width (scans) for that gate; keep well below a real
        peak's width (~1).
    min_smoothing_survival: Fraction of the baseline-subtracted apex height that
        must remain after smoothing (0.5 => stays at least halfway to baseline).
    max_lane_persistence: Drop (never seed from OR recruit) any mass lane whose
        signal is at/above cofeature_threshold in more than this fraction of the
        run's scans — persistent background/contaminant ions (e.g. 279 from
        formic acid). None disables.
    adduct_aware: Loosen the MS1 threshold to loose_corr_threshold when the
        seed<->candidate delta-m/z matches a known adduct/isotope/loss.
    loose_corr_threshold, adduct_ppm_tol, polarity: adduct-loosening params.
    method: Peak-shape scoring metric ('cosine' | 'pearson').
    """
    parent_threshold: float
    cofeature_threshold: float
    ms1_corr_threshold: float
    ms2_corr_threshold: float
    use_rel_intsy: bool = True
    window_strategy: 'WindowStrategy' = "peak_bounds"
    extraction_half_width: int = 10
    min_window_halfwidth: int = 3
    edge_fraction: float = 0.1
    min_prominence: float = 0.5
    min_peak_width: int = 5
    peak_method: str = "prominence"
    baseline_pct: float = 10.0
    min_turn: float = 0.05
    require_smoothing_survival: bool = True
    smoothing_sigma: float = 1.0
    min_smoothing_survival: float = 0.5
    max_lane_persistence: float | None = 0.5
    adduct_aware: bool = True
    loose_corr_threshold: float = 0.8
    adduct_ppm_tol: float = 20.0
    polarity: int = 1
    rt_range: tuple[float, float] | None = None
    # Peak-shape scoring metric for co-feature grouping ('cosine' | 'pearson').
    method: 'CofeatureMetric' = DEFAULT_COFEATURE_METRIC


# Config section holding the persisted auto-generation defaults. Single source of
# truth for both the CLI (`mzkit auto-extract`) and the GUI settings menu.
AUTO_ENSEMBLE_SECTION = "auto_ensemble"


def auto_params_from_config(config: 'ConfigParser') -> AutoEnsembleParams:
    """
    Build an AutoEnsembleParams from the ``[auto_ensemble]`` config section,
    falling back to AutoEnsembleParams' own defaults for any missing key.

    The two composite params are decoded here:
      - ``max_lane_persistence`` is None unless ``reject_persistent_lanes`` is on.
      - ``rt_range`` is None unless ``rt_restrict`` is on and end > start; the
        stored bounds are in MINUTES and converted to the seconds the engine uses.
    """
    s = AUTO_ENSEMBLE_SECTION
    reject = config.getboolean(s, "reject_persistent_lanes", fallback=True)
    max_persist = (
        config.getfloat(s, "max_lane_persistence", fallback=0.5)
        if reject else None
    )

    rt_range = None
    if config.getboolean(s, "rt_restrict", fallback=False):
        rt0 = config.getfloat(s, "rt_start_min", fallback=0.0) * 60.0
        rt1 = config.getfloat(s, "rt_end_min", fallback=0.0) * 60.0
        if rt1 > rt0:
            rt_range = (rt0, rt1)

    return AutoEnsembleParams(
        parent_threshold=config.getfloat(s, "parent_threshold", fallback=10000.0),
        cofeature_threshold=config.getfloat(s, "cofeature_threshold", fallback=1000.0),
        ms1_corr_threshold=config.getfloat(s, "ms1_corr_threshold", fallback=0.9),
        ms2_corr_threshold=config.getfloat(s, "ms2_corr_threshold", fallback=0.8),
        use_rel_intsy=config.getboolean(s, "use_rel_intsy", fallback=True),
        window_strategy=config.get(s, "window_strategy", fallback="peak_bounds"),
        extraction_half_width=config.getint(s, "extraction_half_width", fallback=10),
        min_window_halfwidth=config.getint(s, "min_window_halfwidth", fallback=3),
        edge_fraction=config.getfloat(s, "edge_fraction", fallback=0.1),
        min_prominence=config.getfloat(s, "min_prominence", fallback=0.5),
        min_peak_width=config.getint(s, "min_peak_width", fallback=5),
        peak_method=config.get(s, "peak_method", fallback="prominence"),
        baseline_pct=config.getfloat(s, "baseline_pct", fallback=10.0),
        min_turn=config.getfloat(s, "min_turn", fallback=0.05),
        require_smoothing_survival=config.getboolean(
            s, "require_smoothing_survival", fallback=True),
        smoothing_sigma=config.getfloat(s, "smoothing_sigma", fallback=1.0),
        min_smoothing_survival=config.getfloat(
            s, "min_smoothing_survival", fallback=0.5),
        max_lane_persistence=max_persist,
        adduct_aware=config.getboolean(s, "adduct_aware", fallback=True),
        loose_corr_threshold=config.getfloat(s, "loose_corr_threshold", fallback=0.8),
        adduct_ppm_tol=config.getfloat(s, "adduct_ppm_tol", fallback=20.0),
        polarity=config.getint(s, "polarity", fallback=1),
        rt_range=rt_range,
        method=config.get(s, "method", fallback=DEFAULT_COFEATURE_METRIC),
    )


def auto_params_to_config(
    config: 'ConfigParser',
    params: AutoEnsembleParams,
) -> None:
    """
    Write an AutoEnsembleParams back into the ``[auto_ensemble]`` config section
    (in memory; caller persists via ``save_config``). Inverse of
    ``auto_params_from_config``: the None-able composites are re-encoded as their
    enable flag (+ value), and rt bounds are stored back in minutes. When a
    composite is None its stored value is left untouched, preserving the user's
    last-used bounds for the next time they re-enable it.
    """
    s = AUTO_ENSEMBLE_SECTION
    if not config.has_section(s):
        config.add_section(s)

    def setv(key, value):
        config.set(s, key, str(value))

    setv("method", params.method)
    setv("use_rel_intsy", params.use_rel_intsy)
    setv("ms1_corr_threshold", params.ms1_corr_threshold)
    setv("ms2_corr_threshold", params.ms2_corr_threshold)
    setv("parent_threshold", params.parent_threshold)
    setv("cofeature_threshold", params.cofeature_threshold)
    setv("window_strategy", params.window_strategy)
    setv("extraction_half_width", params.extraction_half_width)
    setv("min_window_halfwidth", params.min_window_halfwidth)
    setv("edge_fraction", params.edge_fraction)
    setv("peak_method", params.peak_method)
    setv("min_peak_width", params.min_peak_width)
    setv("min_prominence", params.min_prominence)
    setv("baseline_pct", params.baseline_pct)
    setv("min_turn", params.min_turn)
    setv("require_smoothing_survival", params.require_smoothing_survival)
    setv("smoothing_sigma", params.smoothing_sigma)
    setv("min_smoothing_survival", params.min_smoothing_survival)
    setv("reject_persistent_lanes", params.max_lane_persistence is not None)
    if params.max_lane_persistence is not None:
        setv("max_lane_persistence", params.max_lane_persistence)
    setv("adduct_aware", params.adduct_aware)
    setv("loose_corr_threshold", params.loose_corr_threshold)
    setv("adduct_ppm_tol", params.adduct_ppm_tol)
    setv("polarity", params.polarity)
    setv("rt_restrict", params.rt_range is not None)
    if params.rt_range is not None:
        setv("rt_start_min", params.rt_range[0] / 60.0)
        setv("rt_end_min", params.rt_range[1] / 60.0)


def auto_generate_ensembles(
    injection: 'Injection',
    params: AutoEnsembleParams,
    progress_callback=None,  # injected by ProcessRunner; unused here
    cancel_event=None,       # injected by ProcessRunner; unused here
) -> list[Ensemble]:
    """
    Automatically discover and extract all ensembles in an Injection's MS1
    ScanArray.

    For DIA (and any non-DDA) injections this is a thin preset over the unified
    engine (``dia_config``): seed from every MS1 lane tallest-first, recruit
    coeluting lanes by peak-shape similarity (optionally adduct-aware), and
    attach MS2 by correlation.

    For DDA injections, MS2 scans are sparse and interleaved across many
    precursors, so correlating an MS2 lane against a continuous MS1 XIC (the
    DIA model) does not hold. Instead this delegates to ``dda_config``: seed
    from MS2-triggering precursor features and attach MS2 as the union of the
    grouped precursors' actual triggering scans — the same construction the
    manual DDA single-click path uses, which the Ensemble Viewer's DDA overlay
    relies on.

    :param injection: Injection with assembled ScanArrays
    :param params: AutoEnsembleParams controlling thresholds
    :return: List of generated Ensembles
    """
    if injection.acquisition_mode == 'dda':
        config = dda_config(
            min_intsy=params.cofeature_threshold,
            parent_threshold=params.parent_threshold,
            cofeature_threshold=params.cofeature_threshold,
            edge_fraction=params.edge_fraction,
            min_prominence=params.min_prominence,
            min_peak_width=params.min_peak_width,
            method=params.method,
            use_rel_intsy=params.use_rel_intsy,
            ms1_corr_threshold=params.ms1_corr_threshold,
            adduct_aware=params.adduct_aware,
            loose_corr_threshold=params.loose_corr_threshold,
            adduct_ppm_tol=params.adduct_ppm_tol,
            polarity=params.polarity,
            rt_range=params.rt_range,
        )
    else:
        config = dia_config(
            parent_threshold=params.parent_threshold,
            cofeature_threshold=params.cofeature_threshold,
            window_strategy=params.window_strategy,
            extraction_half_width=params.extraction_half_width,
            min_window_halfwidth=params.min_window_halfwidth,
            edge_fraction=params.edge_fraction,
            min_prominence=params.min_prominence,
            min_peak_width=params.min_peak_width,
            peak_method=params.peak_method,
            baseline_pct=params.baseline_pct,
            min_turn=params.min_turn,
            require_smoothing_survival=params.require_smoothing_survival,
            smoothing_sigma=params.smoothing_sigma,
            min_smoothing_survival=params.min_smoothing_survival,
            max_lane_persistence=params.max_lane_persistence,
            method=params.method,
            use_rel_intsy=params.use_rel_intsy,
            ms1_corr_threshold=params.ms1_corr_threshold,
            adduct_aware=params.adduct_aware,
            loose_corr_threshold=params.loose_corr_threshold,
            adduct_ppm_tol=params.adduct_ppm_tol,
            polarity=params.polarity,
            ms2_corr_threshold=params.ms2_corr_threshold,
            rt_range=params.rt_range,
        )
    return extract_ensembles(
        injection, config,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
