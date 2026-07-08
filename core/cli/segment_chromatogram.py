"""
Chromatogram segmentation: finding the boundaries of a peak
given its apex index.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def find_peak_boundaries(
    intensity: np.ndarray,
    apex_idx: int,
    edge_fraction: float = 0.1,
    valley_sustain: int = 3,
) -> tuple[int, int]:
    """
    Given a chromatogram and the index of a peak apex,
    returns the (start, end) boundaries of the peak.

    Descends from the apex in both directions, stopping when:
      - intensity hits zero
      - a valley is found (intensity rises again for
        `valley_sustain` consecutive scans)
      - intensity drops below `edge_fraction * apex_intensity`

    :param intensity: 1D intensity array
    :param apex_idx: Index of the peak apex
    :param edge_fraction: Stop when intensity falls below this
        fraction of the apex intensity
    :param valley_sustain: Number of consecutive rising scans
        required to confirm a valley (avoids false valleys from
        noise)
    :return: (start, end) index pair
    """
    apex_intsy = intensity[apex_idx]
    floor = apex_intsy * edge_fraction

    start = _descend(intensity, apex_idx, floor, valley_sustain, direction=-1)
    end = _descend(intensity, apex_idx, floor, valley_sustain, direction=1)

    return (start, end)


def _descend(
    intensity: np.ndarray,
    apex_idx: int,
    floor: float,
    valley_sustain: int,
    direction: int,
) -> int:
    """
    Walk away from apex_idx in the given direction (-1 = left,
    +1 = right), returning the boundary index.

    Stops when:
      - edge of array
      - intensity hits zero
      - intensity drops below floor
      - intensity rises for valley_sustain consecutive steps
        (valley detected)

    Returns the boundary index (inclusive for start, exclusive
    for end — matching slice semantics).
    """
    n = intensity.size
    i = apex_idx
    rising_count = 0
    prev_val = intensity[apex_idx]

    while True:
        next_i = i + direction
        if next_i < 0 or next_i >= n:
            break

        val = intensity[next_i]

        if val <= 0:
            break

        if val < floor:
            i = next_i
            break

        # Valley detection: is intensity rising back up?
        if val > prev_val:
            rising_count += 1
            if rising_count >= valley_sustain:
                # Backtrack to the actual valley minimum
                valley_start = next_i - (valley_sustain * direction)
                valley_region = intensity[
                    min(valley_start, next_i):
                    max(valley_start, next_i) + 1
                ]
                valley_min_offset = np.argmin(valley_region)
                i = min(valley_start, next_i) + valley_min_offset
                break
        else:
            rising_count = 0

        prev_val = val
        i = next_i

    # Return in slice-friendly form
    if direction == -1:
        return i
    else:
        return i + 1


def validate_peak(
    intensity: np.ndarray,
    apex_idx: int,
    seg_start: int,
    seg_end: int,
    min_rise_ratio: float = 2.0,
    min_peak_width: int = 5,
) -> bool:
    """
    Check whether a peak is worth extracting an ensemble from.

    :param intensity: 1D intensity array
    :param apex_idx: Index of the peak apex
    :param seg_start: Left boundary of the peak
    :param seg_end: Right boundary of the peak
    :param min_rise_ratio: Apex must be at least this many times
        the edge intensity. Filters out broad humps.
    :param min_peak_width: Peak must span at least this many scans.
        Filters out noise spikes.
    :return: True if peak is valid
    """
    width = seg_end - seg_start
    if width < min_peak_width:
        return False

    apex_intsy = intensity[apex_idx]
    edge_intsy = max(
        intensity[seg_start],
        intensity[seg_end - 1],
    )

    if edge_intsy <= 0:
        return True

    rise_ratio = apex_intsy / edge_intsy
    return rise_ratio >= min_rise_ratio


def is_peak(
    intensity: np.ndarray,
    apex_idx: int,
    seg_start: int,
    seg_end: int,
    min_peak_width: int = 5,
    min_prominence: float = 0.5,
    baseline_pct: float = 10.0,
    baseline_intensity: 'np.ndarray | None' = None,
    method: str = "prominence",
    min_turn: float = 0.05,
) -> bool:
    """
    Decide whether a segment is a genuine chromatographic peak rather than a
    monotonic slope or a smeared / near-constant background signal.

    A more robust replacement for the apex/edge ``rise_ratio`` test in
    ``validate_peak``, which is toothless in practice: ``find_peak_boundaries``
    cuts the segment at ``edge_fraction * apex``, so the segment edges sit at
    ~that fraction and the rise ratio is ~``1/edge_fraction`` (≈10) almost
    regardless of shape.

    Instead we measure *bilateral prominence*: how far the signal descends below
    the apex on EACH side, with the baseline estimated from a window that looks
    one peak-width BEYOND the cut segment (so it reflects the true surrounding
    level, not the ``edge_fraction`` cut). The weaker of the two shoulders
    governs:

      - a real peak descends to baseline on both sides -> high prominence (kept)
      - a monotonic slope descends on only one side     -> low prominence (dropped)
      - a flat / constant smear never descends          -> low prominence (dropped)

    :param intensity: 1D intensity array used to locate the segment/apex.
    :param apex_idx: Index of the peak apex.
    :param seg_start: Left boundary of the peak (from ``find_peak_boundaries``).
    :param seg_end: Right boundary of the peak (exclusive).
    :param min_peak_width: Minimum segment width in scans (rejects spikes).
    :param min_prominence: Required prominence as a fraction of apex height on
        the weaker shoulder. ``0.5`` => apex must be >= 2x the higher of the two
        side-baselines. This is the single shape knob.
    :param baseline_pct: Percentile used to estimate each side's baseline,
        robust to single-scan dropouts to zero.
    :param baseline_intensity: Optional separate array to measure the baseline
        (and apex height) against. Pass the *raw* chromatogram here when
        ``intensity`` is a greedily-consumed (zeroed) working copy: otherwise the
        zeros punched into already-consumed neighbours read as a false 0
        baseline, making every leftover sliver look infinitely prominent. When
        ``None`` (default), ``intensity`` is used, preserving prior behaviour.
        (Only used by ``method="prominence"``.)
    :param method: Shape test to apply:
        - ``"prominence"`` (default): *bilateral* prominence — the WEAKER
          shoulder must descend to baseline. Strict; best for isolated peaks,
          but rejects a peak fused to a taller neighbour (whose baseline window
          reaches into that neighbour).
        - ``"flank"``: *unilateral*, neighbour-invariant — measured WITHIN the
          segment only, the STEEPER shoulder must descend by ``min_prominence``
          of the apex, and the apex must genuinely turn over (rise into and fall
          out of it by at least ``min_turn``). Keeps fused/shouldered peaks while
          still rejecting whine, which descends only modestly on both sides.
    :param min_turn: (``method="flank"``) minimum fractional descent required on
        BOTH shoulders for the apex to count as a turned-over local maximum
        rather than a monotonic ramp / segment edge.
    :return: True if the segment looks like a real peak.
    """
    width = seg_end - seg_start
    if width < min_peak_width:
        return False

    if method == "flank":
        return _flank_ok(
            intensity, apex_idx, seg_start, seg_end, min_prominence, min_turn
        )
    if method != "prominence":
        raise ValueError(
            f"Unknown is_peak method {method!r}; expected 'prominence' or 'flank'"
        )

    # Judge prominence against the raw signal when supplied, so consumed/zeroed
    # neighbours don't masquerade as baseline.
    base_arr = intensity if baseline_intensity is None else baseline_intensity

    apex_intsy = base_arr[apex_idx]
    if apex_intsy <= 0:
        return False

    # Look one segment-width beyond each cut boundary to see the real baseline.
    left_lo = max(0, seg_start - width)
    right_hi = min(base_arr.size, seg_end + width)

    left_base = np.percentile(base_arr[left_lo:apex_idx + 1], baseline_pct)
    right_base = np.percentile(base_arr[apex_idx:right_hi], baseline_pct)

    # Weaker shoulder governs (bilateral): both sides must descend to baseline.
    base = max(left_base, right_base)
    prominence = (apex_intsy - base) / apex_intsy
    return prominence >= min_prominence


def survives_smoothing(
    intensity: np.ndarray,
    apex_idx: int,
    seg_start: int,
    seg_end: int,
    sigma: float = 1.0,
    min_survival: float = 0.5,
    baseline_pct: float = 10.0,
    baseline_intensity: 'np.ndarray | None' = None,
) -> bool:
    """
    Reject peaks whose apex is carried by high-frequency noise rather than a
    coherent chromatographic band.

    Complements ``is_peak``: prominence/flank ask "does the apex rise above its
    surroundings?", which a spiky bump on a persistent-background plateau can pass
    at a loose threshold. This asks "is that rise made of *coherent* signal?" — a
    genuine band spreads its intensity over several scans, so a narrow Gaussian
    kernel barely lowers its apex (survival ~1); a one/two-scan spike or plateau
    jitter is averaged into its neighbours and collapses toward baseline
    (survival ~0). It does NOT catch already-smooth broad humps — those remain the
    job of ``min_prominence`` / ``baseline_pct``.

    Only a local slice around the segment is smoothed (segment width padded by
    ~``4*sigma`` of context each side), never the whole lane.

    The local baseline is subtracted from both the raw and smoothed apex so the
    ratio measures the coherent excess, not the absolute level: a bump on an
    elevated contaminant floor is judged on the bump, not the floor.

    :param intensity: 1D working chromatogram (locates apex/segment).
    :param apex_idx: Index of the peak apex.
    :param seg_start: Left boundary (from ``find_peak_boundaries``).
    :param seg_end: Right boundary (exclusive).
    :param sigma: Gaussian smoothing width in scans. Keep well below a real
        peak's width (~1) so genuine bands survive while 1-2 scan spikes don't.
    :param min_survival: Minimum fraction of the baseline-subtracted apex height
        that must remain after smoothing. ``0.5`` => the smoothed apex must stay
        at least halfway between baseline and the raw apex.
    :param baseline_pct: Percentile for the local baseline estimate (as
        ``is_peak``), robust to single-scan dropouts to zero.
    :param baseline_intensity: Raw chromatogram to measure against when
        ``intensity`` is a greedily-consumed (zeroed) working copy — the same
        reason ``is_peak`` takes it. When ``None``, ``intensity`` is used.
    :return: True if the peak survives smoothing.
    """
    base_arr = intensity if baseline_intensity is None else baseline_intensity
    width = seg_end - seg_start

    apex_intsy = base_arr[apex_idx]
    if apex_intsy <= 0:
        return False

    # Local baseline from one segment-width beyond each cut boundary (as is_peak).
    left_lo = max(0, seg_start - width)
    right_hi = min(base_arr.size, seg_end + width)
    left_base = np.percentile(base_arr[left_lo:apex_idx + 1], baseline_pct)
    right_base = np.percentile(base_arr[apex_idx:right_hi], baseline_pct)
    base = max(left_base, right_base)

    raw_excess = apex_intsy - base
    if raw_excess <= 0:
        return False

    # Smooth ONLY a local slice: the segment plus enough context that the kernel
    # is well-formed over it. gaussian_filter1d's default 'reflect' mode handles
    # the true array ends when the segment sits near them.
    pad = int(np.ceil(4 * sigma)) + width
    lo = max(0, seg_start - pad)
    hi = min(base_arr.size, seg_end + pad)
    smoothed_local = gaussian_filter1d(base_arr[lo:hi], sigma=sigma)

    # The smoothed apex can drift a scan or two; take the best within the segment.
    smoothed_apex = smoothed_local[seg_start - lo:seg_end - lo].max()
    survival = (smoothed_apex - base) / raw_excess
    return survival >= min_survival


def _flank_ok(
    intensity: np.ndarray,
    apex_idx: int,
    seg_start: int,
    seg_end: int,
    min_descent: float,
    min_turn: float,
) -> bool:
    """
    Neighbour-invariant peak-shape test (``is_peak(method="flank")``).

    Looks ONLY within the segment ``[seg_start, seg_end)`` — never at the
    surrounding signal — so a peak fused to a taller neighbour is judged on its
    own body, not the neighbour's. Requires:

      1. the apex to genuinely turn over: it must rise into the apex from the
         left and fall out of it on the right, each by >= ``min_turn`` of the
         apex height (rejects monotonic ramps / segment edges); and
      2. its STEEPER shoulder to descend by >= ``min_descent`` of the apex.

    A real chromatographic peak plunges to (near) baseline on at least one side,
    even when fused; persistent background 'whine' rides an elevated floor and
    descends only modestly on both sides, so its steeper shoulder stays shallow.
    """
    apex_intsy = intensity[apex_idx]
    if apex_intsy <= 0:
        return False

    left_min = intensity[seg_start:apex_idx + 1].min()
    right_min = intensity[apex_idx:seg_end].min()
    rise = (apex_intsy - left_min) / apex_intsy
    fall = (apex_intsy - right_min) / apex_intsy

    if rise < min_turn or fall < min_turn:
        return False
    return max(rise, fall) >= min_descent
