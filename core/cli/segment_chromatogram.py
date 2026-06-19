"""
Chromatogram segmentation: finding the boundaries of a peak
given its apex index.
"""
import numpy as np


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

    :param intensity: 1D intensity array.
    :param apex_idx: Index of the peak apex.
    :param seg_start: Left boundary of the peak (from ``find_peak_boundaries``).
    :param seg_end: Right boundary of the peak (exclusive).
    :param min_peak_width: Minimum segment width in scans (rejects spikes).
    :param min_prominence: Required prominence as a fraction of apex height on
        the weaker shoulder. ``0.5`` => apex must be >= 2x the higher of the two
        side-baselines. This is the single shape knob.
    :param baseline_pct: Percentile used to estimate each side's baseline,
        robust to single-scan dropouts to zero.
    :return: True if the segment looks like a real peak.
    """
    width = seg_end - seg_start
    if width < min_peak_width:
        return False

    apex_intsy = intensity[apex_idx]
    if apex_intsy <= 0:
        return False

    # Look one segment-width beyond each cut boundary to see the real baseline.
    left_lo = max(0, seg_start - width)
    right_hi = min(intensity.size, seg_end + width)

    left_base = np.percentile(intensity[left_lo:apex_idx + 1], baseline_pct)
    right_base = np.percentile(intensity[apex_idx:right_hi], baseline_pct)

    # Weaker shoulder governs (bilateral): both sides must descend to baseline.
    base = max(left_base, right_base)
    prominence = (apex_intsy - base) / apex_intsy
    return prominence >= min_prominence
