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
