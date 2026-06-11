"""
Generate labeling candidates from a list of Samples.

A Candidate is a (sample, m/z lane, apex scan) tuple paired with
an XIC window suitable for display to a human annotator.

Candidates are deliberately *not* validated — the whole point of
labeling is to include ambiguous and poor signals that an
algorithm might misclassify. Filtering those out before labeling
defeats the exercise.

Diversity sampling (stratify) reorders candidates so consecutive
items span different (intensity, width) bins. This ensures the
annotator sees a representative spread from candidate #1 rather
than exhausting one regime before seeing another.
"""
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Iterator

import numpy as np

from core.cli.segment_chromatogram import find_peak_boundaries

if TYPE_CHECKING:
    from core.data_structs import Sample, ScanArray
    from core.data_structs.uuid_types import SampleUUID

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    sample_uuid: "SampleUUID"
    sample_name: str
    mz_lane_idx: int
    mz: float
    apex_scan_idx: int
    apex_rt: float
    apex_intsy: float
    rough_width: int
    # XIC window to show the annotator
    window_start_scan: int
    window_end_scan: int
    rt_values: np.ndarray
    intsy_values: np.ndarray


def main(
    samples: list["Sample"],
    min_intsy: float,
    window_half_width: int = 60,
    max_per_sample: int | None = None,
    progress_callback=None,  # injected by ProcessRunner; unused here
    cancel_event=None,       # injected by ProcessRunner; unused here
) -> list["Candidate"]:
    """
    Entry point for `ProcessController.start_process` — runs the full
    candidate pipeline (generate + stratify) and returns the ordered
    list ready to hand to LabelingWindow.
    """
    logger.info(
        f"Generating candidates: {len(samples)} sample(s), "
        f"min_intsy={min_intsy}, max_per_sample={max_per_sample}"
    )
    raw = generate_candidates(
        samples=samples,
        min_intsy=min_intsy,
        window_half_width=window_half_width,
        max_per_sample=max_per_sample,
    )
    logger.info(f"Raw candidates: {len(raw)}. Stratifying…")
    ordered = stratify(raw)
    logger.info(f"Stratified candidate queue ready: {len(ordered)} items")
    return ordered


def generate_candidates(
    samples: list["Sample"],
    min_intsy: float,
    window_half_width: int = 60,
    edge_fraction: float = 0.1,
    min_peak_separation: int = 5,
    max_per_sample: int | None = None,
) -> list[Candidate]:
    """
    Scan each Sample's MS1 lanes for local maxima above min_intsy.
    For each maximum, extract an XIC window of ±window_half_width
    scans around the apex.

    Successive apexes within the same lane must be at least
    min_peak_separation scans apart (prevents picking every point
    on a broad plateau as a separate candidate).
    """
    out: list[Candidate] = []
    for sample in samples:
        if sample.injection is None:
            continue
        sa = sample.injection.scan_array_ms1
        n_lanes = sa.intsy_arr.shape[0]

        # Dense per-lane chromatograms are the simplest way to find
        # local maxima. Sparse arrays make this awkward.
        lane_maxes = sa.intsy_arr.max(axis=1).toarray().flatten()
        active_lanes = np.where(lane_maxes >= min_intsy)[0]

        sample_candidates: list[Candidate] = []
        for lane_idx in active_lanes:
            chrom = sa.intsy_arr[int(lane_idx)].toarray().flatten()
            apexes = _find_local_maxima(chrom, min_intsy, min_peak_separation)
            for apex_idx in apexes:
                cand = _build_candidate(
                    sample=sample,
                    scan_array=sa,
                    lane_idx=int(lane_idx),
                    apex_idx=int(apex_idx),
                    chromatogram=chrom,
                    window_half_width=window_half_width,
                    edge_fraction=edge_fraction,
                )
                sample_candidates.append(cand)

        if max_per_sample is not None and len(sample_candidates) > max_per_sample:
            # Keep the top-N by intensity — a coarse cap, further
            # diversified by stratify()
            sample_candidates.sort(key=lambda c: c.apex_intsy, reverse=True)
            sample_candidates = sample_candidates[:max_per_sample]

        logger.info(
            f"Sample {sample.name}: {len(sample_candidates)} candidates"
        )
        out.extend(sample_candidates)

    return out


def stratify(
    candidates: list[Candidate],
    n_intensity_bins: int = 3,
    n_width_bins: int = 3,
    seed: int = 0,
) -> list[Candidate]:
    """
    Reorder candidates so consecutive items span diverse
    (intensity, width) strata. Uses round-robin across bins.

    The shuffle within each bin is deterministic given `seed`
    so sessions are reproducible.
    """
    if not candidates:
        return []

    rng = np.random.default_rng(seed)

    intsys = np.array([c.apex_intsy for c in candidates])
    widths = np.array([c.rough_width for c in candidates])

    intsy_bins = _quantile_bin(intsys, n_intensity_bins)
    width_bins = _quantile_bin(widths, n_width_bins)

    bucketed: dict[tuple[int, int], list[Candidate]] = {}
    for cand, ib, wb in zip(candidates, intsy_bins, width_bins):
        bucketed.setdefault((int(ib), int(wb)), []).append(cand)

    for bucket in bucketed.values():
        rng.shuffle(bucket)

    # Round-robin across all buckets until drained
    iters = [iter(v) for v in bucketed.values()]
    out: list[Candidate] = []
    while iters:
        next_iters = []
        for it in iters:
            try:
                out.append(next(it))
                next_iters.append(it)
            except StopIteration:
                continue
        iters = next_iters

    return out


def _find_local_maxima(
    chrom: np.ndarray, min_intsy: float, min_separation: int,
) -> list[int]:
    """
    Return scan indices of local maxima in `chrom` that exceed
    min_intsy, separated by at least min_separation scans.

    Greedy: pick the tallest untaken point, mark a neighborhood
    as taken, repeat.
    """
    available = chrom >= min_intsy
    apexes: list[int] = []
    # Copy because we mutate
    work = chrom.copy()
    work[~available] = 0

    while True:
        idx = int(np.argmax(work))
        if work[idx] < min_intsy:
            break
        # Confirm it's a local max relative to immediate neighbors
        left = work[idx - 1] if idx > 0 else 0
        right = work[idx + 1] if idx + 1 < work.size else 0
        if work[idx] >= left and work[idx] >= right:
            apexes.append(idx)
        lo = max(0, idx - min_separation)
        hi = min(work.size, idx + min_separation + 1)
        work[lo:hi] = 0

    apexes.sort()
    return apexes


def _build_candidate(
    sample: "Sample",
    scan_array: "ScanArray",
    lane_idx: int,
    apex_idx: int,
    chromatogram: np.ndarray,
    window_half_width: int,
    edge_fraction: float,
) -> Candidate:
    seg_start, seg_end = find_peak_boundaries(
        chromatogram, apex_idx, edge_fraction=edge_fraction,
    )
    rough_width = max(1, seg_end - seg_start)

    n_scans = chromatogram.size
    w_start = max(0, apex_idx - window_half_width)
    w_end = min(n_scans, apex_idx + window_half_width + 1)

    rt_values = scan_array.rt_arr[w_start:w_end]
    intsy_values = chromatogram[w_start:w_end]
    mz = float(scan_array.mz_lane_label[lane_idx])
    apex_rt = float(scan_array.rt_arr[apex_idx])
    apex_intsy = float(chromatogram[apex_idx])

    return Candidate(
        sample_uuid=sample.uuid,
        sample_name=sample.name,
        mz_lane_idx=lane_idx,
        mz=mz,
        apex_scan_idx=apex_idx,
        apex_rt=apex_rt,
        apex_intsy=apex_intsy,
        rough_width=rough_width,
        window_start_scan=w_start,
        window_end_scan=w_end,
        rt_values=rt_values,
        intsy_values=intsy_values,
    )


def _quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each value to a quantile-based bin [0, n_bins)."""
    if values.size == 0 or n_bins <= 1:
        return np.zeros(values.size, dtype=int)
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    thresholds = np.quantile(values, quantiles)
    return np.searchsorted(thresholds, values)
