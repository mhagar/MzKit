"""Choose whole mass lanes for labeling.

Favor seed-capable lanes while retaining some random active lanes. With a
model, oversample and prefer uncertain lanes. Only selected lanes are densified.
"""
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator

if TYPE_CHECKING:
    from core.data_structs import Sample
    from core.data_structs.uuid_types import SampleUUID

logger = logging.getLogger(__name__)

_RNG: Generator = np.random.default_rng(42)


@dataclass
class LaneCandidate:
    sample_uuid: "SampleUUID"
    sample_name: str
    lane_idx: int
    mz: float
    rt_values: np.ndarray
    intsy_values: np.ndarray


def main(
    samples: list["Sample"],
    parent_threshold: float,
    min_intsy: float,
    n_lanes: int = 100,
    max_samples: int = 3,
    random_frac: float = 0.2,
    model_path: str | None = None,
    exclude: set | None = None,
    pool_factor: int = 5,
    explore: float = 0.2,
    progress_callback=None,  # injected by ProcessRunner; unused here
    cancel_event=None,       # injected by ProcessRunner; unused here
) -> list[LaneCandidate]:
    """Return labeling candidates, interleaved across randomly selected samples."""
    model = None
    if model_path:
        from core.labeling import peak_model
        model = peak_model.load_model(model_path)

    selected_samples = select_random_samples(max_samples, samples)
    exclude = exclude or set()
    n_per_sample = int(np.ceil(n_lanes / max(1, len(selected_samples))))
    pool_size = n_per_sample * (pool_factor if model is not None else 1)

    queues = [
        _candidate_queue(
            sample,
            parent_threshold=parent_threshold,
            min_intsy=min_intsy,
            n_per_sample=n_per_sample,
            pool_size=pool_size,
            random_frac=random_frac,
            exclude=exclude,
            model=model,
            explore=explore,
        )
        for sample in selected_samples
    ]
    return _interleave_queues(queues)


def select_random_samples(
    max_samples: int,
    samples: list["Sample"],
) -> list["Sample"]:
    """Choose samples with injections without changing their relative order."""
    samples = [sample for sample in samples if sample.injection is not None]
    if len(samples) <= max_samples:
        return samples

    picks = _RNG.choice(len(samples), size=max_samples, replace=False)
    return [samples[i] for i in sorted(picks)]


def _candidate_queue(
    sample: "Sample",
    *,
    parent_threshold: float,
    min_intsy: float,
    n_per_sample: int,
    pool_size: int,
    random_frac: float,
    exclude: set,
    model: dict | None,
    explore: float,
) -> list[LaneCandidate]:
    scan_array = sample.injection.scan_array_ms1
    lane_maxima = scan_array.intsy_arr.max(axis=1).toarray().ravel()

    excluded_lanes = [lane for uuid, lane in exclude if uuid == sample.uuid]
    lane_maxima[excluded_lanes] = 0
    seedable_lanes = np.flatnonzero(lane_maxima >= parent_threshold)
    active_lanes = np.flatnonzero(lane_maxima >= min_intsy)

    lane_indices = _pick_lane_indices(
        seedable_lanes,
        active_lanes,
        n_lanes=pool_size,
        random_frac=random_frac,
    )
    candidates = [_make_candidate(sample, lane) for lane in lane_indices]

    if model is not None and len(candidates) > n_per_sample:
        candidates = _prioritize_by_uncertainty(
            candidates,
            model=model,
            n_keep=n_per_sample,
            min_intsy=min_intsy,
            explore=explore,
        )

    logger.info(
        f"Sample {sample.name}: {len(lane_indices)} lanes "
        f"({seedable_lanes.size} seed-capable, {active_lanes.size} active)"
    )
    return candidates


def _pick_lane_indices(
    seedable_lanes: np.ndarray,
    active_lanes: np.ndarray,
    *,
    n_lanes: int,
    random_frac: float,
) -> list[int]:
    seedable_slots = n_lanes - round(n_lanes * random_frac)
    selected_lanes = _random_subset(seedable_lanes, seedable_slots)

    remaining_lanes = np.setdiff1d(active_lanes, selected_lanes)
    selected_lanes += _random_subset(
        remaining_lanes,
        n_lanes - len(selected_lanes),
    )

    _RNG.shuffle(selected_lanes)
    return selected_lanes


def _random_subset(values: np.ndarray, size: int) -> list[int]:
    picks = _RNG.choice(values, size=min(size, len(values)), replace=False)
    return [int(value) for value in picks]


def _make_candidate(sample: "Sample", lane_idx: int) -> LaneCandidate:
    scan_array = sample.injection.scan_array_ms1
    intsy_values = scan_array.intsy_arr[lane_idx].toarray().ravel()
    return LaneCandidate(
        sample_uuid=sample.uuid,
        sample_name=sample.name,
        lane_idx=lane_idx,
        mz=float(scan_array.mz_lane_label[lane_idx]),
        rt_values=np.asarray(scan_array.rt_arr, dtype=float),
        intsy_values=intsy_values.astype(float),
    )


def _prioritize_by_uncertainty(
    candidates: list[LaneCandidate],
    *,
    model: dict,
    n_keep: int,
    min_intsy: float,
    explore: float,
) -> list[LaneCandidate]:
    from core.labeling import peak_model

    uncertainties = np.array([
        peak_model.lane_uncertainty(
            model,
            candidate.intsy_values,
            min_intsy,
            rt=candidate.rt_values,
        )
        for candidate in candidates
    ])
    uncertainty_order = np.argsort(-uncertainties)

    n_uncertain = round(n_keep * (1 - explore))
    selected_indices = uncertainty_order[:n_uncertain].tolist()
    selected_indices += _random_subset(
        uncertainty_order[n_uncertain:],
        n_keep - n_uncertain,
    )

    _RNG.shuffle(selected_indices)
    return [candidates[i] for i in selected_indices]


def _interleave_queues(
    queues: list[list[LaneCandidate]],
) -> list[LaneCandidate]:
    shared_length = min((len(queue) for queue in queues), default=0)
    shared = [candidate for group in zip(*queues) for candidate in group]
    tails = [
        candidate
        for queue in queues
        for candidate in queue[shared_length:]
    ]
    return shared + tails
