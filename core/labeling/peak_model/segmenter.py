"""On-demand whole-lane and cropped segmentation for ScanArrays."""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from core.labeling.schema import PEAK, Span

from .inference import (
    EXACT,
    RF,
    containing_span,
    decode,
    forward_features,
    lane_features,
    segment_lane,
    trim_to_signal,
)

CROP_WIDTH = 2 * (RF + EXACT) + 1
ScanResult = tuple[int, Span | None]


@dataclass
class ScanQuery:
    result_index: int
    lane: int
    scan: int


@dataclass
class CropQuery:
    result_index: int
    lane: int
    scan: int
    crop_start: int
    features: np.ndarray
    intsy_values: np.ndarray


class LaneSegmenter:
    """Cache whole lanes for seeds and use cropped inference for nearby queries."""

    def __init__(self, model: dict, scan_array):
        self.model = model
        self.scan_array = scan_array
        self._lane_cache: dict[int, tuple[np.ndarray, list[Span]]] = {}
        self._point_cache: dict[tuple[int, int], ScanResult] = {}

    def _intsy(self, lane: int) -> np.ndarray:
        return self.scan_array.intsy_arr[lane].toarray().ravel()

    def lane(self, lane: int) -> tuple[np.ndarray, list[Span]]:
        if lane not in self._lane_cache:
            self._lane_cache[lane] = segment_lane(
                self.model,
                self._intsy(lane),
                self.scan_array.rt_arr,
            )
        return self._lane_cache[lane]

    def span_at(self, lane: int, scan: int) -> Span | None:
        _, spans = self.lane(lane)
        return containing_span(spans, scan)

    def near(self, lanes, scans) -> list[ScanResult]:
        queries = [
            ScanQuery(index, int(lane), int(scan))
            for index, (lane, scan) in enumerate(zip(lanes, scans))
        ]
        n_scans = self.scan_array.intsy_arr.shape[1]
        if n_scans <= 2 * CROP_WIDTH:
            return [self._whole_lane_result(query) for query in queries]

        results: list[ScanResult | None] = [None] * len(queries)
        crop_queries = []
        for query in queries:
            cached_result = self._cached_result(query)
            if cached_result is not None:
                results[query.result_index] = cached_result
            else:
                crop_queries.append(self._make_crop_query(query))

        if crop_queries:
            features = np.stack([query.features for query in crop_queries])
            batch_logits = forward_features(self.model, features)
            for query, logits in zip(crop_queries, batch_logits):
                result = self._resolve_crop(query, logits)
                self._point_cache[(query.lane, query.scan)] = result
                results[query.result_index] = result

        return cast(list[ScanResult], results)

    def _cached_result(self, query: ScanQuery) -> ScanResult | None:
        if query.lane in self._lane_cache:
            return self._whole_lane_result(query)
        return self._point_cache.get((query.lane, query.scan))

    def _whole_lane_result(self, query: ScanQuery) -> ScanResult:
        class_ids, spans = self.lane(query.lane)
        return int(class_ids[query.scan]), containing_span(spans, query.scan)

    def _make_crop_query(self, query: ScanQuery) -> CropQuery:
        n_scans = self.scan_array.intsy_arr.shape[1]
        crop_start = min(
            max(0, query.scan - RF - EXACT),
            n_scans - CROP_WIDTH,
        )
        crop_end = crop_start + CROP_WIDTH
        intsy_values = self._intsy(query.lane)
        features = lane_features(intsy_values, self.scan_array.rt_arr)
        return CropQuery(
            result_index=query.result_index,
            lane=query.lane,
            scan=query.scan,
            crop_start=crop_start,
            features=features[:, crop_start:crop_end],
            intsy_values=intsy_values[crop_start:crop_end],
        )

    def _resolve_crop(self, query: CropQuery, logits: np.ndarray) -> ScanResult:
        crop_scan = query.scan - query.crop_start
        crop_width = logits.shape[-1]
        n_scans = self.scan_array.intsy_arr.shape[1]
        exact_start, exact_end = _exact_zone(
            query.crop_start,
            crop_scan,
            crop_width,
            n_scans,
        )

        class_ids, spans = decode(logits[:, exact_start:exact_end])
        exact_scan = crop_scan - exact_start
        span = containing_span(spans, exact_scan)
        if _peak_extends_beyond_exact_zone(
            span,
            query.crop_start,
            exact_start,
            exact_end,
            n_scans,
        ):
            scan_query = ScanQuery(query.result_index, query.lane, query.scan)
            return self._whole_lane_result(scan_query)

        lane_offset = query.crop_start + exact_start
        if span is not None:
            crop_intsy = query.intsy_values[exact_start:exact_end]
            span = containing_span(trim_to_signal([span], crop_intsy), exact_scan)

        resolved_span = None
        if span is not None:
            resolved_span = (
                span[0] + lane_offset,
                span[1] + lane_offset,
                span[2],
            )
        return int(class_ids[exact_scan]), resolved_span


def _exact_zone(
    crop_start: int,
    crop_scan: int,
    crop_width: int,
    n_scans: int,
) -> tuple[int, int]:
    exact_start = 0 if crop_start == 0 else max(RF, crop_scan - EXACT)
    if crop_start + crop_width == n_scans:
        exact_end = crop_width
    else:
        exact_end = min(crop_width - RF, crop_scan + EXACT + 1)
    return exact_start, exact_end


def _peak_extends_beyond_exact_zone(
    span: Span | None,
    crop_start: int,
    exact_start: int,
    exact_end: int,
    n_scans: int,
) -> bool:
    if span is None or span[2] != PEAK:
        return False
    extends_left = span[0] == 0 and crop_start + exact_start > 0
    extends_right = (
        span[1] == exact_end - exact_start
        and crop_start + exact_end < n_scans
    )
    return extends_left or extends_right
