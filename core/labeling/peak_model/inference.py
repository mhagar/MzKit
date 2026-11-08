"""NumPy feature extraction and inference for chromatographic lane segmentation."""
from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from core.labeling.schema import PEAK, PLATEAU, Span

OTHER_C, PEAK_C, PLATEAU_C = 0, 1, 2
_CLS_NAME = {PEAK_C: PEAK, PLATEAU_C: PLATEAU}
_CLS_ID = {PEAK: PEAK_C, PLATEAU: PLATEAU_C}

N_IN = 6
CHANNELS = 24
KERNEL = 5
DILATIONS = (1, 2, 4, 8, 16, 32, 64)
RF = (KERNEL - 1) // 2 * sum(DILATIONS)
EXACT = 64


def lane_features(
    intsy: np.ndarray,
    rt: np.ndarray | None = None,
) -> np.ndarray:
    intensity = np.maximum(np.asarray(intsy, dtype=np.float64), 0.0)
    signal_present = (intensity > 0).astype(np.float64)
    run_fraction = _run_fraction(rt, intensity.size)

    local_shape = intensity / np.maximum(maximum_filter1d(intensity, 65), 1e-12)
    absolute_intensity = np.log10(intensity + 1.0) / 7.0
    relative_intensity = intensity / max(intensity.max(), 1e-12)
    persistence = uniform_filter1d(signal_present, 257)

    return np.stack([
        local_shape,
        absolute_intensity,
        relative_intensity,
        signal_present,
        persistence,
        run_fraction,
    ]).astype(np.float32)


def _run_fraction(rt: np.ndarray | None, n_scans: int) -> np.ndarray:
    if rt is not None and len(rt) == n_scans and rt[-1] > rt[0]:
        rt = np.asarray(rt, dtype=np.float64)
        return (rt - rt[0]) / (rt[-1] - rt[0])
    return np.linspace(0.0, 1.0, n_scans)


def lane_targets(
    n_scans: int,
    spans: Iterable[Span],
) -> tuple[np.ndarray, np.ndarray]:
    class_ids = np.zeros(n_scans, dtype=np.int64)
    boundary_targets = np.zeros(n_scans, dtype=np.float32)
    peak_bounds = []

    for start, end, kind in spans:
        class_ids[start:end] = _CLS_ID[kind]
        if kind == PEAK:
            peak_bounds.append((start, end))

    peak_bounds.sort()
    for (_, previous_end), (next_start, _) in pairwise(peak_bounds):
        if previous_end == next_start and 0 < next_start < n_scans:
            boundary_targets[next_start - 1:next_start + 1] = 1.0

    return class_ids, boundary_targets


def _conv(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    d: int = 1,
) -> np.ndarray:
    """Same dilated conv1d over (..., channels_in, scans)."""
    c_out, c_in, k = W.shape
    length = x.shape[-1]
    pad = d * (k - 1) // 2
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)])
    columns = np.stack([
        padded[..., i * d:i * d + length]
        for i in range(k)
    ], axis=-2)
    weights = W.reshape(c_out, -1)
    values = columns.reshape(*x.shape[:-2], c_in * k, length)
    return weights @ values + b[:, None]


def forward_features(model: dict, features: np.ndarray) -> np.ndarray:
    hidden = _conv(features, model["inp.weight"], model["inp.bias"])
    for i, dilation in enumerate(DILATIONS):
        residual = _conv(
            hidden,
            model[f"blocks.{i}.weight"],
            model[f"blocks.{i}.bias"],
            dilation,
        )
        hidden = hidden + np.maximum(residual, 0)
    return _conv(hidden, model["out.weight"], model["out.bias"])


def forward(
    model: dict,
    intsy: np.ndarray,
    rt: np.ndarray | None = None,
) -> np.ndarray:
    return forward_features(model, lane_features(intsy, rt))


def decode(logits: np.ndarray) -> tuple[np.ndarray, list[Span]]:
    cls = logits[:3].argmax(0)
    bnd = logits[3] > 0
    spans: list[Span] = []
    n = cls.size
    i = 0
    while i < n:
        j = i
        while j < n and cls[j] == cls[i]:
            j += 1
        if cls[i] != OTHER_C:
            cuts = [i]
            if cls[i] == PEAK_C:
                k = i + 1
                while k < j:
                    if bnd[k]:
                        m = k
                        while m < j and bnd[m]:
                            m += 1
                        cut = (k + m) // 2
                        if cut - cuts[-1] >= 2 and j - cut >= 2:
                            cuts.append(cut)
                        k = m
                    else:
                        k += 1
            cuts.append(j)
            spans += [
                (a, b, _CLS_NAME[int(cls[i])])
                for a, b in pairwise(cuts)
            ]
        i = j
    return cls, spans


def segment_lane(
    model: dict,
    intsy: np.ndarray,
    rt: np.ndarray | None = None,
) -> tuple[np.ndarray, list[Span]]:
    class_ids, spans = decode(forward(model, intsy, rt))
    return class_ids, trim_to_signal(spans, intsy)


def lane_uncertainty(
    model: dict,
    intsy: np.ndarray,
    min_intsy: float,
    k: int = 20,
    rt: np.ndarray | None = None,
) -> float:
    class_logits = forward(model, intsy, rt)[:3]
    probabilities = np.exp(class_logits - class_logits.max(0))
    probabilities /= probabilities.sum(0)
    top_two = np.sort(probabilities, axis=0)[-2:]
    uncertainty = 1 - (top_two[1] - top_two[0])
    active_uncertainty = uncertainty[np.asarray(intsy) >= min_intsy]
    if not active_uncertainty.size:
        return 0.0
    return float(np.sort(active_uncertainty)[-k:].mean())


def containing_span(spans: list[Span], scan: int) -> Span | None:
    for start, end, kind in spans:
        if start <= scan < end:
            return start, end, kind
    return None


def trim_to_signal(spans: list[Span], intsy: np.ndarray) -> list[Span]:
    signal_present = np.asarray(intsy) > 0
    trimmed = []
    for start, end, kind in spans:
        signal_scans = np.flatnonzero(signal_present[start:end])
        if signal_scans.size:
            trimmed.append((
                start + int(signal_scans[0]),
                start + int(signal_scans[-1]) + 1,
                kind,
            ))
    return trimmed
