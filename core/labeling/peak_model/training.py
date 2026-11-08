"""Prepare labeled lanes and train the optional torch peak model."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from core.labeling.schema import LabelFile, Span

from .inference import CHANNELS, DILATIONS, KERNEL, N_IN, lane_features, lane_targets

RUN_FRAC_DROPOUT = 0.3


class LaneExample(NamedTuple):
    intsy_values: np.ndarray
    spans: list[Span]
    rt_values: np.ndarray | None = None

    @classmethod
    def from_value(cls, value) -> LaneExample:
        if isinstance(value, cls):
            return value
        intsy_values, spans, *optional_rt = value
        rt_values = optional_rt[0] if optional_rt else None
        return cls(
            intsy_values=np.asarray(intsy_values, dtype=float),
            spans=spans,
            rt_values=(
                np.asarray(rt_values, dtype=float)
                if rt_values is not None else None
            ),
        )

    @classmethod
    def from_label(cls, label) -> LaneExample:
        return cls(
            intsy_values=np.asarray(label.intsy_values, dtype=float),
            spans=label.spans,
            rt_values=np.asarray(label.rt_values, dtype=float),
        )


@dataclass
class _PreparedLane:
    features: np.ndarray
    class_targets: np.ndarray
    boundary_targets: np.ndarray

    @property
    def n_scans(self) -> int:
        return self.features.shape[1]


@dataclass
class _TrainingBatch:
    features: np.ndarray
    class_targets: np.ndarray
    boundary_targets: np.ndarray
    valid_scan_mask: np.ndarray


def load_examples(paths: Iterable[Path]) -> list[LaneExample]:
    examples = [
        LaneExample.from_label(label)
        for path in paths
        for label in LabelFile.from_json(Path(path)).lanes
    ]
    if not examples:
        raise ValueError("No labeled lanes")
    return examples


def train(
    examples,
    steps: int = 1500,
    crop: int = 1024,
    batch: int = 16,
    seed: int = 0,
    cancel_event=None,
) -> dict:
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "Training the peak model needs torch: `uv sync --extra train`"
        ) from error

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    prepared_lanes = _prepare_training_lanes(examples)
    class_weights = _calculate_class_weights(prepared_lanes)
    lane_probabilities = _lane_probabilities(prepared_lanes)

    net = _torch_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        total_steps=steps,
    )

    for _ in range(steps):
        if cancel_event is not None and cancel_event.is_set():
            break

        training_batch = _sample_training_batch(
            prepared_lanes,
            lane_probabilities,
            rng,
            crop_size=crop,
            batch_size=batch,
        )
        logits = net(torch.from_numpy(training_batch.features))
        loss = _calculate_training_loss(logits, training_batch, class_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return {
        name: value.detach().numpy().astype(np.float32)
        for name, value in net.state_dict().items()
    }


def _prepare_training_lanes(examples) -> list[_PreparedLane]:
    prepared = []
    for value in examples:
        example = LaneExample.from_value(value)
        class_targets, boundary_targets = lane_targets(
            example.intsy_values.size,
            example.spans,
        )
        prepared.append(_PreparedLane(
            features=lane_features(example.intsy_values, example.rt_values),
            class_targets=class_targets,
            boundary_targets=boundary_targets,
        ))
    return prepared


def _calculate_class_weights(lanes: list[_PreparedLane]):
    import torch

    targets = np.concatenate([lane.class_targets for lane in lanes])
    counts = np.bincount(targets, minlength=3) + 1
    weights = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float32)
    return weights / weights.mean()


def _lane_probabilities(lanes: list[_PreparedLane]) -> np.ndarray:
    lengths = np.array([lane.n_scans for lane in lanes], dtype=float)
    return lengths / lengths.sum()


def _sample_training_batch(
    lanes: list[_PreparedLane],
    lane_probabilities: np.ndarray,
    rng: np.random.Generator,
    *,
    crop_size: int,
    batch_size: int,
) -> _TrainingBatch:
    features = np.zeros((batch_size, N_IN, crop_size), np.float32)
    class_targets = np.zeros((batch_size, crop_size), np.int64)
    boundary_targets = np.zeros((batch_size, crop_size), np.float32)
    valid_scan_mask = np.zeros((batch_size, crop_size), np.float32)

    lane_indices = rng.choice(
        len(lanes),
        batch_size,
        p=lane_probabilities,
    )
    for row, lane_idx in enumerate(lane_indices):
        lane = lanes[lane_idx]
        start = int(rng.integers(0, max(1, lane.n_scans - crop_size + 1)))
        length = min(crop_size, lane.n_scans - start)
        end = start + length

        features[row, :, :length] = lane.features[:, start:end]
        class_targets[row, :length] = lane.class_targets[start:end]
        boundary_targets[row, :length] = lane.boundary_targets[start:end]
        valid_scan_mask[row, :length] = 1

    hidden_position = rng.random(batch_size) < RUN_FRAC_DROPOUT
    features[hidden_position, -1] = -1.0
    return _TrainingBatch(
        features=features,
        class_targets=class_targets,
        boundary_targets=boundary_targets,
        valid_scan_mask=valid_scan_mask,
    )


def _calculate_training_loss(logits, batch: _TrainingBatch, class_weights):
    import torch
    import torch.nn.functional as F

    class_loss = F.cross_entropy(
        logits[:, :3],
        torch.from_numpy(batch.class_targets),
        weight=class_weights,
        reduction="none",
    )
    boundary_loss = F.binary_cross_entropy_with_logits(
        logits[:, 3],
        torch.from_numpy(batch.boundary_targets),
        pos_weight=torch.tensor(20.0),
        reduction="none",
    )
    valid_scan_mask = torch.from_numpy(batch.valid_scan_mask)
    losses = (class_loss + boundary_loss) * valid_scan_mask
    return losses.sum() / valid_scan_mask.sum()


def _torch_net():
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(N_IN, CHANNELS, 1)
            self.blocks = nn.ModuleList(
                nn.Conv1d(
                    CHANNELS,
                    CHANNELS,
                    KERNEL,
                    dilation=dilation,
                    padding=dilation * (KERNEL - 1) // 2,
                )
                for dilation in DILATIONS
            )
            self.out = nn.Conv1d(CHANNELS, 4, 1)

        def forward(self, features):
            hidden = self.inp(features)
            for block in self.blocks:
                hidden = hidden + block(hidden).relu()
            return self.out(hidden)

    return Net()
