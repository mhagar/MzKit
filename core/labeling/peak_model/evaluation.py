"""Evaluate peak models, produce training reports, and review labeled predictions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.labeling.schema import PEAK, PLATEAU, LabelFile, Span

from .inference import PLATEAU_C, lane_targets, segment_lane
from .persistence import save_model
from .training import LaneExample, load_examples, train


@dataclass
class MatchCounts:
    true_positives: int = 0
    n_predicted: int = 0
    n_expected: int = 0

    def __add__(self, other: MatchCounts) -> MatchCounts:
        return MatchCounts(
            true_positives=self.true_positives + other.true_positives,
            n_predicted=self.n_predicted + other.n_predicted,
            n_expected=self.n_expected + other.n_expected,
        )


def evaluate(model: dict, examples, parent_threshold: float) -> list[str]:
    model_matches = MatchCounts()
    rule_matches = MatchCounts()
    plateau_intersection = 0
    plateau_union = 0
    n_below_threshold = 0

    for value in examples:
        example = LaneExample.from_value(value)
        intsy = example.intsy_values
        labeled_peaks = [
            (start, end)
            for start, end, kind in example.spans
            if kind == PEAK
        ]
        expected_peaks = seedable_peak_bounds(
            intsy,
            example.spans,
            parent_threshold,
        )
        n_below_threshold += len(labeled_peaks) - len(expected_peaks)

        class_ids, predicted_spans = segment_lane(
            model,
            intsy,
            example.rt_values,
        )
        predicted_peaks = seedable_peak_bounds(
            intsy,
            predicted_spans,
            parent_threshold,
        )
        model_matches += _peak_match_counts(predicted_peaks, expected_peaks)
        rule_peaks = _rules_peaks(intsy, parent_threshold)
        rule_matches += _peak_match_counts(rule_peaks, expected_peaks)

        target_classes, _ = lane_targets(intsy.size, example.spans)
        predicted_plateau = class_ids == PLATEAU_C
        target_plateau = target_classes == PLATEAU_C
        plateau_intersection += int((predicted_plateau & target_plateau).sum())
        plateau_union += int((predicted_plateau | target_plateau).sum())

    plateau_overlap = (
        f"{plateau_intersection / plateau_union:.2f}"
        if plateau_union else "n/a (none labeled)"
    )
    return [
        (
            f"  (scored on peaks >= parent_threshold={parent_threshold:g}, "
            f"the ones that can seed; {n_below_threshold} labeled peaks below it ignored)"
        ),
        f"  peaks  model: {_format_f1(model_matches)}",
        f"  peaks  rules: {_format_f1(rule_matches)}",
        f"  plateau overlap (model): {plateau_overlap}",
    ]


def seedable_peak_bounds(
    intsy: np.ndarray,
    spans: list[Span],
    parent_threshold: float,
) -> list[tuple[int, int]]:
    return [
        (start, end)
        for start, end, kind in spans
        if kind == PEAK and intsy[start:end].max() >= parent_threshold
    ]


def _rules_peaks(
    intsy: np.ndarray,
    parent_threshold: float,
) -> list[tuple[int, int]]:
    from core.cli.segment_chromatogram import (
        find_peak_boundaries,
        is_peak,
        survives_smoothing,
    )

    remaining = np.asarray(intsy, float).copy()
    peaks = []
    while remaining.size and remaining.max() >= parent_threshold:
        apex = int(remaining.argmax())
        start, end = find_peak_boundaries(remaining, apex)
        end = max(end, apex + 1)
        if (
            is_peak(
                remaining,
                apex,
                start,
                end,
                min_peak_width=5,
                min_prominence=0.2,
                baseline_intensity=intsy,
            )
            and survives_smoothing(
                remaining,
                apex,
                start,
                end,
                baseline_intensity=intsy,
            )
        ):
            peaks.append((start, end))
        remaining[start:end] = 0
    return peaks


def _peak_match_counts(
    predicted: list,
    expected: list,
) -> MatchCounts:
    used = set()
    true_positives = 0
    for predicted_start, predicted_end in predicted:
        for index, (expected_start, expected_end) in enumerate(expected):
            if index in used:
                continue
            intersection = max(
                0,
                min(predicted_end, expected_end) - max(predicted_start, expected_start),
            )
            union = max(predicted_end, expected_end) - min(predicted_start, expected_start)
            if intersection / union >= 0.5:
                used.add(index)
                true_positives += 1
                break
    return MatchCounts(true_positives, len(predicted), len(expected))


def _format_f1(counts: MatchCounts) -> str:
    precision = (
        counts.true_positives / counts.n_predicted
        if counts.n_predicted else float("nan")
    )
    recall = (
        counts.true_positives / counts.n_expected
        if counts.n_expected else float("nan")
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return f"precision {precision:.2f}  recall {recall:.2f}  F1 {f1:.2f}"


def train_and_report(
    label_paths: list[Path],
    out_path: Path,
    holdout: float = 0.2,
    seed: int = 0,
    progress_callback=None,  # injected by ProcessRunner; unused here
    cancel_event=None,
) -> str:
    examples = load_examples(label_paths)
    params = LabelFile.from_json(Path(label_paths[0])).extraction_params
    parent_threshold = float(params.get("parent_threshold", 1e4))
    n_peaks = sum(
        kind == PEAK
        for example in examples
        for *_, kind in example.spans
    )
    n_plateaus = sum(
        kind == PLATEAU
        for example in examples
        for *_, kind in example.spans
    )

    lines = [
        f"{len(examples)} lanes: {n_peaks} peak spans, {n_plateaus} plateau spans",
        (
            "(lanes picked by model uncertainty are harder: scores aren't "
            "comparable across label sets, only model vs rules on the same lanes)"
        ),
    ]
    if len(examples) >= 10:
        training_examples, holdout_examples = _split_holdout(
            examples,
            holdout,
            seed,
        )
        lines.append(
            f"Held-out ({len(holdout_examples)} lanes; peak match = overlap >= 50%):"
        )
        holdout_model = train(
            training_examples,
            seed=seed,
            cancel_event=cancel_event,
        )
        lines += evaluate(holdout_model, holdout_examples, parent_threshold)
    else:
        lines.append("(<10 lanes: skipping held-out evaluation)")

    model = train(examples, seed=seed, cancel_event=cancel_event)
    save_model(model, out_path)
    lines.append(f"Saved model -> {out_path}")
    return "\n".join(lines)


def _split_holdout(
    examples: list[LaneExample],
    holdout: float,
    seed: int,
) -> tuple[list[LaneExample], list[LaneExample]]:
    indices = np.random.default_rng(seed).permutation(len(examples))
    n_holdout = max(2, int(len(examples) * holdout))
    training = [examples[index] for index in indices[n_holdout:]]
    holdout_examples = [examples[index] for index in indices[:n_holdout]]
    return training, holdout_examples


def review_predictions(
    lanes: list,
    parent_threshold: float,
    n_folds: int = 5,
    seed: int = 0,
    progress_callback=None,
    cancel_event=None,
) -> list[dict]:
    if len(lanes) < 2 * n_folds:
        raise ValueError(f"Need at least {2 * n_folds} labeled lanes to review")

    examples = [LaneExample.from_label(label) for label in lanes]
    fold_assignments = np.random.default_rng(seed).permutation(len(examples)) % n_folds
    reviews = []

    for fold in range(n_folds):
        if cancel_event is not None and cancel_event.is_set():
            break
        training_examples = [
            example
            for example, assignment in zip(examples, fold_assignments)
            if assignment != fold
        ]
        model = train(
            training_examples,
            seed=seed,
            cancel_event=cancel_event,
        )
        for index in np.flatnonzero(fold_assignments == fold):
            review = _review_example(
                lanes[index],
                examples[index],
                model,
                parent_threshold,
            )
            if review is not None:
                reviews.append(review)
        if progress_callback is not None:
            progress_callback(
                100.0 * (fold + 1) / n_folds,
                f"fold {fold + 1}/{n_folds}",
            )

    return sorted(reviews, key=lambda review: -review["score"])


def _review_example(
    label,
    example: LaneExample,
    model: dict,
    parent_threshold: float,
) -> dict | None:
    class_ids, predicted_spans = segment_lane(
        model,
        example.intsy_values,
        example.rt_values,
    )
    predicted_peaks = seedable_peak_bounds(
        example.intsy_values,
        predicted_spans,
        parent_threshold,
    )
    expected_peaks = seedable_peak_bounds(
        example.intsy_values,
        example.spans,
        parent_threshold,
    )
    matches = _peak_match_counts(predicted_peaks, expected_peaks)
    false_positives = matches.n_predicted - matches.true_positives
    false_negatives = matches.n_expected - matches.true_positives

    target_classes, _ = lane_targets(example.intsy_values.size, example.spans)
    plateau_diff = int(((class_ids == PLATEAU_C) != (target_classes == PLATEAU_C)).sum())
    score = false_positives + false_negatives + plateau_diff / 20
    if score <= 0:
        return None

    return {
        "key": (label.sample_uuid, label.lane_idx),
        "pred": predicted_spans,
        "fp": false_positives,
        "fn": false_negatives,
        "plateau_diff": plateau_diff,
        "score": score,
    }
