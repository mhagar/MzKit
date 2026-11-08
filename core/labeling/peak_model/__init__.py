"""Chromatographic peak segmentation, training, and evaluation."""
from .evaluation import evaluate, review_predictions, train_and_report
from .inference import (
    CHANNELS,
    DILATIONS,
    EXACT,
    KERNEL,
    N_IN,
    OTHER_C,
    PEAK_C,
    PLATEAU_C,
    RF,
    decode,
    forward,
    forward_features,
    lane_features,
    lane_targets,
    lane_uncertainty,
    segment_lane,
)
from .persistence import default_model_path, load_model, save_model
from .segmenter import LaneSegmenter
from .training import RUN_FRAC_DROPOUT, LaneExample, load_examples, train

__all__ = [
    "CHANNELS",
    "DILATIONS",
    "EXACT",
    "KERNEL",
    "N_IN",
    "OTHER_C",
    "PEAK_C",
    "PLATEAU_C",
    "RF",
    "RUN_FRAC_DROPOUT",
    "LaneExample",
    "LaneSegmenter",
    "decode",
    "default_model_path",
    "evaluate",
    "forward",
    "forward_features",
    "lane_features",
    "lane_targets",
    "lane_uncertainty",
    "load_examples",
    "load_model",
    "review_predictions",
    "save_model",
    "segment_lane",
    "train",
    "train_and_report",
]
