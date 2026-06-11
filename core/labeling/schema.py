"""
Schema for peak-morphology labels.

Labels are stored as JSON for portability — they should be
usable outside the MzKit ecosystem (e.g., as an ML training set).
Each label is self-contained: it carries the XIC window the
annotator actually saw, so the label is reproducible without
re-extracting from the source mzML.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2


class MorphologyClass(str, Enum):
    NOISE = "noise"
    SHARP = "sharp"
    TAILING = "tailing"
    FRONTING = "fronting"
    BROAD = "broad"
    MULTI_PEAK = "multi-peak"
    SATURATED = "saturated-peak"


@dataclass
class Label:
    # Identity within the source file
    mz: float
    rt_apex: float
    intensity_apex: float
    apex_scan_idx: int

    # XIC window shown to the annotator. Arrays are stored directly
    # so the label can be consumed without the original mzML.
    window_start_scan: int
    window_end_scan: int
    rt_values: list[float]
    intsy_values: list[float]

    # The label itself
    morphology: MorphologyClass

    # For multi-peak only: scan indices (relative to the window,
    # i.e. offsets into rt_values/intsy_values) where the annotator
    # drew separators between sub-peaks.
    boundary_splits: list[int] = field(default_factory=list)

    # Per-label provenance — which sample this candidate came from.
    # SampleUUID is an int (uuid4().int), regenerated on every import,
    # so it also catches re-imported-into-new-mzk mistakes without
    # needing content hashing.
    sample_uuid: int = 0
    sample_name: str = ""

    annotator: str = ""
    timestamp: str = ""
    session_id: str = ""
    notes: str = ""


@dataclass
class LabelFile:
    extraction_params: dict        # mz_window, etc. — reproduction info
    sample_uuids: list[int] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_json(self, path: Path) -> None:
        payload = {
            "schema_version": self.schema_version,
            "extraction_params": self.extraction_params,
            # JSON keys must be strings; uuid4().int exceeds JS number
            # safe range anyway, so store as strings to be explicit.
            "sample_uuids": [str(u) for u in self.sample_uuids],
            "labels": [_label_to_dict(lbl) for lbl in self.labels],
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "LabelFile":
        data = json.loads(path.read_text())
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {data.get('schema_version')}"
            )
        return cls(
            extraction_params=data["extraction_params"],
            sample_uuids=[int(u) for u in data.get("sample_uuids", [])],
            labels=[_label_from_dict(d) for d in data["labels"]],
            schema_version=data["schema_version"],
        )

    def has_label_for(self, sample_uuid: int, mz: float,
                      apex_scan_idx: int, mz_tol: float = 1e-4) -> bool:
        for lbl in self.labels:
            if (lbl.sample_uuid == sample_uuid
                    and lbl.apex_scan_idx == apex_scan_idx
                    and abs(lbl.mz - mz) < mz_tol):
                return True
        return False


def _label_to_dict(lbl: Label) -> dict[str, Any]:
    d = asdict(lbl)
    d["morphology"] = lbl.morphology.value
    d["sample_uuid"] = str(lbl.sample_uuid)
    return d


def _label_from_dict(d: dict[str, Any]) -> Label:
    return Label(
        mz=d["mz"],
        rt_apex=d["rt_apex"],
        intensity_apex=d["intensity_apex"],
        apex_scan_idx=d["apex_scan_idx"],
        window_start_scan=d["window_start_scan"],
        window_end_scan=d["window_end_scan"],
        rt_values=list(d["rt_values"]),
        intsy_values=list(d["intsy_values"]),
        morphology=MorphologyClass(d["morphology"]),
        boundary_splits=list(d.get("boundary_splits", [])),
        sample_uuid=int(d.get("sample_uuid", 0)),
        sample_name=d.get("sample_name", ""),
        annotator=d.get("annotator", ""),
        timestamp=d.get("timestamp", ""),
        session_id=d.get("session_id", ""),
        notes=d.get("notes", ""),
    )
