"""
Schema for chromatographic span labels used for training etc.
See docs/labeling-policy.md

The annotator marks spans of `peak`  and `plateau`
( i.e. persistent/smeared/streaking/background whine, excluded from recruitment)

Each labeled lane is self-contained JSON

Everything unmarked is other, recruitable but not seed features.
Saving a lane asserts it was reviewed end to end.
"""
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 4

PEAK = "peak"
PLATEAU = "plateau"
SPAN_CLASSES = (PEAK, PLATEAU)

Span = tuple[int, int, str]  # [start, end) scan offsets into the lane, class


@dataclass
class LaneLabel:
    mz: float
    lane_idx: int
    rt_values: list[float]
    intsy_values: list[float]
    spans: list[Span] = field(default_factory=list)

    # Provenance: SampleUUIDs are regenerated on every import,
    # so a mismatch catches both "wrong .mzk" and "re-imported .mzk".
    sample_uuid: int = 0
    sample_name: str = ""
    annotator: str = ""
    timestamp: str = ""
    session_id: str = ""


@dataclass
class LabelFile:
    extraction_params: dict  # parent_threshold, min_intsy, ... reproduction info
    sample_uuids: list[int] = field(default_factory=list)
    lanes: list[LaneLabel] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_json(
            self,
            path: Path
    ) -> None:
        """
        Save as a .json file
        """
        payload = {
            "schema_version": self.schema_version,
            "extraction_params": self.extraction_params,
            # uuid4().int exceeds the JS safe-integer range: store as strings
            "sample_uuids": [str(u) for u in self.sample_uuids],
            "lanes": [_lane_to_dict(lbl) for lbl in self.lanes],
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def from_json(
            cls,
            path: Path
    ) -> "LabelFile":
        """
        Load as a .json file
        """
        data = json.loads(path.read_text())
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported label schema version {data.get('schema_version')} "
                f"(expected {SCHEMA_VERSION}: whole-lane span labels)"
            )
        return cls(
            extraction_params=data["extraction_params"],
            sample_uuids=[int(u) for u in data.get("sample_uuids", [])],
            lanes=[_lane_from_dict(d) for d in data["lanes"]],
        )

    def has_lane(
            self,
            sample_uuid: int,
            lane_idx: int
    ) -> bool:
        return any(
            lbl.sample_uuid == sample_uuid and lbl.lane_idx == lane_idx
            for lbl in self.lanes
        )


def insert_span(
        spans: list[Span],
        start: int,
        end: int,
        cls: str,
) -> list[Span]:
    """
    New span overwrites whatever it overlaps (trimming/splitting neighbours)
    """
    out: list[Span] = []
    for s, e, c in spans:
        if e <= start or s >= end:
            out.append((s, e, c))
            continue
        if s < start:
            out.append((s, start, c))
        if e > end:
            out.append((end, e, c))
    if end > start:
        out.append((start, end, cls))
    return sorted(out)


def _lane_to_dict(
        lbl: LaneLabel
) -> dict[str, Any]:
    d = asdict(lbl)
    d["spans"] = [list(s) for s in lbl.spans]
    d["sample_uuid"] = str(lbl.sample_uuid)
    return d


def _lane_from_dict(
        d: dict[str, Any]
) -> LaneLabel:
    return LaneLabel(
        mz=d["mz"],
        lane_idx=d["lane_idx"],
        rt_values=list(d["rt_values"]),
        intsy_values=list(d["intsy_values"]),
        spans=[(int(s), int(e), str(c)) for s, e, c in d.get("spans", [])],
        sample_uuid=int(d.get("sample_uuid", 0)),
        sample_name=d.get("sample_name", ""),
        annotator=d.get("annotator", ""),
        timestamp=d.get("timestamp", ""),
        session_id=d.get("session_id", ""),
    )
