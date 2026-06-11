"""
Filter an EnsembleAlignment by evaluating a Python expression
per AlignedAnalyte.

Available variables in the expression namespace:
    n           - number of samples this analyte was found in
    samples     - set of sample names where it was found
    mz          - consensus m/z
    rt          - consensus RT (seconds)
    intensities - dict of sample name -> base intensity
"""
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structs import Sample, SampleUUID
    from core.data_structs.alignment import (
        EnsembleAlignment, AlignedAnalyte,
    )


@dataclass
class FilterResult:
    alignment: 'EnsembleAlignment'
    total_before: int
    total_after: int
    per_sample_before: dict[str, int] = field(default_factory=dict)
    per_sample_after: dict[str, int] = field(default_factory=dict)

    @property
    def removed(self) -> int:
        return self.total_before - self.total_after

    def format_summary(self) -> str:
        lines = [
            f"Kept {self.total_after} / {self.total_before} analytes "
            f"(removed {self.removed})",
            "",
            f"{'Sample':<30} {'Before':>8} {'After':>8} {'Removed':>8}",
            "-" * 58,
        ]
        for name in sorted(self.per_sample_before):
            before = self.per_sample_before[name]
            after = self.per_sample_after.get(name, 0)
            lines.append(
                f"{name:<30} {before:>8} {after:>8} {before - after:>8}"
            )
        return "\n".join(lines)

    def format_html(self) -> str:
        rows = []
        for name in sorted(self.per_sample_before):
            before = self.per_sample_before[name]
            after = self.per_sample_after.get(name, 0)
            removed = before - after
            rows.append(
                f"<tr><td>{name}</td>"
                f"<td align='right'>{before}</td>"
                f"<td align='right'>{after}</td>"
                f"<td align='right'>{removed}</td></tr>"
            )

        return (
            f"<p>Kept <b>{self.total_after}</b> / {self.total_before} "
            f"analytes (removed {self.removed})</p>"
            f"<table border='1' cellpadding='3' cellspacing='0'>"
            f"<tr><th>Sample</th><th>Before</th>"
            f"<th>After</th><th>Removed</th></tr>"
            f"{''.join(rows)}</table>"
        )


def filter_alignment(
    alignment: 'EnsembleAlignment',
    expression: str,
    sample_names: dict['SampleUUID', str],
    sample_lookup: Optional[dict['SampleUUID', 'Sample']] = None,
) -> FilterResult:
    """
    Filter an EnsembleAlignment by evaluating a Python expression
    per AlignedAnalyte.

    :param alignment: The alignment to filter
    :param expression: Python expression evaluated per analyte.
        Must return a truthy value to keep the analyte.
    :param sample_names: Mapping of SampleUUID -> sample name,
        used to populate the `samples` variable.
    :param sample_lookup: Mapping of SampleUUID -> Sample,
        used to resolve ensemble intensities. If None,
        `intensities` will be an empty dict.
    :return: FilterResult with the new alignment and summary stats
    """
    from core.data_structs.alignment import EnsembleAlignment

    compiled = compile(expression, "<filter>", "eval")

    safe_builtins = {
        "any": any, "all": all, "len": len,
        "min": min, "max": max, "abs": abs,
        "sum": sum, "sorted": sorted,
        "True": True, "False": False, "None": None,
    }

    # Count per-sample presence before filtering
    per_sample_before: dict[str, int] = {
        name: 0 for name in sample_names.values()
    }
    for analyte in alignment.analytes:
        for uuid in analyte.ensemble_map:
            name = sample_names.get(uuid, str(uuid))
            per_sample_before[name] = per_sample_before.get(name, 0) + 1

    # Filter
    kept: list['AlignedAnalyte'] = []
    for analyte in alignment.analytes:
        found_names = {
            sample_names[uuid]
            for uuid in analyte.ensemble_map
            if uuid in sample_names
        }

        # Build intensities dict: sample_name -> base_intsy
        intensities: dict[str, float] = {}
        if sample_lookup:
            for uuid, ens_uuid in analyte.ensemble_map.items():
                sample = sample_lookup.get(uuid)
                if sample and sample.injection:
                    ensemble = sample.injection.ensembles.get(ens_uuid)
                    if ensemble:
                        name = sample_names.get(uuid, str(uuid))
                        intensities[name] = ensemble.base_intsy

        namespace = {
            "n": len(analyte.ensemble_map),
            "samples": found_names,
            "mz": analyte.consensus_mz,
            "rt": analyte.consensus_rt,
            "intensities": intensities,
        }

        try:
            if eval(compiled, {"__builtins__": safe_builtins}, namespace):
                kept.append(analyte)
        except Exception:
            # If expression errors on a particular analyte, skip it
            continue

    # Count per-sample presence after filtering
    per_sample_after: dict[str, int] = {
        name: 0 for name in sample_names.values()
    }
    for analyte in kept:
        for uuid in analyte.ensemble_map:
            name = sample_names.get(uuid, str(uuid))
            per_sample_after[name] = per_sample_after.get(name, 0) + 1

    new_alignment = EnsembleAlignment(
        sample_uuids=alignment.sample_uuids,
        analytes=kept,
        parameters=alignment.parameters,
        name=alignment.name,
    )

    return FilterResult(
        alignment=new_alignment,
        total_before=len(alignment.analytes),
        total_after=len(kept),
        per_sample_before=per_sample_before,
        per_sample_after=per_sample_after,
    )
