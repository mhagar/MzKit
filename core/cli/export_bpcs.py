"""
Export base peak chromatograms for all samples in an .mzk file.
"""
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structs import Sample

logger = logging.getLogger(__name__)


def export_bpcs(
    samples: list['Sample'],
) -> dict:
    """
    Build BPC data for all samples that have an injection with MS1 data.

    :return: Dict ready for JSON serialization.
    """
    result = {'samples': []}

    for sample in samples:
        if not sample.injection or not sample.injection.scan_array_ms1:
            logger.warning(f"Skipping {sample.name}: no MS1 data")
            continue

        bpc = sample.injection.scan_array_ms1.get_bpc()
        result['samples'].append({
            'name': sample.name,
            'rt': bpc['rt'].tolist(),
            'intsy': bpc['intsy'].tolist(),
        })

    return result


def export_bpcs_to_file(
    samples: list['Sample'],
    output: Path,
) -> None:
    data = export_bpcs(samples)
    output.write_text(json.dumps(data, indent=2))
    logger.info(
        f"Exported BPCs for {len(data['samples'])} samples to {output}"
    )
