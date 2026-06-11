"""
Export an EnsembleAlignment as a feature table (CSV/TSV).

Rows are analytes, columns are samples, values are base intensity.
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structs import Sample, SampleUUID
    from core.data_structs.alignment import EnsembleAlignment

logger = logging.getLogger(__name__)


def export_feature_table(
    alignment: 'EnsembleAlignment',
    samples: dict['SampleUUID', 'Sample'],
    sample_names: dict['SampleUUID', str],
    separator: str = '\t',
) -> str:
    """
    Build a feature table string from an alignment and samples.

    :param alignment: The EnsembleAlignment to export
    :param samples: Mapping of SampleUUID -> Sample (with Injections)
    :param sample_names: Mapping of SampleUUID -> display name
    :param separator: Column separator (tab or comma)
    :return: The table as a string
    """
    ordered_uuids = [
        uuid for uuid in alignment.sample_uuids
        if uuid in sample_names
    ]
    ordered_names = [sample_names[uuid] for uuid in ordered_uuids]

    lines = []
    header = ['analyte_id', 'consensus_mz', 'consensus_rt'] + ordered_names
    lines.append(separator.join(header))

    for i, analyte in enumerate(alignment.analytes):
        row = [
            str(i),
            f"{analyte.consensus_mz:.5f}",
            f"{analyte.consensus_rt:.1f}",
        ]
        for uuid in ordered_uuids:
            ens_uuid = analyte.ensemble_map.get(uuid)
            if ens_uuid is None:
                row.append('0')
            else:
                sample = samples.get(uuid)
                if sample and sample.injection:
                    ensemble = sample.injection.ensembles.get(ens_uuid)
                    if ensemble:
                        row.append(f"{ensemble.base_intsy:.1f}")
                    else:
                        row.append('0')
                else:
                    row.append('0')

        lines.append(separator.join(row))

    return '\n'.join(lines) + '\n'


def export_feature_table_to_file(
    alignment: 'EnsembleAlignment',
    samples: dict['SampleUUID', 'Sample'],
    sample_names: dict['SampleUUID', str],
    output: Path,
    separator: str = '\t',
) -> None:
    """
    Export a feature table to a file.
    """
    table = export_feature_table(
        alignment=alignment,
        samples=samples,
        sample_names=sample_names,
        separator=separator,
    )
    output.write_text(table)

    n_analytes = len(alignment.analytes)
    n_samples = len([
        uuid for uuid in alignment.sample_uuids
        if uuid in sample_names
    ])
    logger.info(
        f"Exported {n_analytes} analytes x "
        f"{n_samples} samples to {output}"
    )
