"""
Export an EnsembleAlignment as a feature table (CSV/TSV), optionally with
a companion .mgf of the aligned spectra.

Rows are analytes, columns are samples, values are base intensity.

The MGF is built via the shared, Qt-free `core.cli.export_ensemble` machinery
(same code path as the single-compound / SIRIUS exports), so ensemble metadata
(identity -> NAME, formula -> FORMULA, adduct/charge, and any user_metadata)
comes along for free. Two modes:

- ``consensus`` (default): one MGF entry per analyte, using the most-intense
  ensemble (highest base intensity) across all samples it was detected in.
- ``per_sample``: one MGF entry per (analyte, sample) it was detected in, i.e.
  every ensemble participating in the alignment.

In both modes each entry is stamped with ``FEATURE_ID`` = the analyte's row id,
so the table's ``analyte_id`` column cross-references the MGF (GNPS-FBMN style).
"""
import logging
from pathlib import Path
from typing import Literal, Mapping, Optional, TYPE_CHECKING

from core.cli.export_ensemble import build_ensemble_export

if TYPE_CHECKING:
    from core.data_structs import Sample, SampleUUID, Ensemble
    from core.data_structs.uuid_types import EnsembleUUID
    from core.data_structs.alignment import EnsembleAlignment

logger = logging.getLogger(__name__)

MgfMode = Literal['consensus', 'per_sample']


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


def _best_ensemble(
    analyte,
    samples: dict['SampleUUID', 'Sample'],
) -> Optional['Ensemble']:
    """
    Most-intense ensemble (by base intensity) across every sample the
    analyte was detected in, or None.
    """
    best_ensemble = None
    best_intsy = -1.0
    for sample_uuid, ens_uuid in analyte.ensemble_map.items():
        sample = samples.get(sample_uuid)
        if not sample or not sample.injection:
            continue
        ensemble = sample.injection.ensembles.get(ens_uuid)
        if not ensemble:
            continue
        if ensemble.base_intsy > best_intsy:
            best_intsy = ensemble.base_intsy
            best_ensemble = ensemble
    return best_ensemble


def export_feature_mgf(
    alignment: 'EnsembleAlignment',
    samples: dict['SampleUUID', 'Sample'],
    mode: MgfMode = 'consensus',
    normalize: bool = True,
    formulas: Optional[Mapping['EnsembleUUID', str]] = None,
) -> str:
    """
    Build an MGF string for an alignment's aligned spectra.

    :param mode: ``'consensus'`` for one entry per analyte (best ensemble
        across samples) or ``'per_sample'`` for one entry per aligned
        (analyte, sample).
    :param normalize: normalize spectra to 0-100.
    :param formulas: accepted formula per ensemble (e.g.
        DataRegistry.chosen_formulas()), written as FORMULA tags.
    :return: MGF text (empty string if nothing exportable).
    """
    blocks: list[str] = []
    formulas = formulas or {}

    for i, analyte in enumerate(alignment.analytes):
        if mode == 'consensus':
            ensemble = _best_ensemble(analyte, samples)
            if ensemble is None:
                continue
            export = build_ensemble_export(
                ensemble, rt=None, normalize=normalize,
                formula=formulas.get(ensemble.uuid),
            )
            export.metadata['FEATURE_ID'] = str(i)
            text = export.to_mgf_text()
            if text:
                blocks.append(text)
        else:  # per_sample
            for sample_uuid, ens_uuid in analyte.ensemble_map.items():
                sample = samples.get(sample_uuid)
                if not sample or not sample.injection:
                    continue
                ensemble = sample.injection.ensembles.get(ens_uuid)
                if not ensemble:
                    continue
                export = build_ensemble_export(
                    ensemble, rt=None, normalize=normalize,
                    formula=formulas.get(ensemble.uuid),
                )
                export.metadata['FEATURE_ID'] = str(i)
                export.metadata['SAMPLE'] = sample.name
                text = export.to_mgf_text()
                if text:
                    blocks.append(text)

    return '\n\n'.join(blocks) + '\n' if blocks else ''


def export_feature_table_to_file(
    alignment: 'EnsembleAlignment',
    samples: dict['SampleUUID', 'Sample'],
    sample_names: dict['SampleUUID', str],
    output: Path,
    separator: str = '\t',
    write_mgf: bool = True,
    mgf_mode: MgfMode = 'consensus',
    normalize: bool = True,
    formulas: Optional[Mapping['EnsembleUUID', str]] = None,
) -> Optional[Path]:
    """
    Export a feature table to a file, and (by default) a companion .mgf.

    The MGF is written as a sibling of ``output`` with a ``.mgf`` suffix
    (``features.tsv`` -> ``features.mgf``).

    :param write_mgf: also write the companion MGF.
    :param mgf_mode: ``'consensus'`` (best ensemble per analyte) or
        ``'per_sample'`` (every aligned ensemble).
    :param normalize: normalize MGF spectra to 0-100.
    :return: the MGF path if one was written, else None.
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

    if not write_mgf:
        return None

    mgf_text = export_feature_mgf(
        alignment=alignment,
        samples=samples,
        mode=mgf_mode,
        normalize=normalize,
        formulas=formulas,
    )
    mgf_path = output.with_suffix('.mgf')
    mgf_path.write_text(mgf_text)
    logger.info(
        f"Exported {mgf_mode} MGF to {mgf_path}"
    )
    return mgf_path
