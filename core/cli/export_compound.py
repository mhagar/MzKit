"""
Export XIC + spectra for a single analyte from an EnsembleAlignment.

The per-ensemble spectrum/MGF/JSON building is delegated to
`core.cli.export_ensemble` (the single source of truth shared with the
GUI). This module adds the alignment-specific concerns: picking the
most-intense ensemble per analyte, the cross-sample XIC ("detected_in")
summary, and batch export of every analyte.
"""
import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from core.cli.export_ensemble import build_ensemble_export

if TYPE_CHECKING:
    from core.data_structs import Sample, SampleUUID, Ensemble
    from core.data_structs.alignment import EnsembleAlignment
    from core.cli.export_ensemble import EnsembleExport

logger = logging.getLogger(__name__)


def _best_ensemble(
    alignment: 'EnsembleAlignment',
    analyte_index: int,
    samples: dict['SampleUUID', 'Sample'],
) -> Optional['Ensemble']:
    """
    Return the most-intense ensemble (by base intensity) across all
    samples in which the analyte was detected, or None.
    """
    best_ensemble = None
    best_intsy = -1.0

    analyte = alignment.analytes[analyte_index]
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


def _build(
    alignment: 'EnsembleAlignment',
    analyte_index: int,
    samples: dict['SampleUUID', 'Sample'],
    normalize: bool,
) -> Optional['EnsembleExport']:
    """
    Build the shared EnsembleExport for an analyte's best ensemble, with
    FEATURE_ID overridden to the analyte index (for downstream tooling).
    """
    best = _best_ensemble(alignment, analyte_index, samples)
    if best is None:
        return None

    export = build_ensemble_export(best, rt=None, normalize=normalize)
    export.metadata['FEATURE_ID'] = str(analyte_index)
    return export


def export_compound(
    alignment: 'EnsembleAlignment',
    analyte_index: int,
    samples: dict['SampleUUID', 'Sample'],
    normalize: bool = True,
) -> dict:
    """
    Build compound data for a single analyte in an alignment.

    Combines the shared per-ensemble export (spectra + metadata) with an
    alignment-specific per-sample XIC summary.

    :param normalize: If True, normalize spectra to 0-100.
    :return: Dict ready for JSON serialization.
    """
    analyte = alignment.analytes[analyte_index]

    detected_in = []
    for sample_uuid, ens_uuid in analyte.ensemble_map.items():
        sample = samples.get(sample_uuid)
        if not sample or not sample.injection:
            continue
        ensemble = sample.injection.ensembles.get(ens_uuid)
        if not ensemble:
            continue

        xic = ensemble.get_base_chromatogram(ms_level=1)
        detected_in.append({
            'sample_name': sample.name,
            'base_mz': float(ensemble.base_mz),
            'peak_rt': float(ensemble.peak_rt),
            'base_intsy': float(ensemble.base_intsy),
            'xic_rt': xic['rt'].tolist(),
            'xic_intsy': xic['intsy'].tolist(),
        })

    export = _build(alignment, analyte_index, samples, normalize)
    ms1_spectrum = None
    ms2_spectrum = None
    if export is not None:
        spec_json = export.to_json_obj()
        ms1_spectrum = spec_json['ms1_spectrum']
        ms2_spectrum = spec_json['ms2_spectrum']

    return {
        'analyte_index': analyte_index,
        'consensus_mz': analyte.consensus_mz,
        'consensus_rt': analyte.consensus_rt,
        'detected_in': detected_in,
        'ms1_spectrum': ms1_spectrum,
        'ms2_spectrum': ms2_spectrum,
    }


def export_compound_mgf(
    alignment: 'EnsembleAlignment',
    analyte_index: int,
    samples: dict['SampleUUID', 'Sample'],
    normalize: bool = True,
) -> str:
    """
    Build MGF entries (MS1 + MS2) for a single analyte's best ensemble.

    Returns the MGF string (one or two BEGIN IONS blocks), or '' if the
    analyte has no usable ensemble.
    """
    export = _build(alignment, analyte_index, samples, normalize)
    if export is None:
        return ''
    return export.to_mgf_text()


def export_compound_to_file(
    alignment: 'EnsembleAlignment',
    analyte_index: int,
    samples: dict['SampleUUID', 'Sample'],
    output_dir: Path,
    write_json: bool = False,
    normalize: bool = True,
) -> None:
    """
    Export a single compound to MGF (and optionally JSON) in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"compound_{analyte_index:03d}"

    mgf_text = export_compound_mgf(
        alignment, analyte_index, samples, normalize=normalize,
    )
    if mgf_text:
        mgf_path = output_dir / f"{prefix}.mgf"
        mgf_path.write_text(mgf_text)
        logger.info(f"Exported MGF for analyte {analyte_index} to {mgf_path}")

    if write_json:
        data = export_compound(
            alignment, analyte_index, samples, normalize=normalize,
        )
        json_path = output_dir / f"{prefix}.json"
        json_path.write_text(json.dumps(data, indent=2))
        logger.info(
            f"Exported JSON for analyte {analyte_index} "
            f"(m/z {data['consensus_mz']:.4f}) to {json_path}"
        )


def export_all_compounds(
    alignment: 'EnsembleAlignment',
    samples: dict['SampleUUID', 'Sample'],
    output_dir: Path,
    write_json: bool = False,
    normalize: bool = True,
) -> None:
    """
    Export all compounds: single compounds.mgf + optional per-compound JSON
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    mgf_blocks = []
    for i in range(alignment.analyte_count):
        mgf_text = export_compound_mgf(
            alignment, i, samples, normalize=normalize,
        )
        if mgf_text:
            mgf_blocks.append(mgf_text)

        if write_json:
            data = export_compound(
                alignment, i, samples, normalize=normalize,
            )
            json_path = output_dir / f"compound_{i:03d}.json"
            json_path.write_text(json.dumps(data, indent=2))

    mgf_path = output_dir / "compounds.mgf"
    mgf_path.write_text('\n\n'.join(mgf_blocks))
    logger.info(
        f"Exported {len(mgf_blocks)} compounds to {mgf_path}"
    )
    if write_json:
        logger.info(
            f"Exported JSON for {alignment.analyte_count} compounds to {output_dir}"
        )
