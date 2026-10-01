"""
Export an EnsembleAlignment as feature tables (CSV/TSV), with a companion
.mgf of the aligned spectra and a .graphml MS2 similarity network.

From an output path like `features.tsv`, this writes:
- `features_abundance.tsv`:
    rows = analytes
    cols = samples
    vals = base intensity
        ('0' = not detected; empty = detected, but its ensemble
                couldn't be resolved).

- `features_formulas.tsv`:
    each analyte's consensus formula (highest `count` of `total`)
    and per-sample columns w/ accepted formula,
        '0' if present without one, empty if absent.

- `features.mgf`: see below
- `features.graphml`: see `core.cli.export_network`.

The MGF is built via `core.cli.export_ensemble` (same as single cmpd exports)
Two modes:
- `consensus` (default):
    one MGF entry per analyte: its RepresentativeSpectrum (see alignment.py),
     with theconsensus formula as FORMULA

- `per_sample`: one entry per analyte per sample it was detected in

Each analyte's row id is recorded as `FEATURE_ID` to cross-reference the MGF
(GNPS-FBMN style)
"""
import logging
import threading
from pathlib import Path
from typing import Callable, Literal, Mapping, Optional, TYPE_CHECKING

from core.cli.cluster_analytes import ClusterParams
from core.cli.export_ensemble import build_ensemble_export
from core.cli.export_network import build_network_graphml

if TYPE_CHECKING:
    from core.data_structs import Sample, SampleUUID
    from core.data_structs.uuid_types import EnsembleUUID
    from core.data_structs.alignment import EnsembleAlignment

logger = logging.getLogger(__name__)

MgfMode = Literal['consensus', 'per_sample']


def _ordered_samples(
    alignment: 'EnsembleAlignment',
    sample_names: dict['SampleUUID', str],
) -> tuple[list['SampleUUID'], list[str]]:
    """The alignment's sample uuids that have a name, and those names."""
    uuids = [u for u in alignment.sample_uuids if u in sample_names]
    return uuids, [sample_names[u] for u in uuids]


def export_abundance_table(
    alignment: 'EnsembleAlignment',
    samples: dict['SampleUUID', 'Sample'],
    sample_names: dict['SampleUUID', str],
    separator: str = '\t',
) -> str:
    """
    Analyte x sample table of base intensities.

    :param samples: Mapping of SampleUUID -> Sample (with Injections)
    :param sample_names: Mapping of SampleUUID -> display name
    :param separator: Column separator (tab or comma)
    """
    ordered_uuids, ordered_names = _ordered_samples(alignment, sample_names)

    lines = [separator.join(
        ['analyte_id', 'consensus_mz', 'consensus_rt'] + ordered_names
    )]
    for i, analyte in enumerate(alignment.analytes):
        members = analyte.resolve_members(samples.get)
        row = [
            str(i),
            f"{analyte.consensus_mz:.5f}",
            f"{analyte.consensus_rt:.1f}",
        ]
        for uuid in ordered_uuids:
            if uuid not in analyte.ensemble_map:
                row.append('0')
            elif uuid in members:
                row.append(f"{members[uuid].base_intsy:.1f}")
            else:
                row.append('')
        lines.append(separator.join(row))

    return '\n'.join(lines) + '\n'


def export_formula_table(
    alignment: 'EnsembleAlignment',
    samples: dict['SampleUUID', 'Sample'],
    sample_names: dict['SampleUUID', str],
    formulas: Optional[Mapping['EnsembleUUID', str]] = None,
    separator: str = '\t',
) -> str:
    """
    Analyte x sample table of accepted formulas, led by each analyte's
    consensus formula (see `AlignedAnalyte.consensus_formula`).

    :param formulas: accepted formula per ensemble
        (e.g. DataRegistry.chosen_formulas()).
    """
    formulas = formulas or {}
    ordered_uuids, ordered_names = _ordered_samples(alignment, sample_names)

    lines = [separator.join(
        ['analyte_id', 'consensus_mz', 'consensus_rt',
         'consensus_formula', 'count', 'total'] + ordered_names
    )]
    for i, analyte in enumerate(alignment.analytes):
        members = analyte.resolve_members(samples.get)
        consensus = analyte.consensus_formula(members, formulas)
        row = [
            str(i),
            f"{analyte.consensus_mz:.5f}",
            f"{analyte.consensus_rt:.1f}",
            consensus.formula or '',
            str(consensus.count),
            str(consensus.total),
        ]
        for uuid in ordered_uuids:
            ensemble = members.get(uuid)
            if ensemble is None:
                row.append('')
            else:
                row.append(formulas.get(ensemble.uuid, '0'))
        lines.append(separator.join(row))

    return '\n'.join(lines) + '\n'


def export_feature_mgf(
    alignment: 'EnsembleAlignment',
    samples: dict['SampleUUID', 'Sample'],
    mode: MgfMode = 'consensus',
    normalize: bool = True,
    formulas: Optional[Mapping['EnsembleUUID', str]] = None,
) -> str:
    """
    Build an MGF string for an alignment's aligned spectra.

    :param mode: ``'consensus'`` for one entry per analyte (its
        RepresentativeSpectrum) or ``'per_sample'`` for one entry per
        aligned (analyte, sample).
    :param normalize: normalize spectra to 0-100.
    :param formulas: accepted formula per ensemble (e.g.
        DataRegistry.chosen_formulas()), written as FORMULA tags (the
        consensus formula in consensus mode).
    :return: MGF text (empty string if nothing exportable).
    """
    blocks: list[str] = []
    formulas = formulas or {}

    for i, analyte in enumerate(alignment.analytes):
        members = analyte.resolve_members(samples.get)

        if mode == 'consensus':
            rep = analyte.representative_spectrum(members)
            if rep is None:
                continue
            export = build_ensemble_export(
                members[rep.sample_uuid],
                composite=rep.composite,
                normalize=normalize,
                formula=analyte.consensus_formula(members, formulas).formula,
            )
            export.metadata['FEATURE_ID'] = str(i)
            text = export.to_mgf_text()
            if text:
                blocks.append(text)
        else:  # per_sample
            for sample_uuid, ensemble in members.items():
                export = build_ensemble_export(
                    ensemble, rt=None, normalize=normalize,
                    formula=formulas.get(ensemble.uuid),
                )
                export.metadata['FEATURE_ID'] = str(i)
                export.metadata['SAMPLE'] = samples[sample_uuid].name
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
    write_graphml: bool = True,
    network_params: ClusterParams = ClusterParams(),
    min_edge_cosine: float = 0.7,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[Path]:
    """
    Export the abundance + formula tables, and (by default) the companion
    .mgf and .graphml, all named after ``output``
    (``features.tsv`` -> ``features_abundance.tsv``,
    ``features_formulas.tsv``, ``features.mgf``, ``features.graphml``).

    ProcessController-compatible (the network scoring is the slow part).

    :param separator: ``','`` writes .csv tables, anything else .tsv.
    :param write_mgf: also write the companion MGF.
    :param mgf_mode: ``'consensus'`` (RepresentativeSpectrum per analyte)
        or ``'per_sample'`` (every aligned ensemble).
    :param normalize: normalize MGF spectra to 0-100.
    :param write_graphml: also write the MS2 similarity network.
    :param network_params: network scoring parameters
        (see `cluster_params_from_config`).
    :param min_edge_cosine: network edges scoring below this are dropped.
    :return: the written paths (the .graphml is skipped if cancelled).
    """
    stem = output.with_suffix('').name
    suffix = '.csv' if separator == ',' else '.tsv'
    written: list[Path] = []

    def sibling(name: str) -> Path:
        return output.with_name(f"{stem}{name}")

    def write(path: Path, text: str) -> None:
        path.write_text(text)
        written.append(path)
        logger.info(f"Exported {path}")

    write(
        sibling(f"_abundance{suffix}"),
        export_abundance_table(alignment, samples, sample_names, separator),
    )
    write(
        sibling(f"_formulas{suffix}"),
        export_formula_table(
            alignment, samples, sample_names, formulas, separator,
        ),
    )

    if write_mgf:
        write(
            sibling('.mgf'),
            export_feature_mgf(
                alignment=alignment,
                samples=samples,
                mode=mgf_mode,
                normalize=normalize,
                formulas=formulas,
            ),
        )

    if write_graphml:
        graphml = build_network_graphml(
            alignment,
            list(samples.values()),
            params=network_params,
            formulas=formulas,
            min_edge_cosine=min_edge_cosine,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if graphml is None:
            logger.info("Cancelled; skipped the .graphml")
        else:
            write(sibling('.graphml'), graphml)

    return written
