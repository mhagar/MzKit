"""
MzKit CLI entry point.

Usage:
    mzkit import-mzml      - Import .mzML files as samples and save to .mzk
    mzkit import-features  - Import a feature table and generate ensembles
    mzkit filter           - Filter an alignment by expression
    mzkit export-table     - Export alignment as a feature table
    mzkit export-bpcs      - Export base peak chromatograms
    mzkit export-compound  - Export XIC + spectra for a single analyte
"""
import argparse
import csv
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

from core.data_structs.alignment import (
    AlignmentParams,
    EnsembleAlignment,
    AlignedAnalyte,
)
from core.cli.export_bpcs import export_bpcs_to_file
from core.cli.export_compound import export_compound_to_file, export_all_compounds
from core.cli.export_table import export_feature_table_to_file
from core.cli.filter_alignment import filter_alignment
from core.cli.import_feature_table import (
    import_feature_table,
    FeatureCoordinate,
    FeatureTableImportParams,
)
from core.cli.mzml_import import main as mzml_import_main
from core.data_structs import DataRegistry
from core.data_structs.scan_array import ScanArrayParameters
from core.utils.persistence import load_project, save_project

logger = logging.getLogger(__name__)


# --- Helpers ---

def _load_registry(*mzk_paths: Path) -> DataRegistry:
    """Load one or more .mzk files into a DataRegistry."""
    registry = DataRegistry()
    for path in mzk_paths:
        logger.info(f"Loading {path}")
        samples, alignments = load_project(path)
        registry.register_samples(samples)
        for alignment in alignments:
            registry.register_alignment(alignment)
    return registry


def _resolve_output(args: argparse.Namespace, input_path: Path) -> Path:
    """Return --output-mzk if given, otherwise the input path (in-place)."""
    if args.output_mzk:
        output = Path(args.output_mzk)
        if output != input_path:
            shutil.copy2(input_path, output)
        return output
    return input_path


def _get_alignment(
    registry: DataRegistry,
    alignment_name: Optional[str] = None,
) -> EnsembleAlignment:
    """Get an alignment from the registry, by name or the only one."""
    uuids = registry.get_all_alignment_uuids()
    if not uuids:
        raise ValueError("No alignments found in .mzk file")

    if alignment_name:
        for uuid in uuids:
            a = registry.get_alignment(uuid)
            if a.name == alignment_name:
                return a
        raise ValueError(
            f"No alignment named '{alignment_name}'. "
            f"Available: {[registry.get_alignment(u).name for u in uuids]}"
        )

    if len(uuids) == 1:
        return registry.get_alignment(uuids[0])

    raise ValueError(
        f"Multiple alignments found; use --alignment-name to select one. "
        f"Available: {[registry.get_alignment(u).name for u in uuids]}"
    )


def _extract_numeric(
    value,
    regex_pattern: Optional[str],
) -> Optional[float]:
    """
    Extract a numeric value from a cell.

    If regex_pattern is provided, apply it to the string
    representation of the value and use the first capture
    group (or the whole match if no groups). Otherwise,
    convert the value directly to float.
    """
    if regex_pattern is None:
        return float(value)

    text = str(value)
    match = re.search(regex_pattern, text)
    if not match:
        return None

    if match.lastindex:
        return float(match.group(1))
    return float(match.group(0))


def _parse_feature_csv(
    filepath: Path,
    mz_col: str = 'mz',
    rt_col: str = 'rt',
    id_col: Optional[str] = None,
    mz_regex: Optional[str] = None,
    rt_regex: Optional[str] = None,
    rt_in_minutes: bool = False,
) -> list[FeatureCoordinate]:
    """
    Parse a CSV/TSV with configurable column names and optional
    regex extraction.

    Auto-detects delimiter from file extension (.tsv -> tab, else comma).
    """
    delimiter = '\t' if filepath.suffix.lower() == '.tsv' else ','

    features = []
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        # Validate that requested columns exist
        if reader.fieldnames is None:
            raise ValueError(f"Empty or unreadable CSV: {filepath}")

        for col_name, col_label in [
            (mz_col, 'mz'), (rt_col, 'rt'),
        ]:
            if col_name not in reader.fieldnames:
                raise ValueError(
                    f"{col_label} column '{col_name}' not found. "
                    f"Available columns: {', '.join(reader.fieldnames)}"
                )

        for row in reader:
            try:
                mz = _extract_numeric(row[mz_col], mz_regex)
                rt = _extract_numeric(row[rt_col], rt_regex)
            except (ValueError, TypeError):
                continue

            if mz is None or rt is None:
                continue

            if rt_in_minutes:
                rt *= 60.0

            analyte_id = ''
            if id_col and id_col in row:
                analyte_id = str(row[id_col])

            features.append(FeatureCoordinate(
                mz=mz, rt=rt, analyte_id=analyte_id,
            ))

    return features


# --- Subcommands ---

def _read_filepath_list(path: Path) -> list[str]:
    """Read a text file containing one filepath per line."""
    lines = path.read_text().strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def cmd_import_mzml(args: argparse.Namespace) -> None:
    # Resolve input filepaths: either direct arguments or from a text file
    if args.input_list:
        input_filepaths = _read_filepath_list(Path(args.input_list))
    else:
        input_filepaths = args.inputs

    if not input_filepaths:
        raise ValueError("No input files specified")

    # Extract sample names from filenames
    if args.name_regex:
        sample_names = []
        for fp in input_filepaths:
            filename = Path(fp).name
            match = re.search(args.name_regex, filename)
            if not match:
                raise ValueError(
                    f"--name-regex did not match filename: {filename}"
                )
            name = match.group(1) if match.lastindex else match.group(0)
            sample_names.append(name)
    else:
        sample_names = [Path(f).stem for f in input_filepaths]

    # Build ScanArrayParameters
    ms1_params = ScanArrayParameters(
        ms_level=1,
        mz_tolerance=args.ms1_mz_tolerance,
        scan_gap_tolerance=args.ms1_scan_gap,
        min_intsy=args.ms1_min_intsy,
        scan_nums=None,
    )

    ms2_params = None
    if args.import_ms2:
        ms2_params = ScanArrayParameters(
            ms_level=2,
            mz_tolerance=args.ms2_mz_tolerance,
            scan_gap_tolerance=args.ms2_scan_gap,
            min_intsy=args.ms2_min_intsy,
            scan_nums=None,
        )

    scan_array_params = (ms1_params, ms2_params)

    samples = mzml_import_main(
        input_filepaths=input_filepaths,
        sample_names=sample_names,
        scan_array_params=scan_array_params,
    )

    if not samples:
        logger.warning("No samples were imported")
        return

    output = Path(args.output)
    registry = DataRegistry()
    registry.register_samples(samples)
    save_project(output, registry)

    logger.info(f"Saved {len(samples)} sample(s) to {output}")


def cmd_import_features(args: argparse.Namespace) -> None:
    # Parse the feature table CSV/TSV
    feature_path = Path(args.features)
    features = _parse_feature_csv(
        filepath=feature_path,
        mz_col=args.mz_col,
        rt_col=args.rt_col,
        id_col=args.id_col,
        mz_regex=args.mz_regex,
        rt_regex=args.rt_regex,
        rt_in_minutes=(args.rt_unit == 'minutes'),
    )
    logger.info(f"Parsed {len(features)} features from {feature_path}")

    mzk_path = Path(args.mzk)
    registry = _load_registry(mzk_path)

    logger.info(f"Loaded {registry.sample_count()} samples")

    params = FeatureTableImportParams(
        rt_window=args.rt_window,
        mz_window=args.mz_window,
        ms1_corr_threshold=args.ms1_corr,
        ms2_corr_threshold=args.ms2_corr,
        min_intsy=args.min_intsy,
        use_rel_intsy=args.use_rel_intsy,
        pregroup=args.pregroup,
        pregroup_rt_tolerance=args.pregroup_rt_tolerance,
        pregroup_corr_threshold=args.pregroup_corr_threshold,
    )

    all_samples = registry.get_all_samples()
    alignment = import_feature_table(features, all_samples, params)

    if args.name:
        alignment = EnsembleAlignment(
            sample_uuids=alignment.sample_uuids,
            analytes=alignment.analytes,
            parameters=alignment.parameters,
            uuid=alignment.uuid,
            name=args.name,
        )

    registry.register_alignment(alignment)

    output = _resolve_output(args, mzk_path)
    save_project(output, registry)

    logger.info(
        f"Saved to {output}: "
        f"{alignment.analyte_count} analytes across "
        f"{alignment.sample_count} samples"
    )


def cmd_filter(args: argparse.Namespace) -> None:
    mzk_path = Path(args.mzk)
    registry = _load_registry(mzk_path)

    alignment = _get_alignment(registry, args.alignment_name)

    sample_names = {
        s.uuid: s.name for s in registry.get_all_samples()
    }
    sample_lookup = {
        s.uuid: s for s in registry.get_all_samples()
    }

    result = filter_alignment(
        alignment=alignment,
        expression=args.expression,
        sample_names=sample_names,
        sample_lookup=sample_lookup,
    )

    # Replace the old alignment with the filtered one
    registry.remove_alignment(alignment.uuid)
    registry.register_alignment(result.alignment)

    output = _resolve_output(args, mzk_path)
    save_project(output, registry)

    print(result.format_summary())
    logger.info(f"Saved filtered alignment to {output}")


def cmd_export_table(args: argparse.Namespace) -> None:
    registry = _load_registry(Path(args.mzk))
    alignment = _get_alignment(registry, args.alignment_name)

    all_samples = {s.uuid: s for s in registry.get_all_samples()}
    sample_names = {s.uuid: s.name for s in registry.get_all_samples()}

    sep = '\t' if args.format == 'tsv' else ','

    export_feature_table_to_file(
        alignment=alignment,
        samples=all_samples,
        sample_names=sample_names,
        output=Path(args.output),
        separator=sep,
    )


def cmd_export_bpcs(args: argparse.Namespace) -> None:
    samples, _ = load_project(Path(args.mzk))

    export_bpcs_to_file(
        samples=samples,
        output=Path(args.output),
    )


def cmd_export_compound(args: argparse.Namespace) -> None:
    registry = _load_registry(Path(args.mzk))
    alignment = _get_alignment(registry, args.alignment_name)

    sample_lookup = {s.uuid: s for s in registry.get_all_samples()}
    output_dir = Path(args.output_dir)
    normalize = not args.absolute

    if args.all:
        export_all_compounds(
            alignment=alignment,
            samples=sample_lookup,
            output_dir=output_dir,
            write_json=args.json,
            normalize=normalize,
        )
    else:
        export_compound_to_file(
            alignment=alignment,
            analyte_index=args.analyte_index,
            samples=sample_lookup,
            output_dir=output_dir,
            write_json=args.json,
            normalize=normalize,
        )


# --- Argument parser ---

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='mzkit',
        description='MzKit command-line tools',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable debug logging',
    )

    subparsers = parser.add_subparsers(
        dest='command',
        required=True,
    )

    # --- import-mzml ---
    p_mzml = subparsers.add_parser(
        'import-mzml',
        help='Import .mzML files as samples and save to .mzk',
    )

    # Input files: either direct paths or a text file listing them
    mzml_input_group = p_mzml.add_mutually_exclusive_group(required=True)
    mzml_input_group.add_argument(
        '--inputs', '-i',
        nargs='+',
        default=None,
        help='One or more .mzML file paths',
    )
    mzml_input_group.add_argument(
        '--input-list',
        default=None,
        help='Text file with one .mzML path per line',
    )

    # Sample name extraction
    p_mzml.add_argument(
        '--name-regex',
        default=None,
        help='Regex to extract sample name from filename '
             '(uses first capture group, or full match). '
             'Default: filename stem.',
    )

    # Output
    p_mzml.add_argument(
        '-o', '--output',
        required=True,
        help='Output .mzk file',
    )

    # MS1 ScanArrayParameters
    p_mzml.add_argument(
        '--ms1-mz-tolerance',
        type=float, default=0.05,
        help='MS1 m/z tolerance for lane grouping (default: 0.05)',
    )
    p_mzml.add_argument(
        '--ms1-scan-gap',
        type=int, default=3,
        help='MS1 max consecutive empty scans before new lane (default: 3)',
    )
    p_mzml.add_argument(
        '--ms1-min-intsy',
        type=float, default=2000.0,
        help='MS1 minimum intensity / noise threshold (default: 2000.0)',
    )

    # MS2 ScanArrayParameters (opt-in)
    p_mzml.add_argument(
        '--import-ms2',
        action='store_true', default=False,
        help='Also import MS2 data',
    )
    p_mzml.add_argument(
        '--ms2-mz-tolerance',
        type=float, default=0.05,
        help='MS2 m/z tolerance (default: 0.05)',
    )
    p_mzml.add_argument(
        '--ms2-scan-gap',
        type=int, default=3,
        help='MS2 max consecutive empty scans (default: 3)',
    )
    p_mzml.add_argument(
        '--ms2-min-intsy',
        type=float, default=2000.0,
        help='MS2 minimum intensity (default: 2000.0)',
    )

    p_mzml.set_defaults(func=cmd_import_mzml)

    # --- import-features ---
    p_import = subparsers.add_parser(
        'import-features',
        help='Import a feature table (mz/rt CSV) and generate ensembles',
    )
    p_import.add_argument(
        'features',
        help='CSV/TSV file with mz, rt columns',
    )
    p_import.add_argument(
        'mzk',
        help='.mzk file containing sample data',
    )
    p_import.add_argument(
        '--output-mzk',
        default=None,
        help='Output .mzk file (default: modify input in place)',
    )
    p_import.add_argument(
        '--name',
        default='',
        help='Name for the alignment',
    )

    # Column selection
    p_import.add_argument(
        '--mz-col',
        default='mz',
        help='Column name for m/z values (default: mz)',
    )
    p_import.add_argument(
        '--rt-col',
        default='rt',
        help='Column name for RT values (default: rt)',
    )
    p_import.add_argument(
        '--id-col',
        default=None,
        help='Column name for analyte IDs (optional)',
    )

    # Regex extraction
    p_import.add_argument(
        '--mz-regex',
        default=None,
        help='Regex to extract m/z from column (uses first capture group)',
    )
    p_import.add_argument(
        '--rt-regex',
        default=None,
        help='Regex to extract RT from column (uses first capture group)',
    )

    # RT unit
    p_import.add_argument(
        '--rt-unit',
        choices=['seconds', 'minutes'],
        default='seconds',
        help='Unit of RT values in the input file (default: seconds)',
    )

    # Processing parameters
    p_import.add_argument(
        '--rt-window',
        type=float, default=4.0,
        help='RT half-window in seconds (default: 4.0)',
    )
    p_import.add_argument(
        '--mz-window',
        type=float, default=0.01,
        help='m/z tolerance in Da (default: 0.01)',
    )
    p_import.add_argument(
        '--ms1-corr',
        type=float, default=0.8,
        help='MS1 correlation threshold (default: 0.8)',
    )
    p_import.add_argument(
        '--ms2-corr',
        type=float, default=0.7,
        help='MS2 correlation threshold (default: 0.7)',
    )
    p_import.add_argument(
        '--min-intsy',
        type=float, default=1000.0,
        help='Minimum intensity (default: 1000.0)',
    )
    p_import.add_argument(
        '--use-rel-intsy',
        action='store_true', default=True,
        help='Normalize chromatograms before correlation (default: true)',
    )
    p_import.add_argument(
        '--pregroup',
        action='store_true', default=False,
        help='Pre-group redundant features by peak shape',
    )
    p_import.add_argument(
        '--pregroup-rt-tolerance',
        type=float, default=5.0,
        help='RT tolerance for pre-grouping (default: 5.0)',
    )
    p_import.add_argument(
        '--pregroup-corr-threshold',
        type=float, default=0.8,
        help='Correlation threshold for pre-grouping (default: 0.8)',
    )
    p_import.set_defaults(func=cmd_import_features)

    # --- filter ---
    p_filter = subparsers.add_parser(
        'filter',
        help='Filter an alignment by expression',
    )
    p_filter.add_argument(
        'mzk',
        help='.mzk file containing samples and alignment',
    )
    p_filter.add_argument(
        '-e', '--expression',
        required=True,
        help='Python expression to filter analytes (e.g. "n >= 3")',
    )
    p_filter.add_argument(
        '--output-mzk',
        default=None,
        help='Output .mzk file (default: modify input in place)',
    )
    p_filter.add_argument(
        '--alignment-name',
        default=None,
        help='Name of alignment to filter (required if multiple exist)',
    )
    p_filter.set_defaults(func=cmd_filter)

    # --- export-table ---
    p_export = subparsers.add_parser(
        'export-table',
        help='Export alignment as a feature table',
    )
    p_export.add_argument(
        'mzk',
        help='.mzk file containing samples and alignment',
    )
    p_export.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV/TSV file',
    )
    p_export.add_argument(
        '-f', '--format',
        choices=['csv', 'tsv'],
        default='tsv',
        help='Output format (default: tsv)',
    )
    p_export.add_argument(
        '--alignment-name',
        default=None,
        help='Name of alignment to export (required if multiple exist)',
    )
    p_export.set_defaults(func=cmd_export_table)

    # --- export-bpcs ---
    p_bpcs = subparsers.add_parser(
        'export-bpcs',
        help='Export base peak chromatograms for all samples',
    )
    p_bpcs.add_argument(
        'mzk',
        help='.mzk file containing samples',
    )
    p_bpcs.add_argument(
        '-o', '--output',
        required=True,
        help='Output JSON file',
    )
    p_bpcs.set_defaults(func=cmd_export_bpcs)

    # --- export-compound ---
    p_compound = subparsers.add_parser(
        'export-compound',
        help='Export MGF (+ optional JSON) for analyte(s)',
    )
    p_compound.add_argument(
        'mzk',
        help='.mzk file containing samples and alignment',
    )

    target_group = p_compound.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        '--analyte-index',
        type=int, default=None,
        help='Index of a single analyte to export',
    )
    target_group.add_argument(
        '--all',
        action='store_true', default=False,
        help='Export all analytes (single compounds.mgf)',
    )

    p_compound.add_argument(
        '-o', '--output-dir',
        required=True,
        help='Output directory',
    )
    p_compound.add_argument(
        '--json',
        action='store_true', default=False,
        help='Also write JSON files',
    )
    p_compound.add_argument(
        '--absolute',
        action='store_true', default=False,
        help='Use absolute intensities (default: normalize to 0-100)',
    )
    p_compound.add_argument(
        '--alignment-name',
        default=None,
        help='Name of alignment to use (required if multiple exist)',
    )
    p_compound.set_defaults(func=cmd_export_compound)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    args.func(args)


if __name__ == '__main__':
    main()
