# MzKit

LC/MS metabolomics analysis toolkit. PyQt5 GUI + headless CLI.

## Running

- Always use `uv run` instead of bare `python`
- GUI: `uv run python main.py`
- CLI: `uv run mzkit <subcommand>` (import-features, filter, export-table)
- Tests: `uv run pytest`
- Python 3.13 required

## Architecture

See ARCHITECTURE.md for a more thorough breakdown
```
core/               # GUI-independent logic (importable by CLI and GUI)
  cli/              # Processing scripts (ensemble generation, alignment, import, export)
  data_structs/     # Domain model (Sample, Injection, Ensemble, EnsembleAlignment, etc.)
  utils/            # Persistence (.mzk files), config, array helpers
  controllers/      # ProcessController (threaded background task runner)
gui/                # PyQt5 GUI (depends on core/)
  controllers/      # MainController, SampleController, SubWindowManager
  views/            # MDI subwindows (SampleViewer, EnsembleViewer, AlignmentViewer, etc.)
  widgets/          # Reusable plot widgets (ChromPlotWidget, MSPlotWidget, FPrintWidget)
  models/           # Qt models (SampleListModel, AlignmentListModel, etc.)
  resources/        # .ui files and their generated Python (via pyuic5)
  dialogues/        # Wizard dialogs (FormulaFinder, AlignmentFilter, FeatureTableImport)
tests/
```

## Key domain concepts

- **Sample**: top-level container. Has an optional Injection (MS data) and/or Fingerprint (bioactivity).
- **Injection**: raw LC/MS data. Contains ScanArrays (MS1, optionally MS2) and Ensembles.
- **ScanArray**: structured numpy array of (mz, intensity, scan_num) across scans.
- **Ensemble**: a group of coeluting ions (cofeatures) extracted from a ScanArray. The central analytical unit.
- **EnsembleAlignment**: cross-sample alignment of Ensembles. Contains AlignedAnalytes mapping SampleUUID -> EnsembleUUID. Immutable after creation.
- **DataRegistry**: central in-memory store. Holds Samples and EnsembleAlignments. Uses Qt signals for change notification.

## Persistence (.mzk files)

- `.mzk` files are ZIP archives containing JSON + pickle.
- `save_project()` / `load_project()` in `core/utils/persistence.py`.
- `load_project()` returns `(samples, alignments)` tuple.
- Alignment JSON I/O (standalone, outside .mzk): `save_alignment_json()` / `load_alignment_json()` in `core/cli/main.py`.

## CLI tools (core/cli/)

All processing logic lives here as plain functions — no Qt dependency. The GUI calls these via ProcessController (threaded).

- `generate_ensemble.py` - Extract Ensembles from an Injection at given m/z coordinates
- `align_ensembles.py` - Cross-sample alignment by spectral cosine similarity
- `filter_alignment.py` - Filter an EnsembleAlignment by Python expression
- `import_feature_table.py` - Import external feature coordinates (MZmine CSV) and generate Ensembles + alignment
- `export_table.py` - Export an EnsembleAlignment as a feature intensity table (CSV/TSV)
- `find_cofeatures.py` - Core cofeature detection (Pearson correlation of extracted ion chromatograms)
- `segment_chromatogram.py` - Peak boundary detection
- `export_bpcs.py` - Export base peak chromatograms for all samples in an .mzk file (JSON)
- `export_compound.py` - Export XIC + MS1/MS2 spectra for a single analyte (JSON)
- `main.py` - Unified CLI entry point (`mzkit import-features`, `mzkit filter`, `mzkit export-table`, `mzkit export-bpcs`, `mzkit export-compound`)

## GUI resources

- `.ui` files are edited in Qt Designer, then compiled with `pyuic5 -o gui/resources/Foo.py gui/resources/Foo.ui`
- Never hand-edit the generated `.py` files in `gui/resources/`

## UUID system

All domain objects use `int` UUIDs (from `uuid.uuid4().int`). Type aliases in `core/data_structs/uuid_types.py`
(SampleUUID, EnsembleUUID, AlignmentUUID, etc.) for type safety via `NewType`.

## Conventions

- Processing functions should be stateless and Qt-free so they work in both GUI (via ProcessController) and CLI contexts
- Background tasks in the GUI go through `ProcessController.start_process()`, which runs them in a daemon thread
- Config lives in `default_config.ini`, loaded by `core/utils/config.py`
