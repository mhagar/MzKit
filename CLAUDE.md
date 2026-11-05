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
  widgets/          # Reusable widgets (ChromPlotWidget, MSPlotWidget, FPrintWidget, AlignmentPlotWidget, StructureWidget)
  models/           # Qt models (SampleListModel, AlignmentListModel, etc.)
  resources/        # .ui files and their generated Python (via pyuic5)
  dialogues/        # Wizard dialogs (FormulaFinder, AlignmentFilter, FeatureTableImport)
tests/
```

## Key domain concepts

- **Sample**: top-level container. Has an optional Injection (MS data) and/or Fingerprint (bioactivity).
- **Injection**: raw LC/MS data. Contains ScanArrays (MS1, optionally MS2) and Ensembles.
- **ScanArray**: structured numpy array of (mz, intensity, scan_num) across scans.
- **Ensemble**: a group of coeluting ions (cofeatures) extracted from a ScanArray. The central analytical unit. All annotations live on it (`adduct_network`, `formula_assignment`, `annotation`, peak-level annotations); edit them via `DataRegistry.update_ensemble` (emits `sigEnsembleAnnotated`).
- **EnsembleAlignment**: cross-sample alignment of Ensembles (MS1 + RT only). Contains AlignedAnalytes mapping SampleUUID -> EnsembleUUID. Immutable after creation. An analyte's annotations are display-only aggregates over its members.
- **EnsembleGroup**: a user basket of EnsembleUUIDs (title, notes, SMILES), independent of alignments.
- **DataRegistry**: central in-memory store. Holds Samples, EnsembleAlignments, EnsembleGroups and SimilarityMatrices. Uses Qt signals for change notification.

## Persistence (.mzk files)

- `.mzk` files are ZIP archives containing JSON + pickle.
- `save_project()` / `load_project()` in `core/utils/persistence.py`.
- `load_project()` returns `ProjectData(samples, alignments, groups, matrices)`; ensembles carry their own annotations.
- Alignment JSON I/O (standalone, outside .mzk): `save_alignment_json()` / `load_alignment_json()` in `core/cli/main.py`.

## CLI tools (core/cli/)

All processing logic lives here as plain functions — no Qt dependency. The GUI calls these via ProcessController (threaded).

- `generate_ensemble.py` - Extract Ensembles from an Injection at given m/z coordinates
- `align_ensembles.py` - Cross-sample alignment by MS1 cosine similarity within an RT window
- `filter_alignment.py` - Filter an EnsembleAlignment by Python expression
- `import_feature_table.py` - Import external feature coordinates (MZmine CSV) and generate Ensembles + alignment
- `export_table.py` - Export an EnsembleAlignment as abundance + formula tables (CSV/TSV), plus companion .mgf, .graphml and sample similarity (GUI: following the Alignment Viewer's applied clustering settings)
- `export_network.py` - Similarity network of an alignment's analytes as GraphML, from a SimilarityMatrix (computed via `cluster_analytes.py`, MS2DeepScore by default, or imported)
- `cluster_samples.py` - Hierarchical clustering of an alignment's samples: Jaccard of analyte presence (optionally MS2-grouped) or an imported lineage (Alignment Viewer sample cluster mode)
- `cluster_analytes.py` - Hierarchical clustering of an alignment's analytes by MS2DeepScore (default) or MS2 modified cosine (Alignment Viewer cluster mode)
- `rank_candidates.py` - Rank compound-DB candidates for an analyte by GLACIER-predicted MS2 (explained top-N peaks, then cosine); `[glacier]` config (Search (structures) tab, gui/widgets/annotation_tabs.py: Alignment + Ensemble Viewer)
- `search_spectra.py` - Search spectral libraries for an analyte's MS2: identity (cosine within the precursor window) or analogue (embedding cosine: MS2DeepScore or an imported embedder; modified cosine shown, never ranked); `[spectral_search]` config (Search (spectra) tab, same hosts)
- `find_cofeatures.py` - Core cofeature detection (Pearson correlation of extracted ion chromatograms)
- `segment_chromatogram.py` - Peak boundary detection
- `export_bpcs.py` - Export base peak chromatograms for all samples in an .mzk file (JSON)
- `export_compound.py` - Export XIC + MS1/MS2 spectra for a single analyte (JSON)
- `main.py` - Unified CLI entry point (`mzkit import-features`, `mzkit align`, `mzkit filter`, `mzkit export-table`, `mzkit export-bpcs`, `mzkit export-compound`)

## Spectral libraries

- `core/utils/spectrum_db.py`: every `spectra/<library>.mgf` (gitignored folder) parsed by matchms `load_from_mgf` (harmonized keys) into `spectra/spectra.sqlite`; negative mode skipped. Rebuild: `uv run mzkit build-spectral-db` or Tools > Libraries & databases (gui/dialogues/LibrariesDialog.py, which also builds the compound DB cache)
- Embeddings are row-aligned `spectra.<embedder>.npy` (float16): MS2DeepScore at build time, plus external `<library>.<embedder>.npz` (`ids` = spectrum ids, `embeddings`). Query embeddings for an external embedder: the consensus .mgf from export-table, imported in Tools > Libraries & databases (session only, `QUERY_EMBEDDINGS` in gui/widgets/spectral_search_widget.py)

## MS2DeepScore

- NumPy-only inference in `core/utils/ms2deepscore.py`; weights are `.npz` files in the gitignored `models/` folder
- `tools/ms2deepscore/export_weights.py` (dev tool, own locked torch env): `uv run --script tools/ms2deepscore/export_weights.py` dumps the official weights, checks parity, writes `tests/ms2deepscore_parity.npz`

## GLACIER (MS/MS spectrum prediction, ms-pred)

- `core/utils/glacier.py`: three ONNX blocks run in onnxruntime; featurization + fragment glue in NumPy/RDKit. Model files in gitignored `models/glacier/`. `predict()` rows also carry each fragment's formula, H shift and ion term; `ion_formula()` turns a row into e.g. `C7H7O+`
- `tools/glacier/export_onnx.py` (dev tool, own locked torch env): `uv run --script tools/glacier/export_onnx.py --ckpt best.ckpt --ms-pred <checkout>` exports the blocks + `meta.json`, checks parity vs ms-pred's `predict_mol`, writes `tests/glacier_parity.npz`

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
