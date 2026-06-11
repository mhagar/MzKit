# MzKit Architecture

A map of the system for humans and models working on the codebase. For build
and run instructions see [`README.md`](README.md); for terse conventions see
[`CLAUDE.md`](CLAUDE.md). This file explains *how the pieces fit together*.

MzKit is a desktop tool for **inspecting** LC/MS data (PyQt5 GUI + a headless
CLI). It is deliberately not a metabolomics pipeline: the center of gravity is
fast, manual, fine-grained inspection and annotation.

---

## 1. The big picture

```
 GUI (PyQt5, gui/)                          core/ (Qt-free*, importable by CLI and GUI)
 ┌──────────────────┐   start_process(       ┌─────────────────────┐
 │  MainController  │     module_path,       │  ProcessController  │ ─ daemon thread ─▶ core/cli/*.py
 │  + SubWindowMgr  │ ──  fn_name, params) ─▶│  (QTimer polls      │                   (pure functions:
 │  + views/widgets │ ◀── result via ─────── │   every 100 ms)     │ ◀── return value ─ import / align /
 └──────────────────┘     pyqtSignal         └─────────────────────┘                     export / extract)
        │  on_completion_func                                                                  │
        │  mutates registry                                                                    ▼
        ▼                                                                          mutates ─▶ DataRegistry
   Qt models  ◀───────────────── sigSampleAdded / sigAlignmentAdded / ... ──────────────────────┘
                                  (Qt signals broadcast every change)
```

\* `core` is GUI-independent **except** two classes that use Qt *signals only*
(no widgets): `DataRegistry` and `ProcessController`. Importing anything under
`core` must never pull in `gui`.

This is enforceable by eye and should be adhered to.
(to test: `python -c "import core.data_structs; import sys;
assert not [m for m in sys.modules if m.startswith('gui')]"`).

**The central design principle:**
All heavy/processing logic lives as plain, stateless, Qt-free functions in `core/cli/`.
The GUI never calls them directly. It hands a `(module_path, function_name, parameters)`
triple to `ProcessController`, which imports and runs the function on a background daemon
thread and delivers the return value back to an `on_completion_func` on the main thread
via a Qt signal.

That same function is also the CLI subcommand implementation. Write processing
logic once; it serves both front-ends.

---

## 2. Domain model

Everything hangs off a single in-memory store, `DataRegistry`, which owns
`Sample`s and `EnsembleAlignment`s and broadcasts every change as a Qt signal
so views/models stay in sync.

```
DataRegistry                       core/data_structs/data_registry.py
├── Sample                         sample.py    — a unique chemical mixture
│   ├── Injection      [optional]  injection.py — raw LC/MS for one run
│   │   ├── ScanArray (MS1)        scan_array.py
│   │   ├── ScanArray (MS2) [opt]               — DDA/DIA carry precursor metadata
│   │   └── ensembles: {EnsembleUUID -> Ensemble}
│   │        └── Ensemble          ensemble.py  — THE analytical unit
│   │             ├── ms1_cofeatures: [FeaturePointer]   feature_pointer.py
│   │             ├── ms2_cofeatures: [FeaturePointer]
│   │             └── annotations: IonAnnotation / MzDiffAnnotation /
│   │                              IonPairAnnotation / GenericAnnotation
│   └── Fingerprint    [optional]  fingerprint.py — bioassay data, "stapled" on
└── EnsembleAlignment(s)           alignment.py — cross-sample grouping (immutable)
     └── AlignedAnalyte            — maps {SampleUUID -> EnsembleUUID} per sample
```

Key types (all in `core/data_structs/`):

- **Sample** — top-level container. Has an optional `Injection` (MS data) and/or
  `Fingerprint` (bioassay). A Sample may be built empty and filled later; the
  "must have at least one" rule is enforced at registration
  (`DataRegistry.validate_new_sample`), not in the constructor. Samples with the
  same name are *merged* on registration (e.g. MS-only + fingerprint-only → one
  Sample) — see `DataRegistry.merge_samples`.
- **Injection** — raw LC/MS for one run. Builds `ScanArray`s from a pyOpenMS
  `MSExperiment` in `__post_init__` (when given raw `exp`). Holds its Ensembles.
  `acquisition_mode` is `ms1_only` | `dda` | `dia`.
- **ScanArray** — the performance-critical structure: scipy sparse `(mz, intsy)`
  matrices, rows = "m/z lanes", columns = scans. Stored in both CSR and CSC for
  fast chromatogram (`get_bpc`/`get_xic`) *and* spectrum (`get_spectrum`) slices.
  Built by `build_features` (see §5).
- **FeaturePointer** — a lightweight pointer into a ScanArray (a mass-lane index
  + scan indices + source array uuid/shape). Ensembles are made of these rather
  than copies of the data.
- **Ensemble** — a group of coeluting ions (cofeatures) at one RT. The unit the
  user actually inspects and annotates. Caches base peak / RT / m/z on
  `set_injection`. Carries user annotations and user-editable identity/formula.
- **EnsembleAlignment / AlignedAnalyte** — cross-sample grouping. The alignment
  *is* the grouping (there is no separate "analyte table" abstraction); it is
  **immutable** — adding a sample means re-aligning. Produced by the algorithm in
  `core/cli/align_ensembles.py`, but the data types live in
  `core/data_structs/alignment.py` (so `data_registry` and `persistence` can
  depend on them without reaching into `core/cli`).

### UUIDs

Every domain object carries an `int` uuid from `uuid.uuid4().int`. Type aliases
in `core/data_structs/uuid_types.py` (`SampleUUID`, `EnsembleUUID`, …) give
`NewType`-based type safety at zero runtime cost.

---

## 3. Layer responsibilities

```
core/
  data_structs/   Domain model. Pure data + methods over it. The vocabulary.
  cli/            Processing logic as stateless Qt-free functions (the "verbs").
                  Also the unified CLI entry point (main.py -> `mzkit` command).
  controllers/    ProcessController — the threaded task runner (Qt signals only).
  utils/          Persistence (.mzk), config, array helpers, formula formatting.
  interfaces/     Lightweight protocols/ABCs for decoupling.
  labeling/       Peak-morphology labeling tool (schema + candidate generator).

gui/
  controllers/    MainController (orchestrator), SampleController, SubWindowManager,
                  SelectionManager. Wire signals; translate UI intent into
                  ProcessController calls; register results into DataRegistry.
  views/          MDI subwindows: SampleViewer, EnsembleViewer, AlignmentViewer,
                  FingerprintViewer, ProcessMonitor. The big ones are packages
                  with their plot/tool logic split into submodules.
  widgets/        Reusable pyqtgraph plot widgets (Chrom/MS/FPrint plots, overlays).
  models/         Qt item models (SampleListModel, AlignmentListModel, …).
  resources/      .ui files + their pyuic5-generated .py (never hand-edit the .py).
  dialogues/      Wizard dialogs (FormulaFinder, AlignmentFilter, imports).
  utils/          GUI-only helpers (graphics, ms array shaping).
```

**Rule:** dependencies point `gui => core`, never the reverse. `core/cli/*`
functions must stay stateless and Qt-free so they run identically under the GUI
(threaded, via ProcessController) and the CLI.

---

## 4. Execution / threading model

`ProcessController` (`core/controllers/ProcessController.py`):

1. `start_process(module_path, function_name, parameters, on_completion_func)`
   creates a `ProcessRunner` (`core/cli/process_runner.py`) and starts it on a
   **daemon thread**. Each process gets an integer id.
2. The target function may accept injected `progress_callback` and
   `cancel_event` kwargs (cooperative cancellation — Python threads can't be
   force-killed, so long tasks must check `cancel_event` themselves).
3. A `QTimer` polls every 100 ms for output/progress/status changes and emits Qt
   signals; the `ProcessMonitor` window reflects them.
4. On completion, the registered `on_completion_func` runs on the **main
   thread** with the function's return value. This is where results get
   registered into `DataRegistry` (which then emits the relevant `sig…`).

Trade-off to be aware of: the `(module_path, function_name, parameters)`
dispatch is **stringly-typed** — there's no static guarantee the function exists
or that the params match. Rename a CLI function and the only failure is at
runtime inside a thread. A typed wrapper per task would remove that footgun
later; for now, keep param dicts adjacent to their target signatures and test
the round-trip.

---

## 5. Feature extraction (the hot path)

`core/data_structs/scan_array.py` turns a stack of same-MS-level spectra into a
`ScanArray` via a "mass lane" builder:

- `build_features` — dense parallel-array implementation. Output is
  `(n_features, n_scans)` m/z and intensity buffers.
- `_run_feature_kernel` — pure-Python matching kernel (binary-search peak
  matching, claim-once semantics).
- `_run_feature_kernel_numba` — optional `@njit` kernel, dispatched
  automatically when numba is importable; set `MZKIT_DISABLE_NUMBA=1` to force
  the pure-Python path (used for A/B benchmarking).
- `_build_features_legacy` — the original per-`Feature` reference
  implementation. **Not used in production**; kept only as the parity oracle for
  `tests/test_build_features_parity.py`. If you change matching semantics, that
  test is the contract.

---

## 6. Persistence — `.mzk` files

`core/utils/persistence.py`, `save_project()` / `load_project()`.

- A `.mzk` is a **ZIP archive** of JSON (primitives, metadata, alignments) +
  pickle (numpy/scipy arrays: ScanArrays, fingerprint arrays, ensembles).
- `load_project()` returns `(samples, alignments)`.
- Layout inside the zip: `project_metadata.json`, `samples/<name>/…`,
  `alignments/<uuid>.json`.
- **Back-compat is intentional:** every deserializer uses `.get(..., default)`
  for fields added over time, so old `.mzk` files keep loading as the model
  grows. Preserve this when adding fields.

Known trade-offs (fine for V1, flagged for later):

- Pickle means loading an untrusted `.mzk` can execute arbitrary code, and a
  numpy/scipy upgrade can in principle break old files. If users will *share*
  `.mzk` files, plan a migration of the array payloads to `.npz`/arrow.
- Samples are keyed by **name** in the zip path; same-named samples or names
  containing `/` would collide. Keying by uuid is the fix.

---

## 7. Conventions

- Run everything through `uv run` (never bare `python`).
- `.ui` files are edited in Qt Designer and compiled with
  `pyuic5 -o gui/resources/Foo.py gui/resources/Foo.ui`. Never hand-edit the
  `.ui` files or generated `.py`.
- New processing logic => a stateless function in `core/cli/`, exposed as a
  `mzkit` subcommand in `core/cli/main.py` *and* callable from the GUI via
  `ProcessController`.
- Config lives in `default_config.ini`, loaded by `core/utils/config.py`.
- Use `logging`, not `print`, for diagnostics.

---

## 8. Known follow-ups / tech debt

Tracked here so they're explicit rather than surprising:

- **`AnalyteTable` is retired** (done). The old `AnalyteTable` / `Analyte`
  system — data structures, CSV import path, viewer, list model, registry
  machinery, and the selection→PeakOverlay flow — was removed; `EnsembleAlignment`
  is the sole cross-sample path. The alignment list widget/tab were renamed to
  `listViewAlignments` / `tabAlignments` ("Alignments") in `MainWindow.ui` and
  the references in `gui/views/main_view.py` updated to match. Remaining tidy-up:
  delete the orphaned `gui/resources/AnalyteTableImportWizardWindow.ui` and
  `AnalyteTableViewerWindow.ui` (their generated `.py` are already gone).
- **`ProcessController` typed dispatch** — see §4.
- **God-object viewers** — `gui/views/ensemble_viewer/__init__.py` and
  `sample_viewer/__init__.py` still hold a large not-yet-extracted remainder.
  Decomposition has started (`plot_managers.py`, `tool_controllers.py`,
  `dda_overlays.py`); continue lowest-risk-first (export, annotation drawing).
- **Cross-platform** — `main.py` forces `QT_QPA_PLATFORM=xcb` (X11). Must be
  conditioned on Linux before shipping to macOS/Windows users.
- **Annotation caveats** — e.g. `Ensemble.add_ion_pair_annot` does not yet handle
  adducts (see its inline TODO) — don't rely on it for adduct relationships.
