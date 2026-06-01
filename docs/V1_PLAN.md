# MzKit V1 Plan (Pre-ASMS)

This is a handoff document for Claude instances working on V1 polish before the ASMS conference. Read this in full before starting any chunk. Update the **Status** line on each task as you make progress, and append notes under **Notes from prior sessions** at the bottom.

The five workstreams are ordered roughly by `impact / effort`. Each is sized to fit in 1–2 sessions. Items marked **Defer** are post-ASMS unless time appears.

---

## Project context (read this first)

MzKit is an LC/MS metabolomics analysis toolkit. PyQt5 GUI + headless CLI. See `CLAUDE.md` at repo root for full architecture. Quick recap of key invariants:

- `core/` is Qt-free. CLI scripts live in `core/cli/` and are called by the GUI via `ProcessController` (a threaded runner — **not** subprocess).
- Domain objects use `int` UUIDs (see `core/data_structs/uuid_types.py`).
- `.ui` files compile to `gui/resources/*.py` via `pyuic5 -o gui/resources/Foo.py gui/resources/Foo.ui`. Never hand-edit generated `.py` files.
- Always run via `uv run`. Python 3.13.
- Tests: `uv run pytest`.

---

## Workstream 1: Window management overhaul ✅

**Goal:** make MDI behaviour consistent and stop destroying windows on close.

**Status:** complete (manual-tested by user). See "Notes from prior sessions" at bottom for what changed.

**Files:**
- `gui/controllers/subwindow_controller.py`
- `gui/views/sample_viewer/__init__.py` (~ line 655, `_open_ensemble_in_viewer`)
- `gui/controllers/main_controller.py` (`_handle_view_ensemble_request`, ~ line 696)
- `gui/resources/MainWindow.ui` + regenerated `.py`

**Tasks:**

1. **Hide-don't-destroy on close.**
   - `SubWindowManager.close_window()` currently does `del self.windows[...]` and `del self.sub_windows[...]`. Remove that.
   - Install an event filter / override `closeEvent` on each subwindow so it calls `hide()` + `event.ignore()` instead of being destroyed by Qt.
   - Re-showing via `show_window()` already handles the "create if missing" case, so this should be a small change.

2. **Single EnsembleViewer route — kill the second path.**
   - `sample_viewer/__init__.py:655` `_open_ensemble_in_viewer` instantiates a fresh `EnsembleViewer` and `show()`s it floating. This is the inconsistency the user reported.
   - Replace with a signal emit (existing pattern — see `view_ensemble_requested` already wired in `main_controller.py:160` to `_handle_view_ensemble_request`).
   - Delete the `self._ensemble_viewers` reference list.
   - After this change every ensemble click should land in the same MDI subwindow.

3. **Deprecate AnalyteTableViewer.**
   - Remove `'analyte_viewer'` entry from `WINDOW_CONFIGS`.
   - Remove the menu action from `MainWindow.ui`, regenerate.
   - Leave `gui/views/analyte_table_viewer.py` and `gui/models/analyte_table_list_model.py` on disk for now but unhooked — final deletion in a separate cleanup pass once we're sure `AlignmentViewer` covers the workflow.
   - `AnalyteTableImportWizard`: keep available if it covers an import path not covered by alignment-import; otherwise also unhook.

4. **Window menu.**
   - Add a "Window" menu to `MainWindow.ui` listing all known windows with checkmarks for visible. Standard MDI behaviour. Toggling a check shows/hides.

**Estimated effort:** half a session.

**Acceptance:**
- Closing any subwindow doesn't destroy it; reopening from the menu restores it with state intact.
- Right-click → "Open in ensemble viewer" on a sample opens in the existing MDI subwindow, never floating.
- Analyte Table Viewer no longer reachable from the UI.

---

## Workstream 2: Process monitor improvements ✅

**Goal:** real progress display, real cancel button, partial-result preservation on cancel.

**Status:** complete for the MVP scope (mzml_import + import_feature_table). See "Notes from prior sessions" at the bottom.

**Files:**
- `core/cli/process_runner.py`
- `core/controllers/ProcessController.py`
- `gui/views/process_monitor.py`
- All CLI scripts in `core/cli/` (instrument loops)

**Tasks:**

1. **Progress reporting plumbing.**
   - Add an optional `progress_callback: Callable[[float, str], None] | None` parameter convention to all CLI `main()` functions in `core/cli/`.
   - `ProcessRunner` constructs a callback that puts `("progress", percent, message)` items on `output_queue` and passes it via `self.parameters`.
   - `ProcessController._poll_active_processes` routes "progress" items to `self.model.updateProcess(progress=...)` (column already exists).
   - Add a similar `cancel_event: threading.Event | None = None` parameter to the same functions.

2. **Cancel button — cooperative cancellation.**
   - Real thread-killing not viable. Pattern:
     - Add `cancel_event: threading.Event` to `ProcessRunner`, plumb into the wrapped function's parameters.
     - `ProcessController.cancel_process()` calls `event.set()` instead of mutating `process.status` (which is currently a no-op since `run()` re-sets status anyway).
     - Long-running loops in CLI functions check `if cancel_event and cancel_event.is_set(): break` periodically (e.g. every 100 iterations).
   - Wire the "Cancel" button in `process_monitor.py` (currently does nothing — see the `.ui` file) to `process_controller.cancel_process(pid)` for the selected row.

3. **Instrument loops with progress + cancel checks.**
   - Priority order: `mzml_import.py` (per-file), `generate_ensemble.py`, `align_ensembles.py`, `import_feature_table.py`.
   - Each has a clear `for i, x in enumerate(items)` — emit progress as `(i+1) / len(items) * 100`, message `f"Processing {x.name}"`.

4. **Partial-result preservation on cancel** (super-extra-optional).
   - In `mzml_import.main`, on cancel between files: `return samples` (whatever has completed). The completion callback in `main_controller` already registers samples one-by-one — no changes needed downstream.
   - Truly a one-liner once cancellation is cooperative.

**Estimated effort:** ~one session.

**Acceptance:**
- Progress column updates live during mzML import / ensemble generation.
- Cancel button stops the run within a few seconds and partially-imported samples are kept.

---

## Workstream 3: mzML import optimization ✅ (partial — see notes)

**Goal:** 5–10× speedup on large files. Stop choking.

**Status:** core algorithmic refactor + numba kernel landed. **(a) auto noise-threshold was explicitly rejected by the user** — they want a UX path (easy manual threshold setting) instead, deferred to post-ASMS. **(c) multiprocessing not pursued** — landed speedup made it unnecessary. See "Notes from prior sessions" at bottom.

**Files (as touched):**
- `core/data_structs/scan_array.py` (hot loop renamed `build_features_masscube` → `build_features`)
- `tests/test_build_features_parity.py` (new — pure-Python and numba kernels vs legacy oracle)
- `pyproject.toml` / `uv.lock` (numba dep)

**The bottleneck is `build_features`** (formerly `build_features_masscube`), not pyopenms loading. It's a pure-Python double loop over scans × WIP features with per-iteration `np.argmin` over the full WIP list. Three layers, ordered by ROI:

**(a) Noise pre-filter.**
- `_get_peaks_higher_than_intsy` exists; `min_intsy` defaults are typically way too low.
- When user passes `min_intsy=0`, auto-derive: e.g. 5th percentile of nonzero intensities sampled from the first 100 scans.
- Cuts WIP feature list dramatically on noisy Orbitrap data.

**(b) Numba on the hot loop.**
- `_find_closest_idx` + the per-scan loop body is a textbook numba target — pure numpy + scalars, no Python objects on the hot path once refactored.
- Refactor: keep `wip_mzs` as a sorted numpy array, use `np.searchsorted` for matching instead of `np.argmin` over the full array.
- Wrap inner loop in `@njit`.
- Realistic 10–50× on large files.

**(c) Multiprocessing across files.** *(Defer if time tight.)*
- In `mzml_import.main`, with >1 file, use `concurrent.futures.ProcessPoolExecutor` with `max_workers = min(n_files, os.cpu_count() // 2)`.
- Each worker returns a `Sample`. Check pickleability of `oms.MSExperiment`; if not pickleable, drop `exp` before returning (post-init has already populated ScanArrays).
- Has nasty interactions with cancellation and the log-queue routing. Skip if (a) + (b) already get 5–10×.

**Tasks:**
1. Implement (a). Measure speedup.
2. Implement (b). Measure speedup.
3. Add a timing test that pins runtime within a tolerance band on a fixture file.
4. (c) only if time permits.

**Estimated effort:** one session for (a) + (b). Half-session more for (c).

**Acceptance:**
- mzML import of a 500 MB file is at least 5× faster than today on the same hardware.
- Existing `test_mzml_import.py` tests still pass.
- New timing-regression test added.

---

## Workstream 4: Crude DDA handling ✅ (Phases 1 + 2 done; Phase 3 deferred)

**Goal:** surface MS2 spectra with precursor metadata, link them to ensembles, and make DDA datasets viewable in the SampleViewer. "Crude" = no chimeric handling, no deconvolution, no consensus spectra.

**Status:** Phase 1 (DDA viewing in SampleViewer) and Phase 2 (Ensemble DDA integration) are complete. Phase 3 (formula finder + ruler verification on MS2) is deferred to a follow-up session — there are known glitches in those tools on MS2 that the user wants to chase down separately. The dropdown/nav-arrow `.ui` change called for in step 7 was **not made**; the user found the badge-based selection (already used in SampleViewer) sufficient and more intuitive, so the dropdown is redundant. The `export_compound.py` migration (step 10) is intentionally skipped — the user is rewriting compound export from scratch. See "Notes from prior sessions" at the bottom for everything that landed.

### Core design principles (read before touching anything)

The previous design notes for this workstream were wrong in two important ways. The user steered substantially:

- **ScanArray is a storage layout, not a semantic layer.** Mass-lane construction runs identically for MS1, DIA MS2, and DDA MS2. A mass lane in DDA MS2 says "intensity at this m/z, across these scans" — that's true regardless of which precursors triggered those scans. Peak shapes across a DDA mass lane will look weird (interleaved fragmentations of different precursors), but that's a feature: it still lets the user do things like "show me a chromatogram for everything fragmenting between 300–400 m/z." The lane-building algorithm doesn't try to express "which compound" — that's Ensemble's job.
- **Ensemble is the semantic layer.** Compound-aware filtering of MS2 data happens via `FeaturePointer.scan_idxs` sub-selecting the relevant MS2 scans for an ensemble's precursor. So a DDA ensemble's `ms2_cofeatures` are mass lanes in the MS2 ScanArray, but with `scan_idxs` restricted to just the scans whose precursor m/z matches that ensemble. The mass lane itself may span scans from many compounds; the Ensemble narrows it down.
- **Precursor metadata lives on `ScanArray`** as optional fields, populated only for DDA MS2 arrays, `None` otherwise. Persistence rides along with ScanArray pickling — no new top-level object on `Injection`, no new `DDASpectrum` dataclass.

**Cursor mechanic in SampleViewer is unchanged.** It already shows the spectrum corresponding to the currently selected MS level. Clicking a precursor badge on an MS1 spectrum switches MS level to 2 and jumps the cursor to the triggered MS2 scan. No new cursor semantics.

### Files

- `core/data_structs/scan_array.py`
- `core/data_structs/injection.py`
- `core/data_structs/ensemble.py`
- `core/cli/mzml_import.py`
- `core/utils/persistence.py`
- `gui/dialogues/MzMLImportWizard.py` + `gui/resources/MzMLImportWizard.ui`
- `gui/views/sample_viewer/` (precursor badges + isolation window overlays — Python only)
- `gui/widgets/MSPlotWidget.py` (overlay primitives)
- `gui/views/ensemble_viewer/__init__.py` + `gui/resources/EnsembleViewerWindow.ui`
- `core/cli/export_compound.py`

### Data model changes

Add to `ScanArray` (all `Optional[np.ndarray] = None`, length matches `scan_num_arr`):

- `precursor_mz_arr: Optional[np.ndarray[float]]`
- `isolation_lo_arr: Optional[np.ndarray[float]]`
- `isolation_hi_arr: Optional[np.ndarray[float]]`
- `precursor_charge_arr: Optional[np.ndarray[int]]`
- `triggering_ms1_scan_arr: Optional[np.ndarray[int]]` — precomputed at build time. For each MS2 scan, the nearest preceding MS1 scan number. Powers badge rendering on MS1 spectra without per-redraw search.

Add to `Ensemble`:

- `precursor_mz: Optional[float]`
- `precursor_charge: Optional[int]`

`.mzk` persistence: extend pickling for the new ScanArray fields. Version-bump `.mzk` format and confirm older `.mzk` files still load with the new fields defaulting to `None`. See `core/utils/persistence.py`.

### Import pipeline

- **`MzMLImportWizard.ui` (Designer change):** add a `QComboBox acquisitionModeCombo` with items "MS1 only", "DDA", "DIA". **No default selection** — the user must choose every time. Wizard's "Next"/"Finish" stays disabled until a choice is made. (User explicitly does not want autodetection or a guessed default — if it gets mis-specified, that's the user's responsibility for now.)
- **`Injection.assemble_scan_array` for MS2:** runs the existing `build_features` on the MS2 spectra exactly as MS1 does. For DDA, pass `scan_gap_tolerance=∞` (replicate MS2s of the same precursor are interleaved with MS2s of everything else, so any finite gap tolerance would fragment the mass lanes incorrectly; the Ensemble layer does compound-level filtering downstream).
- During MS2 construction, populate the new ScanArray precursor arrays from `oms.Precursor` (m/z, charge, isolation lower/upper). Compute `triggering_ms1_scan_arr` in the same pass.

### Phase 1 — DDA viewing in SampleViewer

The highest-leverage demo win and **independent of Ensembles**. This phase alone makes DDA datasets viewable.

1. Acquisition mode dropdown wired through import wizard (above).
2. MS2 ScanArray construction with precursor metadata (above).
3. SampleViewer plot overlays (Python only, no `.ui` changes):
   - **Precursor badge layer on MS1 spectrum.** For the MS1 scan under the cursor, look up which MS2 scans were triggered from it (via `triggering_ms1_scan_arr`), find the closest peak in the MS1 spectrum to each precursor m/z, render a clickable badge over that peak.
   - **Click handler on badges.** Clicking a precursor badge switches the active MS level to 2 and jumps the cursor to the triggered MS2 scan.
   - **Isolation window overlay on MS2 spectrum.** Shaded m/z band from `isolation_lo_arr` / `isolation_hi_arr` for the displayed MS2 scan.
   - **MS2 header text** showing precursor m/z, charge, RT, scan #.
4. `.mzk` persistence wired and tested with a real DDA file.

### Phase 2 — Ensemble DDA integration

5. Add `precursor_mz`, `precursor_charge` to `Ensemble`.
6. Linkage at ensemble generation: walk MS2 ScanArray's `precursor_mz_arr`, find scans matching the ensemble's m/z within tolerance and RT inside the ensemble's window. Build `ms2_cofeatures` as `FeaturePointer`s pointing at MS2 mass lanes overlapping those scans, with `scan_idxs` restricted to the matched MS2 scans only. (This is the "Ensemble does the compound-level filtering" mechanic.)
7. **`EnsembleViewerWindow.ui` (Designer change):** above the MS2 plot, add a `QHBoxLayout ms2SelectorLayout` containing, left to right:
   - `QToolButton ms2PrevBtn` (~24×24, left arrow icon, fixed size)
   - `QComboBox ms2Selector` (horizontal expanding)
   - `QToolButton ms2NextBtn` (~24×24, right arrow icon, fixed size)
8. Wire MS2 selector in Python: populate combo with each linked MS2 scan, e.g. `"RT=12.34 min  prec=345.12  TIC=1.2e5"`. Pre-select highest-TIC entry. Arrow buttons step through. Selection drives `Ensemble.get_spectrum(scan_num)` into the MS2 plot.
9. Clickable precursor badges on the EnsembleViewer MS1 panel — reuses the SampleViewer overlay code. Clicking a badge selects the corresponding entry in the MS2 dropdown.
10. Migrate `core/cli/export_compound.py` to walk the new linkage when assembling MGF/JSON output.

### Phase 3 — Tools on MS2

11. Confirm formula finder (`gui/views/ensemble_viewer/find_formula.py`) and neutral-loss ruler (`measure_loss.py`) bind correctly to the MS2 plot widget. Expected ~10 LOC of wiring per tool — the widget interface is the same as MS1.

### Summary of `.ui` files needing Designer changes

- **`MzMLImportWizard.ui`** — add `QComboBox acquisitionModeCombo` with items "MS1 only" / "DDA" / "DIA", no default selection, Next/Finish disabled until chosen.
- **`EnsembleViewerWindow.ui`** — `QHBoxLayout ms2SelectorLayout` above the MS2 plot containing `ms2PrevBtn` + `ms2Selector` + `ms2NextBtn`.
- **`SampleViewerWindow.ui`** — no changes; all work is Python overlay code on the existing plot widgets.

**Estimated effort:** 2–3 sessions across the phases.

**Acceptance:**

- Import wizard forces the user to pick an acquisition mode; DDA-mode mzML import populates the new precursor arrays on `Injection.scan_array_ms2`.
- SampleViewer renders precursor badges on MS1 spectra; clicking one switches MS level and jumps to the triggered MS2 scan; MS2 view shows the isolation window overlay and precursor header text.
- An ensemble generated from a DDA injection populates `precursor_mz` / `precursor_charge` and has the right MS2 scans linked.
- EnsembleViewer dropdown + arrow buttons cycle through linked MS2 spectra; clicking a precursor badge on the MS1 panel selects the matching MS2 in the dropdown.
- `export_compound.py` produces MGF output for a DDA ensemble using the new linkage.
- Formula finder and neutral loss ruler work on the MS2 widget.
- A `.mzk` saved with DDA data round-trips through save/load and old `.mzk` files (no precursor arrays) still load.

---

## Workstream 5: MGF viewer (Dropped from V1)

**Goal:** standalone MGF inspection tool, structurally similar to EnsembleViewer.

**Status:** dropped from V1 by the user. Reconsider post-ASMS if there's demand.

**Files (new):**
- `gui/views/mgf_viewer/__init__.py`
- `core/utils/spectrum_export.py` (add `from_mgf` reader — currently write-only)

**Tasks:**
- Add MGF reader to `core/utils/spectrum_export.py`.
- New view that mirrors `ensemble_viewer/` structure: left panel listing spectra grouped by user-selected key field, center `MSPlotWidget`, FormulaFinder dialog usable against the selected spectrum.
- Wire into `SubWindowManager`. New "File → Open MGF" menu action.

**Estimated effort:** 1+ session. Defer unless workstreams 1–4 finish early.

---

## Suggested execution order

1. ~~**Window management**~~ ✅
2. ~~**Process monitor (progress + cancel)**~~ ✅
3. ~~**mzML import optimization (b)**~~ ✅ — (a) rejected, (c) deferred.
4. ~~**Crude DDA handling**~~ ✅ (Phases 1+2) — Phase 3 (formula finder + ruler on MS2) **← next up in a fresh session**, then V1 ships.
5. ~~**MGF viewer**~~ — dropped from V1.

## Cross-cutting

- Maintain `CHANGELOG.md` as you go.
- Add `--version` flag to `mzkit` CLI if not already present.
- Don't forget to regenerate `gui/resources/*.py` from `.ui` after any Qt Designer change: `uv run pyuic5 -o gui/resources/Foo.py gui/resources/Foo.ui`.

---

## Notes from prior sessions

### Workstream 1 — completed

What landed:

- **`gui/controllers/subwindow_controller.py`**
  - Added `_HideOnCloseFilter` (QObject event filter). Intercepts `QEvent.Close` on each `QMdiSubWindow`, calls `obj.hide()` + `event.ignore()`. Also emits `visibilityChanged(bool)` on `ShowToParent` / `HideToParent`.
  - `SubWindowManager` is now a `QObject` and exposes `visibility_changed(str, bool)` aggregated across all filters. Subclassing `QObject` required `super().__init__()` in `__init__`.
  - `add_to_mdi` installs the filter on every newly-added subwindow and lambda-wires its signal to the manager's aggregate signal.
  - `show_window` now calls **both** `sub_window.widget().show()` and `sub_window.show()` then `raise_()` + `setFocus()`. The widget-only show was the original "click menu, nothing happens" bug: callers used to do `widget.show()`, which only un-hides the inner widget while leaving the parent `QMdiSubWindow` itself hidden.
  - New `hide_window(window_type)` and `is_window_visible(window_type)` helpers.
  - `close_window` no longer deletes registry entries — it just hides. Renamed the destructive path to `destroy_window` for the rare case where it's genuinely wanted.
  - Removed the `'analyte_viewer'` entry from `WINDOW_CONFIGS` and its import. Files `gui/views/analyte_table_viewer.py` and `gui/models/analyte_table_list_model.py` left on disk for a later cleanup sweep.

- **`gui/controllers/main_controller.py`**
  - Replaced three `sample_viewer.show()` / `alignment_viewer.show()` / `ensemble_viewer.setFocus()` calls with `self.subwindow_manager.show_window(...)`. These were the inconsistent paths.
  - Added `_connect_sample_viewer_signals` line connecting the new `sigViewEnsembleRequested` to `_handle_view_ensemble_request`.
  - Added `WINDOW_MENU_ACTIONS` class-level dict mapping action object names to `SubWindowManager` keys.
  - Added `_connect_window_menu()` and `_sync_window_menu_check()`. Toggles drive `show_window`/`hide_window`; `visibility_changed` from the manager drives `setChecked` (with `blockSignals` to avoid re-entry).

- **`gui/views/sample_viewer/__init__.py`**
  - Added `sigViewEnsembleRequested = pyqtSignal(object)` (Ensemble).
  - `_open_ensemble_in_viewer` no longer instantiates a fresh `EnsembleViewer` and `.show()`s it floating with a `self._ensemble_viewers` retention list. It now just emits the new signal — the MainController routes to the singleton MDI subwindow.

- **`gui/resources/MainWindow.ui` + `.py`** (edited by user in Qt Designer)
  - New `menuWindow` ("&Window") with five checkable actions:
    `actionWindowSamples`, `actionWindowEnsemble`, `actionWindowAlignment`,
    `actionWindowFingerprint`, `actionWindowProcessMonitor`.

Things future-Claude should be aware of:

- The `Process Monitor` and `Fingerprint Viewer` were previously unreachable from the UI (no menu actions). They're now reachable via the Window menu. Workstream 2 will improve the Process Monitor itself.
- `analyte_table_viewer.py` / `analyte_table_list_model.py` are dead code; safe to delete in a later cleanup pass once we're certain nothing reaches them.
- If you add a new MDI subwindow: register it in `SubWindowManager.WINDOW_CONFIGS`, add an action in `MainWindow.ui` named `actionWindowFoo`, and add a row to `MainController.WINDOW_MENU_ACTIONS`. The rest is automatic.
- The user runs pytest manually / tests via the GUI; do not rely on `uv run pytest` passing — there's a pre-existing `tests/conftest.py` import-path issue (`ModuleNotFoundError: No module named 'core'`) unrelated to this work.

### Workstream 2 — completed (MVP scope)

User scoped this down: cancellation only needed for the long-running tasks (mzML import + feature table import). Everything else just grew no-op kwargs to satisfy the new ProcessRunner contract.

Decision recorded mid-session: went with **explicit kwargs on every CLI function** rather than `inspect.signature`-based injection. Future Claude — if you add a new CLI script callable via `ProcessController.start_process`, its entry function **must** accept `progress_callback=None, cancel_event=None` or the runner will crash with `TypeError`. `core/cli/base.py` defines a `CLITool` ABC that was intended to enforce this; we did not migrate scripts onto it (rushing to MVP). Worth revisiting post-ASMS.

What landed:

- **`core/cli/process_runner.py`**
  - `ProcessRunner` now owns a `progress_queue: queue.Queue` and a `cancel_event: threading.Event`.
  - Always passes `progress_callback=self._emit_progress` and `cancel_event=self.cancel_event` into the wrapped function. `_emit_progress(percent, message)` is `put_nowait` onto `progress_queue` (latest-wins on the consumer side, never raises).
  - Added a `'cancelled'` status. After the wrapped call returns, status is `'cancelled'` if `cancel_event.is_set()`, else `'completed'`. Failed/error paths unchanged.
  - New `get_progress()` drains the queue and returns the most recent `(percent, message)` tuple, or None.

- **`core/controllers/ProcessController.py`**
  - `terminate_process` rewritten: calls `process.cancel_event.set()` instead of mutating `process.status` (which was a no-op since `run()` re-set it anyway).
  - `_poll_active_processes` now also drains `process.get_progress()` and calls `model.updateProcess(progress="42% — message")` — the Progress column was already in the model, just never written to.
  - `'cancelled'` added to the cleanup / process-finished status set, so cancelled processes still trigger the `on_completion_func` callback (with whatever partial result was returned).

- **`gui/views/process_monitor.py`**
  - Renamed button label "Cancel Process" → "Cancel Selected Process".
  - Wired `pushButton.clicked` → `_cancel_selected()`, which reads the table's selected row, parses column 0 as the process id, and calls `process_controller.cancel_process(pid)`.

- **`core/cli/mzml_import.py`** — instrumented:
  - Per-file progress emit before each file: `100 * i / total`, message `Importing {filename} ({i+1}/{total})`.
  - Cancel check at the top of each loop iteration. On cancel, breaks and returns `samples` so far. The completion callback (`sample_controller.on_mzml_import_completion`) was already incremental, so partially-imported samples land in the registry as expected.
  - Final 100% emit only if not cancelled.

- **`core/cli/import_feature_table.py`** — instrumented:
  - Per-feature progress + cancel check inside the outer `for feat in features` loop.
  - On cancel, returns an `EnsembleAlignment` containing whatever analytes have been processed so far. This is *technically* useful (partial alignment is a valid object) but the GUI registers it the same as a complete one — if that turns out to be misleading we can revisit.
  - `pregroup_features` is not instrumented. It's usually fast; revisit if user complaints surface.

- **No-op kwargs added** to the remaining functions the GUI invokes via the runner: `fingerprint_import.main`, `metadata_import.main`, `analyte_table_import.main`, `generate_ensemble.get_cofeature_ensembles`, `generate_ensemble.auto_generate_ensembles`, `align_ensembles.align_ensembles`, `labeling.candidate_generator.main`, `persistence.save_project`, `persistence.load_project`. All two-liner edits, all just absorb the kwargs and ignore.

Smoke tests run during the session:
- All instrumented + shim modules import cleanly.
- All entry points declare `progress_callback` and `cancel_event` in their signature.
- A toy `ProcessRunner` invocation against a synthetic module shows progress flowing through `progress_queue` and `status='completed'` at the end.

Things future-Claude should be aware of:

- The Cancel button currently signals the process whose row is *selected* in the table. If nothing is selected it does nothing — no error, no surprise. Considered showing a confirmation dialog; skipped for MVP.
- `mzml_import` does not check `cancel_event` *inside* a single file — only between files. A multi-GB file will keep going for tens of seconds after cancel. Workstream 3 (numba refactor of `build_features_masscube`) is a natural place to add an inner cancel check.
- `import_feature_table.pregroup_features` is uncancellable. If a user runs pregroup on a huge feature table and hits cancel, the cancel only takes effect once pregroup finishes. Probably fine for now.
- The `'Progress'` column shows strings like `"42% — Importing sample_3.mzML (3/7)"` — no QProgressBar delegate. If you want a real progress bar in the table, that's a `QStyledItemDelegate` exercise; not done for MVP.

### Workstream 3 — completed (partial scope)

User scoped this down mid-session. Plan called for three layers (a, b, c); only (b) was implemented. (a) and (c) were explicitly rejected / deferred.

**(a) auto noise-threshold — rejected.** User dislikes magic auto-thresholding. Prefers a UX path where users set the threshold themselves but more easily (e.g. real-time histogram + draggable threshold line in the import wizard). Out of scope for V1; on user's personal post-ASMS list.

**(c) multiprocessing — deferred.** The (b) speedup already met the bar; the extra complexity (pickling pyopenms objects, interactions with cancellation + log-queue routing) wasn't worth it for V1.

What landed:

- **`core/data_structs/scan_array.py`**
  - `build_features_masscube` renamed to **`build_features`** — we've departed from the original MassCube reference enough that the name was misleading.
  - Inner loop rewritten to operate on **dense parallel-array buffers** instead of the per-feature `Feature` dataclass. Return signature changed from `list[Feature]` to `(out_mz_2d, out_intsy_2d, rt_per_scan)`. `build_scan_array` updated to consume the new shape (skips the wasteful structured-array stacking step).
  - Per-feature m/z matching now uses `np.searchsorted` (pure-Python kernel) / manual binary search (numba kernel) instead of `np.argmin` over the full peak array. **Claim-once semantics preserved exactly:** if the globally-closest peak is already claimed, the feature gap-counts rather than falling back to next-closest — matches legacy `_find_closest_idx` behavior.
  - New `_run_feature_kernel` (pure-Python) and `_run_feature_kernel_numba` (`@njit(cache=True)`). `build_features` auto-dispatches to numba when available; set `MZKIT_DISABLE_NUMBA=1` to force the pure-Python kernel (useful for benchmarking / debugging).
  - **`_build_features_legacy`** (verbatim copy of the old implementation) retained as the parity-test oracle. `_find_closest_idx` retained because legacy still uses it. Both can be deleted once we're confident enough to fly without the safety net.
  - Helpers added: `_flatten_peaks` (list-of-arrays → CSR-style flat input for numba), `_grow_2d` / `_grow_1d_*` (doubling-growth buffers used inside numba). Both pure-Python and numba paths use `np.empty + slice-copy + tail-zero` instead of `np.zeros` to avoid full-buffer memsets on each doubling — matters once the output buffer reaches GB scale.

- **`tests/test_build_features_parity.py`** (new)
  - `MockSpectrum` duck-types `pyopenms.MSSpectrum` (`get_peaks` + `getRT`) so the tests have zero external dependencies.
  - Synthetic spectra generator with Gaussian RT-envelope true features + uniform-random noise peaks, well-separated to avoid tie-break corner cases.
  - Parametrized parity test over 5 seeds, comparing the active path (numba by default) against `_build_features_legacy`. Identical feature count + identical `(mz, intsy)` arrays after row-order normalization.
  - Smoke test for the `<2 peaks per scan → skipped, gap not incremented` legacy quirk.

- **`pyproject.toml`** — `numba>=0.61` added. Compatible with the project's `python>=3.13,<3.14` pin (numba 0.65 supports 3.13).

**Benchmark (real data, 4 mzML files, KINGSTON dataset):**
- Original: ~30 s (extrapolated)
- Pure-Python refactor: 24.0 s (~1.25× from original)
- Numba kernel: 12.5 s (**~2.4× total speedup**)

**Things future-Claude should be aware of:**

- **Numba is slower than pure-Python on pathological inputs.** On a 2000-scan synthetic stress test where features ≈ 64% of total peaks (mostly singletons from random noise), numba came in slower than pure-Python — the per-scan `np.argsort` + buffer growth costs dominated. Real LC/MS data has a much smaller feature/peak ratio so numba wins comfortably. If a future input mix changes this, the `MZKIT_DISABLE_NUMBA` escape hatch is already in place.
- **Stable-sort assumption relaxed in the numba kernel.** The pure-Python kernel uses `np.argsort(-x, kind='stable')` to break intensity ties deterministically (matching the legacy `list.sort`). The numba kernel uses default `np.argsort` (not stable) because stable kinds aren't reliably available in numba. In practice intensity ties on real floating-point data are vanishingly rare and parity holds on synthetic data; if future work surfaces a divergence, this is the place to look.
- **Cancellation hook still missing inside `build_features`.** Workstream 2 notes flagged this. A multi-GB single file will keep going after cancel until the kernel finishes. The numba kernel has no convenient cancel-event check because numba can't read a `threading.Event` cheaply; the right place to add a check is in the per-scan loop of the *pure-Python* outer wrapper. Modest engineering, untouched in this pass.
- **Dense `(n_features, n_scans)` output is still the memory ceiling.** `build_features` immediately hands the dense arrays to `csr_array(...)` which sparsifies them. A future optimization would be to build the CSR directly inside the kernel, skipping the dense intermediate entirely — would help on huge files where the dense buffer pushes RAM limits.
- **`_build_features_legacy` is dead weight in production** but it's the oracle for the parity tests — deleting it deletes the tests' ability to validate against a known-correct reference. Worth keeping until/unless a future refactor obsoletes it.
- **Pre-existing test failures** (`tests/test_ms_data_structures.py::test_mzml_to_injection` — stale signature; `tests/test_persistence.py::test_populate_data_registry` — missing fixture files; `tests/test_find_cofeatures.py` — missing `scan_array` fixture) are unrelated to this work. Worth a separate cleanup pass post-ASMS.

### Workstream 4 — Phases 1 + 2 completed

User scoped the workstream down as it progressed: dropdown/nav-arrow UI in EnsembleViewer dropped in favor of badge-based selection (already implemented for SampleViewer, reused conceptually for EnsembleViewer); compound export migration dropped (user rewriting from scratch); MGF viewer dropped from V1 entirely. Phase 3 (formula finder + neutral-loss ruler verification on MS2 widget) deferred to a follow-up session — see "Loose ends" below.

#### Data model + import pipeline

- **`core/data_structs/scan_array.py`** — `ScanArray` gained five optional precursor-metadata fields (all `Optional[np.ndarray] = None`, length matches `scan_num_arr`):
  - `precursor_mz_arr`, `isolation_lo_arr`, `isolation_hi_arr`, `precursor_charge_arr`, `triggering_ms1_scan_arr`.

- **`core/data_structs/injection.py`** — `Injection` gained `acquisition_mode: Literal['ms1_only','dda','dia'] = 'ms1_only'`. Type alias `AcquisitionMode` exported from this module. `assemble_scan_array` for `ms_level == 2`:
  - Tracks the most recent MS1 scan_num while walking spectra (for `triggering_ms1_scan_arr`).
  - For DDA, harvests `getPrecursors()[0]` per scan into the new arrays.
  - Overrides `scan_gap_tolerance = len(spectra) + 1` for DDA so the mass-lane builder doesn't fragment lanes (interleaved precursors).

- **`core/cli/mzml_import.py`** — `main` and `mzml_to_injection` accept `acquisition_mode: str = 'ms1_only'`, forwarded into `Injection(acquisition_mode=...)`.

- **`gui/resources/MzMLImportWizard.ui` + `.py`** (user edited in Designer) — new `acquisitionModeCombo` on `wizardPage3` with items: `-- Select --`, `MS1 only`, `DIA`, `DDA` (note: **DIA at index 2, DDA at index 3** — user's natural order). The user also unset `groupBoxMS2.checkable` since Python now drives its `setEnabled(...)` off the combo.

- **`gui/dialogues/MzMLImportWizard.py`** — `sigImportParamsGiven` grew a 5th `str` arg (mode). `_acquisition_mode_from_index` does **text-based lookup** off `acquisitionModeCombo.itemText(idx)` rather than hardcoded indices — Designer can reorder items freely without breaking this. `validateCurrentPage` rejects page 3 until a real mode is chosen. `groupBoxMS2.setEnabled(mode in ('dda', 'dia'))`.

- **`gui/controllers/sample_controller.py`** + **`gui/controllers/main_controller.py`** — `sigMzMLImportWizardComplete` grew the `str` arg; `_run_mzml_import_process` adds `acquisition_mode` to the `ProcessRunner` parameters dict.

- **`core/utils/persistence.py`** — `injection.json` now stores `acquisition_mode`; loader uses `.get('acquisition_mode', 'ms1_only')` for old files. ScanArray precursor fields ride along via the existing `__dict__` pickle path — old pickles deserialize fine because dataclass defaults the missing keys to `None`.

#### Phase 1 — SampleViewer DDA overlays

- **`gui/views/sample_viewer/dda_overlays.py`** (new, ~190 LOC) — `DDAOverlayManager`:
  - On MS1: red diamond (`symbol='d'`) badges at the precursor m/z of each MS2 scan triggered from the current MS1 scan. Y position anchored to the nearest peak's intensity in the displayed spectrum (so badges sit on parent ions).
  - On MS2: blue `LinearRegionItem` over the isolation window if `hi - lo > 1e-4`, else a vertical `InfiniteLine` at the precursor m/z (Waters DDA exports zero-width). Header label via `MSPlotWidget.update_label`: `MS2 precursor m/z X.XXXX z=±N RT=… scan #…`.
  - Bails out cleanly when `acquisition_mode != 'dda'` or `precursor_mz_arr is None` — non-DDA samples cost nothing.
  - Badge click → `selection_mgr.set_selected_spectrum_by_scan_num(uuid, ms_level=2, scan_num=ms2_idx)`.
  - Two click-suppression guards in the click handler:
    - `tool_mgr.active_tool != ToolType.NONE` → skip. Prevents collisions with ensemble extraction / XIC selection / spectrum-grab tools.
    - `self.plot.hovered_ms_signal = None` set before any handling. Otherwise `MSPlotWidget.mousePressEvent` (lines 547–554) unconditionally fires `MSSignalClicked()` after super(); since badges sit on peak tops, the hover state always points at a peak and the click would also be interpreted as an MS1-signal-click (firing `on_ms1_signal_clicked` and adding stray chromatograms).
  - `points` from `ScatterPlotItem.sigClicked` is a **numpy array**; check `len(points) == 0` instead of `not points` (the latter raises ambiguous-truth ValueError).

- **`gui/views/sample_viewer/__init__.py`** — instantiates `dda_overlay_mgr`, passes `tool_mgr=self.tool_mgr`. `update_spectrum_plot()` calls `dda_overlay_mgr.update(sample_uuid, ms_level, scan_index)` after `setSpectrumArray`. Subscribes `selection_mgr.sigMSLevelSelected` to a new `_sync_ms_level_combo` slot that uses `blockSignals` to avoid feedback into `on_ms_level_change_requested` — so the MS1/MS2 combobox tracks programmatic level changes (e.g. badge clicks).

#### Phase 2 — Ensemble DDA integration

- **`core/data_structs/ensemble.py`** — `Ensemble` gained `precursor_mz: Optional[float]` and `precursor_charge: Optional[int]`, populated at construction for DDA ensembles.

- **`core/cli/generate_ensemble.py`** — `get_cofeature_ensemble` branches on `injection.acquisition_mode`:
  - DDA → new helper `_dda_link_ms2_cofeatures(injection, ms1_cofeatures, search_ftr_ptr, min_intsy, precursor_mz_tolerance)`. Trigger-based linkage: walk MS2 `precursor_mz_arr`, match against any MS1 cofeature's lane-label m/z within tolerance AND within the search RT window. For each matched scan, find MS2 mass lanes with `max intensity ≥ min_intsy`; build one `FeaturePointer` per active lane with `scan_idxs = matched_ms2_idxs`. Returns `(cofeatures, precursor_mz, precursor_charge)` — precursor_mz is the median across matched scans, charge is the modal non-zero charge.
  - Else (DIA / fallback): original `find_cofeatures_across_scan_array` correlation path.
  - `EnsembleExtractionParams` gained `precursor_mz_tolerance: float = 0.5` (TODO: expose in settings menu post-ASMS).
  - `auto_generate_ensembles` was **intentionally NOT updated** — user is deprecating it.

- **`core/utils/persistence.py`** — `precursor_mz` and `precursor_charge` saved/loaded with `.get(...)` defaulting to `None`. Backwards compatible.

- **`gui/views/ensemble_viewer/dda_overlays.py`** (new, ~190 LOC) — `EnsembleDDAOverlayManager`:
  - Same visual conventions as SampleViewer overlay (red diamonds for badges, blue for isolation).
  - Badges derived from `ensemble.ms2_cofeatures[0].scan_idxs` — all DDA MS2 cofeatures share the same scan_idxs by construction.
  - Takes an `on_select_rt` callback (the viewer provides one); badge click invokes that callback with the target RT instead of relying on `sigPositionChanged`.
  - Same two click-suppression guards as the SampleViewer overlay (numpy `len()` check + clear `hovered_ms_signal`).

- **`gui/views/ensemble_viewer/__init__.py`** — instantiates `dda_overlay_mgr` and threads `on_select_rt=self._select_ensemble_ms2_rt`. Updates the overlay after every `populate_spectrum_plot` (three call sites: `initialize_plots`, `onChromatogramSelectorMoved`, `_select_ensemble_ms2_rt`).
  - **`_snap_rt_to_ensemble(rt)`** — for DDA ensembles, snaps the requested RT to the nearest RT among `ensemble.ms2_cofeatures[0].scan_idxs`. Returns `rt` unchanged for non-DDA. Used at all three populate call sites so the cursor can't "escape" the ensemble and show scans triggered by unrelated precursors.
  - **`_select_ensemble_ms2_rt(rt)`** — programmatic equivalent of dragging the chrom selector. Snaps, moves the indicator with `QSignalBlocker` to avoid re-entry into `onChromatogramSelectorMoved`, then directly calls `populate_spectrum_plot` + `dda_overlay_mgr.update` + `_redraw_annotations_for_current_scan`. One explicit code path; no reliance on signal ordering.

#### Annotation system: snapshot + scan-tied display

This wasn't on the original plan but landed in the same session because the user hit a neutral-loss-ruler bug ("number jumps to 0 or to the mass of the first signal on commit") that was particularly visible on DDA MS2 (sparse mass lanes).

Root cause: `Ensemble.add_mz_diff_annot` was recomputing `delta_mz` from `Ensemble.get_spectrum(scan_rt=self.peak_rt)`. For DDA MS2 lanes that aren't active at `peak_rt`, the lookup returns 0.0 → wrong `delta_mz`. User wanted to additionally make annotations scan-tied (only shown when viewing the scan they were taken at) since `MSPlotItem.setSpectrumArray` already clears them on every spectrum change anyway.

- **`core/data_structs/ensemble.py`**:
  - `MzDiffAnnotation` and `IonAnnotation` both gained `scan_num: Optional[int] = None`.
  - `Ensemble.add_mz_diff_annot` now accepts `delta_mz` directly and stores it (snapshot of what the user saw); dead recomputation code removed. New optional `scan_num` kwarg.
  - `Ensemble.add_ion_annot` gained `scan_num` kwarg.

- **`gui/views/ensemble_viewer/measure_loss.py`** — `sigMzDiffMeasured` grew a `float` arg for the snapshot `delta_mz`. The click handler computes `delta_mz = abs(mz - selected_mz)` from the values the user saw in the preview (which is now what gets committed).

- **`gui/views/ensemble_viewer/__init__.py`**:
  - `_current_scan_num(ms_level)` helper — maps `spectrum_manager.selected_rt` to a scan index via `ScanArray.rt_to_scan_num`.
  - `add_mz_diff_annotation` and `add_formula_annotation` capture `scan_num=self._current_scan_num(ms_level)` at annotation creation.
  - `_restore_annotations` renamed to `_redraw_annotations_for_current_scan`. Filters by `annot.scan_num == _current_scan_num(annot.ms_level)`. `scan_num is None` → always shown (back-compat for pre-existing `.mzk` files).
  - Hooked into every spectrum-update path: `set_ensemble`, `onChromatogramSelectorMoved`, `_select_ensemble_ms2_rt`.

- **`core/utils/persistence.py`** — `MzDiffAnnotation` uses `**dict` so `scan_num` rides along automatically; `IonAnnotation` reconstruction needed one explicit `scan_num=annot_dict.get('scan_num')` line.

#### Things future-Claude should be aware of

- **Sparse MS2 chromatograms for DDA ensembles.** `FeaturePointer.get_intensity_values` slices `scan_start:scan_end` as a contiguous range. For DDA, the matched MS2 scans are interleaved with scans of other precursors — so the chromatogram returned will include intensity from neighboring precursors' MS2 scans. Spectrum display is correct; chromatograms will look noisy. Spectrum-driven workflows are unaffected; future per-precursor MS2 XIC tools will want a different access pattern.

- **MS-community visual conventions** are now hardcoded across both overlay modules: red diamonds for precursor badges, blue for isolation windows. If you change one, change the other.

- **`MSPlotWidget.mousePressEvent` (lines 547–554) is a click-handling minefield.** It unconditionally fires `MSSignalClicked()` after super whenever `hovered_ms_signal` is set. ScatterPlotItem clicks DON'T suppress this because the hover state was set before the click event by the snapping hover detection. The badge handlers clear `hovered_ms_signal` before doing their thing as a workaround. If you add another overlay that responds to clicks, you'll need the same trick.

- **`MzMLImportWizard` combo ordering trap.** Items in the .ui file are currently `-- Select --` / `MS1 only` / `DIA` / `DDA` (i.e. DIA comes before DDA). The Python uses **text-based lookup** so this is safe to reorder in Designer, but if you ever switch to index-based lookup, *check the order*. This caused a real bug ("acq_mode='dia' when user picked DDA") earlier in the session.

- **Snap-to-ensemble is essential for DDA scrubbing UX.** Without it, the chrom cursor lands on whatever MS2 scan is closest by RT — which is frequently a scan triggered by an unrelated precursor. The displayed MS2 spectrum (filtered to this ensemble's mass lanes) then looks nearly empty / nonsense. This was the user's most confusing pre-fix symptom.

- **Annotation rendering invariant.** `MSPlotItem.setSpectrumArray` calls `clear_anchored_labels() / clear_ion_annotations() / clear_delta_brackets()`. Anything added between `populate_spectrum_plot` and `_redraw_annotations_for_current_scan` gets nuked by the next spectrum update. Currently nothing else does that, but it's a thing.

#### Loose ends (Phase 3 + polish for the next session)

- **Formula finder on MS2 widget.** Plan said "~10 LOC of wiring." User reports it "works" but there's something subtly off in how it handles MS2 — needs investigation. No verification done in this session.
- **Neutral-loss ruler on MS2 widget.** The snapshot bug is fixed but the user mentioned "still some glitches" they want to chase down. Likely related to how the ruler interacts with scan-tied annotation display, or to MS2 hovering. Needs reproduction.
- **`precursor_mz_tolerance` UI exposure.** Currently hardcoded at 0.5 Da default in `EnsembleExtractionParams`. Plan calls for it to land in the settings menu post-ASMS, not for V1.
- **Cancellation inside `build_features`.** Still missing (carryover from Workstream 3). Not a V1 blocker.
- **Compound export migration (plan step 10).** User is rewriting from scratch — explicitly out of scope.
- **MGF viewer.** Dropped from V1.
- **`auto_generate_ensembles` is not branched for DDA** — it still uses the correlation path. User is deprecating it, so left alone.

### EnsembleViewer polish pass — completed

Small/medium polish items the user asked for ahead of the more substantial annotations rework. All five landed in one session.

What landed:

- **`gui/widgets/MSPlotWidget.py`**
  - New top-right `bpc_label: QLabel` + `update_bpc_label(text)` method. Hidden when empty. `resizeEvent` repositions it; guarded against the call pyqtgraph's `PlotWidget.__init__` makes *before* our subclass's `__init__` runs (uses `self.__dict__.get('bpc_label')` to bypass pyqtgraph's overridden `__getattr__` that raises on missing attrs).
  - **`MSPlotItem.updateSpectrumPlot`** now calls `self.scene().update()` + `self.plot_widget.viewport().update()` at the end. Fixes a long-standing "stale primitives left painted until the user nudges the viewbox" bug across both SampleViewer and EnsembleViewer spectrum plots. Cheap, runs once per spectrum update.

- **`gui/views/ensemble_viewer/plot_managers.py`** — `SpectrumPlotManager`
  - New `_normalize_spectra` flag + `set_normalize_spectra(bool)`. When on, the spectrum's intensity array is divided by its own max before being handed to the plot — switching between low- and high-abundance precursors no longer rescales the viewbox jarringly.
  - `populate_spectrum_plot` now also emits the un-normalized BPC via `plot.update_bpc_label(f"BPC: {bpc:.2e}")` — visible at-a-glance intensity readout in the top-right of each MS plot. Empty when spectrum has no intensity.

- **`gui/views/ensemble_viewer/plot_managers.py`** — `ChromatogramPlotManager`
  - `_is_dda()` helper.
  - `update_correlation_plot` skips the MS2 series entirely on DDA injections (DDA MS2 features were never grouped by correlation — plotting them was misleading; it was a debug leftover).
  - **Pearson r² + Spearman ρ²** computed via `scipy.stats.{pearsonr, spearmanr}` over the most-recently-matched intensity pair. Degenerate inputs (constant series, len < 3, NaNs) fall back to `"-"` rather than blowing up.
  - New `_update_correlation_title(...)` writes the corr plot's title: `Seed m/z X.XXXX → MSn m/z Y.YYYY    Pearson r² = …    Spearman ρ² = …`. Pre-selection state shows a "click a signal to populate" hint instead.
  - New `set_last_selection(mz, ms_level)` — called from `EnsembleViewer.on_ms1_signal_clicked` / `on_ms2_signal_clicked` so the corr plot title knows which target was selected.
  - **`_apply_selection_bounds()`** — calls `chrom_plot.pi.selection_indicator.setBounds([rt_min, rt_max])` derived from `self.base_chrom['rt'].min/max`. Wired into `populate_chromatogram_plot`. Locks dragging of the chrom cursor to the ensemble's own RT span so it can't wander out into unrelated scans.

- **`gui/views/ensemble_viewer/__init__.py`**
  - `checkNormalizeSpectra.toggled` → `_on_normalize_spectra_toggled` — flips the flag on `spectrum_manager`, re-populates both MS plots, refreshes DDA overlay, redraws annotations.
  - `on_ms1_signal_clicked` / `on_ms2_signal_clicked` both call `chrom_manager.set_last_selection(mz, ms_level)` before `update_correlation_plot()`.

- **`gui/views/{ensemble,sample}_viewer/dda_overlays.py`**
  - DDA badge `ScatterPlotItem`s now do `scatter.getContextMenus = lambda event=None: None` — kills the pyqtgraph default right-click context menu (the "dropdown") that was appearing on badges. Same change made in both overlays for consistency.
  - `scatter.setCursor(QtCore.Qt.PointingHandCursor)` — hand cursor on hover signals clickability without an extra hover-state machine.

Things future-Claude should be aware of:

- **`MSPlotWidget.bpc_label` init-order quirk.** pyqtgraph's `PlotWidget.__init__` calls `resizeEvent` before subclass attributes exist. The guard pattern (`self.__dict__.get('bpc_label')`) matters because pyqtgraph overrides `__getattr__` to raise `AttributeError` rather than return `None` — a plain `hasattr` works but is noisier. Any future widget-positioned-in-resizeEvent additions need the same trick.
- **Spectrum-plot scene update is on the hot path.** Spectrum changes are frequent (every cursor scrub on DDA). The added `scene().update()` + `viewport().update()` runs *every* spectrum redraw. If it ever shows up in profiling, restrict it to cases where stale items are likely (e.g. when annotations were cleared).
- **Correlation title uses the last overlap pair only.** When multiple targets are selected (MS1 + MS2 click history accumulates in `ms1_chroms` / `ms2_chroms`), Pearson/Spearman reflect just the most recent one. The other scatters are still plotted; this is intentional — the title tracks the user's last interaction, not an aggregate.
- **DDA badge context-menu suppression is by attribute reassignment**, not subclassing. If pyqtgraph changes how `getContextMenus` is consumed (e.g. starts treating `None` differently from `[]`), revisit. Currently both pyqtgraph's `ViewBox.getMenu` and the scatter chain handle `None` gracefully.
- **Selection-indicator bounds are derived from `base_chrom` RT min/max**, not from `ensemble.ms2_cofeatures[0].scan_idxs` (which is the DDA-snap source). The two ranges are close but not identical. If you ever need stricter DDA-only locking, that's the place to switch.

### Annotation management — completed

Three-phase rework letting users delete annotations, add free-form labels, and find chromatographic annotated scans at a glance. Plus a neutral-loss-formula option added in a follow-up. All shipped in one session.

Phases (in chronological order):

- **Phase A — Clear buttons.** Two toolbar actions in `EnsembleViewerWindow.ui` (`actionClearScanAnnots`, `actionClearAllAnnots`). "Clear (scan)" wipes annotations whose `scan_num` matches the displayed scan (scan-agnostic annotations preserved). "Clear (all)" wipes everything (mz_diffs, ion_annots, ion_pair_annots, generic_annots) behind a `QMessageBox` confirm.

- **Phase B — Right-click delete.**
  - Added `uuid: int = field(default_factory=…)` to `MzDiffAnnotation` (was the only annotation type missing stable identity). Back-compat is automatic via dataclass defaults — old `.mzk` files deserialize fine.
  - **`gui/widgets/MSPlotWidget.py`** — new `ClickableTextItem(pg.TextItem)` with `on_click(button)` callback (right-button only by default; pointing-hand cursor on hover). `IonAnnotationGraphic` and `DeltaMzBracket` gained optional `on_click` kwargs that they forward to their text label. `create_textitem` gained an `on_click` kwarg that swaps in `ClickableTextItem`. **`MSLabelManager` is now a `QObject`** with `sigAnnotationClicked = pyqtSignal(str plot_id, int button)`. Each `add_*` method builds a closure capturing the annotation's `annot_id` and wires it through the click callback. `MSPlotWidget.sigAnnotationClicked` forwards from the label manager.
  - **EnsembleViewer transient plot-id maps** (`_mz_diff_plot_ids`, `_ion_annot_plot_ids`, `_generic_annot_plot_ids` — all `dict[annot_uuid, plot_id]`). Cleared at the top of every `_redraw_annotations_for_current_scan` because plot IDs become invalid on redraw. Populated as each `_draw_*` runs and captures the return value of `add_*`.
  - **`_on_annotation_clicked(plot_id, button, ms_level)`** — only acts on right-click. Reverse-looks-up the plot_id across the three maps to find `(annot_uuid, kind)`, pops a single-item `QMenu` ("Delete annotation"), removes from the matching collection on confirm, redraws.

- **Phase C — Add-annotation tool.**
  - `GenericAnnotation(cofeature_idx, ms_level, text, uuid, scan_num)` dataclass on `core.data_structs.ensemble`. `Ensemble.generic_annots: dict[int, GenericAnnotation]` + `add_generic_annot(...)` method (validates cofeature_idx range like the other `add_*`s).
  - **`gui/views/ensemble_viewer/add_annot.py`** (new) — `AddAnnotController(BaseToolController)` with `sigGenericAnnotationRequested(int cofeature_idx, int ms_level, str text)`. Lifecycle: `on_activated` → `request_next_stage` (IDLE → SELECTING); click peak → `QInputDialog.getText`; emit the signal if accepted with non-empty text; `request_cancel()` either way (otherwise the next click would pop another dialog with no visual cue). Registered in `tool_controllers` and `tool_map`/`_update_tool_buttons` rows alongside FindFormula/MeasureLoss.
  - `_draw_generic_annotation` uses the existing `add_anchored_label` primitive (no new graphic type needed).
  - **Persistence** — `generic_annots` serializes via `asdict` and reloads via `**dict`, with `.get('generic_annots', {})` for back-compat.

- **Chromatogram annotation markers.** After A/B/C landed, user wanted at-a-glance scan locators on the chrom plot.
  - **`ChromatogramPlotManager.set_annotation_markers(rts_by_level: dict[int, list[float]])`** — stores per-MS-level RT lists, then `_render_annotation_markers()` does the work.
  - Markers are `pg.ScatterPlotItem` with `symbol='t1'` (down-triangle). Y values come from `np.interp(rt, bpc_rt, bpc_intsy)` against the *transformed* background chrom — so they sit on the trace and re-anchor automatically when normalize/diff toggles fire.
  - MS1 = magenta (220,60,220), MS2 = green (40,180,60). Matches the convention used by `update_correlation_plot`.
  - `populate_chromatogram_plot` calls `_render_annotation_markers()` at the end — necessary because rebuilding the chrom plot wipes prior scatter items.
  - **`EnsembleViewer._refresh_chrom_annotation_markers`** collects unique RTs from all three annotation collections (skipping `scan_num is None`), maps `scan_num → scan_array.rt_arr[scan_num]` per ms_level, and pushes to the manager. Hung off `_redraw_annotations_for_current_scan` (which fires for every annotation lifecycle event) plus the three `add_*` paths (which don't redraw).

- **Neutral-loss formula annotation.** Follow-up after the user asked for nice figure labels.
  - `MzDiffAnnotation` gained `formula: Optional[FormulaCandidate] = None`. `Ensemble.add_mz_diff_annot` accepts a `formula` kwarg.
  - **`gui/views/ensemble_viewer/measure_loss.py`** — second-click branch emits `sigMzDiffMeasured` immediately (plain bracket appears), then calls `_open_finder_for_neutral_loss(delta_mz)` which:
    - Sets `spinCharge.setValue(0)` (saves the prior value for restoration).
    - `populate_table([(delta_mz, 1.0)])`.
    - Shows the finder + runs `on_search_execute()`.
    - Marks `_pending_finder_mode = 'neutral_loss'`.
  - `eventFilter` on the finder catches `QEvent.Hide` and calls `request_cancel()` if `_pending_finder_mode` was set — so closing the dialog without picking still exits the tool cleanly.
  - `handle_formula_assigned` routes by `_pending_finder_mode`: neutral_loss → emit `sigMzDiffFormulaAssigned(formula)` + hide finder + restore charge; ion → legacy Enter-press path.
  - **`EnsembleViewer._last_mz_diff_uuid`** tracks the most recently added MzDiffAnnotation (updated in `add_mz_diff_annotation`). `_on_neutral_loss_formula_assigned(formula)` looks up that annotation, sets `target.formula = formula`, clears delta brackets on both plots, and redraws.
  - **Two-line bracket label** when `annotation.formula is not None`: formula HTML (via `format_formula_obj_to_html`) on top, `Δ {delta_mz:.4f}` in 8pt below. Otherwise the plain `Δ {delta_mz:.4f}` label as before.
  - **Persistence** — MzDiffAnnotation no longer round-trips via blanket `asdict` (FormulaCandidate's inner `Formula` object isn't asdict-able). Now serialized field-by-field; formula written as `{formula_str, error_ppm}` mirroring `IonAnnotation`. Loader reconstructs via `Formula(formula_str)` + `FormulaCandidate(...)`. Back-compat via `mz_diff_dict.get('formula')`.

UI changes (Designer, by user):

- **`EnsembleViewerWindow.ui`** — three new QToolButtons in the tool layout: `toolAddAnnot`, `toolClearScanAnnots`, `toolClearAllAnnots`. Three matching QActions: `actionAddAnnot` (checkable, "Annot", Shift+A), `actionClearScanAnnots` (Shift+Backspace), `actionClearAllAnnots` (Ctrl+Shift+Backspace). The two clear actions' default text reads as the object name (cosmetic — would be nicer as "Clear (scan)" / "Clear (all)" in Designer, but the buttons themselves have proper labels).

Things future-Claude should be aware of:

- **Plot IDs are transient.** They live only until the next `_redraw_annotations_for_current_scan` call (every spectrum scrub triggers one). The three plot-id maps are cleared at the top of that method and rebuilt as `_draw_*` runs. Any code that wants to act on a plot_id must do so within one redraw cycle.
- **Annotation `scan_num` is an **index** into `ScanArray.rt_arr`**, not a raw scan number from `scan_num_arr`. `_current_scan_num` returns `int(scan_array.rt_to_scan_num(rt))` and `rt_to_scan_num` is implemented as `np.argmin(abs(rt_arr - rt))`. Mapping back to RT is just `rt_arr[scan_num]`. If `ScanArray` ever stops storing per-row RTs, this assumption breaks.
- **MzDiffAnnotation persistence is no longer `asdict`-based** because of the FormulaCandidate field. If you add a new field, update the manual dict construction in `serialize_ensembles` and `deserialize_injection_ensembles` in `core/utils/persistence.py`.
- **`ClickableTextItem` defaults to right-button only.** Left-clicks still go through pyqtgraph's normal viewport handling (pan, zoom, rubber-band select). If you want left-click semantics on an annotation, pass `accepted_buttons=Qt.RightButton | Qt.LeftButton` to the constructor — but watch out for the existing `MSPlotWidget.mousePressEvent` minefield (the V1_PLAN W4 notes cover that).
- **The `_pending_finder_mode` state on MeasureLossController** is single-mode. If you ever add a third path that opens the same finder, the routing in `handle_formula_assigned` and the Hide event filter both need updating.
- **The neutral-loss search uses the user's existing element constraints** (max_counts, RDBE range, etc.) which are tuned for positive-mode ions. Some plausible neutral losses get filtered out (e.g. an H2O loss is fine, but something like NH3 might be missed if min_counts excludes it). A "loss preset" or auto-relaxation could be nice post-ASMS, but not a V1 blocker — the user explicitly said no validation, this is a labelling aid.
- **`_last_mz_diff_uuid` is a global single-slot state.** It's only valid because the formula finder is window-modal and the Hide event clears the pending mode. If the finder ever becomes non-modal or the user gains another path to create MzDiffAnnotations in parallel, the slot can race.
- **Chrom markers re-render on every spectrum scrub** (because `_redraw_annotations_for_current_scan` triggers the refresh and that runs on every scrub). The marker count rarely changes between scrubs, so this is wasted work — a `_markers_dirty` flag is a clean optimization if it ever shows up in a profile.

#### Loose ends

- **Pre-existing test failures** unchanged from W3 notes. The new annotation code has no automated tests; verification was manual through the UI (the user exercised every Phase A/B/C path + chrom markers + neutral-loss formula round-trip).
- **Two-line bracket label uses raw HTML.** No styling configuration; the size (8pt) is hardcoded in `_draw_mz_diff_annotation`. If the user wants per-figure styling control later, that's the place.
- **Tooltip / action display text** in the .ui currently shows raw object names ("actionClearScanAnnots") because Designer didn't get human-friendly text set. Cosmetic only — the QToolButton labels are correct.
