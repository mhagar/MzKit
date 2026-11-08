"""
Controller for the whole-lane span labeling window.

Bridges `Ui_MainWindow` (from LabelingWindow.ui) to the Qt-free logic in
`core.labeling`.

Shows annotator one mass lane at a time.

Annotator must drag an RT range and mark it peak (Q) or plateau (T)
Unmarked signal counts as "other" (i.e. minor blips).

(Space) saves the lane as fully reviewed.

Hotkeys kept under left hand so mouse can select/zoom.
See docs/labeling-policy.md.
"""
from __future__ import annotations
import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt

from gui.resources.LabelingWindow import Ui_MainWindow
from core.labeling import LaneCandidate, LaneLabel, LabelFile, PEAK, PLATEAU, insert_span
from core.labeling import peak_model
from core.utils.array_types import to_chrom_arr

if TYPE_CHECKING:
    from core.data_structs import Sample

logger = logging.getLogger(__name__)

# Selection edges this close to a span edge snap onto it
_SNAP_SCANS = 2

# Colours
_SPAN_BRUSH = {PEAK: (40, 180, 70, 70), PLATEAU: (220, 50, 50, 70)}
_MODEL_PEN = {PEAK: (20, 120, 40), PLATEAU: (170, 20, 20)}  # review: model outlines

_LEGEND_HTML = (
    "Click + drag to select RT range. Q: label as PEAK. T: label as PLATEAU. "
    "X: delete region under cursor. Z: undo. Space: lane reviewed, next. "
    "Shift+Space: skip. W,A,S,D or middle click: Pan/zoom. F: fit lane. "
    "Unmarked signals can be used as co-features but not seed features."
)


class LabelingWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        samples: list["Sample"],
        candidates: list[LaneCandidate],
        label_file_path: Path,
        session_id: str = "",
        annotator: str = "",
        extraction_params: Optional[dict] = None,
        process_controller=None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.optionalSpectrumFrame.setVisible(False)

        self._current_sample_uuids = sorted(s.uuid for s in samples)
        self.label_file_path = Path(label_file_path)
        self.session_id = session_id or datetime.datetime.now().isoformat()
        self.annotator = annotator
        self.process_controller = process_controller

        if self.label_file_path.exists():
            self.label_file = LabelFile.from_json(self.label_file_path)
            logger.info(
                f"Resumed label file with {len(self.label_file.lanes)} lanes"
            )
            # Lanes carry their own XICs, so one file can grow across projects;
            # sample_uuids just records every sample it has drawn from.
            self.label_file.sample_uuids = sorted(
                set(self.label_file.sample_uuids) | set(self._current_sample_uuids)
            )
        else:
            self.label_file = LabelFile(
                extraction_params=dict(extraction_params or {}),
                sample_uuids=list(self._current_sample_uuids),
            )

        self._candidates = [
            c for c in candidates
            if not self.label_file.has_lane(c.sample_uuid, c.lane_idx)
        ]
        self._current_idx = 0
        self._spans: list = []
        self._undo: list[list] = []
        self._committed: list[int] = []  # candidate idxs saved this session, in order
        self._selection: Optional[tuple[float, float]] = None
        self._cursor_rt: Optional[float] = None
        self._span_items: list[pg.LinearRegionItem] = []
        self._model = self._load_model()

        # "Review mode"
        # Show already-labeled lanes, in order of worst disagreement w/ ground truth labels.
        self._review: Optional[list[dict]] = None   # out-of-fold model results
        self._review_labels: list[LaneLabel] = []
        self._paused_queue = None                    # labeling state to restore
        self._n_corrected = 0

        pw = self.ui.xicPlotWidget
        pw.set_seed_drag_enabled(True)  # left-drag draws an RT range
        pw.sigSeedBarDrawn.connect(self._on_range_drawn)
        pw.sigChromatogramHovered.connect(lambda p: setattr(self, "_cursor_rt", p.x()))

        self._init_stats_table()
        self._connect_menu_actions()
        self._refresh_stats()
        self._show_current_lane()

    def _init_stats_table(self) -> None:
        self.ui.hotkeyLegendText.setHtml(_LEGEND_HTML)
        table = self.ui.classCountsTable
        headers = ["Lanes", "Peak spans", "Plateau spans"]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(1)
        for col in range(len(headers)):
            item = QtWidgets.QTableWidgetItem("0")
            item.setTextAlignment(Qt.AlignCenter)
            table.setItem(0, col, item)

    def _connect_menu_actions(self) -> None:
        ui = self.ui
        ui.actionExit.triggered.connect(self.close)
        ui.actionSave_labels_as_JSON.triggered.connect(self._save_labels)
        # whole-lane view: the per-apex spectrum / y-normalization toggles don't apply
        ui.actionToggle_MS1_spectrum.setVisible(False)
        ui.actionToggle_Y_axis_normalization.setVisible(False)
        # Added in code: the generated Ui file must not be hand-edited
        self._train_action = ui.menuSession.addAction("Train peak model")
        self._train_action.triggered.connect(self._train_model)
        self._train_action.setEnabled(self.process_controller is not None)
        self._review_action = ui.menuSession.addAction("Review model errors")
        self._review_action.triggered.connect(self._start_review)
        self._review_action.setEnabled(self.process_controller is not None)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def _cand(self) -> LaneCandidate:
        return self._candidates[self._current_idx]

    def _show_current_lane(self, spans: Optional[list] = None) -> None:
        self._selection = None
        self._undo = []
        if self._current_idx >= len(self._candidates):
            self._on_session_complete()
            return
        cand = self._cand()
        pw = self.ui.xicPlotWidget
        pw.clear_seed_preview()
        pw.setChromArray(to_chrom_arr(cand.rt_values, cand.intsy_values), replace=True)
        pw.getPlotItem().vb.autoRange()

        # Proposals come only from a model trained on the user's own labels,
        # never from the old rules (whose mistakes would leak into the labels)
        if spans is None and self._review is not None:
            spans = self._review_labels[self._current_idx].spans
        if spans is None:
            spans = (
                peak_model.segment_lane(self._model, cand.intsy_values, cand.rt_values)[1]
                if self._model is not None else []
            )
        self._spans = list(spans)
        self._draw_spans()

        self.ui.sourceLabel.setText(cand.sample_name)
        self.ui.mzLabel.setText(f"m/z {cand.mz:.4f}")
        self.ui.scanLabel.setText(f"lane {cand.lane_idx}")
        self.ui.progressLabel.setText(
            f"{self._current_idx + 1} / {len(self._candidates)}"
        )
        if self._review is not None:
            r = self._review[self._current_idx]
            self.ui.progressLabel.setText(
                f"REVIEW {self._current_idx + 1}/{len(self._review)} — model: "
                f"{r['fp']} extra / {r['fn']} missed peaks, plateau off by "
                f"{r['plateau_diff']} scans (dashed = model)"
            )

    def _draw_spans(self) -> None:
        plot_item = self.ui.xicPlotWidget.getPlotItem()
        for item in self._span_items:
            plot_item.removeItem(item)
        rt = self._cand().rt_values
        self._span_items = []
        for s, e, c in self._spans:
            item = pg.LinearRegionItem(
                values=(rt[s], rt[e - 1]), movable=False,
                brush=pg.mkBrush(*_SPAN_BRUSH[c]),
            )
            item.setZValue(-10)
            plot_item.addItem(item)
            self._span_items.append(item)
        if self._review is not None:  # model's out-of-fold spans as dashed outlines
            for s, e, c in self._review[self._current_idx]["pred"]:
                item = pg.LinearRegionItem(
                    values=(rt[s], rt[e - 1]), movable=False,
                    brush=pg.mkBrush(0, 0, 0, 0),
                    pen=pg.mkPen(_MODEL_PEN[c], width=2, style=Qt.DashLine),
                )
                item.setZValue(-5)
                plot_item.addItem(item)
                self._span_items.append(item)

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------
    def _on_range_drawn(self, rt_start: float, rt_end: float, _intensity: float) -> None:
        self._selection = (rt_start, rt_end)

    def _selection_scans(self) -> Optional[tuple[int, int]]:
        if self._selection is None:
            return None
        rt = self._cand().rt_values
        s = int(np.searchsorted(rt, self._selection[0], "left"))
        e = int(np.searchsorted(rt, self._selection[1], "right"))
        # snap onto nearby span edges so touching peaks share an exact edge
        edges = [x for sp in self._spans for x in sp[:2]]
        for edge in edges:
            if abs(s - edge) <= _SNAP_SCANS:
                s = edge
            if abs(e - edge) <= _SNAP_SCANS:
                e = edge
        return (s, e) if e > s else None

    def _edit(self, new_spans: list) -> None:
        self._undo.append(self._spans)
        self._spans = new_spans
        self._draw_spans()

    def _mark(self, cls: str) -> None:
        scans = self._selection_scans()
        if scans is None:
            self.ui.statusbar.showMessage("Drag an RT range first", 1500)
            return
        self._edit(insert_span(self._spans, *scans, cls))
        self._selection = None
        self.ui.xicPlotWidget.clear_seed_preview()

    def _delete_under_cursor(self) -> None:
        if self._cursor_rt is None:
            return
        scan = int(np.searchsorted(self._cand().rt_values, self._cursor_rt))
        kept = [sp for sp in self._spans if not sp[0] <= scan < sp[1]]
        if len(kept) != len(self._spans):
            self._edit(kept)

    def _undo_last(self) -> None:
        if self._undo:
            self._spans = self._undo.pop()
            self._draw_spans()
            return
        # nothing left to undo in this lane: reopen the last lane saved this
        # session (skipped lanes in between are shown again afterwards)
        if self._committed and self._review is None:
            last = self.label_file.lanes.pop()
            self._save_labels()
            self._refresh_stats()
            self._current_idx = self._committed.pop()
            self._show_current_lane(spans=last.spans)
            self.ui.statusbar.showMessage("Reopened previous lane", 2000)

    def _commit_lane(self) -> None:
        if self._review is not None:  # correct the existing label in place
            label = self._review_labels[self._current_idx]
            if label.spans != self._spans:
                label.spans = list(self._spans)
                self._n_corrected += 1
                self._save_labels()
                self._refresh_stats()
            self._current_idx += 1
            self._show_current_lane()
            return
        cand = self._cand()
        self.label_file.lanes.append(LaneLabel(
            mz=cand.mz,
            lane_idx=cand.lane_idx,
            rt_values=[float(v) for v in cand.rt_values],
            intsy_values=[float(v) for v in cand.intsy_values],
            spans=list(self._spans),
            sample_uuid=cand.sample_uuid,
            sample_name=cand.sample_name,
            annotator=self.annotator,
            timestamp=datetime.datetime.now().isoformat(),
            session_id=self.session_id,
        ))
        self._save_labels()
        self._refresh_stats()
        self._committed.append(self._current_idx)
        self._current_idx += 1
        self._show_current_lane()

    # ------------------------------------------------------------------
    # Keys (all left hand)
    # ------------------------------------------------------------------
    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        if self._current_idx >= len(self._candidates) and ev.key() != Qt.Key_Z:
            return super().keyPressEvent(ev)
        key, shift = ev.key(), bool(ev.modifiers() & Qt.ShiftModifier)
        vb = self.ui.xicPlotWidget.getPlotItem().vb
        (x0, x1), _ = vb.viewRange()
        actions = {
            Qt.Key_Q: lambda: self._mark(PEAK),
            Qt.Key_T: lambda: self._mark(PLATEAU),
            Qt.Key_X: self._delete_under_cursor,
            Qt.Key_Z: self._undo_last,
            Qt.Key_Space: (self._skip_lane if shift else self._commit_lane),
            Qt.Key_A: lambda: vb.setXRange(x0 - (x1 - x0) / 2, x1 - (x1 - x0) / 2, padding=0),
            Qt.Key_D: lambda: vb.setXRange(x0 + (x1 - x0) / 2, x1 + (x1 - x0) / 2, padding=0),
            Qt.Key_W: lambda: vb.scaleBy(x=0.5, y=None),
            Qt.Key_S: lambda: vb.scaleBy(x=2.0, y=None),
            Qt.Key_F: vb.autoRange,
        }
        action = actions.get(key)
        if action is None:
            return super().keyPressEvent(ev)
        action()
        ev.accept()

    def _skip_lane(self) -> None:
        self.ui.statusbar.showMessage("Skipped", 1000)
        self._current_idx += 1
        self._show_current_lane()

    # ------------------------------------------------------------------
    # Persistence, stats, training
    # ------------------------------------------------------------------
    def _save_labels(self) -> None:
        # shortcut: rewrites the whole JSON per lane
        # (~0.9 ms/lane stored, so ~1 s at 1000 lanes)
        # TODO: switch to append-only / per-lane files if Space lags.
        self.label_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.label_file.to_json(self.label_file_path)

    def _refresh_stats(self) -> None:
        spans = [c for lbl in self.label_file.lanes for *_, c in lbl.spans]
        counts = [len(self.label_file.lanes), spans.count(PEAK), spans.count(PLATEAU)]
        for col, n in enumerate(counts):
            self.ui.classCountsTable.item(0, col).setText(str(n))

    def _load_model(self) -> Optional[dict]:
        path = peak_model.default_model_path()
        return peak_model.load_model(str(path)) if path.exists() else None

    def _train_model(self) -> None:
        if not self.label_file.lanes:
            self.ui.statusbar.showMessage("Label some lanes first", 3000)
            return
        self._save_labels()
        self._train_action.setEnabled(False)
        self.ui.statusbar.showMessage("Training peak model (see Process Monitor)", 0)
        self.process_controller.start_process(
            module_path="core.labeling.peak_model",
            function_name="train_and_report",
            parameters={
                "label_paths": [self.label_file_path],
                "out_path": peak_model.default_model_path(),
            },
            on_completion_func=self._on_trained,
        )

    def _on_trained(
            self,
            report
    ) -> None:
        self._train_action.setEnabled(True)
        if not isinstance(report, str):  # failed; the error is in the Process Monitor
            self.ui.statusbar.showMessage("Training failed (see Process Monitor)", 5000)
            return
        self._model = self._load_model()
        self.ui.statusbar.showMessage("Peak model trained", 3000)
        QtWidgets.QMessageBox.information(self, "Peak model", report)
        # refresh proposals on the current lane unless the user already edited it
        if self._current_idx < len(self._candidates) and not self._undo:
            self._show_current_lane()

    # ------------------------------------------------------------------
    # Review mode
    # ------------------------------------------------------------------
    def _start_review(self) -> None:
        if self._review is not None:
            return
        self._review_action.setEnabled(False)
        self.ui.statusbar.showMessage(
            "Cross-validating for review (5 trainings, a few minutes; see "
            "Process Monitor). Keep labeling meanwhile.", 0
        )
        self.process_controller.start_process(
            module_path="core.labeling.peak_model",
            function_name="review_predictions",
            parameters={
                # snapshot: labeling continues while this runs
                "lanes": list(self.label_file.lanes),
                "parent_threshold": float(
                    self.label_file.extraction_params.get("parent_threshold", 1e4)
                ),
            },
            on_completion_func=self._on_review_ready,
        )

    def _on_review_ready(self, results) -> None:
        self._review_action.setEnabled(True)
        if not isinstance(results, list):
            self.ui.statusbar.showMessage("Review failed (see Process Monitor)", 5000)
            return
        by_key = {(l.sample_uuid, l.lane_idx): l for l in self.label_file.lanes}
        results = [r for r in results if tuple(r["key"]) in by_key]  # undone since
        if not results:
            QtWidgets.QMessageBox.information(
                self, "Review", "The model agrees with every labeled lane.")
            return
        if QtWidgets.QMessageBox.question(
            self, "Review ready",
            f"{len(results)} labeled lanes where the model disagrees with you "
            "(worst first). Review now? Your labeling queue resumes afterwards.\n\n"
            "Space = keep your edits / labels, Shift+Space = next, unchanged.",
        ) != QtWidgets.QMessageBox.Yes:
            return
        self._paused_queue = (self._candidates, self._current_idx, self._committed)
        self._review = results
        self._review_labels = [by_key[tuple(r["key"])] for r in results]
        self._candidates = [
            LaneCandidate(
                sample_uuid=l.sample_uuid, sample_name=l.sample_name,
                lane_idx=l.lane_idx, mz=l.mz,
                rt_values=np.asarray(l.rt_values, float),
                intsy_values=np.asarray(l.intsy_values, float),
            )
            for l in self._review_labels
        ]
        self._current_idx, self._committed, self._n_corrected = 0, [], 0
        self._show_current_lane()

    def _end_review(self) -> None:
        n = self._n_corrected
        self._candidates, self._current_idx, self._committed = self._paused_queue
        self._review, self._review_labels, self._paused_queue = None, [], None
        self._show_current_lane()
        self.ui.statusbar.showMessage(
            f"Review done: {n} lanes corrected. Back to labeling", 5000)

    def _on_session_complete(self) -> None:
        if self._review is not None:
            self._end_review()
            return
        self.ui.statusbar.showMessage("Session complete. All lanes labeled", 0)
        self.ui.sourceLabel.setText("—")
        self.ui.mzLabel.setText("")
        self.ui.scanLabel.setText("")
        self.ui.progressLabel.setText(f"{len(self.label_file.lanes)} lanes labeled")
