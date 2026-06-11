"""
Controller for the peak-morphology labeling window.

Bridges `Ui_MainWindow` (from LabelingWindow.ui) to the Qt-free
logic in `core.labeling`. Handles:
  - Candidate display (XIC window + optional MS1 spectrum)
  - Hotkey-driven classification
  - Tier-2 boundary labeling for multi-peak class
  - Append-on-keystroke persistence
"""
from __future__ import annotations
import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from gui.resources.LabelingWindow import Ui_MainWindow
from core.labeling import (
    Candidate,
    Label,
    LabelFile,
    MorphologyClass,
)
from core.utils.array_types import to_chrom_arr

if TYPE_CHECKING:
    from core.data_structs import Sample

logger = logging.getLogger(__name__)


# Column order in the stats QTableWidget matches the .ui file's
# header order
_CLASS_COLUMNS: dict[MorphologyClass, int] = {
    MorphologyClass.SHARP: 0,
    MorphologyClass.BROAD: 1,
    MorphologyClass.NOISE: 2,
    MorphologyClass.TAILING: 3,
    MorphologyClass.FRONTING: 4,
    MorphologyClass.MULTI_PEAK: 5,
    MorphologyClass.SATURATED: 6,
}

# Hotkeys
_KEY_TO_CLASS: dict[int, MorphologyClass] = {
    Qt.Key_N: MorphologyClass.NOISE,
    Qt.Key_S: MorphologyClass.SHARP,
    Qt.Key_T: MorphologyClass.TAILING,
    Qt.Key_F: MorphologyClass.FRONTING,
    Qt.Key_B: MorphologyClass.BROAD,
    Qt.Key_M: MorphologyClass.MULTI_PEAK,
    Qt.Key_X: MorphologyClass.SATURATED,
}


class LabelingWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        samples: list["Sample"],
        candidates: list[Candidate],
        label_file_path: Path,
        session_id: str = "",
        annotator: str = "",
        min_intsy: float = 1000.0,
        window_half_width: int = 60,
        max_per_sample: Optional[int] = 200,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.samples_by_uuid = {s.uuid: s for s in samples}
        self._current_sample_uuids = sorted(self.samples_by_uuid.keys())
        self.label_file_path = Path(label_file_path)
        self.session_id = session_id or datetime.datetime.now().isoformat()
        self.annotator = annotator

        # View toggles
        self._normalize_y = False
        self._show_spectrum = False
        self.ui.optionalSpectrumFrame.setVisible(False)

        # Multi-peak split mode state
        self._split_mode_active = False
        self._split_scan_idxs: list[int] = []
        self._split_marker_items: list[pg.InfiniteLine] = []
        self._scene_click_connected = False

        # Candidates are generated up-front (via ProcessController so
        # the work + logs show up in the Process Monitor) and passed in.
        self._candidates: list[Candidate] = list(candidates)

        # Load or create the label file
        if self.label_file_path.exists():
            self.label_file = LabelFile.from_json(self.label_file_path)
            logger.info(
                f"Resumed label file with {len(self.label_file.labels)} "
                f"existing labels"
            )
            if not self._check_sample_uuid_match():
                # User chose to abort after seeing the mismatch warning
                raise RuntimeError("Labeling session aborted: sample mismatch")
        else:
            self.label_file = LabelFile(
                extraction_params={
                    "min_intsy": min_intsy,
                    "window_half_width": window_half_width,
                    "max_per_sample": max_per_sample,
                },
                sample_uuids=list(self._current_sample_uuids),
            )

        # Drop candidates already labeled in a prior session
        self._candidates = [
            c for c in self._candidates
            if not self.label_file.has_label_for(
                c.sample_uuid, c.mz, c.apex_scan_idx,
            )
        ]

        self._current_idx = 0

        self._init_stats_table()
        self._connect_menu_actions()
        self._refresh_stats()
        self._show_current_candidate()

    # ------------------------------------------------------------------
    # Sample-set validation on resume
    # ------------------------------------------------------------------
    def _check_sample_uuid_match(self) -> bool:
        """
        SampleUUIDs are regenerated on every import, so a mismatch
        means the user pointed at the wrong .mzk or re-imported the
        mzMLs. Either way the existing labels aren't comparable to
        what this session would produce. Return False if the user
        chooses to abort.
        """
        stored = set(self.label_file.sample_uuids)
        current = set(self._current_sample_uuids)
        if not stored:
            # Legacy file or freshly created elsewhere — nothing to check.
            return True
        if stored == current:
            return True

        missing = stored - current
        extra = current - stored
        msg = (
            "The loaded samples don't match the ones this label file "
            "was created against.\n\n"
            f"Missing from current session: {len(missing)}\n"
            f"New in current session: {len(extra)}\n\n"
            "Resuming will produce labels that aren't comparable to "
            "the existing ones. Continue anyway?"
        )
        reply = QtWidgets.QMessageBox.warning(
            self,
            "Label file / sample mismatch",
            msg,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return reply == QtWidgets.QMessageBox.Yes

    # ------------------------------------------------------------------
    # UI initialization
    # ------------------------------------------------------------------
    def _init_stats_table(self) -> None:
        table = self.ui.classCountsTable
        table.setRowCount(1)
        for col in range(table.columnCount()):
            item = QtWidgets.QTableWidgetItem("0")
            item.setTextAlignment(Qt.AlignCenter)
            table.setItem(0, col, item)

    def _connect_menu_actions(self) -> None:
        ui = self.ui
        ui.actionExit.triggered.connect(self.close)
        ui.actionSave_labels_as_JSON.triggered.connect(self._save_labels)
        ui.actionToggle_MS1_spectrum.triggered.connect(self._toggle_spectrum)
        ui.actionToggle_Y_axis_normalization.triggered.connect(
            self._toggle_normalize
        )

    # ------------------------------------------------------------------
    # Candidate display
    # ------------------------------------------------------------------
    def _show_current_candidate(self) -> None:
        self._clear_split_markers()

        if self._current_idx >= len(self._candidates):
            self._on_session_complete()
            return

        cand = self._candidates[self._current_idx]

        rt_arr = np.asarray(cand.rt_values, dtype=np.float64)
        intsy_arr = np.asarray(cand.intsy_values, dtype=np.float64)
        if self._normalize_y and intsy_arr.max() > 0:
            intsy_arr = intsy_arr / intsy_arr.max()

        chrom = to_chrom_arr(rt_arr, intsy_arr)
        self.ui.xicPlotWidget.setChromArray(chrom, replace=True)
        self.ui.xicPlotWidget.getPlotItem().vb.autoRange()

        # Mark the apex
        self.ui.xicPlotWidget.setSelectionIndicator(cand.apex_rt)
        self.ui.xicPlotWidget.setSelectionIndicatorVisible(True)

        # Header
        self.ui.sourceLabel.setText(cand.sample_name)
        self.ui.mzLabel.setText(f"m/z {cand.mz:.4f}")
        self.ui.scanLabel.setText(f"scan {cand.apex_scan_idx}")
        self.ui.progressLabel.setText(
            f"{self._current_idx + 1} / {len(self._candidates)}"
        )
        self.ui.statusbar.clearMessage()

        if self._show_spectrum:
            self._show_ms1_spectrum(cand)

    def _show_ms1_spectrum(self, cand: Candidate) -> None:
        sample = self.samples_by_uuid.get(cand.sample_uuid)
        if sample is None or sample.injection is None:
            return
        sa = sample.injection.scan_array_ms1
        spec = sa.get_spectrum(scan_num=cand.apex_scan_idx)
        self.ui.spectrumPlotWidget.setSpectrumArray(spec)

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------
    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        if self._split_mode_active:
            self._handle_split_mode_key(ev)
            return

        key = ev.key()
        if key == Qt.Key_Space:
            self._skip_current()
            ev.accept()
            return
        if key == Qt.Key_Z:
            self._undo_last()
            ev.accept()
            return

        cls = _KEY_TO_CLASS.get(key)
        if cls is None:
            super().keyPressEvent(ev)
            return

        if cls is MorphologyClass.MULTI_PEAK:
            self._enter_split_mode()
            ev.accept()
            return

        self._commit_label(cls, [])
        ev.accept()

    # ------------------------------------------------------------------
    # Labeling actions
    # ------------------------------------------------------------------
    def _commit_label(
        self,
        morphology: MorphologyClass,
        boundary_splits: list[int],
    ) -> None:
        cand = self._candidates[self._current_idx]
        label = Label(
            mz=cand.mz,
            rt_apex=cand.apex_rt,
            intensity_apex=cand.apex_intsy,
            apex_scan_idx=cand.apex_scan_idx,
            window_start_scan=cand.window_start_scan,
            window_end_scan=cand.window_end_scan,
            rt_values=[float(v) for v in cand.rt_values],
            intsy_values=[float(v) for v in cand.intsy_values],
            morphology=morphology,
            boundary_splits=list(boundary_splits),
            sample_uuid=cand.sample_uuid,
            sample_name=cand.sample_name,
            annotator=self.annotator,
            timestamp=datetime.datetime.now().isoformat(),
            session_id=self.session_id,
        )
        self.label_file.labels.append(label)
        self._save_labels()
        self._refresh_stats()
        self.ui.statusbar.showMessage(
            f"Labeled as {morphology.value}", 1500
        )
        self._current_idx += 1
        self._show_current_candidate()

    def _skip_current(self) -> None:
        self.ui.statusbar.showMessage("Skipped", 1000)
        self._current_idx += 1
        self._show_current_candidate()

    def _undo_last(self) -> None:
        if not self.label_file.labels:
            self.ui.statusbar.showMessage("Nothing to undo", 1500)
            return
        last = self.label_file.labels.pop()
        self._save_labels()
        self._refresh_stats()
        self._current_idx = max(0, self._current_idx - 1)
        self.ui.statusbar.showMessage(
            f"Undid: {last.morphology.value} @ m/z {last.mz:.4f}",
            2000,
        )
        self._show_current_candidate()

    # ------------------------------------------------------------------
    # Multi-peak tier-2: split boundary drawing
    # ------------------------------------------------------------------
    def _enter_split_mode(self) -> None:
        self._split_mode_active = True
        self._split_scan_idxs = []
        self.ui.statusbar.showMessage(
            "Multi-peak: click to add splits, Enter to commit, "
            "Esc to cancel, Backspace to remove last"
        )
        scene = self.ui.xicPlotWidget.getPlotItem().scene()
        scene.sigMouseClicked.connect(self._on_scene_click_for_split)
        self._scene_click_connected = True

    def _exit_split_mode(self) -> None:
        self._split_mode_active = False
        if self._scene_click_connected:
            try:
                scene = self.ui.xicPlotWidget.getPlotItem().scene()
                scene.sigMouseClicked.disconnect(
                    self._on_scene_click_for_split
                )
            except TypeError:
                pass
            self._scene_click_connected = False
        self._clear_split_markers()

    def _on_scene_click_for_split(self, ev) -> None:
        if not self._split_mode_active:
            return
        # Only respond to primary-button clicks inside the plot
        if ev.button() != Qt.LeftButton:
            return
        pw = self.ui.xicPlotWidget
        vb = pw.getPlotItem().vb
        scene_pos = ev.scenePos()
        if not pw.getPlotItem().sceneBoundingRect().contains(scene_pos):
            return

        view_pos = vb.mapSceneToView(scene_pos)
        rt_click = view_pos.x()

        cand = self._candidates[self._current_idx]
        rt_arr = np.asarray(cand.rt_values, dtype=np.float64)
        rel_idx = int(np.argmin(np.abs(rt_arr - rt_click)))

        if rel_idx in self._split_scan_idxs:
            return
        self._split_scan_idxs.append(rel_idx)

        marker = pg.InfiniteLine(
            pos=float(rt_arr[rel_idx]),
            angle=90,
            pen=pg.mkPen('g', width=2),
        )
        pw.getPlotItem().addItem(marker)
        self._split_marker_items.append(marker)

    def _handle_split_mode_key(self, ev: QtGui.QKeyEvent) -> None:
        key = ev.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if not self._split_scan_idxs:
                self.ui.statusbar.showMessage(
                    "No splits added — click to add or Esc to cancel",
                    2000,
                )
                ev.accept()
                return
            splits = sorted(set(self._split_scan_idxs))
            self._exit_split_mode()
            self._commit_label(MorphologyClass.MULTI_PEAK, splits)
            ev.accept()
            return
        if key == Qt.Key_Escape:
            self._exit_split_mode()
            self.ui.statusbar.showMessage("Multi-peak cancelled", 1500)
            self._show_current_candidate()
            ev.accept()
            return
        if key == Qt.Key_Backspace and self._split_scan_idxs:
            self._split_scan_idxs.pop()
            marker = self._split_marker_items.pop()
            self.ui.xicPlotWidget.getPlotItem().removeItem(marker)
            ev.accept()
            return
        super().keyPressEvent(ev)

    def _clear_split_markers(self) -> None:
        plot_item = self.ui.xicPlotWidget.getPlotItem()
        for marker in self._split_marker_items:
            plot_item.removeItem(marker)
        self._split_marker_items = []
        self._split_scan_idxs = []

    # ------------------------------------------------------------------
    # Persistence + stats
    # ------------------------------------------------------------------
    def _save_labels(self) -> None:
        self.label_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.label_file.to_json(self.label_file_path)

    def _refresh_stats(self) -> None:
        counts: dict[MorphologyClass, int] = {c: 0 for c in MorphologyClass}
        for lbl in self.label_file.labels:
            counts[lbl.morphology] = counts.get(lbl.morphology, 0) + 1

        table = self.ui.classCountsTable
        for cls, col in _CLASS_COLUMNS.items():
            item = table.item(0, col)
            if item is not None:
                item.setText(str(counts[cls]))

    # ------------------------------------------------------------------
    # Menu action handlers
    # ------------------------------------------------------------------
    def _toggle_normalize(self) -> None:
        self._normalize_y = not self._normalize_y
        self._show_current_candidate()

    def _toggle_spectrum(self) -> None:
        self._show_spectrum = not self._show_spectrum
        self.ui.optionalSpectrumFrame.setVisible(self._show_spectrum)
        if self._show_spectrum and self._current_idx < len(self._candidates):
            self._show_ms1_spectrum(self._candidates[self._current_idx])

    def _on_session_complete(self) -> None:
        self.ui.statusbar.showMessage(
            "Session complete — all candidates labeled.", 0
        )
        self.ui.sourceLabel.setText("—")
        self.ui.mzLabel.setText("")
        self.ui.scanLabel.setText("")
        self.ui.progressLabel.setText(
            f"{len(self.label_file.labels)} labeled"
        )
