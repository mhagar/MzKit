"""
Small pre-session dialog for chromatogram labeling tool
(i.e. for for generating training data for chrom segmentation)

Asks the annotator for:
  - Output label file path (resumes if it exists)
  - Seed feature threshold (lanes reaching it are favoured)
  - Intensity floor (min lane max for the random share of lanes)
  - Number of lanes to queue
  - Annotator name

Kept as a plain QDialog in code rather than a .ui file for now.
# TODO: turn this into .ui when tool is mature
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtWidgets


@dataclass
class LabelingSessionParams:
    label_file_path: Path
    parent_threshold: float
    min_intsy: float
    n_lanes: int
    annotator: str


class LabelingSessionDialog(QtWidgets.QDialog):
    def __init__(
        self,
        default_label_file: Optional[Path] = None,
        default_parent_threshold: float = 10000.0,
        default_min_intsy: float = 1000.0,
        default_n_lanes: int = 100,
        default_annotator: str = "",
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("New labeling session")
        self.resize(480, 220)

        self._label_file_edit = QtWidgets.QLineEdit(
            str(default_label_file) if default_label_file else ""
        )
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_label_file)

        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self._label_file_edit, 1)
        path_row.addWidget(browse_btn)

        self._parent_spin = QtWidgets.QDoubleSpinBox()
        self._parent_spin.setRange(0.0, 1e12)
        self._parent_spin.setDecimals(1)
        self._parent_spin.setValue(default_parent_threshold)
        self._parent_spin.setSingleStep(1000.0)

        self._min_intsy_spin = QtWidgets.QDoubleSpinBox()
        self._min_intsy_spin.setRange(0.0, 1e12)
        self._min_intsy_spin.setDecimals(1)
        self._min_intsy_spin.setValue(default_min_intsy)
        self._min_intsy_spin.setSingleStep(100.0)

        self._n_lanes_spin = QtWidgets.QSpinBox()
        self._n_lanes_spin.setRange(1, 10_000)
        self._n_lanes_spin.setValue(default_n_lanes)

        self._annotator_edit = QtWidgets.QLineEdit(default_annotator)

        form = QtWidgets.QFormLayout()
        form.addRow("Label file:", path_row)
        form.addRow("Seed feature threshold (favoured lanes):", self._parent_spin)
        form.addRow("Activity floor (random lanes):", self._min_intsy_spin)
        form.addRow("Lanes to queue (from <= random samples):", self._n_lanes_spin)
        form.addRow("Annotator:", self._annotator_edit)

        self._hint = QtWidgets.QLabel(
            "If label file exists, already labeled candidates will be skipped."
        )
        self._hint.setWordWrap(True)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._try_accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._hint)
        layout.addStretch(1)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    def _browse_label_file(self) -> None:
        current = self._label_file_edit.text() or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select label file",
            current,
            "Label files (*.json);;All files (*)",
            options=QtWidgets.QFileDialog.DontConfirmOverwrite,
        )
        if path:
            self._label_file_edit.setText(path)

    def _try_accept(self) -> None:
        text = self._label_file_edit.text().strip()
        if not text:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing label file",
                "Please choose a path for the label file.",
            )
            return
        self.accept()

    # ------------------------------------------------------------------
    def params(self) -> LabelingSessionParams:
        return LabelingSessionParams(
            label_file_path=Path(self._label_file_edit.text().strip()),
            parent_threshold=float(self._parent_spin.value()),
            min_intsy=float(self._min_intsy_spin.value()),
            n_lanes=int(self._n_lanes_spin.value()),
            annotator=self._annotator_edit.text().strip(),
        )
