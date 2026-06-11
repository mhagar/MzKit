"""
Small pre-session dialog for the peak-morphology labeling tool.

Asks the annotator for:
  - Output label file path (resumes if it exists)
  - Candidate floor (min intensity to consider)
  - Max candidates per sample
  - Annotator name

Kept as a plain QDialog in code rather than a .ui file — the
field list is short and likely to evolve while the tool matures.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtWidgets


@dataclass
class LabelingSessionParams:
    label_file_path: Path
    min_intsy: float
    max_per_sample: Optional[int]
    annotator: str


class LabelingSessionDialog(QtWidgets.QDialog):
    def __init__(
        self,
        default_label_file: Optional[Path] = None,
        default_min_intsy: float = 1000.0,
        default_max_per_sample: int = 200,
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

        self._min_intsy_spin = QtWidgets.QDoubleSpinBox()
        self._min_intsy_spin.setRange(0.0, 1e12)
        self._min_intsy_spin.setDecimals(1)
        self._min_intsy_spin.setValue(default_min_intsy)
        self._min_intsy_spin.setSingleStep(100.0)

        self._max_per_sample_spin = QtWidgets.QSpinBox()
        self._max_per_sample_spin.setRange(0, 100_000)
        self._max_per_sample_spin.setValue(default_max_per_sample)
        self._max_per_sample_spin.setSpecialValueText("No limit")

        self._annotator_edit = QtWidgets.QLineEdit(default_annotator)

        form = QtWidgets.QFormLayout()
        form.addRow("Label file:", path_row)
        form.addRow("Candidate floor (min intensity):", self._min_intsy_spin)
        form.addRow("Max candidates per sample:", self._max_per_sample_spin)
        form.addRow("Annotator:", self._annotator_edit)

        self._hint = QtWidgets.QLabel(
            "If the label file exists, already-labeled candidates will be skipped."
        )
        self._hint.setStyleSheet("color: gray;")
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
        max_per = self._max_per_sample_spin.value()
        return LabelingSessionParams(
            label_file_path=Path(self._label_file_edit.text().strip()),
            min_intsy=float(self._min_intsy_spin.value()),
            max_per_sample=max_per if max_per > 0 else None,
            annotator=self._annotator_edit.text().strip(),
        )
