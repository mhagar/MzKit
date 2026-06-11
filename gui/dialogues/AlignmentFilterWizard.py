"""
QWizard for filtering an EnsembleAlignment using a Python expression.

Shows a text editor for the expression, a preview button that
displays before/after counts per sample, and a help tab with
available variables.
"""
from PyQt5 import QtWidgets, QtCore

from gui.resources.AlignmentFilterWizard import Ui_Wizard

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structs import Sample, SampleUUID
    from core.data_structs.alignment import EnsembleAlignment
    from core.cli.filter_alignment import FilterResult


class AlignmentFilterWizard(
    QtWidgets.QWizard,
    Ui_Wizard,
):
    sigFilterAccepted = QtCore.pyqtSignal(
        object,  # FilterResult
    )

    def __init__(
        self,
        alignment: 'EnsembleAlignment',
        sample_names: dict['SampleUUID', str],
        sample_lookup: dict['SampleUUID', 'Sample'] = None,
    ):
        super().__init__()
        self.setupUi(self)

        self._alignment = alignment
        self._sample_names = sample_names
        self._sample_lookup = sample_lookup or {}
        self._last_result: Optional['FilterResult'] = None

        self.pushPreview.clicked.connect(self._on_preview)
        self.textFilterExpression.setPlainText("n >= 2")

    def _on_preview(self):
        from core.cli.filter_alignment import filter_alignment

        expression = self.textFilterExpression.toPlainText().strip()
        if not expression:
            self.textPreview.setHtml("<p>No expression entered.</p>")
            return

        try:
            result = filter_alignment(
                self._alignment, expression, self._sample_names,
                self._sample_lookup,
            )
        except Exception as e:
            self.textPreview.setHtml(
                f"<p style='color: red;'>Error: {e}</p>"
            )
            self._last_result = None
            return

        self._last_result = result
        self.textPreview.setHtml(result.format_html())

    def accept(self):
        from core.cli.filter_alignment import filter_alignment

        expression = self.textFilterExpression.toPlainText().strip()
        if not expression:
            QtWidgets.QMessageBox.warning(
                self, "No expression",
                "Please enter a filter expression.",
            )
            return

        # Run the filter (or reuse preview result if expression unchanged)
        try:
            result = filter_alignment(
                self._alignment, expression, self._sample_names,
                self._sample_lookup,
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Expression error",
                f"Error evaluating expression:\n{e}",
            )
            return

        self.sigFilterAccepted.emit(result)
        super().accept()
