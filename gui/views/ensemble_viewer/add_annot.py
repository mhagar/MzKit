"""
Controller for the 'Add Annotation' tool. Lets the user attach a
free-form text label to any peak in either MS plot.
"""
from PyQt5 import QtCore, QtWidgets

from gui.views.ensemble_viewer.tools import ToolType, ToolStage
from gui.views.ensemble_viewer.tool_controllers import BaseToolController

from typing import Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from gui.views.ensemble_viewer import EnsembleViewer


class AddAnnotController(BaseToolController):
    """
    Click a peak → modal text-input dialog → annotation created.
    """

    # Emitted when the user confirms a label.
    # Args: (cofeature_idx, ms_level, text).
    sigGenericAnnotationRequested = QtCore.pyqtSignal(int, int, str)

    def __init__(self, ensemble_viewer: 'EnsembleViewer'):
        super().__init__(
            viewer=ensemble_viewer,
            tool_type=ToolType.ADDANNOT,
        )

    def on_activated(self):
        # IDLE → SELECTING. Same lifecycle as FindFormula / MeasureLoss.
        self.viewer.tool_manager.request_next_stage()

    def on_cancelled(self):
        # Nothing to clean up — the tool holds no transient state.
        self.sigSelectionCleared.emit()

    def handle_ms_signal_clicked(
        self,
        data: tuple[float, float, int],  # mz, intsy, spec_idx
        ms_level: Literal[1, 2],
    ):
        if self.viewer.tool_manager.active_stage != ToolStage.SELECTING:
            return

        mz, _intsy, spec_idx = data

        text, ok = QtWidgets.QInputDialog.getText(
            self.viewer,
            "Add annotation",
            f"Label for peak m/z {mz:.4f} (MS{ms_level}):",
        )
        # Whether the user accepted or cancelled, exit the tool —
        # otherwise the next click pops another input dialog with no
        # visible cue that we're still in tool-mode.
        if ok and text.strip():
            self.sigGenericAnnotationRequested.emit(
                int(spec_idx), int(ms_level), text.strip(),
            )

        self.viewer.tool_manager.request_cancel()
