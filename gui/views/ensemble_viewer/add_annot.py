"""
Controller for the 'Add Annotation' tool. Lets the user attach a
free-form text label to one or more peaks of either MS plot.
"""
from PyQt5 import QtCore, QtWidgets

from gui.views.ensemble_viewer.tools import ToolType
from gui.views.ensemble_viewer.tool_controllers import SelectingToolController

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gui.views.ensemble_viewer import EnsembleViewer


class AddAnnotController(SelectingToolController):
    """
    Click peaks (one MS level) → Enter → text dialog → annotation created.
    """

    # Emitted when the user confirms a label.
    # Args: (cofeature_idxs, tallest first; ms_level; text).
    sigGenericAnnotationRequested = QtCore.pyqtSignal(list, int, str)

    def __init__(self, ensemble_viewer: 'EnsembleViewer'):
        super().__init__(
            viewer=ensemble_viewer,
            tool_type=ToolType.ADDANNOT,
        )

    def on_enter_pressed(self):
        if self.selected_signals:
            n = len(self.selected_signals)
            text, ok = QtWidgets.QInputDialog.getText(
                self.viewer,
                "Add annotation",
                f"Label for {n} peak{'s' if n > 1 else ''} (MS{self.selected_ms_level}):",
            )
            if ok and text.strip():
                tallest_first = sorted(self.selected_signals, key=lambda x: -x[1])
                self.sigGenericAnnotationRequested.emit(
                    [int(x[2]) for x in tallest_first],
                    int(self.selected_ms_level), text.strip(),
                )
        # Either way, exit the tool (cancel clears the selection)
        self.viewer.tool_manager.request_cancel()
