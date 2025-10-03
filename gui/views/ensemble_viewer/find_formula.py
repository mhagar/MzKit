"""
Interactive formula finding. Includes a controller that manages
the ensemble viewer displays
"""
from gui.dialogues.formula_finder import FormulaFinderDialog
from gui.views.ensemble_viewer.tools import (
    ToolType, Mode, ToolStage
)

from PyQt5 import QtCore, QtWidgets

from typing import Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from gui.views.ensemble_viewer import EnsembleViewer

class Controller(QtCore.QObject):
    sigSignalSelected = QtCore.pyqtSignal(
        tuple, #data
        int,  # level
        bool, # added/removed
    )
    sigSelectionCleared = QtCore.pyqtSignal()

    def __init__(
        self,
        ensemble_viewer: 'EnsembleViewer'
    ):
        super().__init__()
        self.viewer = ensemble_viewer
        self.selected_ms_level: Literal[None, 1, 2] = None
        self.selected_signals: list = []
        self.formula_finder_menu: FormulaFinderDialog = FormulaFinderDialog()


    def handle_ms_signal_clicked(
        self,
        data: tuple[int, float],  # [spec_idx, mz_float]
        ms_level: Literal[1, 2],
    ):
        if self.selected_ms_level != ms_level:
            if not self.selected_ms_level:
                self.selected_ms_level = ms_level
            else:
                self.handle_clear_selections()

        if self.viewer.tool_manager.active_stage == ToolStage.SELECTING:
            if data in self.selected_signals:
                # Remove signal from selection
                self.selected_signals.remove(data)
                self.sigSignalSelected.emit(
                    data,
                    self.selected_ms_level,
                    False,
                )
            else:
                # Add signal to selection
                self.selected_signals.append(data)
                self.sigSignalSelected.emit(
                    data,
                    self.selected_ms_level,
                    True,
                )

    def handle_clear_selections(self):
        self.selected_signals.clear()
        self.selected_ms_level = None
        self.sigSelectionCleared.emit()

    def handle_show_finder_menu(self):
        self.formula_finder_menu.show()