"""
Interactive formula finding. Includes a controller that manages
the ensemble viewer displays
"""
from gui.dialogues.formula_finder import FormulaFinderDialog
from gui.views.ensemble_viewer.tools import (
    ToolType, Mode, ToolStage
)
from gui.views.ensemble_viewer.tool_controllers import BaseToolController

from PyQt5 import QtCore, QtWidgets
from find_mfs import FormulaCandidate

from typing import Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from gui.views.ensemble_viewer import EnsembleViewer


class FindFormulaController(BaseToolController):
    """
    Controller for the 'Find Formula' tool
    """
    sigFormulaAssigned = QtCore.pyqtSignal(
        object,      # find_mfs.FormulaCandidate
        int,       # ms_level
        list,      # cofeature_idxs
    )

    def __init__(
        self,
        ensemble_viewer: 'EnsembleViewer'
    ):
        super().__init__(
            viewer=ensemble_viewer,
            tool_type=ToolType.FINDFORMULA,
        )
        self.selected_ms_level: Literal[None, 1, 2] = None
        self.selected_signals: list[tuple[float, float, int]] = []
        self.formula_finder_menu: FormulaFinderDialog = FormulaFinderDialog(
            parent=ensemble_viewer,
            config=ensemble_viewer.config,
            modal=True,
        )

        self._connect_signals()

    def _connect_signals(self):
        self.formula_finder_menu.sigFormulaAssigned.connect(
            self.handle_formula_assigned
        )

    def on_activated(self):
        """
        Called when Find Formula tool is activated
        """
        self.handle_clear_selections()
        # Request next stage (IDLE -> SELECTING)
        self.viewer.tool_manager.request_next_stage()

    def on_enter_pressed(self):
        """
        Called when user presses Enter while selecting signals
        """
        self.handle_show_finder_menu()
        self.viewer.tool_manager.request_next_stage()

    def on_cancelled(self):
        """
        Called when tool is cancelled
        """
        self.handle_clear_selections()

    def handle_ms_signal_clicked(
        self,
        data: tuple[float, float, int],  # mz, intsy, spec_idx
        ms_level: Literal[1, 2],
    ):
        if self.selected_ms_level != ms_level:
            if not self.selected_ms_level:
                self.selected_ms_level = ms_level
            else:
                self.handle_clear_selections()

        if self.viewer.tool_manager.active_stage == ToolStage.SELECTING:
            mz, intsy, spec_idx = data

            if data in self.selected_signals:
                # Remove signal from selection
                self.selected_signals.remove(data)
                self.sigSignalSelected.emit(
                    mz,
                    intsy,
                    spec_idx,
                    self.selected_ms_level,
                    False,
                )
            else:
                # Add signal to selection
                self.selected_signals.append(data)
                self.sigSignalSelected.emit(
                    mz,
                    intsy,
                    spec_idx,
                    self.selected_ms_level,
                    True,
                )

    def handle_clear_selections(self):
        self.selected_signals.clear()
        self.selected_ms_level = None
        self.sigSelectionCleared.emit()

    def handle_show_finder_menu(self):
        self.formula_finder_menu.show()

        # Send selected_signals to formula finder
        self.formula_finder_menu.populate_table(
            [(x[0], x[1]) for x in self.selected_signals]
        )

        self.formula_finder_menu.on_search_execute()

    def handle_formula_assigned(
        self,
        formula: 'FormulaCandidate'
    ):
        """
        Called when user selects a formula from finder dialogue.

        Passes the signal outwards, and reverts to IDLE stage.
        """
        self.sigFormulaAssigned.emit(
            formula,  # FormulaCandidate
            self.selected_ms_level,  # int
            [x[2] for x in self.selected_signals],  # feature coidxs (ints)
        )

        self.handle_clear_selections()




