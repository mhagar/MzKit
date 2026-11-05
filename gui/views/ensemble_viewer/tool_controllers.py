"""
Base tool controller class and implementations for ensemble viewer tools
"""
from PyQt5 import QtCore
from gui.views.ensemble_viewer.tools import ToolType, ToolStage

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from gui.views.ensemble_viewer import EnsembleViewer


class BaseToolController(QtCore.QObject):
    """
    Each tool should extend this class and override the methods
    to implement tool-specific behavior
    """
    sigSignalSelected = QtCore.pyqtSignal(
        float,  # mz
        float,  # intsy
        int,    # spec_idx
        int,    # ms_level
        bool,   # is_selected
    )
    sigSelectionCleared = QtCore.pyqtSignal()

    def __init__(
        self,
        viewer: 'EnsembleViewer',
        tool_type: ToolType,
    ):
        super().__init__()
        self.viewer = viewer
        self.tool_type = tool_type

    def on_activated(self):
        """
        Called when this tool is selected/activated.
        Override to implement tool-specific initialization.
        """
        pass

    def on_enter_pressed(self):
        """
        Called when user presses Enter while in SELECTING stage.
        Override to implement tool-specific confirmation behavior.
        """
        pass

    def on_cancelled(self):
        """
        Called when tool is cancelled (e.g., Escape pressed).
        Override to implement tool-specific cleanup.
        """
        pass

    def on_stage_changed(self, stage: ToolStage):
        """
        Called when tool stage changes.
        Override to react to stage transitions.
        """
        pass

    def handle_ms_signal_clicked(
        self,
        data: tuple[float, float, int],  # mz, intsy, spec_idx
        ms_level: Literal[1, 2],
    ):
        """
        Called when user clicks on a signal in MS1/MS2 spectrum.
        Override to implement tool-specific signal handling.
        """
        pass

    def handle_ms_signal_hovered(
        self,
        data: tuple[float, float, int],  # mz, intsy, spec_idx
        ms_level: Literal[1, 2],
    ):
        """
        Called when user hovers on a signal in MS1/MS2 spectrum.
        Override to implement tool-specific signal handling.
        """
        pass


class SelectingToolController(BaseToolController):
    """
    A tool that collects signals of one MS level: a click toggles a signal,
    clicking another level starts over, Enter confirms (`on_enter_pressed`).
    """
    def __init__(self, viewer: 'EnsembleViewer', tool_type: ToolType):
        super().__init__(viewer=viewer, tool_type=tool_type)
        self.selected_ms_level: Literal[None, 1, 2] = None
        self.selected_signals: list[tuple[float, float, int]] = []  # mz, intsy, spec_idx

    def on_activated(self):
        self.handle_clear_selections()
        self.viewer.tool_manager.request_next_stage()   # IDLE -> SELECTING

    def on_cancelled(self):
        self.handle_clear_selections()

    def handle_ms_signal_clicked(
        self,
        data: tuple[float, float, int],  # mz, intsy, spec_idx
        ms_level: Literal[1, 2],
    ):
        if self.selected_ms_level != ms_level:
            self.handle_clear_selections()
            self.selected_ms_level = ms_level

        if self.viewer.tool_manager.active_stage != ToolStage.SELECTING:
            return
        selected = data not in self.selected_signals
        if selected:
            self.selected_signals.append(data)
        else:
            self.selected_signals.remove(data)
        mz, intsy, spec_idx = data
        self.sigSignalSelected.emit(
            mz, intsy, spec_idx, self.selected_ms_level, selected,
        )

    def handle_clear_selections(self):
        self.selected_signals.clear()
        self.selected_ms_level = None
        self.sigSelectionCleared.emit()
