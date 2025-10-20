"""
Base tool controller class and implementations for ensemble viewer tools
"""
from PyQt5 import QtCore
from gui.views.ensemble_viewer.tools import ToolType, ToolStage

from typing import TYPE_CHECKING, Literal, Optional
if TYPE_CHECKING:
    from gui.views.ensemble_viewer import EnsembleViewer


class BaseToolController(QtCore.QObject):
    """
    Base class for tool controllers.

    Each tool should extend this class and override the methods
    to implement tool-specific behavior.
    """

    # Signals that tool controllers can emit
    sigSignalSelected = QtCore.pyqtSignal(
        tuple,  # data (spec_idx, mz)
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
        data: tuple[int, float],
        ms_level: Literal[1, 2],
    ):
        """
        Called when user clicks on a signal in MS1/MS2 spectrum.
        Override to implement tool-specific signal handling.
        """
        pass


class MeasureLossController(BaseToolController):
    """
    Controller for the 'Measure Loss' tool.

    This is a stub implementation - expand as needed.
    """

    def __init__(
        self,
        viewer: 'EnsembleViewer',
    ):
        super().__init__(
            viewer=viewer,
            tool_type=ToolType.MEASURELOSS,
        )
        # Add tool-specific state here

    def on_activated(self):
        """Called when Measure Loss tool is activated"""
        # TODO: Implement tool initialization
        self.viewer.tool_manager.request_next_stage()

    def on_enter_pressed(self):
        """Called when user presses Enter while selecting signals"""
        # TODO: Implement confirmation behavior
        pass

    def on_cancelled(self):
        """Called when tool is cancelled"""
        # TODO: Implement cleanup
        pass

    def handle_ms_signal_clicked(
        self,
        data: tuple[int, float],
        ms_level: Literal[1, 2],
    ):
        """Handle signal clicks for measure loss tool"""
        # TODO: Implement signal selection logic
        pass
