"""
Script containing a ToolManager
"""
from enum import Enum, auto
import logging

from PyQt5 import QtCore


class ToolType(Enum):
    NONE = auto()
    FINDFORMULA = auto()
    MEASURELOSS = auto()
    ADDANNOT = auto()
    DELETEANNOT = auto()


class ToolStage(Enum):
    IDLE = auto()  # i.e. ToolType.NONE
    SELECTING = auto()
    CONFIGURING = auto()


class Mode(Enum):
    SCAN = auto()
    COMPOSITE = auto()


# This defines which 'mode' each tool can be accessed from
TOOL_GROUPS = {
    Mode.SCAN: [],
    Mode.COMPOSITE: [
        ToolType.FINDFORMULA,
        ToolType.MEASURELOSS,
        ToolType.ADDANNOT,
        ToolType.DELETEANNOT,
    ]
}

STAGES = (
    ToolStage.IDLE,
    ToolStage.SELECTING,
    ToolStage.CONFIGURING,
)

logger = logging.getLogger()


class ToolManager(QtCore.QObject):
    # Signals for broadcasting state changes
    sigModeChanged = QtCore.pyqtSignal(Mode)
    sigToolChanged = QtCore.pyqtSignal(ToolType)
    sigStageChanged = QtCore.pyqtSignal(ToolStage)
    sigToolReset = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self._current_mode = Mode.COMPOSITE
        self._current_tool = ToolType.NONE
        self._current_stage = ToolStage.IDLE

    def register_listener(
        self,
        listener: 'QtCore.QObject'
    ):
        """
        Connects an object to this tool manager's signals.

        If the object has any of these functions:
            - on_tool_changed()
            - on_mode_changed()
            - on_stage_changed()
            - on_tool_reset()

        .. they will be connected to appropriate signals in tool manager
        """
        for signal, func in [
            (self.sigModeChanged, 'on_tool_mode_changed'),
            (self.sigToolChanged, 'on_tool_type_changed'),
            (self.sigStageChanged, 'on_tool_stage_changed'),
            (self.sigToolReset, 'on_tool_reset'),
        ]:
            if hasattr(listener, func):
                signal.connect(
                    getattr(listener, func)
                )

    def request_mode(
        self,
        mode: Mode
    ):
        if mode != self._current_mode:

            self._current_mode = mode
            self.sigModeChanged.emit(
                self._current_mode
            )

            self.request_cancel()


    def request_tool(
        self,
        tool: ToolType,
    ):
        # Check if in currently in appropriate mode, and switch if appropriate
        if tool not in TOOL_GROUPS[self._current_mode]:
            success = self._switch_to_correct_mode(tool)
            if not success:
                return

        self._set_tool(tool)


    def request_next_stage(self):
        # Conditional branching in case I want to handle each stage differently
        match self._current_stage:
            case ToolStage.IDLE:
                self._set_stage(ToolStage.SELECTING)

            case ToolStage.SELECTING:
                self._set_stage(ToolStage.CONFIGURING)

            case ToolStage.CONFIGURING:
                self._set_stage(ToolStage.IDLE)

            case _:
                raise ValueError(
                    f"Invalid current_stage: {self._current_stage}"
                )


    def request_cancel(self):
        self._set_stage(ToolStage.IDLE)
        self._set_tool(ToolType.NONE)
        self.sigToolReset.emit()


    @property
    def active_tool(self) -> ToolType:
        return self._current_tool

    @property
    def active_stage(self) -> ToolStage:
        return self._current_stage

    @property
    def active_mode(self) -> Mode:
        return self._current_mode

    def _switch_to_correct_mode(
        self,
        tool: ToolType,
    ) -> bool:
        correct_mode = None
        for mode, tools in TOOL_GROUPS.items():
            if tool in tools:
                correct_mode = mode
                break
        if not correct_mode:
            logger.error(
                f"Requested tool {tool} not in TOOL_GROUPS"
            )
            return False

        self.request_mode(correct_mode)
        return True


    def _set_tool(
        self,
        tool: ToolType,
    ):
        self._set_stage(ToolStage.IDLE)

        self._current_tool = tool
        self.sigToolChanged.emit(self._current_tool)


    def _set_stage(
        self,
        stage: ToolStage
    ):
        self._current_stage = stage
        self.sigStageChanged.emit(
            self._current_stage
        )



