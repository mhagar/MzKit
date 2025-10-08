"""
Placing this here to avoid circular import
"""
from PyQt5 import QtCore

from enum import Enum, auto
from typing import Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5 import QtWidgets


class ToolType(Enum):
    NONE = auto()
    GETSPECTRUM = auto()
    GETCOMPOUND = auto()
    GETXIC = auto()


class ExtractionMode(Enum):
    NONE = auto()
    XIC = auto()
    BPC = auto()


class ToolStage(Enum):
    IDLE = auto()
    SELECTING = auto()
    CONFIGURING = auto()
    READY_TO_EXECUTE = auto()


class XICMode(Enum):
    NONE = auto()
    XIC = auto()
    BPC = auto()


class ToolStateListener(Protocol):
    """
    Protocol for objects that need to know
    about tool state
    """
    sigSelectionMade: QtCore.pyqtSignal
    sigConfigurationMade: QtCore.pyqtSignal

    def on_tool_type_changed(self, tool: ToolType) -> None: ...
    def on_tool_stage_changed(self, stage: ToolStage) -> None: ...
    def on_xic_mode_changed(self, mode: XICMode) -> None: ...


class ToolManagerNew(QtCore.QObject):
    """
    State manager for mutually exclusive multi-stage tools
    """
    # Signals
    stage_changed = QtCore.pyqtSignal(ToolStage)
    tool_changed = QtCore.pyqtSignal(ToolType)

    selection_ready = QtCore.pyqtSignal(object)
    configuration_ready = QtCore.pyqtSignal(dict)

    operation_complete = QtCore.pyqtSignal()

    sigReset = QtCore.pyqtSignal()

    xic_mode_changed = QtCore.pyqtSignal(XICMode)


    def __init__(self):
        super().__init__()

        # Tool states
        self._stage = ToolStage.IDLE
        self._active_tool = ToolType.NONE
        self._selection_data: Optional[any] = None
        self._config_data: Optional[dict] = None

        # These tools follow the idle -> selection -> configuration -> execute
        self._multistep_tools = {
            ToolType.GETCOMPOUND
        }

        # Functions that are called when tool is executed
        self._tool_callbacks: dict[ToolType, callable] = {}

        # XIC state
        self._xic_mode: XICMode = XICMode.NONE


    @property
    def stage(self) -> ToolStage:
        return self._stage


    @property
    def active_tool(self) -> ToolType:
        return self._active_tool


    @property
    def xic_mode(self) -> XICMode:
        return self._xic_mode


    def _set_stage(
        self,
        new_stage: ToolStage
    ):
        if self._stage != new_stage:
            self._stage = new_stage

            self.stage_changed.emit(
                new_stage
            )


    def activate_tool(
        self,
        tool: ToolType
    ):
        """
        Activate a tool. Cancels currently active tool
        """
        if self._stage != ToolStage.IDLE:
            self.cancel()

        self._active_tool = tool

        if self._active_tool != ToolType.NONE:
            self._set_stage(ToolStage.SELECTING)

        self.tool_changed.emit(tool)


    def set_selection(
        self,
        selection_data: any = None
    ) -> bool:
        """
        Handle selection. Returns True if accepted, False if invalid state
        """
        if self._stage != ToolStage.SELECTING:
            return False

        self._selection_data = selection_data
        self.selection_ready.emit(selection_data)

        # Check if tool is multi-step
        if self._active_tool in self._multistep_tools:
            self._set_stage(ToolStage.CONFIGURING)
        else:
            # Skip to processing for simple tools
            self._process()

        return True


    def set_configuration(
        self,
        config_data: dict[any],
    ) -> bool:
        """
        Handle configuration.
        Returns True if accepted, False if invalid state
        """
        if self._stage != ToolStage.CONFIGURING:
            return False

        self._config_data = config_data
        self.configuration_ready.emit(config_data)

        self._process()

        return True


    def _process(self):
        """
        Execute a tool operation and return to idle
        """
        self._set_stage(
            ToolStage.READY_TO_EXECUTE
        )

        # Execute registered operation, if exists
        operation = self._tool_callbacks.get(self._active_tool)
        if operation:
            operation(self._selection_data, self._config_data)

        self.operation_complete.emit()
        self._reset()


    def cancel(self):
        """
        Cancel current tool, return to idle
        """
        if self._stage != ToolStage.IDLE:
            self._reset()


    def _reset(self):
        """
        Reset to ToolStage.IDLE and ToolType.NONE
        """
        self._set_stage(ToolStage.IDLE)
        self.activate_tool(ToolType.NONE)

        self._selection_data = None
        self._config_data = None

        self.sigReset.emit()


    def set_xic_mode(
        self,
        mode: XICMode
    ):
        self._xic_mode = mode

        if self.active_tool != ToolType.NONE:
            self.activate_tool(ToolType.NONE)

        self.xic_mode_changed.emit(
            self._xic_mode
        )


    def register_tool_operation(
        self,
        tool: ToolType,
        operation: callable,
    ):
        """
        Register a callback/function that will be called when a tool
        is invoked
        :param tool:
        :param operation:
        :return:
        """
        self._tool_callbacks[tool] = operation


    def register_tool_listener(
        self,
        listener: ToolStateListener,
    ):
        """
        Register a listener for tool state changes.
        The listener should implement ToolStateListener protocol
        """
        self.tool_changed.connect(listener.on_tool_type_changed)
        self.stage_changed.connect(listener.on_tool_stage_changed)
        self.xic_mode_changed.connect(listener.on_xic_mode_changed)

        # Force sync
        listener.on_tool_type_changed(self._active_tool)
        listener.on_tool_stage_changed(self._stage)
        listener.on_xic_mode_changed(self._xic_mode)

        listener.sigSelectionMade.connect(
            self.set_selection
        )

        listener.sigConfigurationMade.connect(
            self.set_configuration
        )











