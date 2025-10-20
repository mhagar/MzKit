from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import pyqtgraph as pg

from core.data_structs import Ensemble
from gui.resources.EnsembleViewerWindow import Ui_Form
from gui.views.ensemble_viewer.tools import (
    ToolType, ToolStage, Mode,
    ToolManager,
)
from gui.views.ensemble_viewer.find_formula import FindFormulaController
from gui.views.ensemble_viewer.tool_controllers import (
    BaseToolController,
    MeasureLossController,
)
from gui.views.ensemble_viewer.plot_managers import (
    ChromatogramPlotManager,
    SpectrumPlotManager,
)

from typing import TYPE_CHECKING, Optional, Literal

if TYPE_CHECKING:
    from core.interfaces.data_sources import SampleDataSource


class EnsembleViewer(
    QtWidgets.QWidget,
    Ui_Form,
):
    sigSelectionMade = QtCore.pyqtSignal()
    sigConfigurationMade = QtCore.pyqtSignal()

    def __init__(
        self,
        data_source: 'SampleDataSource'
    ):
        super().__init__()
        self.setupUi(self)
        self.data_source = data_source
        self.tool_manager = ToolManager()

        self.ensemble: Optional['Ensemble'] = None

        # Plot managers
        self.chrom_manager = ChromatogramPlotManager(
            chrom_plot_widget=self.chromPlotWidget,
            corr_plot_widget=self.corrPlotWidget,
        )
        self.spectrum_manager = SpectrumPlotManager(
            ms1_plot_widget=self.ms1_plot,
            ms2_plot_widget=self.ms2_plot,
        )

        # Tool controllers
        self.tool_controllers: dict[ToolType, BaseToolController] = {
            ToolType.FINDFORMULA: FindFormulaController(self),
            ToolType.MEASURELOSS: MeasureLossController(self),
        }

        self._setup_plots()
        self._setup_actions()
        self._setup_tool_listeners()
        self._connect_signals()
        self._hide_misc_plots()

    def _connect_signals(self):
        # Connect spectrum manager signals
        self.spectrum_manager.sigMS1SignalClicked.connect(
            self.on_ms1_signal_clicked
        )

        self.spectrum_manager.sigMS2SignalClicked.connect(
            self.on_ms2_signal_clicked
        )

        # Connect transform checkboxes
        self.checkNormalize.clicked.connect(
            self._on_transform_settings_changed
        )

        self.checkDiff.clicked.connect(
            self._on_transform_settings_changed
        )

        # *** TOOLS ***
        # Connect all tool controller signals
        for controller in self.tool_controllers.values():
            controller.sigSignalSelected.connect(
                self._update_signal_selection_graphics
            )
            controller.sigSelectionCleared.connect(
                self._clear_signal_selection_graphics
            )

    def _setup_plots(self):
        """
        """
        # Configure MS plots
        ms2_vb = self.ms2_plot.pi.vb
        ms1_vb = self.ms1_plot.pi.vb
        ms2_vb.setXLink(ms1_vb)

        # Configure corr plot
        self.corrPlotWidget.addItem(
            pg.InfiniteLine(
                angle=45,
            )
        )

    def _setup_actions(self):
        """
        Configure action group for mutually exclusive tools
        """
        # This is for switching between 'composite' and 'scan' mode
        self.mode_action_group = QtWidgets.QActionGroup(self)
        self.mode_action_group.triggered.connect(
            self.on_mode_action_triggered
        )

        # This is for switching between 'find formula' and 'measure loss'
        self.tool_action_group = QtWidgets.QActionGroup(self)
        self.tool_action_group.triggered.connect(
            self.on_tool_action_triggered
        )

        for action, action_type, btn in [
            (self.actionScan,       'mode', self.toolScan),
            (self.actionComposite,  'mode', self.toolComposite),
            (self.actionFindFormula,'tool', self.toolFindFormula),
            (self.actionMeasureLoss,'tool', self.toolMeasureLoss),
        ]:
            action: QtWidgets.QAction
            action_type: Literal['mode', 'tool']
            btn: QtWidgets.QToolButton

            btn.setDefaultAction(action)

            match action_type:
                case 'mode':
                    self.mode_action_group.addAction(
                        action
                    )

                case 'tool':
                    self.tool_action_group.addAction(
                        action
                    )

        # Initialize in Composite mode
        self.actionComposite.trigger()

    def _setup_tool_listeners(self):
        """
        Connect the tool manager to whatever objects need to respond to
        tool activations
        """
        for listener in [
            self,
            self.ms1_plot,
            self.ms2_plot,
        ]:
            self.tool_manager.register_listener(listener)

    def on_mode_action_triggered(
        self,
        action: QtWidgets.QAction,
    ):
        """
        If user switches to scan/composite mode,
        tells ToolManager to either reset or switch to scan tool
        """
        mode = {
            self.actionScan: Mode.SCAN,
            self.actionComposite: Mode.COMPOSITE,
        }.get(action)

        if not mode:
            raise ValueError(
                f"QAction has no corresponding mode: {action}"
            )

        self.tool_manager.request_mode(mode)

    def on_tool_action_triggered(
        self,
        action: QtWidgets.QAction,
    ):
        """
        Tells ToolManager that user wants to switch to a tool
        """
        tool_map = {
            self.actionFindFormula: ToolType.FINDFORMULA,
            self.actionMeasureLoss: ToolType.MEASURELOSS,
        }

        tool_type = tool_map.get(
            action,
            ToolType.NONE,  # Default if invalid tool
        )

        self.tool_manager.request_tool(tool_type)

    def on_tool_type_changed(
        self,
        tool: ToolType,
    ):
        """
        Controls this widget's behaviour when tool is changed.
        Delegates to the appropriate tool controller.
        """
        if tool == ToolType.NONE:
            return

        # Delegate to the tool controller
        controller = self.tool_controllers.get(tool)
        if controller:
            controller.on_activated()

        self._update_tool_buttons()
        self._clear_signal_selection_graphics()

    def on_tool_stage_changed(
        self,
        stage: ToolStage,
    ):
        """
        Notify active tool controller of stage change
        """
        self._update_tool_buttons()

        # Notify active controller
        active_tool = self.tool_manager.active_tool
        controller = self.tool_controllers.get(active_tool)
        if controller:
            controller.on_stage_changed(stage)

    def on_tool_mode_changed(
        self,
        mode: Mode
    ):
        self._update_tool_buttons()

    def on_tool_reset(self):
        """
        Called when tool is reset/cancelled
        """
        # Notify active controller
        active_tool = self.tool_manager.active_tool
        controller = self.tool_controllers.get(active_tool)
        if controller:
            controller.on_cancelled()

        self._update_tool_buttons()
        self._clear_signal_selection_graphics()

    def _update_tool_buttons(
        self,
    ):
        """
        Updates the tool buttons to match the state of ToolManager,
        ***without emitting signals***!!
        """
        _ = (
            ( self.toolScan,         'mode',      Mode.SCAN),
            ( self.toolComposite,    'mode',      Mode.COMPOSITE),
            ( self.toolFindFormula,  'tool',  ToolType.FINDFORMULA),
            ( self.toolMeasureLoss,  'tool',  ToolType.MEASURELOSS),
        )

        for btn, activation_type, activation_condition in _:
            activation_type: Literal['mode', 'tool']
            btn: QtWidgets.QToolButton

            with QtCore.QSignalBlocker(btn):
                match activation_type:
                    case 'mode':
                        btn.setChecked(
                            activation_condition == self.tool_manager.active_mode
                        )

                    case 'tool':
                        btn.setChecked(
                            activation_condition == self.tool_manager.active_tool
                        )

    def set_ensemble(
        self,
        ensemble: 'Ensemble'
    ):
        self.ensemble = ensemble

        # Update plot managers with ensemble
        self.chrom_manager.set_ensemble(ensemble)
        self.spectrum_manager.set_ensemble(ensemble)

        self.initialize_plots()

    def _hide_misc_plots(self):
        self.checkShowMiscPlots.setChecked(False)
        self.tabWidget.setVisible(False)

    def _show_misc_plots(self):
        self.checkShowMiscPlots.setChecked(True)
        self.tabWidget.setVisible(True)

    def initialize_plots(
        self,
    ):
        if not self.ensemble:
            return

        self._update_transform_settings()

        # Populate plots using managers
        self.spectrum_manager.populate_spectrum_plot(
            scan_rt=self.ensemble.peak_rt
        )
        self.chrom_manager.populate_chromatogram_plot(
            peak_rt=self.ensemble.peak_rt
        )

        # Connect chromatogram selector signal
        self.chromPlotWidget.pi.selection_indicator.sigPositionChanged.connect(
            self.onChromatogramSelectorMoved
        )

    def onChromatogramSelectorMoved(
        self,
        slide_selector: 'pg.InfiniteLine',
    ):
        """
        Called when user moves the chromatogram selector
        """
        new_xpos = slide_selector.getXPos()

        if new_xpos != self.chrom_manager.selected_rt:

            self.spectrum_manager.populate_spectrum_plot(scan_rt=new_xpos)
            self.chrom_manager.selected_rt = new_xpos

    def _on_transform_settings_changed(self):
        """
        Called when normalize or diff checkboxes change
        """
        self._update_transform_settings()
        self.chrom_manager.populate_chromatogram_plot(
            peak_rt=self.ensemble.peak_rt
        )
        self.chrom_manager.update_chromatogram_plot()

    def _update_transform_settings(self):
        """
        Update the chromatogram manager with current transform settings
        """
        self.chrom_manager.set_transform_settings(
            normalize=self.checkNormalize.isChecked(),
            diff=self.checkDiff.isChecked(),
        )

    def on_ms1_signal_clicked(
        self,
        data: tuple[int, float], # [spec_idx, mz_float]
    ):
        """
        Called when user clicks on a signal in the MS1 spectrum
        """
        spec_idx, mz = data

        if not spec_idx:
            return

        # Delegate to active tool controller
        active_tool = self.tool_manager.active_tool
        controller = self.tool_controllers.get(active_tool)
        if controller:
            controller.handle_ms_signal_clicked(
                data=data,
                ms_level=1,
            )

        # Get chromatograms and update manager
        ms1_chroms = self.ensemble.get_chromatograms(
            ms_level=1,
            idxs=slice(spec_idx, spec_idx + 1),
        )
        self.chrom_manager.set_ms1_chroms(ms1_chroms)

        self.chrom_manager.update_chromatogram_plot()
        self.chrom_manager.update_correlation_plot()

    def on_ms2_signal_clicked(
        self,
        data: tuple[int, float], # [spec_idx, mz_float]
    ):
        """
        Called when user clicks on a signal in the MS2 spectrum
        """
        spec_idx, mz = data

        if not spec_idx:
            return

        # Delegate to active tool controller
        active_tool = self.tool_manager.active_tool
        controller = self.tool_controllers.get(active_tool)
        if controller:
            controller.handle_ms_signal_clicked(
                data=data,
                ms_level=2,
            )

        # Get chromatograms and update manager
        ms2_chroms = self.ensemble.get_chromatograms(
            ms_level=2,
            idxs=slice(spec_idx, spec_idx + 1),
        )
        self.chrom_manager.set_ms2_chroms(ms2_chroms)

        self.chrom_manager.update_correlation_plot()
        self.chrom_manager.update_chromatogram_plot()

    def _update_signal_selection_graphics(
        self,
        data: tuple[int, float],
        level: Literal[1, 2, None],
        is_selected: bool,
    ):
        """
        Called whenever user selects/deselects a signal.
        Called with `level = None` if user switches from MS1 <-> MS2 spectrum
        """
        if not level:
            return

        # Delegate to spectrum manager
        if is_selected:
            self.spectrum_manager.add_signal_marker(
                spec_idx=data[0],
                ms_level=level,
            )
        else:
            self.spectrum_manager.remove_signal_marker(
                spec_idx=data[0],
                ms_level=level,
            )

    def _clear_signal_selection_graphics(self):
        """Clear all signal selection markers"""
        self.spectrum_manager.clear_signal_markers()

    def keyPressEvent(self, event):
        """
        Intercepts key press events:
        - Escape to exit tool modes
        - Enter to accept tool config
        """
        if event.key() == Qt.Key_Escape:
            if self.tool_manager.active_tool != ToolType.NONE:
                # Cancel current tool and return to view mode
                self.tool_manager.request_cancel()
                event.accept()
                return

        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self._on_press_enter()
            event.accept()
            return

        # Pass event for normal handling
        super().keyPressEvent(event)

    def _on_press_enter(
        self,
    ):
        """
        Called when user hits Enter.
        Delegates to active tool controller.
        """
        if not self.tool_manager.active_stage == ToolStage.SELECTING:
            return

        # Delegate to active tool controller
        active_tool = self.tool_manager.active_tool
        controller = self.tool_controllers.get(active_tool)
        if controller:
            controller.on_enter_pressed()
