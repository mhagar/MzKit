from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
from pyqtgraph import exporters

from core.data_structs import Ensemble, IonAnnotation
from core.data_structs.ensemble import MzDiffAnnotation, GenericAnnotation
from find_mfs.isotopes.envelope import get_isotope_envelope
from core.utils.spectrum_export import to_sirius_ms, to_mgf
from core.utils.config import load_config
from gui.utils.formula_formatting import format_formula_obj_to_html
from gui.resources.EnsembleViewerWindow import Ui_Form
from gui.views.ensemble_viewer.tools import (
    ToolType, ToolStage, Mode,
    ToolManager,
)
from gui.views.ensemble_viewer.dda_overlays import EnsembleDDAOverlayManager
from gui.views.ensemble_viewer.add_annot import AddAnnotController
from gui.views.ensemble_viewer.find_formula import FindFormulaController
from gui.views.ensemble_viewer.measure_loss import MeasureLossController
from gui.views.ensemble_viewer.plot_managers import (
    ChromatogramPlotManager,
    SpectrumPlotManager,
)
from gui.dialogues.CompoundExportWizard import NCICompoundExportWizard
from gui.models.ensemble_properties_model import EnsemblePropertiesModel

from pathlib import Path
import json
import subprocess
import threading
from typing import TYPE_CHECKING, Optional, Literal

if TYPE_CHECKING:
    from gui.views.ensemble_viewer.tool_controllers import (
        BaseToolController,
    )
    from configparser import ConfigParser
    from core.interfaces.data_sources import SampleDataSource
    from find_mfs import FormulaCandidate


class EnsembleViewer(
    QtWidgets.QWidget,
    Ui_Form,
):
    sigSelectionMade = QtCore.pyqtSignal()
    sigConfigurationMade = QtCore.pyqtSignal()

    def __init__(
        self,
        data_source: 'SampleDataSource',
        config: Optional['ConfigParser'] = None
    ):
        super().__init__()
        self.setupUi(self)
        self.data_source = data_source

        # TODO: TEMPORARY. NEED TO REFACTOR WINDOW MGMT
        self.config = load_config()
        self.tool_manager = ToolManager()

        self.ensemble: Optional['Ensemble'] = None
        self.properties_model: Optional['EnsemblePropertiesModel'] = None

        self.wizard: Optional['NCICompoundExportWizard'] = None

        # Transient plot-id → annotation-uuid maps. These are populated
        # by the _draw_* methods and cleared at the top of every
        # _redraw_annotations_for_current_scan call, because the plot
        # IDs become invalid the moment the spectrum redraws.
        # Shape: {annot_uuid: {ms_level: plot_id}}
        self._mz_diff_plot_ids: dict[int, str] = {}
        self._ion_annot_plot_ids: dict[int, str] = {}
        self._generic_annot_plot_ids: dict[int, str] = {}

        # Tracks the uuid of the most recently created MzDiffAnnotation
        # so the neutral-loss-formula signal (fired asynchronously when
        # the user picks a candidate) knows which annotation to attach
        # the formula to.
        self._last_mz_diff_uuid: Optional[int] = None

        # Plot managers
        self.chrom_manager = ChromatogramPlotManager(
            chrom_plot_widget=self.chromPlotWidget,
            corr_plot_widget=self.corrPlotWidget,
        )
        self.spectrum_manager = SpectrumPlotManager(
            ms1_plot_widget=self.ms1_plot,
            ms2_plot_widget=self.ms2_plot,
        )

        # DDA overlay layer (no-op for non-DDA ensembles).
        self.dda_overlay_mgr = EnsembleDDAOverlayManager(
            ms1_plot=self.ms1_plot,
            ms2_plot=self.ms2_plot,
            chrom_plot=self.chromPlotWidget,
            on_select_rt=self._select_ensemble_ms2_rt,
        )

        # Tool controllers
        self.tool_controllers: dict[ToolType, 'BaseToolController'] = {
            ToolType.FINDFORMULA: FindFormulaController(self),
            ToolType.MEASURELOSS: MeasureLossController(self),
            ToolType.ADDANNOT: AddAnnotController(self),
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

        self.spectrum_manager.sigMSSignalHovered.connect(
            self.on_ms_signal_hovered
        )


        # Connect transform checkboxes
        self.checkNormalize.clicked.connect(
            self._on_transform_settings_changed
        )

        self.checkDiff.clicked.connect(
            self._on_transform_settings_changed
        )

        self.checkNormalizeSpectra.toggled.connect(
            self._on_normalize_spectra_toggled
        )

        # *** TOOLS ***
        # Connect all generic tool controller signals
        for controller in self.tool_controllers.values():
            controller.sigSignalSelected.connect(
                self._update_signal_selection_graphics
            )
            controller.sigSelectionCleared.connect(
                self._clear_signal_selection_graphics
            )

        # Connect signal for when formula finder assigns a formula
        self.tool_controllers[ToolType.FINDFORMULA].sigFormulaAssigned.connect(
            self.add_formula_annotation
        )

        # Connect signal for when measure loss tool measures a delta m/z
        self.tool_controllers[ToolType.MEASURELOSS].sigMzDiffMeasured.connect(
            self.add_mz_diff_annotation
        )

        # Neutral-loss formula assignment: attaches to the most recently
        # created MzDiffAnnotation.
        self.tool_controllers[ToolType.MEASURELOSS].sigMzDiffFormulaAssigned.connect(
            self._on_neutral_loss_formula_assigned
        )

        # Connect signal for the add-annotation tool
        self.tool_controllers[ToolType.ADDANNOT].sigGenericAnnotationRequested.connect(
            self.add_generic_annotation
        )

        # Right-click on any drawn annotation routes through here. Same
        # handler for both plots — ms_level is inferred from which map
        # the plot_id lives in.
        self.ms1_plot.sigAnnotationClicked.connect(
            lambda plot_id, btn: self._on_annotation_clicked(plot_id, btn, 1)
        )
        self.ms2_plot.sigAnnotationClicked.connect(
            lambda plot_id, btn: self._on_annotation_clicked(plot_id, btn, 2)
        )
        # Connect metadata buttons
        self.pushAddMetadataField.clicked.connect(
            self._on_add_metadata_clicked
        )
        self.pushRemoveMetadataField.clicked.connect(
            self._on_remove_metadata_clicked
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
            (self.actionAddAnnot,   'tool', self.toolAddAnnot),
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

        # Set up EXPORT buttons
        self.toolExportSpec.setDefaultAction(
            self.actionExportSpec
        )
        self.actionExportSpec.triggered.connect(
            self.show_export_cmpd_wizard
        )

        # Clear-annotation triggers (not modes — just one-shot actions).
        self.toolClearScanAnnots.setDefaultAction(self.actionClearScanAnnots)
        self.toolClearAllAnnots.setDefaultAction(self.actionClearAllAnnots)
        self.actionClearScanAnnots.triggered.connect(
            self._on_clear_scan_annotations
        )
        self.actionClearAllAnnots.triggered.connect(
            self._on_clear_all_annotations
        )

        # self.toolExportChrom.setDefaultAction(
        #     self.actionExportChrom
        # )
        # self.actionExportChrom.triggered.connect(
        #     self.export_chromatogram
        # )

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
            self.actionAddAnnot:    ToolType.ADDANNOT,
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
            ( self.toolAddAnnot,     'tool',  ToolType.ADDANNOT),
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
        self.dda_overlay_mgr.set_ensemble(ensemble)

        self.initialize_plots()
        self._redraw_annotations_for_current_scan()
        self.initialize_property_table()

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

        # For DDA, snap the initial spectrum RT onto a matched MS2 scan
        # so we never open onto a scan that doesn't belong to this
        # ensemble. For non-DDA, peak_rt passes through unchanged.
        initial_rt = self._snap_rt_to_ensemble(self.ensemble.peak_rt)

        # Populate plots using managers
        self.spectrum_manager.populate_spectrum_plot(
            scan_rt=initial_rt
        )
        self.dda_overlay_mgr.update(scan_rt=initial_rt)
        self.chrom_manager.populate_chromatogram_plot(
            peak_rt=initial_rt
        )

        # Connect chromatogram selector signal
        self.chromPlotWidget.pi.selection_indicator.sigPositionChanged.connect(
            self.onChromatogramSelectorMoved
        )

    def initialize_property_table(
        self,
    ):
        """
        Sets up the tableView to show the Ensemble's properties
        """
        if not self.ensemble:
            return

        sample_name: str = self.data_source.get_sample(
            self.ensemble.injection.sample_uuid
        ).name

        self.properties_model = EnsemblePropertiesModel(
            ensemble=self.ensemble,
            sample_name=sample_name
        )
        self.tableViewProperties.setModel(self.properties_model)

        # Configure table appearance
        header = self.tableViewProperties.horizontalHeader()
        header.setStretchLastSection(True)
        self.tableViewProperties.verticalHeader().setVisible(False)

        # Resize columns to content
        self.tableViewProperties.resizeColumnsToContents()

    def _on_add_metadata_clicked(self):
        """
        Add a new metadata field row
        """
        if not self.properties_model:
            return

        new_row = self.properties_model.add_metadata_field()

        # Select the new row for editing
        index = self.properties_model.index(new_row, 0)
        self.tableViewProperties.setCurrentIndex(index)
        self.tableViewProperties.edit(index)

    def _on_remove_metadata_clicked(self):
        """
        Remove the selected metadata field row
        """
        if not self.properties_model:
            return

        current_index = self.tableViewProperties.currentIndex()
        if not current_index.isValid():
            return

        row = current_index.row()
        if not self.properties_model.is_metadata_row(row):
            # Can only remove metadata rows
            return

        self.properties_model.remove_metadata_field(row)

    def onChromatogramSelectorMoved(
        self,
        slide_selector: 'pg.InfiniteLine',
    ):
        """
        Called when user moves the chromatogram selector
        """
        new_xpos = slide_selector.getXPos()

        # For DDA, snap the cursor to the nearest RT that actually belongs
        # to this ensemble — otherwise the MS2 panel would 'escape' the
        # ensemble and render scans triggered by unrelated precursors.
        new_xpos = self._snap_rt_to_ensemble(new_xpos)

        if new_xpos != self.chrom_manager.selected_rt:

            self.spectrum_manager.populate_spectrum_plot(scan_rt=new_xpos)
            self.dda_overlay_mgr.update(scan_rt=new_xpos)
            self.chrom_manager.selected_rt = new_xpos
            self._redraw_annotations_for_current_scan()

    def _snap_rt_to_ensemble(self, rt: float) -> float:
        """
        If the ensemble is from a DDA injection, snap the requested RT to
        the nearest RT among the ensemble's matched MS2 scans. Returns
        the original RT for non-DDA ensembles.
        """
        if (
            not self.ensemble
            or not self.ensemble.injection
            or self.ensemble.injection.acquisition_mode != 'dda'
            or not self.ensemble.ms2_cofeatures
        ):
            return rt

        ms2_arr = self.ensemble.injection.scan_array_ms2
        if ms2_arr is None or ms2_arr.rt_arr is None:
            return rt

        # All MS2 cofeatures share the same scan_idxs by construction.
        scan_idxs = self.ensemble.ms2_cofeatures[0].scan_idxs
        if scan_idxs.size == 0:
            return rt

        candidate_rts = ms2_arr.rt_arr[scan_idxs]
        nearest = int(np.argmin(np.abs(candidate_rts - rt)))
        return float(candidate_rts[nearest])

    def _select_ensemble_ms2_rt(self, rt: float) -> None:
        """
        Programmatic equivalent of the user dragging the chrom selector
        to `rt`. Used by the DDA badge click handler so the click goes
        through the same snap-and-update path as a user drag.
        """
        rt = self._snap_rt_to_ensemble(rt)
        # Block sigPositionChanged so we don't re-enter
        # onChromatogramSelectorMoved when we move the indicator.
        indicator = self.chromPlotWidget.pi.selection_indicator
        with QtCore.QSignalBlocker(indicator):
            indicator.setPos(rt)
        self.spectrum_manager.populate_spectrum_plot(scan_rt=rt)
        self.dda_overlay_mgr.update(scan_rt=rt)
        self.chrom_manager.selected_rt = rt
        self._redraw_annotations_for_current_scan()

    def _on_normalize_spectra_toggled(self, checked: bool):
        """
        Re-populate both spectrum plots with the new normalize setting.
        """
        self.spectrum_manager.set_normalize_spectra(checked)
        if not self.ensemble:
            return
        rt = self.chrom_manager.selected_rt or self.ensemble.peak_rt
        self.spectrum_manager.populate_spectrum_plot(scan_rt=rt)
        self.dda_overlay_mgr.update(scan_rt=rt)
        self._redraw_annotations_for_current_scan()

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
        data: tuple[float, float, int], # mz, intsy, spec_idx
    ):
        """
        Called when user clicks on a signal in the MS1 spectrum
        """
        mz, intsy, spec_idx = data

        print(
            f"Selected: {data}"
        )

        if spec_idx is None:
            return

        # Delegate to active tool controller
        active_tool = self.tool_manager.active_tool
        controller = self.tool_controllers.get(
            active_tool
        )
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
        self.chrom_manager.set_last_selection(mz=mz, ms_level=1)

        self.chrom_manager.update_chromatogram_plot()
        self.chrom_manager.update_correlation_plot()

    def on_ms2_signal_clicked(
        self,
        data: tuple[float, float, int], # [spec_idx, mz_float]
    ):
        """
        Called when user clicks on a signal in the MS2 spectrum
        """
        mz, intsy, spec_idx = data

        if spec_idx is None:
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
        self.chrom_manager.set_last_selection(mz=mz, ms_level=2)

        self.chrom_manager.update_correlation_plot()
        self.chrom_manager.update_chromatogram_plot()

    def on_ms_signal_hovered(
        self,
        data: tuple[float, float, int, Literal[1, 2]],  # mz, intsy, spec_idx, ms_level
    ):
        """
        Called when user hovers on a signal in the MS1 or MS2 spectrum
        """
        mz, intsy, spec_idx, ms_level = data
        if spec_idx is None:
            return

        # Delegate to active tool controller
        active_tool = self.tool_manager.active_tool
        controller = self.tool_controllers.get(
            active_tool
        )
        if controller:
            controller.handle_ms_signal_hovered(
                data=(mz, intsy, spec_idx),
                ms_level=ms_level,
            )

    def _update_signal_selection_graphics(
        self,
        mz: float,
        intsy: float,
        spec_idx: int,
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
                spec_idx=spec_idx,
                ms_level=level,
            )
        else:
            self.spectrum_manager.remove_signal_marker(
                spec_idx=spec_idx,
                ms_level=level,
            )

    def _clear_signal_selection_graphics(self):
        """Clear all signal selection markers"""
        self.spectrum_manager.clear_signal_markers()

        # TODO: This is spaghetti
        self.tool_controllers[ToolType.MEASURELOSS]._clear_transient_bracket()

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

    def add_formula_annotation(
        self,
        formula_candidate: 'FormulaCandidate',
        ms_level,
        cofeature_idxs,
    ):
        if not self.ensemble:
            return

        annotation: IonAnnotation = self.ensemble.add_ion_annot(
            cofeature_idxs=cofeature_idxs,
            ms_level=ms_level,
            formula=formula_candidate,
            label="",
            scan_num=self._current_scan_num(ms_level),
        )

        self._draw_ion_annotation(annotation)
        self._refresh_chrom_annotation_markers()

    def add_generic_annotation(
        self,
        cofeature_idx: int,
        ms_level: Literal[1, 2],
        text: str,
    ):
        if not self.ensemble:
            return

        annotation = self.ensemble.add_generic_annot(
            cofeature_idx=cofeature_idx,
            ms_level=ms_level,
            text=text,
            scan_num=self._current_scan_num(ms_level),
        )
        self._draw_generic_annotation(annotation)
        self._refresh_chrom_annotation_markers()

    def add_mz_diff_annotation(
        self,
        cofeature_a_idx: int,
        cofeature_b_idx: int,
        delta_mz: float,
        ms_level: Literal[1, 2],
    ):
        if not self.ensemble:
            return

        annotation: MzDiffAnnotation = self.ensemble.add_mz_diff_annot(
            cofeature_a_idx=cofeature_a_idx,
            cofeature_b_idx=cofeature_b_idx,
            ms_level=ms_level,
            delta_mz=delta_mz,
            scan_num=self._current_scan_num(ms_level),
        )
        self._last_mz_diff_uuid = annotation.uuid

        self._draw_mz_diff_annotation(annotation)
        self._refresh_chrom_annotation_markers()

    def _on_neutral_loss_formula_assigned(
        self,
        formula: 'FormulaCandidate',
    ):
        """
        Attach a neutral-loss formula to the most recently created
        MzDiffAnnotation (the one whose bracket was just drawn). No
        validation — purely a labelling aid.
        """
        if not self.ensemble or self._last_mz_diff_uuid is None:
            return

        target = next(
            (a for a in self.ensemble.mz_diffs
             if a.uuid == self._last_mz_diff_uuid),
            None,
        )
        if target is None:
            return

        target.formula = formula

        # The bracket was already drawn with the plain Δ label; clear
        # and redraw so the new two-line label takes effect.
        for plot in (self.ms1_plot, self.ms2_plot):
            plot.clear_delta_brackets()
        self._redraw_annotations_for_current_scan()

    def _draw_mz_diff_annotation(self, annotation: 'MzDiffAnnotation'):
        ms_plot = {
            1: self.ms1_plot,
            2: self.ms2_plot,
        }[annotation.ms_level]

        # When a neutral-loss formula is attached, give the formula
        # prominence and demote the \u0394 m/z to a small subtitle line.
        if annotation.formula is not None:
            formula_html = format_formula_obj_to_html(
                annotation.formula.formula
            )
            label = (
                f'<div style="text-align:center;">'
                f'{formula_html}'
                f'<br><span style="font-size:8pt;">'
                f'\u0394 {annotation.delta_mz:.4f}'
                f'</span></div>'
            )
        else:
            label = f"\u0394 {annotation.delta_mz:.4f}"

        plot_id = ms_plot.add_delta_bracket(
            spec_idx_a=annotation.cofeature_a_idx,
            spec_idx_b=annotation.cofeature_b_idx,
            text=label,
        )
        self._mz_diff_plot_ids[annotation.uuid] = plot_id

    def _current_scan_num(self, ms_level: Literal[1, 2]) -> Optional[int]:
        """
        Scan index (column in the ms_level ScanArray) for the currently
        displayed spectrum. Used as the anchor when committing an
        annotation, and when filtering which annotations to re-draw.
        """
        if not self.ensemble or not self.ensemble.injection:
            return None
        scan_array = self.ensemble.injection.get_scan_array(ms_level)
        if scan_array is None:
            return None
        return int(scan_array.rt_to_scan_num(self.spectrum_manager.selected_rt))

    def _draw_ion_annotation(self, annotation: 'IonAnnotation'):
        """
        Draws an ion annotation graphic (label + isotope envelope)
        on the appropriate MS plot.
        """
        ms_plot = {
            1: self.ms1_plot,
            2: self.ms2_plot,
        }[annotation.ms_level]

        envelope = get_isotope_envelope(
            formula=annotation.formula.formula,
            mz_tolerance=0.05,
            threshold=0.005,
        )

        plot_id = ms_plot.add_ion_annotation(
            spec_idxs=annotation.cofeature_idxs,
            text=annotation.format_string,
            envelope=envelope,
        )
        self._ion_annot_plot_ids[annotation.uuid] = plot_id

    def _draw_generic_annotation(self, annotation: 'GenericAnnotation'):
        """
        Draws a free-form user annotation as an anchored text label.
        """
        ms_plot = {
            1: self.ms1_plot,
            2: self.ms2_plot,
        }[annotation.ms_level]

        plot_id = ms_plot.add_anchored_label(
            spec_idx=annotation.cofeature_idx,
            text=annotation.text,
        )
        self._generic_annot_plot_ids[annotation.uuid] = plot_id

    def _refresh_chrom_annotation_markers(self):
        """
        Collect every annotation's RT (grouped by ms_level) and push
        them to the chrom plot manager as down-triangle markers. Skip
        scan-agnostic annotations (`scan_num is None`) since they don't
        anchor to any specific RT.
        """
        if not self.ensemble or not self.ensemble.injection:
            return

        rts_by_level: dict[int, set[float]] = {1: set(), 2: set()}
        for annot in (
            list(self.ensemble.mz_diffs)
            + list(self.ensemble.ion_annots.values())
            + list(self.ensemble.generic_annots.values())
        ):
            if annot.scan_num is None:
                continue
            scan_array = self.ensemble.injection.get_scan_array(annot.ms_level)
            if scan_array is None:
                continue
            try:
                rts_by_level[annot.ms_level].add(
                    float(scan_array.rt_arr[annot.scan_num])
                )
            except (IndexError, TypeError, KeyError):
                continue

        self.chrom_manager.set_annotation_markers({
            ms_level: sorted(rts)
            for ms_level, rts in rts_by_level.items()
        })

    def _is_annot_on_current_scan(self, annot) -> bool:
        """
        Predicate: does this annotation belong to the currently-displayed
        spectrum? `scan_num is None` = scan-agnostic (back-compat for
        pre-scan_num .mzk files), always considered on-scan.
        """
        if annot.scan_num is None:
            return True
        target = self._current_scan_num(1 if annot.ms_level == 1 else 2)
        return annot.scan_num == target

    def _redraw_annotations_for_current_scan(self):
        """
        Redraw the annotations that belong on the currently-displayed
        spectrum. `MSPlotItem.setSpectrumArray` clears all annotations on
        every spectrum change, so this must be called after each
        `populate_spectrum_plot`.
        """
        if not self.ensemble:
            return

        # Reset transient plot-id maps \u2014 the prior IDs were invalidated
        # by the spectrum redraw / clear_*().
        self._mz_diff_plot_ids.clear()
        self._ion_annot_plot_ids.clear()
        self._generic_annot_plot_ids.clear()

        for annotation in self.ensemble.ion_annots.values():
            if self._is_annot_on_current_scan(annotation):
                self._draw_ion_annotation(annotation)

        for annotation in self.ensemble.mz_diffs:
            if self._is_annot_on_current_scan(annotation):
                self._draw_mz_diff_annotation(annotation)

        for annotation in self.ensemble.generic_annots.values():
            if self._is_annot_on_current_scan(annotation):
                self._draw_generic_annotation(annotation)

        # Keep chromatogram-side markers in sync with whatever lives on
        # the ensemble right now. Cheap (one ScatterPlotItem swap) and
        # routes every annotation lifecycle event through one place.
        self._refresh_chrom_annotation_markers()

    def _on_clear_scan_annotations(self):
        """
        Delete all annotations whose `scan_num` matches the currently-
        displayed scan (per ms_level). Scan-agnostic annotations
        (scan_num is None) are left alone \u2014 they don't 'belong' to any
        scan, so 'clear this scan' shouldn't remove them.
        """
        if not self.ensemble:
            return

        def belongs_to_current_scan(annot) -> bool:
            if annot.scan_num is None:
                return False
            return annot.scan_num == self._current_scan_num(
                1 if annot.ms_level == 1 else 2
            )

        self.ensemble.mz_diffs = [
            a for a in self.ensemble.mz_diffs
            if not belongs_to_current_scan(a)
        ]
        self.ensemble.ion_annots = {
            k: v for k, v in self.ensemble.ion_annots.items()
            if not belongs_to_current_scan(v)
        }
        self.ensemble.generic_annots = {
            k: v for k, v in self.ensemble.generic_annots.items()
            if not belongs_to_current_scan(v)
        }

        for plot in (self.ms1_plot, self.ms2_plot):
            plot.clear_ion_annotations()
            plot.clear_delta_brackets()
            plot.clear_anchored_labels()
        self._redraw_annotations_for_current_scan()

    def _on_clear_all_annotations(self):
        """
        Wipe every annotation on the ensemble. Confirms first because
        this is destructive and not scoped.
        """
        if not self.ensemble:
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear all annotations",
            "Delete every annotation on this ensemble? "
            "This cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        self.ensemble.mz_diffs.clear()
        self.ensemble.ion_annots.clear()
        self.ensemble.ion_pair_annots.clear()
        self.ensemble.generic_annots.clear()

        for plot in (self.ms1_plot, self.ms2_plot):
            plot.clear_ion_annotations()
            plot.clear_delta_brackets()
            plot.clear_anchored_labels()
        self._redraw_annotations_for_current_scan()

    def _on_annotation_clicked(
        self,
        plot_id: str,
        button: int,
        ms_level: Literal[1, 2],
    ):
        """
        Routed from MSPlotWidget.sigAnnotationClicked. Reverse-lookups
        the plot_id against the transient maps to find the underlying
        annotation, then offers a right-click context menu to delete it.

        Left-click is currently a no-op (kept reserved for future use).
        """
        if button != int(Qt.RightButton):
            return
        if not self.ensemble:
            return

        # Reverse-lookup: plot_id → (annot_uuid, kind). The maps were
        # populated by _draw_* in the most recent redraw; if the user
        # right-clicks a transient bracket (e.g. the measure-loss
        # preview), no mapping exists and we silently bail.
        annot_uuid: Optional[int] = None
        kind: Optional[Literal['mz_diff', 'ion', 'generic']] = None
        for uuid_, pid in self._mz_diff_plot_ids.items():
            if pid == plot_id:
                annot_uuid, kind = uuid_, 'mz_diff'
                break
        if annot_uuid is None:
            for uuid_, pid in self._ion_annot_plot_ids.items():
                if pid == plot_id:
                    annot_uuid, kind = uuid_, 'ion'
                    break
        if annot_uuid is None:
            for uuid_, pid in self._generic_annot_plot_ids.items():
                if pid == plot_id:
                    annot_uuid, kind = uuid_, 'generic'
                    break

        if annot_uuid is None:
            return

        menu = QtWidgets.QMenu(self)
        delete_action = menu.addAction("Delete annotation")
        chosen = menu.exec_(QtGui.QCursor.pos())
        if chosen is not delete_action:
            return

        # Strike from the data model, then trigger a full redraw of the
        # current scan (which also resets the transient plot-id maps).
        if kind == 'mz_diff':
            self.ensemble.mz_diffs = [
                a for a in self.ensemble.mz_diffs if a.uuid != annot_uuid
            ]
        elif kind == 'ion':
            self.ensemble.ion_annots.pop(annot_uuid, None)
        elif kind == 'generic':
            self.ensemble.generic_annots.pop(annot_uuid, None)

        for plot in (self.ms1_plot, self.ms2_plot):
            plot.clear_ion_annotations()
            plot.clear_delta_brackets()
            plot.clear_anchored_labels()
        self._redraw_annotations_for_current_scan()

    def _print_envelope_arr(
        self,
        ms_level: Literal[1, 2],
        spec_idxs: list[int],
        scan_window: int = 5,
    ):
        """
        TODO: THIS IS JUST FOR TESTING. Prints an array of m/z and intsy values
        for +5 and -5 scans from base scan for the selected spec idxs
        """
        print(f"TESTING: printing envelope +- {scan_window}")
        for scan_num in range(
            max(self.ensemble.base_scan_num - scan_window, 1),
            self.ensemble.base_scan_num + scan_window,
        ):
            envelope = self.ensemble.get_spectrum(
                ms_level=ms_level,
                scan_num=scan_num,
            )[spec_idxs]

            print(f"Scan: {scan_num}")
            for row in envelope:
                print(f"{row['mz']}, {row['intsy']}")



    def _run_annotation_async(self, compound_path: Path, ionmode: str = 'positive'):
        """
        Runs misc/annotate_compound.py then misc/generate_compound_reports.py
        on the given compound folder in a background thread, then opens the report.
        """
        misc_dir = Path("/home/mh/Dropbox/MzKit/misc")
        annotate_script = misc_dir / "annotate_compound.py"
        report_script   = misc_dir / "generate_compound_reports.py"
        template        = misc_dir / "compound_report_template.html"

        def _run():
            import webbrowser

            print(f"Starting annotation pipeline for: {compound_path.name}")
            try:
                result = subprocess.run(
                    [
                        "python", str(annotate_script), str(compound_path),
                        "--ionmode", ionmode,
                        "--sirius-bin", "/home/mh/Applications/sirius/bin/sirius"
                    ],
                    capture_output=False,
                )
                if result.returncode != 0:
                    print(f"Annotation pipeline exited with code {result.returncode}")
                    return

                print(f"Generating report for: {compound_path.name}")
                result = subprocess.run(
                    [
                        "python", str(report_script), str(compound_path),
                        "--template", str(template),
                    ],
                    capture_output=False,
                )
                if result.returncode != 0:
                    print(f"Report generation exited with code {result.returncode}")
                    return

                html_files = list(compound_path.glob("*_report.html"))
                if html_files:
                    webbrowser.open(html_files[0].as_uri())
                else:
                    print("Report generation finished but no HTML file found")

            except Exception as e:
                print(f"Annotation pipeline failed: {e}")

        threading.Thread(target=_run, daemon=True).start()

    def show_export_cmpd_wizard(self):
        self.wizard = NCICompoundExportWizard()
        self.wizard.show()
        self.wizard.sigCompoundExportParamsGiven.connect(
            self.handle_export_request
        )

    def handle_export_request(
        self,
        params: dict
    ):
        compound_path = Path(params['export_dir']) / params['compound_name']
        compound_path.mkdir(exist_ok=True)

        # Save MS1/MS2 spectra
        self.export_spectrum(
            compound_path
        )

        # Save chromatogram .svg
        self.export_chromatogram(
            compound_path
        )

        # Save everything else:
        self.export_cmpd_metadata(
            params, compound_path
        )

        print(
            f"Compound saved to {compound_path}"
        )

        # Run annotation pipeline in background
        ionmode = params.get('ionmode', 'positive')
        self._run_annotation_async(compound_path, ionmode)

    def export_spectrum(
        self,
        directory: Path,
    ):
        """
        Rough placeholder, need roger's plots asap
        """
        # Shit out .svgs
        for ms_level, plot_item in [
            ('ms1', self.ms1_plot.pi),
            ('ms2', self.ms2_plot.pi),
        ]:
            svg_exporter = exporters.SVGExporter(
                plot_item
            )

            filename = Path(directory) / f'{self.ensemble.format_string}_{ms_level}.svg'
            svg_exporter.export(
                filename
            )

            print(
                f"Exported {filename}"
            )

        # Shit out .ms file
        ms1_spec_arr = self.ensemble.get_spectrum(
            ms_level=1,
            scan_rt=self.spectrum_manager.selected_rt,
        )

        # This ought to be user-configured but whatever
        base_mz = ms1_spec_arr["mz"][
            np.argmax(ms1_spec_arr["intsy"])
        ]

        ms2_spec_arr = self.ensemble.get_spectrum(
            ms_level=2,
            scan_rt=self.spectrum_manager.selected_rt,
        )

        sirius_format = to_sirius_ms(
            compound=self.ensemble.format_string,
            parent_mz=base_mz,
            ms1_spec_arr=ms1_spec_arr,
            ms2_spec_arr=ms2_spec_arr,
        )

        mgf_format = f"{
        to_mgf(
            pepmass=base_mz,
            charge=1,  # TODO: make user configurable
            mslevel=1,
            spec_arr=ms1_spec_arr,
            metadata={
                'TITLE': self.ensemble.format_string
            },
        )}\n{
        to_mgf(
            pepmass=base_mz,
            charge=1,  # TODO: make user configurable
            mslevel=2,
            spec_arr=ms2_spec_arr,
            metadata={
                'TITLE': self.ensemble.format_string
            },
        )}\n"

        # Save as .mgf and .ms
        ms_files = [
            (sirius_format, f"{self.ensemble.format_string}.ms"),
            (mgf_format, f"{self.ensemble.format_string}.mgf"),
        ]
        for contents, filename in ms_files:
            with open(Path(directory) / filename, 'w') as f:
                f.write(contents)

            print(
                f"Exported {Path(directory) / filename}"
            )

    def export_chromatogram(
        self,
        directory: Path,
    ):
        """
        Rough placeholder, need roger's plots asap
        """
        svg_exporter = exporters.SVGExporter(
            self.chromPlotWidget.pi
        )

        filename = Path(directory) / f'{self.ensemble.format_string}_chromatogram.svg'
        svg_exporter.export(
            filename
        )

        print(
            f"Exported {filename}"
        )


    def export_cmpd_metadata(
        self,
        params: dict,
        directory: Path,
    ):
        file_path = directory / "metadata.json"

        with open(file_path, 'w') as f:
            json.dump(params, f, indent=2)


