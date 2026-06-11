from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QPointF
import pyqtgraph as pg

from core.utils.config import load_config
from gui.views.sample_viewer.dda_overlays import DDAOverlayManager
from gui.views.sample_viewer.ensemble_extraction import EnsembleExtractionManager
from gui.views.sample_viewer.menus import FingerprintDisplayMenu
from gui.views.sample_viewer.model import SampleViewerItemModel
from gui.views.sample_viewer.spectrum_selection import SelectionManager
from gui.views.sample_viewer.tools import (
    ToolType, ExtractionMode,
    ToolStage, ToolManagerNew,
    XICMode,
)
from gui.resources.SampleViewerWindow import Ui_Form

from typing import TYPE_CHECKING, Optional, Union, Literal

if TYPE_CHECKING:
    from core.data_structs import (
        Sample, SampleUUID,
        Injection,
        FeaturePointer,
        ScanArray,
        EnsembleUUID, Ensemble,
    )
    from core.interfaces.data_sources import SampleDataSource
    from configparser import ConfigParser

class SampleViewer(
    QtWidgets.QWidget,
    Ui_Form
):
    """
    Window for viewing injection/fingerprint data
    """

    sigMSLevelChanged = QtCore.pyqtSignal(int)
    sigEnsembleExtractionRequested = QtCore.pyqtSignal(
        object,
    )
    sigAutoEnsembleRequested = QtCore.pyqtSignal(
        object,  # SampleUUID
    )
    sigAlignEnsemblesRequested = QtCore.pyqtSignal(
        object,  # list[SampleUUID]
    )
    sigViewEnsembleRequested = QtCore.pyqtSignal(
        object,  # Ensemble
    )

    def __init__(
        self,
        data_source: 'SampleDataSource',
    ):
        super().__init__()
        self.setupUi(self)
        self.config = load_config()
        self.data_source = data_source

        # Subscribe to samples being *updated* in the data registry
        self.data_source.subscribe_to_changes(
            addition_callback=lambda x: None,  # Don't react to sample addition/removal
            removal_callback=lambda x: None,
            update_callback=self.update_sample,
            change_type='Sample',
        )

        # Model for keeping track of which samples are loaded in viewer
        self.model = SampleViewerItemModel(
            sample_data_source=data_source,
        )

        # Tracks/broadcasts what entities are selected
        self.selection_mgr = SelectionManager(
            data_source=self.data_source
        )

        self.fprint_display_params_menu = FingerprintDisplayMenu(self)
        # self.ensemble_extraction_settings_menu = EnsembleExtractionSettingsMenu(self)

        # Tool-state tracking
        self.tool_mgr = ToolManagerNew()
        self.tool_action_group = QtWidgets.QActionGroup(self)
        self.tool_action_group.triggered.connect(
            self.on_tool_action_triggered
        )
        self.extraction_action_group = QtWidgets.QActionGroup(self)
        self.extraction_action_group.triggered.connect(
            self.on_extraction_mode_triggered
        )

        # TreeView actions
        self.viewSampleTree.customContextMenuRequested.connect(
            self.show_treeview_context_menu
        )

        # Fingerprint viewing
        self.actionToggleFprint.toggled.connect(
            self.viewSampleStack.on_fprints_toggled
        )
        self.toolFprintMenu.clicked.connect(
            self.show_fprint_display_params_menu
        )

        # Chromatogram extraction
        self.plotMS.sigExtractionRegionChanged.connect(
            self.on_extraction_region_changed
        )

        # View/hide ensembles
        self.checkShowEnsembles.toggled.connect(
            self.viewSampleStack.on_show_ensembles_toggled
        )

        # View/hide ensemble preview
        self.checkShowEnsemblePreview.toggled.connect(
            self.toggle_ensemble_preview
        )

        # Ensemble Extraction
        self.ensemble_extraction_mgr = EnsembleExtractionManager(
            data_source=self.data_source
        )

        # DDA overlay layer for the MS spectrum plot
        self.dda_overlay_mgr = DDAOverlayManager(
            plot=self.plotMS,
            selection_mgr=self.selection_mgr,
            data_source=self.data_source,
            tool_mgr=self.tool_mgr,
        )

        # Ensemble peak interaction state
        self._selected_ensemble: Optional[tuple['SampleUUID', 'EnsembleUUID']] = None

        self._setup_actions()
        self._add_status_bar()
        self._setup_views()
        self._setup_tool_signals()
        self._setup_selection_signals()
        self._setup_ensemble_extraction_signals()
        self._setup_manual_xic_signals()
        self._setup_ensemble_interaction_signals()


    # ***CONFIGURATION/QT SIGNALS***
    def _add_status_bar(self):
        """
        Can't do this in Qt Designer unfortunately
        """
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.setMaximumHeight(15) # pixels

        self.verticalLayout_5.addWidget(
            self.status_bar
        )

    def _setup_views(self):
        # Configure Tree sidebar
        # Define dragging behaviour
        self.viewSampleTree.setDragEnabled(
            True
        )
        self.viewSampleTree.setDragDropMode(
            self.viewSampleTree.InternalMove  # Reorder within same view
        )
        self.viewSampleTree.setDefaultDropAction(
            Qt.MoveAction
        )
        self.viewSampleTree.setModel(
            self.model,
        )

        # Configure chrom plot stack view
        self.viewSampleStack.setModel(
            self.model,
        )
        self.viewSampleStack.sigChromatogramHovered.connect(
            self.on_chromatogram_hovered
        )

        self.sigMSLevelChanged.connect(
            self.viewSampleStack.chrom_mgr.set_ms_level
        )

        self.plotMS.sigMSSignalHovered.connect(
            self.on_ms_signal_hovered
        )
        self.plotMS.sigMSpectrumLeaved.connect(
            self.on_ms_signal_leaved
        )
        self.plotMS.sigMSSignalClicked.connect(
            self.on_ms_signal_clicked
        )

        # Configure fingerprint display menu
        self.fprint_display_params_menu.colorbar.sigLevelsChanged.connect(
            self.viewSampleStack.link_colorbar_to_fprint_plots
        )

        # Configure ensemble MS preview
        self.plotEnsembleMS.hide()
        self.plotEnsembleMS.pi.getAxis('bottom').hide()
        self.plotEnsembleMS.pi.vb.setXLink(
            self.plotMS.pi.vb
        )

    def _setup_selection_signals(self):
        self.selection_mgr.sigSpectrumSelected.connect(
            self.update_spectrum_plot
        )

        self.selection_mgr.sigSpectrumSelected.connect(
            self.viewSampleStack.on_spectrum_selected
        )

        # Keep the MS-level combobox in sync when the selection switches
        # MS level programmatically (e.g. clicking a precursor badge).
        self.selection_mgr.sigMSLevelSelected.connect(
            self._sync_ms_level_combo
        )

    def _sync_ms_level_combo(self, ms_level: int) -> None:
        # combo indices: 0 = MS1, 1 = MS2
        idx = 0 if ms_level == 1 else 1
        if self.comboMSLevel.currentIndex() == idx:
            return
        self.comboMSLevel.blockSignals(True)
        try:
            self.comboMSLevel.setCurrentIndex(idx)
        finally:
            self.comboMSLevel.blockSignals(False)
        # Mirror what on_ms_level_change_requested would have done, so
        # downstream listeners (chrom_mgr etc.) also see the change.
        self.sigMSLevelChanged.emit(ms_level)

    def _setup_tool_signals(self):
        """
        Connect the tool manager to whatever objects need to respond
        to tool activations
        """
        self.tool_mgr.register_tool_listener(
            listener=self.viewSampleStack
        )

        self.tool_mgr.register_tool_listener(
            listener=self.plotMS
        )

        # TODO: Change register_tool_listener to use ducktyping or somth
        self.tool_mgr.tool_changed.connect(
            self.on_tool_type_changed
        )

        self.tool_mgr.stage_changed.connect(
            self.on_tool_stage_changed
        )

    def _setup_actions(self):
        # Action group for mutually exclusive tools
        self.tool_action_group.setExclusive(True)
        for action, btn in [
            (self.actionView, self.toolView),
            (self.actionGetSpectrum, self.toolGetSpectrum),
            (self.actionGetCompound, self.toolGetCmpd),
        ]:
            action: QtWidgets.QAction
            btn: QtWidgets.QToolButton

            self.tool_action_group.addAction(
                action
            )

            btn.setDefaultAction(action)

        self.actionView.setChecked(True)
        # self.new_tool_manager.sigReset.connect(
        #     lambda: self.actionView.setChecked(True)
        # )

        # Action group for mutually exclusive chrom extraction modes
        self.extraction_action_group.setExclusive(True)
        for action, btn in [
            (self.actionModeNone, self.toolExtNone),
            (self.actionModeXIC, self.toolExtXIC),
            (self.actionModeBPC, self.toolExtBPC),
        ]:
            action: QtWidgets.QAction
            btn: QtWidgets.QToolButton
            self.extraction_action_group.addAction(
                action
            )

            btn.setDefaultAction(action)

        self.actionModeNone.setChecked(True)

        # Fingerprint visibility toggling:
        self.toolFprint.setDefaultAction(
            self.actionToggleFprint
        )

    def _setup_ensemble_extraction_signals(self):
        # Route signal upwards (to main ctrlr)
        self.ensemble_extraction_mgr.sigEnsembleExtractionRequested.connect(
            self.sigEnsembleExtractionRequested.emit
        )

        get_cmpd_btn_pos = lambda: (
            self.mapToGlobal(self.toolGetCmpd.pos()),
            self.toolGetCmpd.height()
        )

        self.toolGetCmpdMenu.clicked.connect(
            lambda: self.ensemble_extraction_mgr.showExtractionMenu(
                *get_cmpd_btn_pos()
            )
        )

    def _setup_manual_xic_signals(self):
        self.spinExtractTarget.valueChanged.connect(
            self.on_manual_extraction_region_entry
        )

        self.spinExtractWindow.valueChanged.connect(
            self.on_manual_extraction_region_entry
        )

    def _setup_ensemble_interaction_signals(self):
        """Connect ensemble peak interaction signals"""
        self.viewSampleStack.sigEnsemblePeakHovered.connect(
            self.on_ensemble_peak_hovered
        )
        self.viewSampleStack.sigEnsemblePeakClicked.connect(
            self.on_ensemble_peak_clicked
        )

    # ***STATE TOGGLING***
    def on_tool_action_triggered(
        self,
        action: QtWidgets.QAction,
    ):
        """
        Called whenever the user switches the active tool
         (i.e. spectrum selector, compound selector, etc)
        :return:
        """
        tool_map = {
            self.actionView: ToolType.NONE,
            self.actionGetSpectrum: ToolType.GETSPECTRUM,
            self.actionGetCompound: ToolType.GETCOMPOUND,
        }

        tool_type = tool_map.get(
            action,
            ToolType.NONE, # Default if invalid tool
        )

        self.tool_mgr.activate_tool(tool_type)

    def _fix_ui_based_on_current_tool(
        self,
    ):
        """
        Automatically "fixes" inappropriate UI elements based on the tool

        For example, GETCMPD can only be used in MS1, so switches to MS1
        """
        match self.tool_mgr.active_tool:
            case ToolType.NONE:
                self.status_bar.showMessage("")

            case ToolType.GETSPECTRUM:
                self.status_bar.showMessage(
                    "Spectrum Selection Mode. Select a point in chromatogram"
                )

            case ToolType.GETCOMPOUND:
                # Switch to MS1
                if self.selection_mgr.selected_ms_level != 1:
                    self.comboMSLevel.setCurrentIndex(0)

                self.status_bar.showMessage(
                    "Ensemble Extraction Mode. Select a reference MS signal"
                )

            case ToolType.GETXIC:
                self.status_bar.showMessage(
                    "Chromatogram Extraction Mode. Select a reference MS signal"
                )

        # Reset any errant states
        self.viewSampleStack.ensemble_ui_mgr.clear_scan_window_selector()

    def on_extraction_mode_triggered(
        self,
        action: QtWidgets.QAction,
    ):
        """
        Called when user selects either 'NONE', 'XIC', or 'BPC'
        """
        mode_map = {
            self.actionModeNone: XICMode.NONE,
            self.actionModeXIC: XICMode.XIC,
            self.actionModeBPC: XICMode.BPC,
        }

        mode = mode_map.get(
            action,
            XICMode.NONE,  # Default to NONE if unrecognized action
        )

        self.tool_mgr.set_xic_mode(
            mode
        )

        if mode != XICMode.NONE:
            self.tool_mgr.cancel()
            self.tool_mgr.activate_tool(
                ToolType.GETXIC
            )

    def on_tool_type_changed(
        self,
        tool: ToolType,
    ):
        self._fix_ui_based_on_current_tool()

    def on_tool_stage_changed(
        self,
        stage: ToolStage,
    ):
        if stage != ToolStage.CONFIGURING:
            self.selection_mgr.clear_selected_ms_lane_idx()

    def on_ms_level_change_requested(
        self,
        idx: int,
    ):
        """
        Triggered when user changes the MS level combo box
        """
        # Convert combo values {0, 1} to ms_levels {1, 2}
        ms_level: Literal[1, 2] = { # type: ignore
            0: 1,
            1: 2,
        }.get(idx, 1)

        self.selection_mgr.set_ms_level(ms_level)
        self.sigMSLevelChanged.emit(ms_level)

    def on_samples_per_window_requested(
        self,
        num: int,
    ):
        """
        Triggered when user changes the 'Samples per window' spinner
        """
        self.viewSampleStack.sample_wdgt_mgr.set_samples_per_window(num)

    def update_spectrum_plot(self):
        """
        Updates MS Plot to match whatever is in Spectrum Selection Manager.

        Also updates the Ensemble MS preview plot, and the DDA overlay
        layer (precursor badges / isolation window).
        """
        spec_array = self.selection_mgr.get_selected_spectrum_array()
        if spec_array is None:
            self.dda_overlay_mgr.clear()
            return

        self.plotMS.setSpectrumArray(
            spec_array
        )

        sample_uuid = self.selection_mgr.selected_sample_uuid
        scan_idx = self.selection_mgr.get_selected_scan_num()
        if sample_uuid is not None and scan_idx is not None:
            self.dda_overlay_mgr.update(
                sample_uuid=sample_uuid,
                ms_level=self.selection_mgr.selected_ms_level,
                scan_index=scan_idx,
            )

    # ***ADDING/REMOVING/UPDATING SAMPLES***
    def add_samples(
        self,
        uuids: list['SampleUUID'],
        visible: bool = True,
    ):
        """
        Add Samples to be displayed
        """
        for uuid in uuids:
            self.model.addSample(
                uuid,
                visible=visible
            )

        self.spinSamplesPerWindow.setMaximum(
            self.model.rowCount()
        )

    def remove_samples(
        self,
        uuids: list['SampleUUID'],
    ):
        for uuid in uuids:
            self.model.removeSample(uuid)

        self.spinSamplesPerWindow.setMaximum(
            self.model.rowCount()
        )

        self.viewSampleStack.rebuild_plots()

    def reset_for_new_project(self):
        """
        Remove all loaded samples (and their plots) when the workspace
        is cleared, so no stale content from a previous project remains.
        """
        loaded_uuids = [
            self.model.item(row).data(self.model.UuidRole)
            for row in range(self.model.rowCount())
        ]
        if loaded_uuids:
            self.remove_samples(loaded_uuids)

    def update_sample(
        self,
        sample: 'Sample',
    ):
        """
        Called whenever a Sample is notified as being updated
        """
        self.viewSampleStack.refresh_plot(
            sample.uuid
        )

    # ***CHROMATOGRAM SCANNING***
    def on_chromatogram_hovered(
        self,
        uuid: 'SampleUUID',
        pos: QPointF,
    ):
        """
        Called when user hovers over a chromatogram
        :param uuid:
        :param pos:
        :return:
        """
        match self.tool_mgr.active_tool:
            case ToolType.GETSPECTRUM:
                self.selection_mgr.set_selected_spectrum_by_rt(
                    uuid=uuid,
                    ms_level=self.selection_mgr.selected_ms_level,
                    rt=pos.x(),
                )

            case ToolType.NONE:
                # TODO: Display some useful info maybe?
                return

    # ***ENSEMBLE PEAK INTERACTION***
    def _get_ensemble(
        self,
        sample_uuid: 'SampleUUID',
        ensemble_uuid: 'EnsembleUUID',
    ) -> Optional['Ensemble']:
        injection = self.model.getInjection(sample_uuid)
        if not injection:
            return None

        ensemble = injection.ensembles.get(ensemble_uuid)
        if not ensemble:
            return None

        return ensemble

    def on_ensemble_peak_hovered(
        self,
        sample_uuid: 'SampleUUID',
        ensemble_uuid: 'EnsembleUUID',
        pos: QPointF,
    ):
        """
        Called when user hovers over an ensemble peak.
        Shows tooltip with ensemble info in the status bar.
        """
        ensemble = self._get_ensemble(
            sample_uuid, ensemble_uuid
        )

        if not ensemble:
            return

        # Update status bar with ensemble info
        tooltip_text = (
            f"Ensemble: {ensemble.base_mz:.4f} m/z @ {ensemble.peak_rt:.2f}s | "
            f"{len(ensemble.ms1_cofeatures)} MS1 + {len(ensemble.ms2_cofeatures)} MS2 features"
        )
        self.status_bar.showMessage(tooltip_text)

    def on_ensemble_peak_clicked(
        self,
        sample_uuid: 'SampleUUID',
        ensemble_uuid: 'EnsembleUUID',
        button: int,
    ):
        """
        Handle ensemble peak clicks:
        - Left click: Select ensemble immediately
        - Right click: Context menu
        """
        if button == Qt.RightButton:
            self._show_ensemble_context_menu(sample_uuid, ensemble_uuid)
            return

        if button == Qt.LeftButton:
            self._select_ensemble(sample_uuid, ensemble_uuid)

    def _select_ensemble(
        self,
        sample_uuid: 'SampleUUID',
        ensemble_uuid: 'EnsembleUUID'
    ):
        """
        Select an ensemble (visual feedback only for now)
        """
        # Clear previous selection
        if self._selected_ensemble:
            old_sample_uuid, old_ensemble_uuid = self._selected_ensemble
            old_widget = self.viewSampleStack.sample_wdgt_mgr.get_widget(old_sample_uuid)
            if old_widget:
                old_widget.chromPlotWidget.pi.set_peak_selected(None)

        # Apply new selection
        widget = self.viewSampleStack.sample_wdgt_mgr.get_widget(sample_uuid)
        if widget:
            widget.chromPlotWidget.pi.set_peak_selected(ensemble_uuid)

            # Trigger as if chromatogram was clicked
            widget.on_chromatogram_clicked()

        self._selected_ensemble = (sample_uuid, ensemble_uuid)

        # Update status bar
        ensemble = self._get_ensemble(
            sample_uuid, ensemble_uuid
        )
        sample_name = self.data_source.get_sample(sample_uuid).name

        self.status_bar.showMessage(
            f"Selected: {ensemble.base_mz:.4f} m/z @ {ensemble.peak_rt:.2f}s "
            f"({sample_name})"
        )

        # Update preview MS plot
        self.plotEnsembleMS.setSpectrumPlotPen(
            pg.mkPen(pg.intColor(   # Temporary color gen
                ensemble_uuid,      # TODO: use func in ensemble_ui_manager.py
                hues=12,
                minValue=150,
                maxValue=255,
                sat=128,
            ))
        )
        self.plotEnsembleMS.update_label(
            f"Selected: {ensemble.base_mz:.4f} m/z @ {ensemble.peak_rt:.2f}s "
            f"({sample_name})"
        )
        self.plotEnsembleMS.setSpectrumArray(
            ensemble.get_spectrum(
                ms_level=self.selection_mgr.selected_ms_level,
                scan_rt=ensemble.peak_rt,
            )
        )

    def _show_ensemble_context_menu(
        self,
        sample_uuid: 'SampleUUID',
        ensemble_uuid: 'EnsembleUUID'
    ):
        """Show right-click context menu for ensemble"""
        from PyQt5.QtGui import QCursor
        menu = QtWidgets.QMenu(self)

        action_open = menu.addAction("Open in Ensemble Viewer")
        action_export = menu.addAction("Export...")
        menu.addSeparator()
        action_delete = menu.addAction("Delete Ensemble")

        action = menu.exec_(QCursor.pos())

        if action == action_open:
            self._open_ensemble_in_viewer(sample_uuid, ensemble_uuid)
        elif action == action_delete:
            self._delete_ensemble(sample_uuid, ensemble_uuid)
        # elif action == action_export: ...

    def _open_ensemble_in_viewer(
        self,
        sample_uuid: 'SampleUUID',
        ensemble_uuid: 'EnsembleUUID'
    ):
        """
        Open an ensemble in the shared EnsembleViewer MDI subwindow by
        emitting sigViewEnsembleRequested. The MainController routes this
        to SubWindowManager so we always land in the same singleton
        viewer rather than spawning floating copies.
        """
        injection = self.model.getInjection(sample_uuid)
        if not injection:
            return

        ensemble = injection.ensembles.get(ensemble_uuid)
        if not ensemble:
            return

        self.sigViewEnsembleRequested.emit(ensemble)

    def _delete_ensemble(
        self,
        sample_uuid: 'SampleUUID',
        ensemble_uuid: 'EnsembleUUID'
    ):
        """Delete ensemble from injection"""
        # Confirm with user
        reply = QtWidgets.QMessageBox.question(
            self,
            'Delete Ensemble',
            'Are you sure you want to delete this ensemble?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            injection = self.model.getInjection(sample_uuid)
            injection.remove_ensemble(ensemble_uuid)

            # Remove from UI
            widget = self.viewSampleStack.sample_wdgt_mgr.get_widget(sample_uuid)
            if widget:
                widget.removePeak(ensemble_uuid)

            # Clear selection if this was selected
            if self._selected_ensemble == (sample_uuid, ensemble_uuid):
                self._selected_ensemble = None

    def toggle_ensemble_preview(
        self,
        toggle: bool,
    ):
        """
        Called when user clicks the checkbox 'Show Ensemble Preview'
        """
        match toggle:
            case True:
                self.plotEnsembleMS.show()
            case False:
                self.plotEnsembleMS.hide()


    # ***XIC/BPC SLICING***
    def on_extraction_region_changed(
        self,
        region: tuple,
    ):
        """
        Called when user modifies the 'extraction region' in spectrum plot
        :param region:
        :return:
        """
        match self.tool_mgr.xic_mode:
            case XICMode.NONE:
                pass

            case XICMode.BPC:
                self.viewSampleStack.set_extraction_range(region)


            case XICMode.XIC:
                self.viewSampleStack.set_extraction_range(region)

    def on_manual_extraction_region_entry(self):
        """
        Called when user manually changes the spinners
        """
        # Calculate what the display region would be
        region = (
            self.spinExtractTarget.value() - self.spinExtractWindow.value(),
            self.spinExtractTarget.value() + self.spinExtractWindow.value(),
        )
        self.plotMS.move_region_selector(
            region
        )

    # ***FINGERPRINT***
    def show_fprint_display_params_menu(self):
        button_pos = self.toolFprintMenu.mapToGlobal(
            QtCore.QPoint(0, 0)
        )

        self.fprint_display_params_menu.move(
            button_pos.x() - self.fprint_display_params_menu.size().width(),
            button_pos.y() + self.toolFprintMenu.height(),
        )

        self.fprint_display_params_menu.show()

    # ***SELECTING MS SIGNALS***
    def on_ms_signal_hovered(
        self,
        signal: tuple[int, float],
    ):
        match self.tool_mgr.active_tool:
            case ToolType.NONE:
                return

            case ToolType.GETCOMPOUND:
                # Draw preview trace
                hovered_mass_lane_idx: int = signal[0]

                scan_array = self.selection_mgr.get_selected_scan_array()

                self.viewSampleStack.chrom_mgr.update_chrom_highlights(
                    [
                        (
                            self.selection_mgr.selected_sample_uuid,
                            scan_array,
                            hovered_mass_lane_idx,
                        )
                    ]
                )

    def on_ms_signal_leaved(self):
        """
        Called when user stops hovering on a signal
        """
        self.viewSampleStack.chrom_mgr.clear_chrom_highlights(
            uuid=self.selection_mgr.selected_sample_uuid
        )

    def on_ms_signal_clicked(
        self,
        signal: tuple[int, float],
    ):
        match self.tool_mgr.active_tool:
            case ToolType.NONE:
                return
            
            case ToolType.GETCOMPOUND:
                selected_mass_lane_idx: int = signal[0]

                self.tool_mgr.set_selection()
                self.selection_mgr.set_selected_ms_lane_idx(
                    selected_mass_lane_idx
                )

                self.viewSampleStack.ensemble_ui_mgr.show_scan_window_selector(
                    uuid=self.selection_mgr.selected_sample_uuid,
                    ms_level=self.selection_mgr.selected_ms_level,
                    mass_lane_idx=self.selection_mgr.selected_ms_lane_idx,
                    initial_scan_window=5,  # TODO: Don't hardcode
                )

    def keyPressEvent(self, event):
        """
        Handle key press events:
        - Escape to exit tool modes
        - Enter to accept tool config
        """
        if event.key() == Qt.Key_Escape:

            if self.tool_mgr.active_tool != ToolType.NONE:
                # Cancel current tool and return to view mode
                self.actionView.trigger()
                event.accept()
                return

        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.tool_mgr.stage == ToolStage.CONFIGURING:
                self.on_confirm_configuration()

            event.accept()
            return
        
        # Pass event for normal handling
        super().keyPressEvent(event)

    def on_confirm_configuration(
        self,
    ):
        """
        Only works during ToolStage.CONFIGURING
        """
        match self.tool_mgr.active_tool:

            case ToolType.GETCOMPOUND:
                rt_bounds = self.viewSampleStack.ensemble_ui_mgr.get_selected_scan_window(
                    uuid=self.selection_mgr.selected_sample_uuid,
                )

                if self.selection_mgr.selected_ms_level != 1:
                    raise NotImplementedError(
                        "Can only generate Ensembles using reference features"
                        "from MS1 scans"
                    )

                if rt_bounds == (0, 0):
                    # No scan window selector for some reason
                    return

                if not self.selection_mgr.selected_ms_lane_idx:
                    # No mass lane has been selected
                    return

                self.ensemble_extraction_mgr.request_using_current_params(
                    sample_uuid=self.selection_mgr.selected_sample_uuid,
                    mass_lane_idx=self.selection_mgr.selected_ms_lane_idx,
                    rt_bounds=rt_bounds,
                )

                self.actionView.trigger()

    def show_treeview_context_menu(
        self,
        position: 'QtCore.QPoint',
    ):
        # Get item at clicked position
        idx: 'QtCore.QModelIndex' = self.viewSampleTree.indexAt(position)
        if not idx.isValid():
            return  # No item clicked

        context_menu = QtWidgets.QMenu(self)
        remove_action = context_menu.addAction(
            "Remove Selected Samples"
        )
        toggle_vis_action = context_menu.addAction(
            "Toggle Selected Samples Visibility"
        )
        context_menu.addSeparator()
        auto_ensemble_action = context_menu.addAction(
            "Auto-generate Ensembles"
        )
        align_action = context_menu.addAction(
            "Align Ensembles Across Samples"
        )

        # Show menu, get selected action
        action = context_menu.exec_(
            self.viewSampleTree.mapToGlobal(position)
        )

        if action == remove_action:
            self.remove_selected_samples()
        elif action == toggle_vis_action:
            self.toggle_selected_sample_visibility()
        elif action == auto_ensemble_action:
            self._request_auto_ensemble_generation()
        elif action == align_action:
            self._request_ensemble_alignment()

    def remove_selected_samples(self):
        """
        Removes samples that are selected in the tree view
        :return:
        """
        selected_idxs: list['QtCore.QModelIndex'] = self.viewSampleTree.selectedIndexes()
        uuids_to_remove: list['SampleUUID'] = []
        for idx in selected_idxs:
            # Get the sample UUID
            uuid = idx.data(self.model.UuidRole)
            if uuid:
                uuids_to_remove.append(
                    uuid
                )

        self.remove_samples(uuids_to_remove)

    def toggle_selected_sample_visibility(self):
        """
        Toggles visibility of samples that are selected in tree view
        """
        selected_idxs: list['QtCore.QModelIndex'] = self.viewSampleTree.selectedIndexes()
        for idx in selected_idxs:
            item = self.model.itemFromIndex(idx)
            match item.checkState():
                case Qt.CheckState.Checked:
                    item.setCheckState(
                        Qt.CheckState.Unchecked
                    )

                case Qt.CheckState.Unchecked:
                    item.setCheckState(
                        Qt.CheckState.Checked
                    )

    def _request_auto_ensemble_generation(self):
        """
        Emit auto-ensemble signal for each selected sample
        """
        selected_idxs: list['QtCore.QModelIndex'] = self.viewSampleTree.selectedIndexes()
        for idx in selected_idxs:
            uuid = idx.data(self.model.UuidRole)
            if uuid:
                sample = self.data_source.get_sample(uuid)
                if sample and sample.injection:
                    self.sigAutoEnsembleRequested.emit(uuid)

    def _request_ensemble_alignment(self):
        """
        Emit alignment signal with all selected sample UUIDs
        """
        selected_idxs: list['QtCore.QModelIndex'] = self.viewSampleTree.selectedIndexes()
        uuids: list['SampleUUID'] = []
        for idx in selected_idxs:
            uuid = idx.data(self.model.UuidRole)
            if uuid:
                sample = self.data_source.get_sample(uuid)
                if sample and sample.injection:
                    if len(sample.injection.ensembles) > 0:
                        uuids.append(uuid)

        if len(uuids) >= 2:
            self.sigAlignEnsemblesRequested.emit(uuids)
