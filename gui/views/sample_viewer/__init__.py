from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QPointF

from core.interfaces.data_sources import AnalyteTableSource
from gui.views.sample_viewer.ensemble_extraction import EnsembleExtractionManager
from gui.views.sample_viewer.menus import FingerprintDisplayMenu, EnsembleExtractionSettingsMenu
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
        Injection, Analyte, AnalyteID,
        FeaturePointer,
        ScanArray,
    )
    from core.interfaces.data_sources import (
        SampleDataSource, AnalyteTableSource,
        AnalyteSource,
    )

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

    def __init__(
        self,
        data_source: Union[
            'SampleDataSource', 'AnalyteTableSource', 'AnalyteSource'
        ]
    ):
        super().__init__()
        self.setupUi(self)
        self.data_source = data_source

        # Model for keeping track of which samples are loaded in viewer
        self.model = SampleViewerItemModel(
            sample_data_source=data_source,
        )

        # Tracks/broadcasts what entities are selected
        self.selection_mgr = SelectionManager(
            data_source=self.data_source
        )

        self.fprint_display_params_menu = FingerprintDisplayMenu(self)
        self.ensemble_extraction_settings_menu = EnsembleExtractionSettingsMenu(self)

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

        # Ensemble Extraction
        self.ensemble_extraction_mgr = EnsembleExtractionManager(
            data_source=self.data_source
        )

        self._setup_actions()
        self._add_status_bar()
        self._setup_views()
        self._setup_tool_signals()
        self._setup_selection_signals()
        self._setup_ensemble_extraction_signals()
        self._setup_manual_xic_signals()


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

    def _setup_selection_signals(self):
        self.selection_mgr.sigSpectrumSelected.connect(
            self.update_spectrum_plot
        )

        self.selection_mgr.sigSpectrumSelected.connect(
            self.viewSampleStack.on_spectrum_selected
        )

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

        self.toolGetCmpdMenu.clicked.connect(
            lambda: self.ensemble_extraction_mgr.showExtractionMenu(
                pos=self.toolGetCmpd.pos(),
                height=self.toolGetCmpd.height(),
            )
        )

    def _setup_manual_xic_signals(self):
        self.spinExtractTarget.valueChanged.connect(
            self.on_manual_extraction_region_entry
        )

        self.spinExtractWindow.valueChanged.connect(
            self.on_manual_extraction_region_entry
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

    def update_spectrum_plot(self):
        """
        Updates MS Plot to match whatever is in Spectrum Selection Manager
        """
        spec_array = self.selection_mgr.get_selected_spectrum_array()
        if spec_array is None:
            return

        self.plotMS.setSpectrumArray(
            spec_array
        )

    # ***ADDING/REMOVING SAMPLES***
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

    def remove_samples(
        self,
        uuids: list['SampleUUID'],
    ):
        for uuid in uuids:
            self.model.removeSample(uuid)

        self.viewSampleStack.rebuild_plots()

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

    # ***ANALYTE SELECTION***
    def on_analyte_selection(
        self,
        selection: dict,
    ):
        # TODO: PLACEHOLDER
        print(
            "TODO: TESTING ANALYTE SELECTION DRAWING"
        )
        for table_uuid, analyte_ids in selection.items():
            analyte_ids: list['AnalyteID']

            analyte_table: 'AnalyteSource' = self.data_source.get_analyte_table(
                table_uuid
            )
            analyte: 'Analyte' = analyte_table.get_analyte(
                analyte_id=analyte_ids[0]
            )

            self.viewSampleStack.set_extraction_range(
                (( analyte.mz - 1 ),
                 (analyte.mz + 1),)
            )

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

        # Show menu, get selected action
        action = context_menu.exec_(
            self.viewSampleTree.mapToGlobal(position)
        )

        if action == remove_action:
            self.remove_selected_samples()
        elif action == toggle_vis_action:
            self.toggle_selected_sample_visibility()

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
