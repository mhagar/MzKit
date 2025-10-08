from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QPointF
import numpy as np

from core.interfaces.data_sources import AnalyteTableSource
from core.cli.generate_ensemble import InputParams as EnsembleInputParams
from core.utils.array_types import SpectrumArray
from gui.views.sample_viewer.menus import FingerprintDisplayMenu, EnsembleExtractionSettingsMenu
from gui.views.sample_viewer.model import SampleViewerItemModel
from gui.views.sample_viewer.tools import (
    ToolType, ExtractionMode,
    ToolStage, ToolManagerNew,
    XICMode,
)
from gui.resources.SampleViewerWindow import Ui_Form

from typing import TYPE_CHECKING, Optional, Union

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
    sigEnsembleGenerationRequested = QtCore.pyqtSignal(
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

        self.fprint_display_params_menu = FingerprintDisplayMenu(self)
        self.ensemble_extraction_settings_menu = EnsembleExtractionSettingsMenu(self)

        # Tool-state tracking
        self.new_tool_manager = ToolManagerNew()
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

        # MS Plot state
        # TODO: Maybe refactor into tool manager?
        self.selected_spectrum_sample_uuid: Optional['SampleUUID'] = None
        self.selected_ms_level: int = 1
        self.selected_scan_num: Optional[int] = None
        self.selected_ms_lane_idx: Optional[int] = None

        # Ensemble Extraction
        # TODO: Refactor into separate manager?
        self.toolGetCmpdMenu.clicked.connect(
            self.show_ensemble_extraction_settings_menu
        )
        self.ensemble_extraction_params: dict = self.ensemble_extraction_settings_menu.get_params()

        self._setup_actions()
        self._add_status_bar()
        self._setup_views()
        self._setup_tool_signals()


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
        # self.viewSampleStack.sigSpectrumSelected.connect(
        #     self.on_spectrum_selected
        # )
        self.sigMSLevelChanged.connect(
            self.viewSampleStack.on_ms_level_changed
        )
        # self.tool_manager.tool_changed.connect(
        #     self.viewSampleStack.on_tool_changed
        # )
        # self.tool_manager.extraction_mode_changed.connect(
        #     self.viewSampleStack.on_extraction_mode_changed
        # )


        # Configure the MS plot
        # self.tool_manager.extraction_mode_changed.connect(
        #     self.plotMS.on_extraction_mode_changed
        # )
        # self.tool_manager.tool_changed.connect(
        #     self.plotMS.on_tool_changed
        # )
        self.plotMS.sigMSSignalHovered.connect(
            self.on_ms_signal_hovered
        )
        self.plotMS.sigMSpectrumLeaved.connect(
            self.on_ms_signal_leaved
        )
        self.plotMS.sigMSSignalClicked.connect(
            self.on_ms_signal_selected
        )

        # Configure fingerprint display menu
        self.fprint_display_params_menu.colorbar.sigLevelsChanged.connect(
            self.viewSampleStack.link_colorbar_to_fprint_plots
        )

        # Configure ensemble extraction settings menu
        self.ensemble_extraction_settings_menu.sigSettingsChanged.connect(
            lambda x: self.ensemble_extraction_params.update(x)
        )

    def _setup_tool_signals(self):
        """
        Connect the tool manager to whatever objects need to respond
        to tool activations
        """
        self.new_tool_manager.register_tool_listener(
            listener=self.viewSampleStack
        )

        self.new_tool_manager.register_tool_listener(
            listener=self.plotMS
        )

        # TODO: Change register_tool_listener to use ducktyping or somth
        self.new_tool_manager.tool_changed.connect(
            self.on_tool_type_changed
        )

        self.new_tool_manager.stage_changed.connect(
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

        self.new_tool_manager.activate_tool(tool_type)

    def _fix_UI_based_on_current_tool(
        self,
    ):
        """
        Automatically "fixes" inappropriate UI elements based on the tool

        For example, GETCMPD can only be used in MS1, so switches to MS1
        """
        match self.new_tool_manager.active_tool:
            case ToolType.NONE:
                self.status_bar.showMessage("")

            case ToolType.GETSPECTRUM:
                self.status_bar.showMessage(
                    "Spectrum Selection Mode. Select a point in chromatogram"
                )

            case ToolType.GETCOMPOUND:
                # Switch to MS1
                if self.selected_ms_level != 1:
                    self.comboMSLevel.setCurrentIndex(0)

                self.status_bar.showMessage(
                    "Ensemble Extraction Mode. Select a reference MS signal"
                )

            case ToolType.GETXIC:
                self.status_bar.showMessage(
                    "Chromatogram Extraction Mode. Select a reference MS signal"
                )

        # Reset any errant states
        self.clear_ensemble_selection_ui()

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

        self.new_tool_manager.set_xic_mode(
            mode
        )

        if mode != XICMode.NONE:
            self.new_tool_manager.cancel()
            self.new_tool_manager.activate_tool(
                ToolType.GETXIC
            )

    def on_tool_type_changed(
        self,
        tool: ToolType,
    ):
        self._fix_UI_based_on_current_tool()

    def on_tool_stage_changed(
        self,
        stage: ToolStage,
    ):
        if stage != ToolStage.CONFIGURING:
            self.selected_ms_lane_idx = None

    def on_ms_level_change_requested(
        self,
        idx: int,
    ):
        """
        Triggered when user changes the MS level

        Sets the state variables accordingly, and converts the
        selected scan number to whatever has the nearest retention time
        to whatever is currently selected
        """
        ms_level = {
            0: 1,
            1: 2,
        }.get(idx)

        if not self.selected_spectrum_sample_uuid:
            # User has not selected a sample yet. Do nothing
            self.selected_ms_level = ms_level
            return

        # Get rt before changing to new scan array
        rt_before_ms_level_change: float = self._get_selected_rt()

        self.selected_ms_level = ms_level
        self.selected_scan_num = self._get_selected_scan_array().rt_to_scan_num(
            rt_before_ms_level_change
        )

        self.update_spectrum_plot()
        self.sigMSLevelChanged.emit(ms_level)

    def _get_selected_scan_array(
        self,
    ) -> 'ScanArray':
        return self.model.getSample(
            self.selected_spectrum_sample_uuid
        ).injection.get_scan_array(
            self.selected_ms_level
        )

    def _get_selected_injection(self) -> 'Injection':
        sample: 'Sample' = self.model.getSample(
            self.selected_spectrum_sample_uuid
        )
        injection = sample.injection
        return injection

    def _get_selected_rt(
        self,
    ) -> float:
        """
        Returns the rt of the currently selected spectrum
        """
        idx = self.selected_scan_num
        scan_array = self._get_selected_scan_array()

        return scan_array.rt_arr[idx]

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
    def set_selected_spectrum(
        self,
        uuid: Optional['SampleUUID'] = None,
        rt: Optional[float] = None,
        idx: Optional[int] = None,
    ):
        """
        Sets the viewer's currently selected spectrum.

        Either rt or scan idx can be provided. If rt is given,
        will calculate the scan idx based on the currently selected ms_level

        :param uuid:
        :param rt:
        :param idx:
        :return:
        """
        self.selected_spectrum_sample_uuid = uuid

        if not self.selected_ms_level:
            raise ValueError(
                "No ms_level selected"
            )

        if not rt and not idx:
            raise ValueError(
                "Neither rt nor idx specified"
            )

        if idx:
            self.selected_scan_num = idx

        else:  # rt given
            # Calculate scan idx based on selected ms_level
            scan_array = self._get_selected_scan_array()

            self.selected_scan_num = scan_array.rt_to_scan_num(
                rt
            )


        self.viewSampleStack.set_selected_scan(
            selected_uuid=self.selected_spectrum_sample_uuid,
            scan_num=self.selected_scan_num,
            ms_level=self.selected_ms_level,
        )
        # self.new_tool_manager.set_selection()

    def update_spectrum_plot(
        self,
        uuid: Optional['SampleUUID'] = None,
        ms_level: Optional[int] = None,
        idx: Optional[int] = None,
    ) -> None:
        """
        Given an Sample UUID, ms_level, and either rt or idx,
        updates the spectrum plot to display the specified MS spectrum

        If not given any arguments, retrieves them from the object's state

        :param uuid:
        :param ms_level:
        :param idx:
        :return:
        """
        if not idx:
            idx = self.selected_scan_num

        if not ms_level:
            ms_level = self.selected_ms_level

        if not uuid:
            uuid = self.selected_spectrum_sample_uuid

        injection = self.model.getSample(uuid).injection
        scan_array = injection.get_scan_array(ms_level)

        spectrum_array: SpectrumArray = scan_array.get_spectrum(
            scan_num=idx,
        )

        self.plotMS.setSpectrumArray(
            spectrum_array
        )

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
        match self.new_tool_manager.active_tool:
            case ToolType.GETSPECTRUM:
                self.set_selected_spectrum(
                    uuid=uuid,
                    rt=pos.x()
                )

                self.update_spectrum_plot()

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
        match self.new_tool_manager.xic_mode:
            case XICMode.NONE:
                pass

            case XICMode.BPC:
                self.viewSampleStack.set_extraction_range(region)

            case XICMode.XIC:
                self.viewSampleStack.set_extraction_range(region)

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
        match self.new_tool_manager.active_tool:
            case ToolType.NONE:
                return

            case ToolType.GETCOMPOUND:
                # Draw preview trace
                hovered_mass_lane_idx: int = signal[0]

                scan_array = self._get_selected_scan_array()

                hovered_ftr_ptr: 'FeaturePointer' = scan_array.make_feature_pointer(
                    mass_lane_idx=hovered_mass_lane_idx,
                    scan_idxs=None,  # Whole lane
                )

                self.viewSampleStack.update_chrom_highlights(
                    [
                        (self.selected_spectrum_sample_uuid,
                         hovered_ftr_ptr)
                    ]
                )

    def on_ms_signal_leaved(self):
        """
        Called when user stops hovering on a signal
        """
        self.viewSampleStack.clear_chrom_highlights(
            uuid=self.selected_spectrum_sample_uuid
        )

    def on_ms_signal_selected(
        self,
        signal: tuple[int, float],
    ):
        match self.new_tool_manager.active_tool:
            case ToolType.NONE:
                return
            
            case ToolType.GETCOMPOUND:
                selected_mass_lane_idx: int = signal[0]
                mz: float = signal[1]

                self.new_tool_manager.set_selection()
                self.selected_ms_lane_idx = selected_mass_lane_idx

                self.show_ensemble_selection_ui(
                    selected_mass_lane_idx,
                )

    def show_ensemble_selection_ui(
        self,
        mass_lane_idx: int,
    ):
        """
        Triggers showing 'ensemble selection' tools in the plot stack.

        :param mass_lane_idx: idx of mass lane to use as reference
        :return:
        """
        self.viewSampleStack.clear_scan_window_selector()

        scan_array = self._get_selected_scan_array()

        ftr_ptr: 'FeaturePointer' = scan_array.make_feature_pointer(
            mass_lane_idx=mass_lane_idx,
            scan_idxs=None,  # Whole lane
        )

        # Get default rt edges to place the window initially
        apex_idx: int = ftr_ptr.get_max_intsy_scan_num(scan_array)
        rt_start = scan_array.rt_arr[apex_idx - 5]
        rt_end = scan_array.rt_arr[apex_idx + 5]

        # Place window
        self.viewSampleStack.show_scan_window_selector(
            uuid=self.selected_spectrum_sample_uuid,
            bounds=(rt_start, rt_end),
            display_arr=ftr_ptr.get_chrom_array(scan_array),
        )

    def clear_ensemble_selection_ui(
        self,
    ):
        """
        Clears the 'ensemble selection' ui
        """
        self.viewSampleStack.clear_scan_window_selector()

    def keyPressEvent(self, event):
        """
        Handle key press events:
        - Escape to exit tool modes
        - Enter to accept tool config
        """
        if event.key() == Qt.Key_Escape:

            if self.new_tool_manager.active_tool != ToolType.NONE:
                # Cancel current tool and return to view mode
                self.actionView.trigger()
                event.accept()
                return

        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.on_press_enter()
            event.accept()
            return
        
        # Pass event for normal handling
        super().keyPressEvent(event)

    def on_press_enter(
        self,
    ):
        """
        Only works during ToolStage.CONFIGURING
        """
        if not self.new_tool_manager.stage == ToolStage.CONFIGURING:
            return

        match self.new_tool_manager.active_tool:

            case ToolType.GETCOMPOUND:
                rt_bounds = self.viewSampleStack.get_selected_scan_window(
                    uuid=self.selected_spectrum_sample_uuid
                )

                if rt_bounds == (0, 0):
                    # No scan window selector for some reason
                    return

                if not self.selected_ms_lane_idx:
                    # No mass lane has been selected
                    return

                # Convert rt_bounds to scan number
                scan_array = self._get_selected_scan_array()
                scan_window: tuple[int, int] = tuple( # type: ignore
                    scan_array.rt_to_scan_num(x) for x in rt_bounds
                )

                self.request_ensemble_generation(
                    injection=self._get_selected_injection(),
                    mass_lane_idx=self.selected_ms_lane_idx,
                    scan_window=scan_window,
                    **self.ensemble_extraction_params,
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



    def request_ensemble_generation(
        self,
        injection: 'Injection',
        mass_lane_idx: int,
        scan_window: tuple[int, int],
        ms1_corr_threshold: float,
        ms2_corr_threshold: float,
        min_intsy: float,
        use_rel_intsy: bool,
    ):
        if self.selected_ms_level != 1:
            raise NotImplementedError(
                "Can only generate Ensembles using reference features"
                "from MS1 scans"
            )

        scan_array = injection.get_scan_array(ms_level=1)
        search_ftr_ptr = scan_array.make_feature_pointer(
            mass_lane_idx=mass_lane_idx,
            scan_idxs=np.arange(scan_window[0], scan_window[1] + 1),
        )

        input_params: 'EnsembleInputParams' = EnsembleInputParams(
            search_ftr_ptr=search_ftr_ptr,
            injection=injection,
            ms1_corr_threshold=ms1_corr_threshold,
            ms2_corr_threshold=ms2_corr_threshold,
            min_intsy=min_intsy,
            use_rel_intsy=use_rel_intsy,
        )

        print(
            f"Selected scan number:"
            f"{self.selected_scan_num}"
        )

        self.sigEnsembleGenerationRequested.emit(
            input_params
        )

    # *** ENSEMBLE EXTRACTION ***
    # TODO: Refactor into separate class?
    def show_ensemble_extraction_settings_menu(self):
        button_pos = self.toolGetCmpdMenu.mapToGlobal(
            QtCore.QPoint(0, 0)
        )

        self.ensemble_extraction_settings_menu.move(
            button_pos.x() - self.ensemble_extraction_settings_menu.size().width(),
            button_pos.y() + self.toolGetCmpdMenu.height(),
            )

        self.ensemble_extraction_settings_menu.show()



