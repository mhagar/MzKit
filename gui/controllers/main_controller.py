from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog

from gui.controllers.sample_controller import SampleController
from gui.controllers.subwindow_controller import SubWindowManager
from gui.controllers.selection_manager import SelectionManager
from core.controllers.ProcessController import ProcessController
from core.data_structs import DataRegistry

from gui.views.main_view import MainView

from pathlib import Path
import logging
from typing import Optional, Literal, TYPE_CHECKING


if TYPE_CHECKING:
    from PyQt5.QtWidgets import QApplication
    from configparser import ConfigParser
    from core.data_structs import Sample
    from core.data_structs.fingerprint import FingerprintImportParams
    from core.data_structs.scan_array import ScanArrayParameters
    from core.cli.generate_ensemble import EnsembleExtractionParams
    import core.data_structs as data_structs
    from gui.views.sample_viewer import SampleViewer
    from gui.views.process_monitor import ProcessMonitorWindow
    from gui.views.fingerprint_viewer import FingerprintViewerWindow
    from gui.views.ensemble_viewer import EnsembleViewer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


class MainController:
    def __init__(
            self,
            app: 'QApplication',
            config: 'ConfigParser',
    ) -> None:
        self.app = app
        self.config: 'ConfigParser' = config
        self.main_view: MainView = MainView()

        self.data_registry: DataRegistry = DataRegistry()

        # Sub-controllers
        self.process_controller: ProcessController = ProcessController(
            main_controller=self,  # TODO: Decouple this
        )
        self.sample_controller: SampleController = SampleController(
            config=config,
            data_registry=self.data_registry,
        )

        # Selections
        self.selection_manager: SelectionManager = SelectionManager()

        # Sub-windows
        self.subwindow_manager = SubWindowManager(
            mdi_area=self.main_view.mdiArea,
            process_controller=self.process_controller,
            data_registry=self.data_registry,
        )
        self.subwindow_manager.initialize_all_windows()

        # Connect QSignals
        self._connect_view_signals()
        self._connect_sample_controller_signals()
        self._connect_sample_viewer_signals()

        # Initialize controllers (must be done at end)
        self.sample_controller.initialize_sample_model()
        self.sample_controller.initialize_analyte_table_model()
        self.sample_controller.initialize_alignment_model()


    def _connect_view_signals(self) -> None:
        self.main_view.sigImportFingerprintsRequested.connect(
            self._handle_import_fingerprints_request
        )

        self.main_view.sigImportMzMLsRequested.connect(
            self._handle_import_mzmls_request
        )

        self.main_view.sigImportMetadataRequested.connect(
            self._handle_import_metadata_request
        )

        self.main_view.sigShowSampleViewerRequested.connect(
            self._handle_view_samples_request
        )

        self.main_view.sigImportAnalyteTableRequested.connect(
            self._handle_import_analyte_table_request
        )

        self.main_view.sigShowAnalyteTableRequested.connect(
            self._handle_view_analyte_table_request
        )

        self.main_view.actionSaveProject.triggered.connect(
            self._handle_save_project_request
        )

        self.main_view.actionLoadProject.triggered.connect(
            self._handle_load_project_request
        )

        self.main_view.sigImportFeatureTableRequested.connect(
            self._handle_import_feature_table_request
        )

        self.main_view.sigFilterAlignmentRequested.connect(
            self._handle_filter_alignment_request
        )

        self.main_view.sigExportAlignmentRequested.connect(
            self._handle_export_alignment_request
        )

        self.main_view.sigSampleFilterChanged.connect(
            self._handle_sample_filter_changed
        )

        self.main_view.sigLabelingRequested.connect(
            self._handle_labeling_request
        )

        self._connect_window_menu()


    # Mapping of Window-menu actions to SubWindowManager keys
    WINDOW_MENU_ACTIONS = {
        'actionWindowSamples':         'sample_viewer',
        'actionWindowEnsemble':        'ensemble_viewer',
        'actionWindowAlignment':       'alignment_viewer',
        'actionWindowFingerprint':     'fingerprint_viewer',
        'actionWindowProcessMonitor':  'process_monitor',
    }

    def _connect_window_menu(self) -> None:
        """
        Wire the Window menu so each checkable action toggles its
        corresponding subwindow's visibility, and stays in sync with
        whatever the user does to the window (clicking X, etc.).
        """
        for action_name, window_type in self.WINDOW_MENU_ACTIONS.items():
            action = getattr(self.main_view, action_name, None)
            if action is None:
                continue

            # Initialize check state from current visibility
            action.setChecked(
                self.subwindow_manager.is_window_visible(window_type)
            )

            # Toggling the action shows/hides the subwindow
            action.toggled.connect(
                lambda checked, wt=window_type: (
                    self.subwindow_manager.show_window(wt)
                    if checked
                    else self.subwindow_manager.hide_window(wt)
                )
            )

        # When a subwindow's visibility changes from any source
        # (X button, programmatic show, etc.), sync the menu check state.
        self.subwindow_manager.visibility_changed.connect(
            self._sync_window_menu_check
        )

    def _sync_window_menu_check(
        self,
        window_type: str,
        is_visible: bool,
    ) -> None:
        for action_name, wt in self.WINDOW_MENU_ACTIONS.items():
            if wt != window_type:
                continue
            action = getattr(self.main_view, action_name, None)
            if action is None:
                return
            # Block signals so setChecked doesn't re-trigger toggled and
            # call show/hide redundantly.
            action.blockSignals(True)
            action.setChecked(is_visible)
            action.blockSignals(False)
            return


    def _connect_sample_controller_signals(self) -> None:
        # TODO *** THESE NEXT FOUR SIGNALS SHOULD BE REFACTORED, REPEATING PATTERN
        self.sample_controller.sigMzMLImportWizardComplete.connect(
            self._run_mzml_import_process
        )

        self.sample_controller.sigFingerprintImportWizardComplete.connect(
            self._run_fingerprint_import_process
        )

        self.sample_controller.sigMetadataImportWizardComplete.connect(
            self._run_metadata_import_process
        )

        self.sample_controller.sigAnalyteTableImportWizardComplete.connect(
            self._run_analyte_table_import_process
        )

        self.sample_controller.sigFeatureTableImportWizardComplete.connect(
            self._run_feature_table_import_process
        )

        self.sample_controller.sigModelChanged.connect(
            self._on_model_changed
        )

        self.sample_controller.sigViewEnsemble.connect(
            self._handle_view_ensemble_request
        )


    def _connect_sample_viewer_signals(self) -> None:
        """
        TODO: refactor this out. Maybe establish an interface?
        Should be called whenever a SampleViewer window is created
        """
        sample_viewer: 'SampleViewer' = self.subwindow_manager.get_window(
            'sample_viewer'
        )
        if not sample_viewer:
            return

        sample_viewer.sigEnsembleExtractionRequested.connect(
            self._handle_generate_ensemble_request
        )

        sample_viewer.sigAutoEnsembleRequested.connect(
            self._handle_auto_ensemble_request
        )

        sample_viewer.sigAlignEnsemblesRequested.connect(
            self._handle_align_ensembles_request
        )

        sample_viewer.sigViewEnsembleRequested.connect(
            self._handle_view_ensemble_request
        )

    def _on_model_changed(
        self,
        model_type: Literal['Sample', 'AnalyteTable'],
        model,
    ) -> None:
        """
        Just passes signal on to MainView
        """
        self.main_view.set_sample_list_model(
            model_type, model
        )

    def _handle_sample_filter_changed(
        self,
        filter_text: str,
        show_injections: bool,
        show_fingerprints: bool,
    ) -> None:
        """
        Handle sample filter changes from MainView
        """
        self.sample_controller.set_sample_filter_criteria(
            filter_text,
            show_injections,
            show_fingerprints,
        )


    def _handle_import_mzmls_request(self) -> None:
        self.sample_controller.show_mzml_import_wizard()


    def _run_mzml_import_process(
        self,
        input_filepaths: str | list[str],
        sample_names: str | list[str],
        scan_array_params: tuple[
            'ScanArrayParameters', Optional['ScanArrayParameters']
        ],
        acquisition_mode: str,
    ):
        self.process_controller.start_process(
            module_path="core.cli.mzml_import",
            function_name="main",
            parameters={
                "input_filepaths": input_filepaths,
                "sample_names": sample_names,
                "scan_array_params": scan_array_params,
                "acquisition_mode": acquisition_mode,
            },
            on_completion_func=self.sample_controller.on_mzml_import_completion
        )


    def _handle_import_fingerprints_request(self) -> None:
        self.sample_controller.show_fingerprint_import_wizard()


    def _run_fingerprint_import_process(
        self,
        import_params: 'FingerprintImportParams'
    ):
        self.process_controller.start_process(
            module_path="core.cli.fingerprint_import",
            function_name="main",
            parameters={
                "params": import_params,
            },
            on_completion_func=(
                self.sample_controller.on_fingerprint_import_completion
            )
        )


    def _handle_import_metadata_request(self) -> None:
        self.sample_controller.show_metadata_import_wizard()


    def _run_metadata_import_process(
        self,
        csv_filepath: str,
        samplename_column: str,
        metadata_columns: list[str],
        samples: list['Sample']
    ):
        # Call the project saving process
        self.process_controller.start_process(
            module_path="core.cli.metadata_import",
            function_name="main",
            parameters={
                "csv_filepath": Path(csv_filepath),
                "samplename_column": samplename_column,
                "metadata_columns": metadata_columns,
                "samples": samples,
            },
            on_completion_func=self.sample_controller.on_metadata_import_completion,
        )


    def _run_analyte_table_import_process(
        self,
        analyte_table_csv_filepath: Path,
        analyte_id_column: str,
        sample_name_columns: list[str],
        metadata_table_csv_filepath: Path,
        metadata_id_column: str,
        field_columns: list[str],

    ):
        self.process_controller.start_process(
            module_path="core.cli.analyte_table_import",
            function_name="main",
            parameters={
                "analyte_table_csv_filepath": Path(analyte_table_csv_filepath),
                "analyte_id_column": analyte_id_column,
                "sample_name_columns": sample_name_columns,
                "metadata_table_csv_filepath": Path(metadata_table_csv_filepath),
                "metadata_id_column": metadata_id_column,
                "field_columns": field_columns,
            },
            on_completion_func=self.sample_controller.on_analyte_table_import_completion,
        )


    def _handle_import_analyte_table_request(self) -> None:
        self.sample_controller.show_analyte_table_import_wizard()

    def _handle_import_feature_table_request(self) -> None:
        self.sample_controller.show_feature_table_import_wizard()

    def _handle_labeling_request(self) -> None:
        from PyQt5.QtWidgets import QMessageBox
        from gui.dialogues.LabelingSessionDialog import LabelingSessionDialog

        samples = [
            s for s in self.data_registry.get_all_samples()
            if s.injection is not None
        ]
        if not samples:
            QMessageBox.warning(
                self.main_view,
                "No samples",
                "Load at least one sample with MS data before labeling.",
            )
            return

        dialog = LabelingSessionDialog(parent=self.main_view)
        if dialog.exec_() != dialog.Accepted:
            return
        params = dialog.params()

        # Generate candidates on a background thread so the work + its
        # logs appear in the Process Monitor. The labeling window is
        # created in the completion callback with the returned list.
        self.process_controller.start_process(
            module_path="core.labeling.candidate_generator",
            function_name="main",
            parameters={
                "samples": samples,
                "min_intsy": params.min_intsy,
                "window_half_width": 60,
                "max_per_sample": params.max_per_sample,
            },
            on_completion_func=lambda candidates: (
                self._open_labeling_window(samples, candidates, params)
            ),
        )

    def _open_labeling_window(
        self,
        samples: list,
        candidates: list,
        params,
    ) -> None:
        from gui.labeling import LabelingWindow

        if not candidates:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                self.main_view,
                "No candidates",
                "Candidate generation produced no peaks. "
                "Try lowering the minimum intensity.",
            )
            return

        # Keep a reference so the window isn't garbage-collected
        self._labeling_window = LabelingWindow(
            samples=samples,
            candidates=candidates,
            label_file_path=params.label_file_path,
            annotator=params.annotator,
            min_intsy=params.min_intsy,
            max_per_sample=params.max_per_sample,
        )
        self._labeling_window.setAttribute(
            QtCore.Qt.WA_DeleteOnClose, True,
        )
        self._labeling_window.destroyed.connect(
            lambda *_: setattr(self, "_labeling_window", None)
        )
        self._labeling_window.show()

    def _run_feature_table_import_process(
        self,
        features,  # list[FeatureCoordinate]
        params,    # FeatureTableImportParams
    ):
        samples = self.data_registry.get_all_samples()
        if not samples:
            return

        self.process_controller.start_process(
            module_path="core.cli.import_feature_table",
            function_name="import_feature_table",
            parameters={
                "features": features,
                "samples": samples,
                "params": params,
            },
            on_completion_func=self._on_feature_table_import_complete,
        )

    def _on_feature_table_import_complete(
        self,
        alignment,  # EnsembleAlignment
    ):
        self.data_registry.register_alignment(alignment)

        multi = sum(
            1 for a in alignment.analytes
            if len(a.ensemble_map) > 1
        )
        print(
            f"\n=== Feature Table Import Complete ===\n"
            f"Samples: {alignment.sample_count}\n"
            f"Total analytes: {alignment.analyte_count}\n"
            f"Matched across samples: {multi}\n"
            f"Singletons: {alignment.analyte_count - multi}\n"
        )

    def _handle_filter_alignment_request(
        self,
        indexes: list[QtCore.QModelIndex],
    ):
        index = indexes[0]
        alignment = self.sample_controller.get_alignment_by_index(index)
        if not alignment:
            return

        # Build UUID -> name and UUID -> Sample mappings
        sample_names = {}
        sample_lookup = {}
        for uuid in alignment.sample_uuids:
            sample = self.data_registry.get_sample(uuid)
            if sample:
                sample_names[uuid] = sample.name
                sample_lookup[uuid] = sample
            else:
                sample_names[uuid] = str(uuid)

        from gui.dialogues.AlignmentFilterWizard import AlignmentFilterWizard
        wizard = AlignmentFilterWizard(alignment, sample_names, sample_lookup)
        wizard.sigFilterAccepted.connect(self._on_filter_alignment_accepted)
        wizard.show()
        self._filter_wizard = wizard  # prevent GC

    def _on_filter_alignment_accepted(self, result):
        self.data_registry.register_alignment(result.alignment)
        print(result.format_summary())

    def _handle_export_alignment_request(
        self,
        indexes: list[QtCore.QModelIndex],
    ):
        index = indexes[0]
        alignment = self.sample_controller.get_alignment_by_index(index)
        if not alignment:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self.main_view,
            "Export Feature Table",
            "",
            "TSV files (*.tsv);;CSV files (*.csv)",
        )
        if not filepath:
            return

        from pathlib import Path
        from core.cli.export_table import export_feature_table_to_file

        path = Path(filepath)
        separator = ',' if path.suffix.lower() == '.csv' else '\t'

        sample_names = {}
        sample_lookup = {}
        for uuid in alignment.sample_uuids:
            sample = self.data_registry.get_sample(uuid)
            if sample:
                sample_names[uuid] = sample.name
                sample_lookup[uuid] = sample
            else:
                sample_names[uuid] = str(uuid)

        export_feature_table_to_file(
            alignment=alignment,
            samples=sample_lookup,
            sample_names=sample_names,
            output=path,
            separator=separator,
        )
        print(f"Exported to {path}")

    def _handle_view_samples_request(
        self,
        indexes: list[QtCore.QModelIndex]
    ) -> None:
        selected_samples: list['data_structs.Sample'] = (
            self.sample_controller.get_samples_by_index(indexes)
        )

        sample_viewer: 'SampleViewer' = self.subwindow_manager.get_window(
            'sample_viewer'
        )

        sample_viewer.add_samples(
            [x.uuid for x in selected_samples],
            visible=True,
        )

        self.selection_manager.sigAnalyteSelectionChanged.connect(
            sample_viewer.on_analyte_selection
        )

        self.subwindow_manager.show_window('sample_viewer')


    def _handle_view_analyte_table_request(
        self,
        index: list[QtCore.QModelIndex], # Qt signal is a list
    ):
        index = index[0]

        alignment = self.sample_controller.get_alignment_by_index(index)
        if not alignment:
            return

        from gui.views.alignment_viewer import AlignmentViewer
        alignment_viewer: 'AlignmentViewer' = self.subwindow_manager.get_window(
            'alignment_viewer'
        )

        alignment_viewer.set_alignment(alignment)
        self.subwindow_manager.show_window('alignment_viewer')

    def _handle_save_project_request(self):
        filename, _ = QFileDialog.getSaveFileName(
            parent=self.main_view,
            caption="Select location to save project file",
            filter="*.mzk",
            initialFilter="*.mzk",
        )

        if not filename:
            return

        # Call the project saving process
        filepath = Path(filename).with_suffix(".mzk")
        self.process_controller.start_process(
            module_path="core.utils.persistence",
            function_name="save_project",
            parameters={
                "filepath": Path(filepath),
                "data_registry": self.data_registry,
                # "injection_model": self.injection_controller.model,
                # "fingerprint_model": self.fingerprint_controller.model,
            },
            on_completion_func=lambda x: None,
        )

    def _handle_load_project_request(self):
        filepath, _ = QFileDialog.getOpenFileName(
            parent=self.main_view,
            caption="Select project file to load",
            filter="*.mzk",
            initialFilter="*.mzk",
        )

        if not filepath:
            return

        # Call the project loading process
        self.process_controller.start_process(
            module_path="core.utils.persistence",
            function_name="load_project",
            parameters={
                "filepath": Path(filepath),
            },
            on_completion_func=self._on_project_loaded,
        )

    def _on_project_loaded(
        self,
        result: tuple[list['Sample'], list],
    ):
        samples, alignments = result
        self.data_registry.register_samples(samples)
        for alignment in alignments:
            self.data_registry.register_alignment(alignment)

    def _handle_generate_ensemble_request(
        self,
        input_params: 'EnsembleExtractionParams',  # Qt signal is named tuple
    ):
        self.process_controller.start_process(
            module_path="core.cli.generate_ensemble",
            function_name="get_cofeature_ensembles",
            parameters={
                "search_ftr_ptrs": [ input_params.search_ftr_ptr ],
                "injection": input_params.injection,
                "ms1_corr_threshold": input_params.ms1_corr_threshold,
                "ms2_corr_threshold": input_params.ms2_corr_threshold,
                "min_intsy": input_params.min_intsy,
                "use_rel_intsy": input_params.use_rel_intsy,
                "precursor_mz_tolerance": input_params.precursor_mz_tolerance,
            },
            on_completion_func=self.sample_controller.on_ensemble_generation,
        )

    def _handle_auto_ensemble_request(
        self,
        sample_uuid: 'data_structs.SampleUUID',
    ):
        sample = self.data_registry.get_sample(sample_uuid)
        if not sample or not sample.injection:
            return

        from core.cli.generate_ensemble import AutoEnsembleParams

        # TODO: expose these params in the GUI
        params = AutoEnsembleParams(
            parent_threshold=2e4,
            cofeature_threshold=1000,
            ms1_corr_threshold=0.8,
            ms2_corr_threshold=0.7,
            use_rel_intsy=True,
            rt_range=(60,380),
        )

        self.process_controller.start_process(
            module_path="core.cli.generate_ensemble",
            function_name="auto_generate_ensembles",
            parameters={
                "injection": sample.injection,
                "params": params,
            },
            on_completion_func=self.sample_controller.on_ensemble_generation,
        )

    def _handle_align_ensembles_request(
        self,
        sample_uuids: list['data_structs.SampleUUID'],
    ):
        samples = [
            self.data_registry.get_sample(uuid)
            for uuid in sample_uuids
        ]
        samples = [s for s in samples if s is not None]

        if len(samples) < 2:
            return

        from core.cli.align_ensembles import AlignmentParams

        # TODO: expose these params in the GUI
        params = AlignmentParams(
            rt_tolerance=10.0,
            mz_tolerance=0.01,
            ms1_similarity_threshold=0.7,
            ms2_similarity_threshold=0.6,
        )

        self.process_controller.start_process(
            module_path="core.cli.align_ensembles",
            function_name="align_ensembles",
            parameters={
                "samples": samples,
                "params": params,
            },
            on_completion_func=self._on_alignment_complete,
        )

    def _on_alignment_complete(
        self,
        alignment,
    ):
        self.data_registry.register_alignment(alignment)

        # Debug summary
        multi_sample = sum(
            1 for a in alignment.analytes
            if len(a.ensemble_map) > 1
        )
        print(
            f"\n=== Alignment Complete ===\n"
            f"Samples: {alignment.sample_count}\n"
            f"Total analytes: {alignment.analyte_count}\n"
            f"Matched across samples: {multi_sample}\n"
            f"Singletons: {alignment.analyte_count - multi_sample}\n"
        )

    def _handle_view_ensemble_request(
        self,
        ensemble: 'data_structs.Ensemble'
    ):
        ensemble_viewer: 'EnsembleViewer' = self.subwindow_manager.get_window(
            'ensemble_viewer'
        )

        ensemble_viewer.set_ensemble(ensemble)
        self.subwindow_manager.show_window('ensemble_viewer')




