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

from ..views.analyte_table_viewer import AnalyteTableViewerWindow

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QApplication
    from configparser import ConfigParser
    from core.data_structs import Sample
    from core.data_structs.fingerprint import FingerprintImportParams
    from core.data_structs.scan_array import ScanArrayParameters
    from core.cli.generate_ensemble import InputParams as EnsembleInputParams
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

        self.main_view.sigSampleFilterChanged.connect(
            self._handle_sample_filter_changed
        )


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

        sample_viewer.sigEnsembleGenerationRequested.connect(
            self._handle_generate_ensemble_request
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
    ):
        self.process_controller.start_process(
            module_path="core.cli.mzml_import",
            function_name="main",
            parameters={
                "input_filepaths": input_filepaths,
                "sample_names": sample_names,
                "scan_array_params": scan_array_params,
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

        sample_viewer.show()


    def _handle_view_analyte_table_request(
        self,
        index: list[QtCore.QModelIndex], # Qt signal is a list
    ):
        index = index[0]

        selected_analyte_table = self.sample_controller.get_analyte_table_by_index(
            index
        )
        if not selected_analyte_table:
            return

        analyte_table_viewer: 'AnalyteTableViewerWindow' = self.subwindow_manager.get_window(
            'analyte_viewer'
        )

        analyte_table_viewer.set_analyte_table(
            selected_analyte_table
        )

        analyte_table_viewer.sigAnalytesSelected.connect(
            self.selection_manager.on_analyte_selection
        )

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

        # Call the project saving process
        self.process_controller.start_process(
            module_path="core.utils.persistence",
            function_name="load_project",
            parameters={
                "filepath": Path(filepath),
            },
            on_completion_func=self.data_registry.register_samples,
        )

    def _handle_generate_ensemble_request(
        self,
        input_params: 'EnsembleInputParams',  # Qt signal is named tuple
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
            },
            on_completion_func=self.sample_controller.on_ensemble_generation,
        )

    def _handle_view_ensemble_request(
        self,
        ensemble: 'data_structs.Ensemble'
    ):
        ensemble_viewer: 'EnsembleViewer' = self.subwindow_manager.get_window(
            'ensemble_viewer'
        )

        ensemble_viewer.set_ensemble(ensemble)
        ensemble_viewer.setFocus()




