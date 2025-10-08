#Temporary:
import pyqtgraph as pg
from gui.utils.ms_arrays import zero_pad_arrays

from PyQt5 import QtCore, QtWidgets

from core.utils.natural_sort import natural_sort_key
from gui.models.analyte_table_list_model import AnalyteTableListModel
from gui.models.sample_list_model import SampleListModel
from gui.models.sample_proxy_model import SampleProxyModel
from gui.dialogues.MzMLImportWizard import MzMLImportWizard
from gui.dialogues.MetadataImportWizard import MetadataImportWizard
from gui.dialogues.FingerprintImportWizard import FingerprintImportWizard
from gui.dialogues.AnalyteTableImportWizard import AnalyteTableImportWizard

import logging
from typing import Optional, Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from core.data_structs import (
        DataRegistry,
        Sample,
        SampleUUID,
        AnalyteTable,
        Ensemble,
    )
    from core.utils.array_types import SpectrumArray
    from core.interfaces.data_sources import AnalyteTableSource
    from core.data_structs.scan_array import ScanArrayParameters
    from core.data_structs.fingerprint import FingerprintImportParams
    from configparser import ConfigParser

logger = logging.getLogger()

class SampleController(QtCore.QObject):
    sigMzMLImportWizardComplete = QtCore.pyqtSignal(
        list,  #list[str] (Paths)
        list,       # list[str] (Samplenames)
        object,     # tuple(ScanArrayParameters (MS1), optional MS2)
    )

    sigFingerprintImportWizardComplete = QtCore.pyqtSignal(
        object,    # FingerprintImportParams
    )

    sigMetadataImportWizardComplete = QtCore.pyqtSignal(
        object,  # csv filepath
        str,       # samplename_column
        object, # metadata_columns
        object,    # list[Sample]
    )

    sigAnalyteTableImportWizardComplete = QtCore.pyqtSignal(
        object,  # analyte table csv filepath
        object, # analyte_id column
        object, # sample name columns
        object, # analyte metadata table csv filepath
        object, # analyte metadata id column
        object, # field columns
    )

    sigViewEnsemble = QtCore.pyqtSignal(
        object, # Ensemble
    )

    sigModelChanged = QtCore.pyqtSignal(
        str,   # Model type: 'Sample' or 'AnalyteTable'
        object,  # SampleListModel
    )

    def __init__(
        self,
        config: 'ConfigParser',
        data_registry: 'DataRegistry',
    ):
        super().__init__()
        self.mzml_import_wizard = MzMLImportWizard()
        self.fingerprint_import_wizard = FingerprintImportWizard()
        self.metadata_import_wizard = MetadataImportWizard()
        self.analyte_table_import_wizard = AnalyteTableImportWizard()
        self.data_registry: 'DataRegistry' = data_registry
        self.config = config

        self.sample_list_model: Optional[SampleListModel] = None
        self.sample_proxy_model: Optional[SampleProxyModel] = None
        self.analyte_table_list_model: Optional[AnalyteTableListModel] = None


    def initialize_sample_model(self):
        self.sample_list_model = SampleListModel(
            registry=self.data_registry
        )
        
        self.sample_proxy_model = SampleProxyModel()
        self.sample_proxy_model.setSourceModel(
            self.sample_list_model
        )
        
        self.sample_proxy_model.sort(0)

        self.sigModelChanged.emit(
            'Sample',
            self.sample_proxy_model,
        )

    def initialize_analyte_table_model(self):
        self.analyte_table_list_model = AnalyteTableListModel(
            registry=self.data_registry,
        )
        self.sigModelChanged.emit(
            'AnalyteTable',
            self.analyte_table_list_model,
        )

    def show_fingerprint_import_wizard(self):
        self.fingerprint_import_wizard = FingerprintImportWizard()
        self.fingerprint_import_wizard.sigImportParamsGiven.connect(
            self.fingerprint_wizard_complete
        )
        self.fingerprint_import_wizard.show()

    def fingerprint_wizard_complete(
        self,
        params: 'FingerprintImportParams',
    ):
        """
        Called when user completes fingerprint import *Wizard*,
        i.e. is done defining parameters for fingerprint import
        :param params:
        :return:
        """
        logger.debug(
            f"Fingerprint import wizard complete. Results: \n"
            f"{params}"
        )
        self.sigFingerprintImportWizardComplete.emit(params)

    def on_fingerprint_import_completion(
        self,
        samples: list['Sample'],
    ):
        for sample in samples:
            self.data_registry.register_sample(
                sample
            )
        
        # Enable sorting if we have samples and proxy model exists
        if self.sample_proxy_model and samples:
            self.sample_proxy_model.sort(0)

    def show_mzml_import_wizard(self):
        self.mzml_import_wizard = MzMLImportWizard(
            config=self.config,
        )
        self.mzml_import_wizard.sigImportParamsGiven.connect(
            # self.sigMzMLImportWizardComplete.emit
            self.mzml_wizard_completed
        )
        self.mzml_import_wizard.show()

    def mzml_wizard_completed(
        self,
        filepaths: list[str],
        samplenames: list[str],
        ms1_params: 'ScanArrayParameters',
        ms2_params: Optional['ScanArrayParameters'],
    ):
        """
        Called when user completes the *Wizard*, i.e.
        is done definining parameters for mzml import
        :param filepaths:
        :param samplenames:
        :param ms1_params:
        :param ms2_params:
        :return:
        """
        self.sigMzMLImportWizardComplete.emit(
            filepaths, samplenames, (ms1_params, ms2_params)
        )

    def on_mzml_import_completion(
        self,
        samples: list['Sample'],
    ):
        """
        Called when an mzml import process is complete.
        Adds the resulting samples to the model
        :param samples:
        :return:
        """
        for sample in samples:
            self.data_registry.register_sample(
                sample
            )
        
        # Sort proxy model
        self.sample_proxy_model.sort(0)


    def show_metadata_import_wizard(self):
        self.metadata_import_wizard = MetadataImportWizard()
        self.metadata_import_wizard.sigImportParamsGiven.connect(
            self.metadata_wizard_completed
        )
        self.metadata_import_wizard.show()

    def metadata_wizard_completed(
        self,
        arg_dict: dict
    ):
        self.sigMetadataImportWizardComplete.emit(
            arg_dict['csv_filepath'],
            arg_dict['samplename_column'],
            arg_dict['metadata_columns'],
            self.data_registry.get_all_samples(),
        )

    def on_metadata_import_completion(
        self,
        results: dict['SampleUUID', dict[str, any]]
    ):
        for uuid, metadata in results.items():
            self.data_registry.update_sample_metadata(
                uuid=uuid,
                metadata=metadata,
            )

    def show_analyte_table_import_wizard(self):
        """
        Placeholder
        """
        self.analyte_table_import_wizard = AnalyteTableImportWizard()
        self.analyte_table_import_wizard.sigImportParamsGiven.connect(
            self.analyte_table_wizard_completed
        )
        self.analyte_table_import_wizard.show()


    def analyte_table_wizard_completed(
        self,
        arg_dict: dict
    ):
        self.sigAnalyteTableImportWizardComplete.emit(
            *arg_dict.values()
        )

    def on_analyte_table_import_completion(
        self,
        results: 'AnalyteTable'
    ):
        self.data_registry.register_analyte_table(
            results
        )

    def on_ensemble_generation(
        self,
        ensembles: list['Ensemble'],
    ):
        print("Sending SIRIUS input to clipboard")

        app = QtWidgets.QApplication.instance()
        clipboard = app.clipboard()

        output = []
        for idx, ensemble in enumerate( ensembles ):
            output.append(f">compound {idx}")
            output.append(f">parentmass {ensemble.base_mz}")
            output.append("")
            output.append(">ms1")
            output += _format_spectrum_array(
                arr=ensemble.get_spectrum(ms_level=1)
            )

            output.append("")
            output.append(">collision 60")
            output += _format_spectrum_array(
                arr=ensemble.get_spectrum(ms_level=2)
            )

        clipboard.setText("\n".join(output))

        # self._plot_ensemble_spectra(ensembles[0])
        self.sigViewEnsemble.emit(ensembles[0])


    def _plot_ensemble_spectra(
        self,
        ensemble: 'Ensemble',
    ):
        """
        Temporary function; need to work fast to make shit for roger
        """
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        for i in [1, 2]:
            i: Literal[1, 2]
            print(f"Showing MS{i} plot")
            spec_array = ensemble.get_spectrum(ms_level=i)

            mz, intsy = zero_pad_arrays(
                spec_array['mz'],
                spec_array['intsy'],
            )

            pg.plot(
                mz,
                intsy,
                connect='pairs',
                title=f'MS{i} plot',
            )

        print("Showing chrom plot")
        scan_array = ensemble.injection.get_scan_array(1)
        bpc_array = scan_array.get_bpc(
            mz_range=(
                ensemble.base_mz - 0.05,
                ensemble.base_mz + 0.05,
            ),
            # rt_range=(
            #     ensemble.peak_rt - 50,
            #     ensemble.peak_rt + 50
            # )
        )


        pw = pg.plot(
            title='Ensemble Plot'
        )
        for ftr_ptr in ensemble.ms1_cofeatures:
            if ftr_ptr.get_max_intsy(scan_array) < 2e3:
                continue

            intsy = ftr_ptr.get_intensity_values(scan_array)
            rt = ftr_ptr.get_retention_times(scan_array)

            pw.plot(
                rt,
                intsy,
                symbol='o',
                symbolSize=2,
                symbolPen=None,
            )


    def get_samples_by_index(
        self,
        indexes: list[QtCore.QModelIndex]
    ) -> list['Sample']:
        """
        Called when user requests some samples are displayed
        in the Sample Viewer

        :param indexes:
        :return:
        """
        samples: list['Sample'] = []

        for proxy_idx in indexes:
            # Convert proxy index to source index
            source_idx = self.sample_proxy_model.mapToSource(proxy_idx)
            sample = self.sample_list_model.getSampleAtIndex(source_idx)
            if sample:
                samples.append(sample)

        return sorted(
            samples,
            key=lambda x: natural_sort_key(x.name)
        )

    def get_analyte_table_by_index(
        self,
        index: QtCore.QModelIndex,
    ) -> 'AnalyteTable':
        """
        Called when user requests an analyte table is displyaed
        :param index:
        :return:
        """
        return self.analyte_table_list_model.getAnalyteTableAtIndex(index)

    def set_sample_filter(
        self,
        filter_text: str
    ) -> None:
        """
        Set the filter text for sample searching
        :param filter_text:
        """
        self.sample_proxy_model.set_filter_text(
            filter_text
        )

    def set_sample_filter_criteria(
        self,
        filter_text: str,
        show_injections: bool,
        show_fingerprints: bool,
    ) -> None:
        """
        Set filter criteria (text and data types)
        :param filter_text: Text to filter sample names
        :param show_injections: Whether to show samples with injection data
        :param show_fingerprints: Whether to show samples with fingerprint data
        """
        if self.sample_proxy_model:
            self.sample_proxy_model.set_filter_criteria(
                filter_text,
                show_injections,
                show_fingerprints,
            )


def _format_spectrum_array(
    arr: 'SpectrumArray',
) -> list[str]:
    """
    Bullshit function, just need to go fast
    """
    output = []
    for mz, intsy in zip(
        arr['mz'],
        arr['intsy'],
    ):
        output.append(f"{mz:.5f} {intsy:.0f}")

    return output

