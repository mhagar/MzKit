import numpy as np
from PyQt5 import QtCore

from core.cli.generate_ensemble import EnsembleExtractionParams
from gui.views.sample_viewer.menus import EnsembleExtractionSettingsMenu

from typing import TYPE_CHECKING, Union, Optional, Literal

if TYPE_CHECKING:
    from core.data_structs import (
        Sample,
        SampleUUID,
        Injection,
        Analyte,
        AnalyteID,
        FeaturePointer,
        ScanArray,
    )
    from core.utils.array_types import (
        SpectrumArray
    )
    from core.interfaces.data_sources import (
        SampleDataSource,
        AnalyteTableSource,
        AnalyteSource,
    )

class EnsembleExtractionManager(
    QtCore.QObject,
):
    """
    Coordinates ensemble extraction
    """
    sigEnsembleExtractionGraphicsRequested = QtCore.pyqtSignal(
        object,  # sample_uuid
        tuple,   # (rt_start, rt_end)
        object,  # chrom_array to display
    )

    sigEnsembleExtractionRequested = QtCore.pyqtSignal(
        object
    )

    def __init__(
        self,
        data_source: Union[
            'SampleDataSource',
            'AnalyteTableSource',
            'AnalyteSource',
        ],
    ):
        super().__init__()
        self.data_source = data_source
        # self.ensemble_extraction_settings_menu = EnsembleExtractionSettingsMenu(self)
        self.settings_menu = EnsembleExtractionSettingsMenu()

    def request_using_current_params(
        self,
        sample_uuid: 'SampleUUID',
        rt_bounds: tuple[float, float],
        mass_lane_idx: int,
    ):
        self.request_ensemble_generation(
            sample_uuid=sample_uuid,
            rt_bounds=rt_bounds,
            mass_lane_idx=mass_lane_idx,
            **self.settings_menu.get_params(),
        )

    def request_ensemble_generation(
        self,
        sample_uuid: 'SampleUUID',
        rt_bounds: tuple[float, float],
        mass_lane_idx: int,
        ms1_corr_threshold: float,
        ms2_corr_threshold: float,
        min_intsy: float,
        use_rel_intsy: bool,
    ):
        """
        Emits a signal that queues ensemble extraction
        """
        if rt_bounds == (0, 0):
            return

        if not mass_lane_idx:
            return

        # Convert rt_bounds into scan numbers
        injection = self.data_source.get_sample(sample_uuid).injection
        scan_array = injection.get_scan_array(ms_level=1)
        scan_window: tuple[int, int] = tuple( #type: ignore
            scan_array.rt_to_scan_num(x) for x in rt_bounds
        )

        search_ftr_ptr = scan_array.make_feature_pointer(
            mass_lane_idx=mass_lane_idx,
            scan_idxs=np.arange(
                scan_window[0], scan_window[1] + 1
            )
        )

        ensemble_extraction_params = EnsembleExtractionParams(
            search_ftr_ptr=search_ftr_ptr,
            injection=injection,
            ms1_corr_threshold=ms1_corr_threshold,
            ms2_corr_threshold=ms2_corr_threshold,
            min_intsy=min_intsy,
            use_rel_intsy=use_rel_intsy,
        )

        self.sigEnsembleExtractionRequested.emit(
            ensemble_extraction_params
        )

    def showExtractionMenu(
        self,
        pos: QtCore.QPoint,
        height: int,
    ):

        self.settings_menu.move(
            pos.x() - self.settings_menu.size().width(),
            pos.y() + height,
            )

        self.settings_menu.show()