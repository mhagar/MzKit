import numpy as np
from PyQt5 import QtCore

from core.cli.generate_ensemble import (
    EnsembleExtractionParams,
    auto_params_from_config,
)

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from configparser import ConfigParser
    from core.formula.params import FindMfsParams
    from core.data_structs import (
        SampleUUID,
    )
    from core.interfaces.data_sources import SampleDataSource


def _manual_params_from_config(config: Optional['ConfigParser']) -> dict:
    """
    Default *manual* single-ensemble extraction params, sourced from the shared
    ``[auto_ensemble]`` scoring settings so config stays the single source of
    truth. Falls back to sane constants when no config is available.
    """
    if config is None:
        return {
            'ms1_corr_threshold': 0.9,
            'ms2_corr_threshold': 0.8,
            'min_intsy': 1000.0,
            'use_rel_intsy': True,
            'method': 'cosine',
        }
    p = auto_params_from_config(config)
    return {
        'ms1_corr_threshold': p.ms1_corr_threshold,
        'ms2_corr_threshold': p.ms2_corr_threshold,
        'min_intsy': p.cofeature_threshold,
        'use_rel_intsy': p.use_rel_intsy,
        'method': p.method,
    }


class EnsembleExtractionManager(
    QtCore.QObject,
):
    """
    Coordinates manual (Cmpd-tool) ensemble extraction.

    Holds the current manual-extraction params (set by the shared
    EnsembleExtractionDialog in SINGLE mode) and, optionally, find-mfs params to
    chain after the ensemble is created.
    """
    sigEnsembleExtractionRequested = QtCore.pyqtSignal(
        object,  # EnsembleExtractionParams
        object,  # find-mfs params dict, or None
    )

    def __init__(
        self,
        data_source: 'SampleDataSource',
        config: Optional['ConfigParser'] = None,
    ):
        super().__init__()
        self.data_source = data_source
        self.config = config

        # Updated whenever the user OKs the settings dialog in SINGLE mode.
        self.manual_params: dict = _manual_params_from_config(config)
        self.pending_findmfs: Optional['FindMfsParams'] = None

    def set_manual_params(
        self,
        params: dict,
        findmfs: Optional['FindMfsParams'] = None,
    ) -> None:
        """Store the params the next manual extraction should use."""
        self.manual_params = params
        self.pending_findmfs = findmfs

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
            **self.manual_params,
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
        method: str = "cosine",
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
            method=method,
        )

        self.sigEnsembleExtractionRequested.emit(
            ensemble_extraction_params,
            self.pending_findmfs,
        )
