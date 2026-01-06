from PyQt5 import QtCore

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

class SelectionManager(
    QtCore.QObject,
):
    """
    Coordinates selections across the viewer
    """
    sigSampleSelected = QtCore.pyqtSignal(
        object,  # uuid
    )

    sigMSLevelSelected = QtCore.pyqtSignal(
        int
    )

    sigSpectrumSelected = QtCore.pyqtSignal(
        object,  # uuid
        int,  # ms_level
        int,  # scan_num
    )

    sigMSLaneSelected = QtCore.pyqtSignal(
        int,  # ms_lane_idx
    )

    sigEnsembleSelected = QtCore.pyqtSignal(
        object,  # EnsembleUUID
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

        self._data_source = data_source

        self._selected_sample_uuid: Optional['SampleUUID'] = None
        self._selected_ms_level: Literal[1, 2] = 1
        self._selected_scan_num: Optional[int] = None
        self._selected_ms_lane: Optional[int] = None

    # ***SAMPLE SELECTION***
    def set_selected_sample(
        self,
        uuid: 'SampleUUID'
    ):
        if uuid == self._selected_sample_uuid:
            return

        self._selected_sample_uuid = uuid

        # Clear dependent selections
        self._clear_spectrum_selection()

        self.sigSampleSelected.emit(uuid)

    @property
    def selected_sample_uuid(self) -> Optional['SampleUUID']:
        return self._selected_sample_uuid

    # ***MS_LEVEL***
    def set_ms_level(
        self,
        ms_level: Literal[1, 2],
    ):
        """
        Sets the ms level, and changes the selected spectrum
        to closest one by rt
        """
        if ms_level == self._selected_ms_level:
            # Do nothing
            return

        if self._selected_sample_uuid:
            rt_before_ms_level_change: float = self._get_scan_array(
                uuid=self._selected_sample_uuid,
                ms_level=self._selected_ms_level,
            ).rt_arr[self._selected_scan_num]

            self.set_selected_spectrum_by_rt(
                uuid=self._selected_sample_uuid,
                ms_level=ms_level,  # Use NEW ms_level
                rt=rt_before_ms_level_change,
            )

        self._selected_ms_level = ms_level
        self.sigMSLevelSelected.emit(
            self._selected_ms_level
        )

    @property
    def selected_ms_level(self) -> Literal[1, 2]:
        return self._selected_ms_level

    # ***SPECTRUM SELECTION***
    def set_selected_spectrum_by_rt(
        self,
        uuid: 'SampleUUID',
        ms_level: int,
        rt: float,
    ) -> None:
        """
        Converts rt into scan number, then sets selected spec
        """
        scan_num: int = self._get_scan_array(
            uuid, ms_level
        ).rt_to_scan_num(rt)

        self._set_spectrum(
            uuid=uuid,
            ms_level=ms_level,
            scan_num=scan_num,
        )

    def set_selected_spectrum_by_scan_num(
        self,
        uuid: 'SampleUUID',
        ms_level: int,
        scan_num: int,
    ) -> None:
        """
        Set selected spec
        """
        self._set_spectrum(
            uuid, ms_level, scan_num
        )

    def _set_spectrum(
        self,
        uuid: 'SampleUUID',
        ms_level: int,
        scan_num: int,
    ) -> None:
        """
        Updates state
        """
        self._selected_sample_uuid = uuid
        self._selected_ms_level = ms_level
        self._selected_scan_num = scan_num

        self.sigSpectrumSelected.emit(
            uuid, ms_level, scan_num
        )
        self.sigSampleSelected.emit(
            uuid
        )
        self.sigMSLevelSelected.emit(
            ms_level
        )

    def _clear_spectrum_selection(self):
        # TODO: consider implementing?
        pass

    def get_selected_spectrum_array(
        self,
    ) -> Optional['SpectrumArray']:
        """
        Returns the currently selected spectrum as a
        SpectrumArray.

        Returns None if nothing selected
        """
        if not self._selected_sample_uuid or self._selected_scan_num is None:
            return None

        return self.get_selected_scan_array().get_spectrum(
            scan_num=self._selected_scan_num
        )

    # ***SELECTED SCAN ARRAY***
    def get_selected_scan_array(
        self,
    ) -> Optional['ScanArray']:
        """
        Returns the currently selected ScanArray.

        Returns None if nothing selected
        """
        if not self._selected_sample_uuid or not self._selected_ms_level:
            return None

        return self._get_scan_array(
            uuid=self._selected_sample_uuid,
            ms_level=self._selected_ms_level,
        )

    def _get_scan_array(
        self,
        uuid: 'SampleUUID',
        ms_level: int,
    ) -> 'ScanArray':
        sample = self._data_source.get_sample(uuid)
        return sample.injection.get_scan_array(ms_level)

    # ***SELECTED MS LANE***
    def set_selected_ms_lane_idx(
        self,
        idx: Optional[int],
    ):
        if idx == self._selected_ms_lane:
            return

        self._selected_ms_lane = idx
        self.sigMSLaneSelected.emit(self._selected_ms_lane)

    @property
    def selected_ms_lane_idx(self) -> Optional[int]:
        return self._selected_ms_lane

    def clear_selected_ms_lane_idx(
        self,
    ):
        self._selected_ms_lane = None
        self.sigMSLaneSelected.emit(None)

    # ***SELECTED SCAN NUM***
    def get_selected_scan_num(
        self
    ) -> int:
        return self._selected_scan_num




