from PyQt5 import QtCore
import pyqtgraph as pg

from gui.views.sample_viewer.tools import (
    XICMode,
)
from gui.utils.ms_arrays import strip_empty_values

from typing import Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from gui.widgets.ChromPlotWidget import ChromViewBox
    from gui.views.sample_viewer.sample_widget_manager import SampleWidgetManager
    from gui.views.sample_viewer.model import SampleViewerItemModel
    from core.utils.array_types import ChromArray
    from core.data_structs import (
        SampleUUID,
        FeaturePointer,
        ScanArray,
    )

class ChromatogramDataManager(QtCore.QObject):
    """
    Manages chromatogram and fingerprint updates
    """
    def __init__(
        self,
        model: Optional['SampleViewerItemModel'],
        widget_manager: 'SampleWidgetManager',
    ):
        super().__init__()
        self._model = model
        self._widget_mgr = widget_manager
        self._current_ms_level: Literal[1, 2] = 1
        self._xic_mode: XICMode = XICMode.NONE
        self._extraction_range: Optional[tuple[float, float]] = None

        # TODO:
        print("Defaulting to link_y = True; expose to user!!")

    def set_model(
        self,
        model: 'SampleViewerItemModel',
    ):
        self._model = model

    def set_ms_level(
        self,
        ms_level: Literal[1, 2],
    ):
        """
        Change MS level and update all chromatograms
        """
        self._current_ms_level = ms_level
        self.update_all_chromatograms()

    def get_ms_level(self) -> Literal[1, 2]:
        return self._current_ms_level

    def set_xic_mode(
        self,
        mode: XICMode,
        extraction_range: Optional[tuple] = None,
    ):
        """
        Change XIC mode and update all chromatograms
        """
        self._xic_mode = mode
        if extraction_range:
            self._extraction_range = extraction_range
        self.update_all_chromatograms()

    def set_extraction_range(
        self,
        extraction_range: tuple[float, float],
    ):
        """
        Update extraction range and update all chromatograms
        """
        self._extraction_range = extraction_range
        self.update_all_chromatograms()

    def update_all_chromatograms(self):
        """
        Updates chromatograms for all visible widgets
        """
        for uuid, widget in self._widget_mgr.get_all_widgets().items():
            self.update_chromatogram(uuid)

        self.link_chrom_widget_axes()

    def update_chromatogram(
        self,
        uuid: "SampleUUID",
    ):
        injection = self._model.getInjection(uuid)
        if not injection:
            return

        scan_array: "ScanArray" = injection.get_scan_array(self._current_ms_level)
        chrom_array: "ChromArray" = self._get_chrom_array(scan_array)

        self._widget_mgr.get_widget(uuid).setChromArray(strip_empty_values(chrom_array))
        self.link_chrom_widget_axes()

    def _get_chrom_array(
        self,
        scan_array: 'ScanArray',
    ) -> 'ChromArray':
        """
        Get chromatogram based on current mode
        """
        match self._xic_mode:
            case XICMode.NONE:
                return scan_array.get_bpc() # type: ignore

            case XICMode.BPC:
                return scan_array.get_bpc(  # type: ignore
                    mz_range=self._extraction_range
                )

            case XICMode.XIC:
                return scan_array.get_xic(  # type: ignore
                    mz_range=self._extraction_range
                )

            case _:
                raise ValueError(
                    f"Invalid chromatogram type specified: "
                    f"{self._xic_mode}"
                )

    def link_chrom_widget_axes(
        self,
        link_x: bool = True,
        link_y: bool = True,
    ):
        """
        Iterates over the chrom_widgets and links their axes together
        :return:
        """
        previous_chrom_vb: Optional['ChromViewBox'] = None
        previous_fprint_vb: Optional['pg.ViewBox'] = None
        for uuid, sample_widget in self._widget_mgr.get_all_widgets().items():
            if not previous_chrom_vb:
                previous_chrom_vb = sample_widget.chromPlotWidget.pi.vb
                previous_fprint_vb = sample_widget.fprintPlotWidget.pi.vb
                continue

            vb_chrom: ChromViewBox = sample_widget.chromPlotWidget.pi.vb
            vb_fprint: pg.ViewBox = sample_widget.fprintPlotWidget.pi.vb
            if link_x:
                vb_chrom.setXLink(previous_chrom_vb)
                vb_fprint.setXLink(previous_fprint_vb)

            if link_y:
                vb_chrom.setYLink(previous_chrom_vb)
                vb_fprint.setYLink(previous_fprint_vb)

    def update_all_fingerprints(self):
        """
        Update fingerprints for all visible widgets
        """
        for uuid, widget in self._widget_mgr.get_all_widgets().items():
            self.update_fingerprint(uuid)

    def update_fingerprint(
        self,
        uuid: 'SampleUUID'
    ):
        fprint = self._model.getFingerprint(uuid)
        if fprint:
            self._widget_mgr.get_widget(uuid).setFprintArray(
                fprint.array,
                fprint.descriptors,
            )

    def update_all_plots(self):
        if not self._model:
            return

        self.update_all_chromatograms()
        self.update_all_fingerprints()

    def update_chrom_highlights(
        self,
        highlights: list[tuple['SampleUUID', 'ScanArray', int]]
    ):
        """
        Given a list of (SampleUUID, ScanArray, mass_lane_idx) tuples,
        updates the appropriate plots to display a little "highlight trace".

        These are intended to be transient.
        """
        for uuid, scan_array, mass_lane_idx in highlights:
            sample_widget = self._widget_mgr.get_widget(uuid)

            ftr_ptr: 'FeaturePointer' = scan_array.make_feature_pointer(
                mass_lane_idx=mass_lane_idx,
                scan_idxs=None,  # Whole lane
            )

            chrom = ftr_ptr.get_chrom_array(
                scan_array=self._model.getInjection(uuid).get_scan_array(
                    ms_level=self._current_ms_level,
                )
            )

            sample_widget.addHighlight(
                chrom=chrom,
                replace=True,
            )

    def clear_chrom_highlights(
        self,
        uuid: 'SampleUUID',
    ):
        """
        Removes highlights from selected sample
        """
        sample_widget = self._widget_mgr.get_widget(uuid)
        if sample_widget:
            sample_widget.clearHighlights()

