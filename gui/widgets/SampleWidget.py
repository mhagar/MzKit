"""
Widget containing a ChromPlotWidget, as well as controls
(i.e. buttons for MS level, XIC/BPC, etc)
"""
from gui.resources.ChromWidget import Ui_Form

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from core.data_structs import (
        Sample, SampleUUID, EnsembleUUID,
        Injection, InjectionUUID,
        Fingerprint, FingerprintUUID,
    )


class SampleWidget(
    QtWidgets.QWidget,
    Ui_Form,
):
    """
    Widget containing a ChromPlotWidget, as well as controls
    (i.e. buttons for MS level, XIC/BPC, etc.)
    """
    sigChromatogramHovered = QtCore.pyqtSignal(
        object, # UUID; Note: specifying `int` converts to 32bit (bad)
        QtCore.QPointF, # Mouse position in scene coords
    )

    sigChromatogramLeaved = QtCore.pyqtSignal(
        object, # UUID; Note: specifying `int` converts to 32bit (bad)
    )

    sigChromatogramClicked = QtCore.pyqtSignal(
        object, # UUID; Note: specifying `int` converts to 32bit (bad)
    )

    sigFPrintHovered = QtCore.pyqtSignal(
        object, # UUID; Note: specifying `int` converts to 32bit (bad)
        int,  # Index of fingerprint feature being hovered
    )

    # Ensemble peak interaction signals
    sigEnsemblePeakHovered = QtCore.pyqtSignal(
        object,  # SampleUUID
        object,  # EnsembleUUID
        QtCore.QPointF,  # Position
    )
    sigEnsemblePeakLeaved = QtCore.pyqtSignal(
        object,  # SampleUUID
    )
    sigEnsemblePeakClicked = QtCore.pyqtSignal(
        object,  # SampleUUID
        object,  # EnsembleUUID
        int,  # MouseButton
    )


    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.UUID: Optional['SampleUUID'] = None     # UUID of displayed sample

        self.setupUi(self)

        # Hide FPrint by default
        self.fprintPlotWidget.setVisible(False)

        # Adjust chromPlotWidget appearance (cleaner)
        self.chromPlotWidget.pi.setContentsMargins(0, 0, 0, 0)
        self.chromPlotWidget.pi.vb.setContentsMargins(0, 0, 0, 0)

        # Connect signals
        self.chromPlotWidget.sigChromatogramHovered.connect(
            self.on_chromatogram_hovered
        )
        self.chromPlotWidget.sigChromatogramLeaved.connect(
            self.on_chromatogram_leaved
        )
        self.chromPlotWidget.sigChromatogramClicked.connect(
            self.on_chromatogram_clicked
        )
        self.fprintPlotWidget.sigFPrintHovered.connect(
            self.on_fprint_hovered
        )

        # Ensemble peak interaction signals
        self.chromPlotWidget.sigEnsemblePeakHovered.connect(
            self.on_ensemble_peak_hovered
        )
        self.chromPlotWidget.sigEnsemblePeakLeaved.connect(
            self.on_ensemble_peak_leaved
        )
        self.chromPlotWidget.sigEnsemblePeakClicked.connect(
            self.on_ensemble_peak_clicked
        )

        # TODO: Expose to user
        # self.setFixedHeight(150)

    def setName(
        self,
        name: str,
    ):
        self.chromPlotWidget.update_label(name)


    def setSampleUuid(
        self,
        uuid: 'SampleUUID',
    ):
        self.UUID = uuid


    def setChromArray(
        self,
        chroms: list[np.ndarray] | np.ndarray,
        max_opacity: Optional[int] = None,
        replace: bool = True,
    ):
        """
        Wrapper around `ChromPlotWidget.setChromArray`
        """
        self.chromPlotWidget.setChromArray(
            chroms=chroms,
            max_opacity=max_opacity,
            replace=replace,
        )


    def addHighlight(
        self,
        chrom: np.ndarray,
        replace: bool = True,
    ):
        """
        Wrapper around 'ChromPlotWidget.addHighlight'
        :param chrom:
        :param replace:
        :return:
        """
        self.chromPlotWidget.addHighlight(
            chrom,
            replace,
        )


    def clearHighlights(
        self,
    ):
        self.chromPlotWidget.clearHighlights()


    def addWindowSelector(
        self,
        bounds: tuple[float, float],
        display_arr: Optional[np.ndarray],
    ):
        """
        Wrapper around 'ChromPlotWidget.addWindowSelector'
        """
        self.chromPlotWidget.addWindowSelector(
            bounds,
            display_arr,
        )


    def getWindowSelectorBounds(
        self,
    ) -> tuple[float, float]:
        window_selector: pg.LinearRegionItem = self.chromPlotWidget.window_selector
        if window_selector:
            return window_selector.getRegion()

        return 0., 0.


    def clearWindowSelector(self):
        self.chromPlotWidget.clearWindowSelector()

    def addPeak(
        self,
        chrom: np.ndarray,
        uuid: 'EnsembleUUID',
        color: Optional[str] = None,
    ):
        """
        Wrapper around ChromPlotWidget.addPeak
        """
        self.chromPlotWidget.addPeak(
            chrom=chrom,
            uuid=uuid,
            color=color,
        )

    def removePeak(
        self,
        uuid: 'EnsembleUUID',
    ):
        """
        Wrapper around ChromPlotWidget.removePeak
        """
        self.chromPlotWidget.removePeak(uuid)

    def clearPeaks(
        self,
    ):
        """
        Wrapper around ChromPlotWidget.clearPeaks
        """
        self.chromPlotWidget.clearPeaks()

    def setFprintArray(
        self,
        array: np.ndarray[float],
        descriptors: list[str],
    ):
        self.fprintPlotWidget.setFPrint(
            array,
            descriptors,
        )


    def on_chromatogram_hovered(
        self,
        pos: QtCore.QPointF,
    ):
        # Called when user hovers over a chromatogram
        self.sigChromatogramHovered.emit(
            self.UUID,
            pos,
        )


    def on_chromatogram_leaved(
        self,
    ):
        # Called when user stops hovering over a chromatogram
        self.sigChromatogramLeaved.emit(
            self.UUID
        )


    def on_chromatogram_clicked(
        self,
    ):
        """
        Called when a user clicks on a chromatogram
        :return:
        """
        self.sigChromatogramClicked.emit(
            self.UUID,
        )


    def on_fprint_hovered(
        self,
        idx: int,
    ):
        self.sigFPrintHovered.emit(
            self.UUID,
            idx,
        )

    def on_ensemble_peak_hovered(
        self,
        ensemble_uuid: 'EnsembleUUID',
        pos: QtCore.QPointF,
    ):
        """Propagate ensemble peak hover event with SampleUUID context"""
        self.sigEnsemblePeakHovered.emit(
            self.UUID,
            ensemble_uuid,
            pos,
        )

    def on_ensemble_peak_leaved(self):
        """Propagate ensemble peak leave event with SampleUUID context"""
        self.sigEnsemblePeakLeaved.emit(self.UUID)

    def on_ensemble_peak_clicked(
        self,
        ensemble_uuid: 'EnsembleUUID',
        button: int,
    ):
        """Propagate ensemble peak click event with SampleUUID context"""
        self.sigEnsemblePeakClicked.emit(
            self.UUID,
            ensemble_uuid,
            button,
        )


    def setFprintLabel(
        self,
        text: str,
    ):
        self.fprintPlotWidget.setLabel(text)


    def setSliderSelector(
        self,
        xpos: float,
    ):
        """
        Wrapped around `ChromPlotWidget.setSliderSelector`
        """
        self.chromPlotWidget.setSliderSelector(xpos=xpos)


    def setSliderSelectorVisible(
        self,
        visible: bool,
    ):
        slider: pg.InfiniteLine = self.chromPlotWidget.pi.slider_selector

        if not slider:
            return

        slider.setVisible(visible)


    def setSelectionIndicator(
        self,
        xpos: float,
    ):
        """
        Wrapper around chromPlotWidget -> chromPlotItem
        """
        self.chromPlotWidget.setSelectionIndicator(xpos)


    def setSelectionIndicatorVisible(
        self,
        visible: bool,
    ):
        self.chromPlotWidget.setSelectionIndicatorVisible(visible)

