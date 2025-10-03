import pyqtgraph as pg
import numpy as np
from PyQt5.QtCore import QPointF, pyqtSignal
from PyQt5.QtWidgets import QLabel

from typing import Iterable, Optional

class FPrintWidget(pg.PlotWidget):
    """
    Custom PlotWidget for displaying activity fingerprints
    """
    sigFPrintHovered = pyqtSignal(
        int, # FPrint idx
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setBackground(None)
        self.pi: pg.PlotItem = self.getPlotItem()

        self._descriptors: list[str] = []
        self._array: np.ndarray = np.empty(1)

        # ImageItem for displaying fingerprint
        self.ImageItem = pg.ImageItem()
        self.ImageItem.setColorMap('CET-D1A')
        self.ImageItem.setLevels(
            (-0.5, 0.5)
        )
        self.addItem(self.ImageItem)

        # Floating label for displaying info
        self.floating_label = QLabel(self)
        self.floating_label.setStyleSheet(
            "QLabel { color: black; background-color: rgba(225, 225, 225, 128) }"
        )
        self.floating_label.move(10, 5)
        self.floating_label.raise_()

        self._configure_plotitem()


    def _configure_plotitem(self):
        """
        Set default plot configs
        :return:
        """
        # Hide Y-axis and X-axis
        x_axis: pg.AxisItem = self.pi.getAxis('bottom')
        y_axis: pg.AxisItem = self.pi.getAxis('left')
        for axis in (x_axis, y_axis):
            axis.setVisible(False)

        # Set maximum view limits
        self.pi.vb.setLimits(
            xMin=0,
            xMax=len(self._array),
            yMin=0,
            yMax=1,
        )


    def mouseMoveEvent(self, ev):
        # Call original parent behaviour
        super().mouseMoveEvent(ev)

        # Emit mouse location in scene coordinates
        pos: QPointF = ev.pos()
        if not self.sceneBoundingRect().contains(pos):
            return

        vb: pg.ViewBox = self.getViewBox()
        fprint_idx: int = int(vb.mapSceneToView(pos).x() // 1)

        self.sigFPrintHovered.emit(
              int(fprint_idx),
        )


    def setFPrint(
        self,
        array: np.ndarray[float],
        descriptors: list[str],
    ):
        """
        Given an iterable collection of float values, sets the
        fingerprint displayed in this widget
        :param descriptors:
        :param array:
        :return:
        """
        if len(descriptors) != len(array):
            raise ValueError(
                f"fprint array and descriptors have mismatched lengths. \n"
                f"(descriptors: {descriptors},\n"
                f"array: {array})"
            )

        self._array = np.array(array).reshape(-1, 1)
        self._descriptors = descriptors
        self.updateImage()

    def updateImage(self):
        """
        Given a numpy array, plots a horizontal strip of coloured blocks
        :return:
        """
        self.ImageItem.clear()
        self._configure_plotitem()

        self.ImageItem.setImage(
            self._array
        )


    def setLabel(
        self,
        text: str,
    ):
        self.floating_label.setText(text)
        self.floating_label.raise_()

        # Force refresh of the widget stack
        self.floating_label.setParent(None)
        self.floating_label.setParent(self)
        self.floating_label.show()


