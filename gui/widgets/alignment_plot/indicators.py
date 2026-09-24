"""
Overlay graphics items that highlight the current hover / selection without
touching the base strip drawing (colour is never mutated — highlights are an
extra layer drawn on top).
"""
from typing import Optional

import pyqtgraph as pg
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QPen, QPolygonF


class OutlineIndicator(pg.GraphicsObject):
    """
    Outlines any number of strips and analyte lines (used for both the
    hovered target and the - possibly multiple - selected targets).
    """

    def __init__(self, pen: QPen, z_value: float = 100):
        super().__init__()
        self._pen = pen
        self._polylines: list[QPolygonF] = []
        self._rects: list[QRectF] = []
        self.setZValue(z_value)

    def set_shapes(
        self,
        polylines: list[QPolygonF],
        rects: list[QRectF],
    ):
        self.prepareGeometryChange()
        self._polylines = [QPolygonF(p) for p in polylines]
        self._rects = [QRectF(r) for r in rects]
        self.update()

    def clear(self):
        self.set_shapes([], [])

    def paint(self, p, *args):
        p.setPen(self._pen)
        for polyline in self._polylines:
            p.drawPolyline(polyline)
        for rect in self._rects:
            p.drawRect(rect)

    def boundingRect(self):
        rect = QRectF()
        for shape in [p.boundingRect() for p in self._polylines] + self._rects:
            rect = rect.united(shape) if rect.isValid() else QRectF(shape)
        return rect.adjusted(-2, -2, 2, 2) if rect.isValid() else QRectF()


class LaneIndicator(pg.GraphicsObject):
    """Translucent band spanning a whole sample lane (lane hover)."""

    def __init__(self, brush):
        super().__init__()
        self._brush = brush
        self._rect: Optional[QRectF] = None
        self.setZValue(-10)  # behind the strips

    def set_lane(self, y: Optional[float], x_range: tuple = (0.0, 1.0)):
        self.prepareGeometryChange()
        if y is None:
            self._rect = None
        else:
            x0, x1 = x_range
            self._rect = QRectF(x0, y - 0.5, x1 - x0, 1.0)
        self.update()

    def paint(self, p, *args):
        if self._rect is not None:
            p.setPen(pg.mkPen(None))
            p.setBrush(self._brush)
            p.drawRect(self._rect)

    def boundingRect(self):
        return QRectF(self._rect) if self._rect is not None else QRectF()
