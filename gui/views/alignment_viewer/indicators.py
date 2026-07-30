"""
Overlay graphics items that highlight the current hover / selection without
touching the base strip drawing (colour is never mutated — highlights are an
extra layer drawn on top).
"""
from typing import Optional

import pyqtgraph as pg
from PyQt5.QtCore import QRectF, QPointF
from PyQt5.QtGui import QPen, QPolygonF


class RectIndicator(pg.GraphicsObject):
    """Outlines a single strip (used for ensemble hover/selection)."""

    def __init__(self, pen: QPen):
        super().__init__()
        self._pen = pen
        self._rect: Optional[QRectF] = None
        self.setZValue(100)

    def set_rect(self, rect: Optional[QRectF]):
        self.prepareGeometryChange()
        self._rect = QRectF(rect) if rect is not None else None
        self.update()

    def paint(self, p, *args):
        if self._rect is not None:
            p.setPen(self._pen)
            p.drawRect(self._rect)

    def boundingRect(self):
        if self._rect is None:
            return QRectF()
        return self._rect.adjusted(-2, -2, 2, 2)


class AnalyteIndicator(pg.GraphicsObject):
    """Outlines a whole analyte: its connecting line + each of its strips."""

    def __init__(self, pen: QPen):
        super().__init__()
        self._pen = pen
        self._polyline: Optional[QPolygonF] = None
        self._rects: list[QRectF] = []
        self.setZValue(90)

    def set_target(self, polyline: Optional[QPolygonF], rects: list[QRectF]):
        self.prepareGeometryChange()
        self._polyline = QPolygonF(polyline) if polyline is not None else None
        self._rects = [QRectF(r) for r in rects]
        self.update()

    def clear(self):
        self.set_target(None, [])

    def paint(self, p, *args):
        p.setPen(self._pen)
        if self._polyline is not None:
            p.drawPolyline(self._polyline)
        for rect in self._rects:
            p.drawRect(rect)

    def boundingRect(self):
        rect = QRectF()
        if self._polyline is not None:
            rect = self._polyline.boundingRect()
        for r in self._rects:
            rect = rect.united(r) if rect.isValid() else QRectF(r)
        return rect.adjusted(-2, -2, 2, 2) if rect.isValid() else QRectF()


class LaneIndicator(pg.GraphicsObject):
    """Translucent band spanning a whole sample lane (sample selection)."""

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
