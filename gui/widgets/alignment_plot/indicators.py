"""
Overlay graphics items that highlight the current hover / selection without
touching the base strip drawing (colour is never mutated — highlights are an
extra layer drawn on top).
"""
from typing import Optional

import pyqtgraph as pg
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QColor, QPen, QPolygonF


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


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of (lo, hi) intervals, sorted (so translucent bands don't stack)."""
    merged: list[list[float]] = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


class BandIndicator(pg.ItemGroup):
    """
    Translucent bands spanning the whole view (LinearRegionItems): horizontal
    (sample lanes, by y) or vertical (columns, by x). Add it with
    `ignoreBounds=True`.
    """

    def __init__(self, vertical: bool):
        super().__init__()
        self._orientation = 'vertical' if vertical else 'horizontal'
        self._bands: list[tuple[float, float, QColor]] = []   # merged
        self._regions: list[pg.LinearRegionItem] = []
        self.setZValue(-10)  # behind everything else

    def set_bands(self, bands: list[tuple[float, float, QColor]]):
        """(lo, hi, colour) intervals; overlapping ones of a colour are merged."""
        by_color: dict = {}
        for lo, hi, color in bands:
            by_color.setdefault(color.rgba(), (color, []))[1].append((lo, hi))
        self._bands = [
            (lo, hi, color)
            for color, intervals in by_color.values()
            for lo, hi in merge_intervals(intervals)
        ]
        for region in self._regions:
            region.setParentItem(None)
            if region.scene() is not None:
                region.scene().removeItem(region)
        self._regions = []
        for lo, hi, color in self._bands:
            region = pg.LinearRegionItem(
                (lo, hi), self._orientation, brush=color,
                pen=pg.mkPen(None), movable=False,
            )
            region.setParentItem(self)
            self._regions.append(region)
