"""
The pyqtgraph PlotItem + ViewBox for the alignment feature-map.

Owns the strips graphics object and the hover/selection overlay indicators,
turns raw hover/click targets into the three selectable entity types
(ensemble / analyte / sample) and re-emits them upward.
"""
from typing import Optional, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtGui import QColor, QPen, QBrush

from gui.views.alignment_viewer.layout import HoverTarget, build_lane_map
from gui.views.alignment_viewer.strips_item import EnsembleStripsItem
from gui.views.alignment_viewer.indicators import (
    RectIndicator,
    AnalyteIndicator,
    LaneIndicator,
)

if TYPE_CHECKING:
    from core.data_structs import DataRegistry
    from core.data_structs.alignment import EnsembleAlignment
    from gui.views.alignment_viewer.params import RenderParams


def _cosmetic_pen(color: QColor, width: int) -> QPen:
    pen = pg.mkPen(color, width=width)
    pen.setCosmetic(True)
    return pen


class AlignmentViewBox(pg.ViewBox):
    """ViewBox whose scroll wheel zooms the RT (x) axis only."""

    def wheelEvent(self, ev, axis=None):
        if axis == 1:  # let explicit y-axis wheel behave normally
            return super().wheelEvent(ev, axis=axis)
        s = 1.02 ** (ev.delta() * self.state['wheelScaleFactor'])
        center = pg.Point(
            pg.functions.invertQTransform(self.childGroup.transform())
            .map(ev.pos())
        )
        self._resetTarget()
        self.scaleBy([s, 1.0], center=center)
        ev.accept()
        self.sigRangeChangedManually.emit(self.state['mouseEnabled'])


class AlignmentPlotItem(pg.PlotItem):
    """Feature-map plot for a single EnsembleAlignment."""

    sigEnsembleSelected = QtCore.pyqtSignal(object)   # Ensemble
    sigAnalyteSelected = QtCore.pyqtSignal(object)    # AlignedAnalyte
    sigSampleSelected = QtCore.pyqtSignal(object)     # SampleUUID
    sigEnsembleActivated = QtCore.pyqtSignal(object)  # Ensemble (double-click)
    sigContextRequested = QtCore.pyqtSignal(object, object)  # HoverTarget, QPoint

    def __init__(self, *args, **kwargs):
        super().__init__(*args, viewBox=AlignmentViewBox(), **kwargs)
        self.getViewBox().invertY(True)      # lane 0 at the top
        self.getViewBox().setMenuEnabled(False)
        self.setMouseEnabled(x=True, y=True)

        self._strips: Optional[EnsembleStripsItem] = None

        # Hover overlays (dimmer) + selection overlays (brighter).
        self._hover_rect = RectIndicator(_cosmetic_pen(QColor(255, 255, 255, 200), 2))
        self._hover_analyte = AnalyteIndicator(_cosmetic_pen(QColor(255, 255, 255, 160), 2))
        self._hover_lane = LaneIndicator(QBrush(QColor(255, 255, 255, 22)))
        self._sel_rect = RectIndicator(_cosmetic_pen(QColor(255, 255, 255, 255), 3))
        self._sel_analyte = AnalyteIndicator(_cosmetic_pen(QColor(255, 255, 255, 220), 3))
        self._sel_lane = LaneIndicator(QBrush(QColor(255, 255, 255, 45)))

    # -- population ---------------------------------------------------------

    def set_alignment(
        self,
        alignment: 'EnsembleAlignment',
        data_registry: 'DataRegistry',
        params: 'RenderParams',
        preserve_view: bool = False,
    ):
        prev_range = self.getViewBox().viewRange() if preserve_view else None

        self.clear()
        self._strips = None

        if alignment.sample_count == 0:
            return

        lane_map = build_lane_map(alignment, data_registry)

        # Y axis = sample lanes.
        ticks = [
            (y, lane_map.name_of_sample[u][-18:])
            for u, y in lane_map.y_of_sample.items()
        ]
        left = self.getAxis('left')
        left.setWidth(130)
        left.setTicks([ticks])
        self.showGrid(x=True, y=False, alpha=0.15)

        strips = EnsembleStripsItem(alignment, data_registry, lane_map, params)
        self.addItem(strips)
        self._strips = strips

        for item in (
            self._hover_lane, self._sel_lane,          # behind strips
            self._hover_analyte, self._sel_analyte,
            self._hover_rect, self._sel_rect,          # on top
        ):
            self.addItem(item)

        strips.sigHovered.connect(self._on_hover)
        strips.sigSelected.connect(self._on_select)
        strips.sigActivated.connect(self._on_activate)
        strips.sigContextRequested.connect(self.sigContextRequested)

        self.getViewBox().setLimits(yMin=-1, yMax=alignment.sample_count)
        if prev_range is not None:
            (px0, px1), (py0, py1) = prev_range
            self.setXRange(px0, px1, padding=0)
            self.setYRange(py0, py1, padding=0)
        else:
            x0, x1 = strips.x_range
            pad = 0.02 * (x1 - x0) if x1 > x0 else 1.0
            self.setXRange(x0 - pad, x1 + pad, padding=0)
            self.setYRange(-0.5, alignment.sample_count - 0.5, padding=0)

    def clear_alignment(self):
        self.clear()
        self._strips = None

    # -- hover / selection --------------------------------------------------

    def _on_hover(self, target: Optional[HoverTarget]):
        self._apply_indicators(target, self._hover_rect,
                               self._hover_analyte, self._hover_lane)

    def _on_select(self, target: Optional[HoverTarget]):
        self._apply_indicators(target, self._sel_rect,
                               self._sel_analyte, self._sel_lane)
        if target is None:
            return
        if target.kind == 'ensemble':
            self.sigEnsembleSelected.emit(target.ensemble)
        elif target.kind == 'analyte':
            self.sigAnalyteSelected.emit(target.analyte)
        elif target.kind == 'sample':
            self.sigSampleSelected.emit(target.sample_uuid)

    def _on_activate(self, target: Optional[HoverTarget]):
        self._on_select(target)
        if target is not None and target.kind == 'ensemble' and target.ensemble:
            self.sigEnsembleActivated.emit(target.ensemble)

    def _apply_indicators(
        self,
        target: Optional[HoverTarget],
        rect_ind: RectIndicator,
        analyte_ind: AnalyteIndicator,
        lane_ind: LaneIndicator,
    ):
        rect_ind.set_rect(None)
        analyte_ind.clear()
        lane_ind.set_lane(None)
        if target is None or self._strips is None:
            return

        if target.kind == 'ensemble':
            rect_ind.set_rect(
                self._strips.rect_for(target.analyte_id, target.sample_uuid)
            )
        elif target.kind == 'analyte':
            analyte_ind.set_target(
                self._strips.polyline_for(target.analyte_id),
                self._strips.rects_for_analyte(target.analyte_id),
            )
        elif target.kind == 'sample':
            span = self._strips.lane_span(target.sample_uuid)
            if span is not None:
                y, x_range = span
                lane_ind.set_lane(y, x_range)
