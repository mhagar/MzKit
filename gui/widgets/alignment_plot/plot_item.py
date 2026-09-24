"""
The pyqtgraph PlotItem + ViewBox for the alignment feature-map.

Owns the strips graphics object, the hover/selection overlay indicators and
the selection state. The selection is an ordered list of ensemble and/or
analyte targets (the first is the "anchor" for comparisons):

 - click a strip / line        -> select just that ensemble / analyte
 - Ctrl+click                  -> toggle it in/out of the selection
 - click a lane's background, or Esc -> clear
 - arrow keys                  -> step to a neighbouring ensemble / analyte
"""
from typing import Optional, TYPE_CHECKING

import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtGui import QColor, QPen, QBrush

from gui.widgets.alignment_plot.layout import (
    HoverTarget,
    XAccessor,
    build_lane_map,
    ensemble_rt,
)
from gui.widgets.alignment_plot.strips_item import EnsembleStripsItem
from gui.widgets.alignment_plot.indicators import (
    OutlineIndicator,
    LaneIndicator,
)

if TYPE_CHECKING:
    from core.data_structs.alignment import EnsembleAlignment
    from core.interfaces.data_sources import SampleDataSource
    from gui.widgets.alignment_plot.params import RenderParams


def _cosmetic_pen(color: QColor, width: int) -> QPen:
    pen = pg.mkPen(color, width=width)
    pen.setCosmetic(True)
    return pen


def target_key(target: HoverTarget) -> tuple:
    """Identity of a selectable target, independent of the objects it holds."""
    return target.kind, target.analyte_id, target.sample_uuid


class AlignmentViewBox(pg.ViewBox):
    """
    ViewBox whose scroll wheel zooms the x axis only.

    NB: must not accept left clicks - pyqtgraph offers them to the ViewBox
    before the strips item, so accepting here would swallow every selection.
    """

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

    sigHovered = QtCore.pyqtSignal(object)            # HoverTarget | None
    sigSelectionChanged = QtCore.pyqtSignal(object)   # list[HoverTarget]
    sigEnsembleActivated = QtCore.pyqtSignal(object)  # Ensemble (double-click)
    sigContextRequested = QtCore.pyqtSignal(object, object)  # HoverTarget, QPoint

    def __init__(self, *args, **kwargs):
        super().__init__(*args, viewBox=AlignmentViewBox(), **kwargs)
        self.getViewBox().invertY(True)      # lane 0 at the top
        self.getViewBox().setMenuEnabled(False)
        self.setMouseEnabled(x=True, y=True)

        self._strips: Optional[EnsembleStripsItem] = None
        self._alignment_uuid: Optional[int] = None
        self._selection: list[HoverTarget] = []

        # Hover overlays (dimmer) + selection overlay (brighter).
        self._hover_outline = OutlineIndicator(
            _cosmetic_pen(QColor(255, 255, 255, 170), 2), z_value=90,
        )
        self._hover_lane = LaneIndicator(QBrush(QColor(255, 255, 255, 22)))
        self._sel_outline = OutlineIndicator(
            _cosmetic_pen(QColor(255, 255, 255, 255), 3), z_value=100,
        )

    # -- population ---------------------------------------------------------

    def set_alignment(
        self,
        alignment: 'EnsembleAlignment',
        data_source: 'SampleDataSource',
        params: 'RenderParams',
        preserve_view: bool = False,
        x_of: XAccessor = ensemble_rt,
    ):
        """
        (Re)draw an alignment. With `preserve_view`, the view range and -
        if it is the same alignment - the selection are kept.
        """
        prev_range = self.getViewBox().viewRange() if preserve_view else None
        same_alignment = alignment.uuid == self._alignment_uuid
        prev_keys = (
            [target_key(t) for t in self._selection]
            if preserve_view and same_alignment else []
        )

        self.clear()
        self._strips = None
        self._selection = []
        self._alignment_uuid = alignment.uuid

        if alignment.sample_count == 0:
            self.sigSelectionChanged.emit([])
            return

        lane_map = build_lane_map(alignment, data_source)

        # Y axis = sample lanes.
        ticks = [
            (y, lane_map.name_of_sample[u][-18:])
            for u, y in lane_map.y_of_sample.items()
        ]
        left = self.getAxis('left')
        left.setWidth(130)
        left.setTicks([ticks])
        self.showGrid(x=True, y=False, alpha=0.15)

        strips = EnsembleStripsItem(
            alignment, data_source, lane_map, params, x_of,
        )
        self.addItem(strips)
        self._strips = strips

        for item in (
            self._hover_lane,                      # behind strips
            self._hover_outline, self._sel_outline,
        ):
            self.addItem(item)

        strips.sigHovered.connect(self._on_hover)
        strips.sigSelected.connect(self._on_click)
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

        self._set_selection(
            [t for t in (self._target_from_key(k) for k in prev_keys) if t]
        )

    def clear_alignment(self):
        self.clear()
        self._strips = None
        self._alignment_uuid = None
        self._set_selection([])

    # -- selection ----------------------------------------------------------

    @property
    def selection(self) -> list[HoverTarget]:
        return list(self._selection)

    def clear_selection(self):
        self._set_selection([])

    def select(self, target: Optional[HoverTarget], additive: bool = False):
        """
        Select an ensemble / analyte target. `additive` toggles it in or
        out of the current selection instead of replacing it. Lane (sample)
        targets and None clear the selection (unless additive).
        """
        if target is None or target.kind not in ('ensemble', 'analyte'):
            if not additive:
                self._set_selection([])
            return

        if not additive:
            self._set_selection([target])
            return

        key = target_key(target)
        kept = [t for t in self._selection if target_key(t) != key]
        if len(kept) == len(self._selection):
            kept.append(target)
        self._set_selection(kept)

    def _set_selection(self, targets: list[HoverTarget]):
        self._selection = list(targets)
        self._update_selection_indicator()
        self.sigSelectionChanged.emit(list(self._selection))

    def _target_from_key(self, key: tuple) -> Optional[HoverTarget]:
        if self._strips is None:
            return None
        kind, aid, sample_uuid = key
        if kind == 'ensemble' and self._strips.rect_for(aid, sample_uuid):
            return self._strips.ensemble_target(aid, sample_uuid)
        if kind == 'analyte' and self._strips.analyte_for(aid) is not None:
            return self._strips.analyte_target(aid)
        return None

    # -- hover / click ------------------------------------------------------

    def _on_hover(self, target: Optional[HoverTarget]):
        self._hover_outline.clear()
        self._hover_lane.set_lane(None)
        if target is not None and self._strips is not None:
            if target.kind == 'sample':
                span = self._strips.lane_span(target.sample_uuid)
                if span is not None:
                    self._hover_lane.set_lane(*span)
            else:
                self._hover_outline.set_shapes(*self._shapes_for([target]))
        self.sigHovered.emit(target)

    def _on_click(self, target: Optional[HoverTarget], additive: bool):
        self.select(target, additive)

    def _on_activate(self, target: Optional[HoverTarget]):
        if target is not None and target.kind == 'ensemble' and target.ensemble:
            self.select(target)
            self.sigEnsembleActivated.emit(target.ensemble)

    def _shapes_for(self, targets: list[HoverTarget]) -> tuple[list, list]:
        polylines, rects = [], []
        for t in targets:
            if t.kind == 'ensemble':
                rect = self._strips.rect_for(t.analyte_id, t.sample_uuid)
                if rect is not None:
                    rects.append(rect)
            elif t.kind == 'analyte':
                polyline = self._strips.polyline_for(t.analyte_id)
                if polyline is not None:
                    polylines.append(polyline)
                rects.extend(self._strips.rects_for_analyte(t.analyte_id))
        return polylines, rects

    def _update_selection_indicator(self):
        if self._strips is None:
            self._sel_outline.clear()
            return
        self._sel_outline.set_shapes(*self._shapes_for(self._selection))

    # -- keyboard navigation ------------------------------------------------

    def handle_key(self, key: int) -> bool:
        """
        Arrow-key navigation from the most recently selected target.
        Returns True if the key was handled.

        Ensemble: Left/Right -> neighbouring ensemble in the same lane;
                  Up/Down    -> same analyte in the next lane that has it.
        Analyte:  Left/Right -> neighbouring multi-sample analyte;
                  Up/Down    -> its top-most / bottom-most member ensemble.
        """
        if key == QtCore.Qt.Key_Escape:
            self.clear_selection()
            return True

        if key not in (
            QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
            QtCore.Qt.Key_Up, QtCore.Qt.Key_Down,
        ):
            return False
        if self._strips is None or not self._selection:
            return True

        current = self._selection[-1]
        step = -1 if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Up) else 1
        horizontal = key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right)

        if current.kind == 'ensemble':
            nxt = (
                self._step_in_lane(current, step) if horizontal
                else self._step_across_lanes(current, step)
            )
        else:
            nxt = (
                self._step_analyte(current, step) if horizontal
                else self._analyte_member(current, first=step < 0)
            )

        if nxt is not None:
            self.select(nxt)
            self._ensure_visible(nxt)
        return True

    def _step_in_lane(self, current: HoverTarget, step: int):
        strips = self._strips
        lane = [
            (c.x(), aid)
            for aid, by_sample in strips.rect_centers.items()
            for su, c in by_sample.items() if su == current.sample_uuid
        ]
        lane.sort()
        aids = [aid for _, aid in lane]
        if current.analyte_id not in aids:
            return None
        idx = aids.index(current.analyte_id) + step
        if not 0 <= idx < len(aids):
            return None
        return strips.ensemble_target(aids[idx], current.sample_uuid)

    def _step_across_lanes(self, current: HoverTarget, step: int):
        strips = self._strips
        members = strips.rect_centers.get(current.analyte_id, {})
        ordered = sorted(members, key=lambda su: members[su].y())
        if current.sample_uuid not in ordered:
            return None
        idx = ordered.index(current.sample_uuid) + step
        if not 0 <= idx < len(ordered):
            return None
        return strips.ensemble_target(current.analyte_id, ordered[idx])

    def _step_analyte(self, current: HoverTarget, step: int):
        strips = self._strips
        ordered = sorted(
            strips.polylines,
            key=lambda aid: strips.analyte_center(aid).x(),
        )
        if current.analyte_id not in ordered:
            return None
        idx = ordered.index(current.analyte_id) + step
        if not 0 <= idx < len(ordered):
            return None
        return strips.analyte_target(ordered[idx])

    def _analyte_member(self, current: HoverTarget, first: bool):
        strips = self._strips
        members = strips.rect_centers.get(current.analyte_id, {})
        if not members:
            return None
        pick = min if first else max
        sample_uuid = pick(members, key=lambda su: members[su].y())
        return strips.ensemble_target(current.analyte_id, sample_uuid)

    def _ensure_visible(self, target: HoverTarget):
        """Pan (without zooming) so the target's centre is in view."""
        if target.kind == 'ensemble':
            point = self._strips.center_for(target.analyte_id, target.sample_uuid)
        else:
            point = self._strips.analyte_center(target.analyte_id)
        if point is None:
            return
        vb = self.getViewBox()
        (x0, x1), (y0, y1) = vb.viewRange()
        dx = 0.0 if x0 <= point.x() <= x1 else point.x() - 0.5 * (x0 + x1)
        dy = 0.0 if y0 <= point.y() <= y1 else point.y() - 0.5 * (y0 + y1)
        if dx or dy:
            vb.translateBy(x=dx, y=dy)
