"""
The single high-performance graphics object that draws every ensemble strip
and every analyte connecting-line in one precomputed ``QPicture``, then
hit-tests the cursor against them.

Modelled on ms2analyte's ``ReplicateAnalyteStacksItem`` — one paint call for
the whole plot, with cached rects / centre-points / polylines for cheap
hover + click resolution.
"""
from typing import Optional, TYPE_CHECKING

import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import QRectF, QPointF, QLineF
from PyQt5.QtGui import QPicture, QPainter, QPolygonF, QColor

from gui.widgets.alignment_plot.layout import (
    LaneMap,
    HoverTarget,
    XAccessor,
    ensemble_rt,
    resolve_ensemble,
    strip_geometry,
    scale_line_opacity,
    analyte_color,
)

if TYPE_CHECKING:
    from core.interfaces.data_sources import SampleDataSource
    from core.data_structs.alignment import EnsembleAlignment
    from gui.widgets.alignment_plot.params import RenderParams


# Pixel radius for the "near a strip centre" fallback hit-test (helps when
# zoomed out and strips are sub-pixel).
NEAR_CENTER_PX_SQ = 225.0     # 15 px, squared
# Pixel half-width of the cursor's polyline hit test.
LINE_HIT_PX = 4.0


class EnsembleStripsItem(pg.GraphicsObject):
    """
    Draws all strips + analyte lines; emits hover / click / context signals
    carrying a :class:`HoverTarget`.
    """
    sigHovered = QtCore.pyqtSignal(object)              # HoverTarget | None
    sigSelected = QtCore.pyqtSignal(object, bool)       # HoverTarget | None, additive (Ctrl)
    sigActivated = QtCore.pyqtSignal(object)            # HoverTarget | None
    sigContextRequested = QtCore.pyqtSignal(object, object)  # target, QPoint

    def __init__(
        self,
        alignment: 'EnsembleAlignment',
        data_source: 'SampleDataSource',
        lane_map: LaneMap,
        params: 'RenderParams',
        x_of: XAccessor = ensemble_rt,
        width_scale: float = 1.0,
        stagger: bool = True,
    ):
        super().__init__()
        self._alignment = alignment
        self._data_source = data_source
        self._lanes = lane_map
        self._params = params
        self._x_of = x_of
        self._width_scale = width_scale
        self._stagger = stagger

        # Hit-test caches, keyed by analyte index.
        self.rects: dict[int, dict] = {}          # aid -> {sample_uuid: QRectF}
        self.rect_centers: dict[int, dict] = {}   # aid -> {sample_uuid: QPointF}
        self.polylines: dict[int, QPolygonF] = {} # aid -> polyline (multi only)
        self._ensembles: dict[tuple, object] = {} # (aid, sample_uuid) -> Ensemble
        self._analytes: dict[int, object] = {}    # aid -> AlignedAnalyte

        self._x_range: tuple[float, float] = (0.0, 1.0)
        self._picture = QPicture()
        self._hovered: Optional[HoverTarget] = None

        self._build_picture()

    # -- geometry / painting ------------------------------------------------

    @property
    def x_range(self) -> tuple[float, float]:
        return self._x_range

    def _build_picture(self):
        painter = QPainter(self._picture)
        min_x, max_x = None, None

        for aid, analyte in enumerate(self._alignment.analytes):
            self._analytes[aid] = analyte
            samples = [
                u for u in self._alignment.sample_uuids
                if u in analyte.ensemble_map
            ]
            is_singleton = len(samples) < 2
            color = analyte_color(aid, is_singleton)

            painter.setPen(pg.mkPen(color))
            fill = QColor(color)
            fill.setAlpha(110)
            painter.setBrush(pg.mkBrush(fill))

            self.rects[aid] = {}
            self.rect_centers[aid] = {}
            centers = []
            max_intsy = 0.0

            for stagger_idx, sample_uuid in enumerate(samples):
                ensemble = resolve_ensemble(
                    analyte, sample_uuid, self._data_source
                )
                if ensemble is None:
                    continue
                cx = self._x_of(aid, analyte, ensemble)
                if cx is None:
                    continue

                lane_y = self._lanes.y_of_sample[sample_uuid]
                rect, center = strip_geometry(
                    ensemble, cx, lane_y, aid, self._params,
                    self._width_scale, self._stagger,
                )
                painter.drawRect(rect)

                self.rects[aid][sample_uuid] = rect
                self.rect_centers[aid][sample_uuid] = center
                self._ensembles[(aid, sample_uuid)] = ensemble
                centers.append(center)
                max_intsy = max(max_intsy, ensemble.base_intsy)

                min_x = rect.left() if min_x is None else min(min_x, rect.left())
                max_x = rect.right() if max_x is None else max(max_x, rect.right())

            # Connecting line only makes sense across >= 2 samples. Its
            # opacity scales with the analyte's brightest ensemble.
            if len(centers) >= 2:
                line_color = QColor(color)
                line_color.setAlpha(scale_line_opacity(max_intsy, self._params))
                line_pen = pg.mkPen(line_color)
                line_pen.setWidth(0)  # cosmetic hairline
                painter.setPen(line_pen)
                polyline = QPolygonF(sorted(centers, key=lambda p: p.y()))
                painter.drawPolyline(polyline)
                self.polylines[aid] = polyline

        painter.end()
        if min_x is not None:
            self._x_range = (min_x, max_x)

    def paint(self, p, *args):
        p.drawPicture(0, 0, self._picture)

    def boundingRect(self):
        rect = self._picture.boundingRect()
        return QRectF(rect) if rect.isValid() else QRectF()

    # -- hit testing --------------------------------------------------------

    def _smallest_rect_at(self, point: QPointF):
        """Return (aid, sample_uuid) of the *smallest* rect containing point."""
        best = None
        best_area = None
        for aid, by_sample in self.rects.items():
            for sample_uuid, rect in by_sample.items():
                if rect.contains(point):
                    area = rect.width() * rect.height()
                    if best_area is None or area < best_area:
                        best, best_area = (aid, sample_uuid), area
        return best

    def _polyline_at(self, point: QPointF) -> Optional[int]:
        vb = self.getViewBox()
        if vb is None:
            return None
        px, _ = vb.viewPixelSize()
        tol_sq = (LINE_HIT_PX * px) ** 2
        for aid, polygon in self.polylines.items():
            if _point_near_polyline(point, polygon, tol_sq):
                return aid
        return None

    def _near_center(self, point: QPointF):
        vb = self.getViewBox()
        if vb is None:
            return None
        px, py = vb.viewPixelSize()
        mx, my = point.x() / px, point.y() / py
        best = None
        best_d = NEAR_CENTER_PX_SQ
        for aid, by_sample in self.rect_centers.items():
            for sample_uuid, c in by_sample.items():
                dx = c.x() / px - mx
                dy = c.y() / py - my
                d = dx * dx + dy * dy
                if d < best_d:
                    best, best_d = (aid, sample_uuid), d
        return best

    def _lane_at(self, y: float):
        for sample_uuid, ly in self._lanes.y_of_sample.items():
            if abs(y - ly) <= 0.5:
                return sample_uuid
        return None

    def _target_at(self, pos) -> Optional[HoverTarget]:
        point = QPointF(pos)

        hit = self._smallest_rect_at(point)
        if hit is None:
            hit = self._near_center(point)
        if hit is not None:
            return self.ensemble_target(*hit)

        aid = self._polyline_at(point)
        if aid is not None:
            return self.analyte_target(aid)

        sample_uuid = self._lane_at(point.y())
        if sample_uuid is not None:
            return HoverTarget(kind='sample', sample_uuid=sample_uuid)

        return None

    # -- events -------------------------------------------------------------

    def hoverEvent(self, ev):
        if ev.exit:
            self._set_hover(None)
            return
        self._set_hover(self._target_at(ev.pos()))

    def _set_hover(self, target: Optional[HoverTarget]):
        if target != self._hovered:
            self._hovered = target
            self.sigHovered.emit(target)

    def mouseClickEvent(self, ev):
        ev.accept()
        target = self._hovered
        if ev.button() == QtCore.Qt.RightButton:
            self.sigContextRequested.emit(target, ev.screenPos().toPoint())
        elif ev.double():
            self.sigActivated.emit(target)
        else:
            additive = bool(ev.modifiers() & QtCore.Qt.ControlModifier)
            self.sigSelected.emit(target, additive)

    # -- lookups for indicators / consumers ---------------------------------

    def rect_for(self, aid: int, sample_uuid) -> Optional[QRectF]:
        return self.rects.get(aid, {}).get(sample_uuid)

    def polyline_for(self, aid: int) -> Optional[QPolygonF]:
        return self.polylines.get(aid)

    def rects_for_analyte(self, aid: int) -> list:
        return list(self.rects.get(aid, {}).values())

    def ensemble_for(self, aid: int, sample_uuid):
        return self._ensembles.get((aid, sample_uuid))

    def analyte_for(self, aid: int):
        return self._analytes.get(aid)

    def center_for(self, aid: int, sample_uuid) -> Optional[QPointF]:
        return self.rect_centers.get(aid, {}).get(sample_uuid)

    def analyte_center(self, aid: int) -> Optional[QPointF]:
        """Mean of an analyte's strip centres."""
        centers = list(self.rect_centers.get(aid, {}).values())
        if not centers:
            return None
        return QPointF(
            sum(c.x() for c in centers) / len(centers),
            sum(c.y() for c in centers) / len(centers),
        )

    def ensemble_target(self, aid: int, sample_uuid) -> HoverTarget:
        return HoverTarget(
            kind='ensemble',
            analyte_id=aid,
            sample_uuid=sample_uuid,
            ensemble=self._ensembles.get((aid, sample_uuid)),
            analyte=self._analytes.get(aid),
        )

    def analyte_targets(self, analyte_uuids) -> list[HoverTarget]:
        """Targets for the given AnalyteUUIDs that have strips drawn."""
        aid_of = {a.uuid: aid for aid, a in self._analytes.items()}
        return [
            self.analyte_target(aid_of[u]) for u in analyte_uuids
            if u in aid_of and self.rects.get(aid_of[u])
        ]

    def analyte_target(self, aid: int) -> HoverTarget:
        return HoverTarget(
            kind='analyte',
            analyte_id=aid,
            analyte=self._analytes.get(aid),
        )

    def lane_span(self, sample_uuid) -> Optional[tuple]:
        y = self._lanes.y_of_sample.get(sample_uuid)
        if y is None:
            return None
        return y, self._x_range


def _point_near_polyline(
    point: QPointF,
    polygon: QPolygonF,
    tol_sq: float,
) -> bool:
    """True if point lies within sqrt(tol_sq) of any polyline segment."""
    for i in range(polygon.size() - 1):
        seg = QLineF(polygon[i], polygon[i + 1])
        lvx = seg.p2().x() - seg.p1().x()
        lvy = seg.p2().y() - seg.p1().y()
        pvx = point.x() - seg.p1().x()
        pvy = point.y() - seg.p1().y()
        length_sq = lvx * lvx + lvy * lvy
        if length_sq == 0:
            continue
        t = max(0.0, min(1.0, (pvx * lvx + pvy * lvy) / length_sq))
        cx = seg.p1().x() + t * lvx
        cy = seg.p1().y() + t * lvy
        dx = point.x() - cx
        dy = point.y() - cy
        if dx * dx + dy * dy <= tol_sq:
            return True
    return False
