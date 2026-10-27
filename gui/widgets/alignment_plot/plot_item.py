"""
The pyqtgraph PlotItem + ViewBox for the alignment feature-map.

Owns the strips graphics object, the hover/selection overlay indicators and
the SelectionModel (selection, focus and anchor; see selection.py), which
clicks and keys edit:

 - click an unselected strip / line -> select just that ensemble / analyte
 - click a selected one             -> focus it
 - Ctrl+click                       -> toggle it in/out of the selection
 - click a lane's background, or Esc -> clear the selection (not the anchor)
 - arrow keys -> one selected: step to a neighbouring ensemble / analyte;
                 several: move the focus through them
"""
from typing import Optional, TYPE_CHECKING

import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtGui import QColor, QPen

from gui.widgets.alignment_plot.layout import (
    ANCHOR_COLOR,
    CURRENT_COLOR,
    HoverTarget,
    XAccessor,
    build_lane_map,
    ensemble_rt,
)
from gui.widgets.alignment_plot.selection import (
    SelectionModel,
    display_order,
    target_key,
)
from gui.widgets.alignment_plot.strips_item import EnsembleStripsItem
from gui.widgets.alignment_plot.indicators import (
    BandIndicator,
    OutlineIndicator,
)

if TYPE_CHECKING:
    from core.data_structs.alignment import EnsembleAlignment
    from core.interfaces.data_sources import SampleDataSource
    from gui.widgets.alignment_plot.params import RenderParams


# Lane / column bands behind the selected items (grey reads on light and dark);
# only columns get the focus colour
SELECTION_BAND = QColor(128, 128, 128, 40)
FOCUS_BAND = QColor(CURRENT_COLOR.red(), CURRENT_COLOR.green(), CURRENT_COLOR.blue(), 45)


def _cosmetic_pen(color: QColor, width: int) -> QPen:
    pen = pg.mkPen(color, width=width)
    pen.setCosmetic(True)
    return pen


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
    sigEnsembleActivated = QtCore.pyqtSignal(object)  # Ensemble (double-click)
    sigContextRequested = QtCore.pyqtSignal(object, object)  # HoverTarget, QPoint
    # The selection's column bands, for the x-linked dendrogram / group track
    sigColumnsChanged = QtCore.pyqtSignal(object)     # list[(x0, x1, QColor)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, viewBox=AlignmentViewBox(), **kwargs)
        self.getViewBox().invertY(True)      # lane 0 at the top
        self.getViewBox().setMenuEnabled(False)
        self.setMouseEnabled(x=True, y=True)

        self._strips: Optional[EnsembleStripsItem] = None
        self._alignment_uuid: Optional[int] = None
        self.selection_model = SelectionModel(self)
        for signal in (
            self.selection_model.sigSelectionChanged,
            self.selection_model.sigFocusChanged,
            self.selection_model.sigAnchorChanged,
        ):
            signal.connect(lambda _: self._update_selection_indicator())

        # Hover overlays (dimmer) + selection overlay (brighter).
        self._hover_outline = OutlineIndicator(
            _cosmetic_pen(QColor(255, 255, 255, 170), 2), z_value=90,
        )
        self._lane_bands = BandIndicator(vertical=False)
        self._column_bands = BandIndicator(vertical=True)
        self._sel_outline = OutlineIndicator(
            _cosmetic_pen(QColor(255, 255, 255, 255), 3), z_value=100,
        )
        # The anchor, and the focus within a multi-selection
        self._anchor_outline = OutlineIndicator(
            _cosmetic_pen(ANCHOR_COLOR, 3), z_value=101,
        )
        self._focus_outline = OutlineIndicator(
            _cosmetic_pen(CURRENT_COLOR, 3), z_value=102,
        )

    # -- population ---------------------------------------------------------

    def set_alignment(
        self,
        alignment: 'EnsembleAlignment',
        data_source: 'SampleDataSource',
        params: 'RenderParams',
        preserve_range: bool = False,
        preserve_selection: bool = False,
        x_of: XAccessor = ensemble_rt,
        width_scale: float = 1.0,
        x_axis_visible: bool = True,
        stagger: bool = True,
        sample_order: Optional[list] = None,
    ):
        """
        (Re)draw an alignment. `preserve_range` keeps the view range;
        `preserve_selection` keeps the selection (and focus, anchor) if it
        is the same alignment; otherwise they are cleared. `x_of` / `width_scale` place the strips (RT by default);
        `x_axis_visible` shows/hides the bottom axis + x grid; `stagger`
        toggles the within-lane vertical stagger of strips; `sample_order`
        sets the lane order (default: by name, see `build_lane_map`).
        """
        prev_range = self.getViewBox().viewRange() if preserve_range else None
        preserve = preserve_selection and alignment.uuid == self._alignment_uuid

        self.clear()
        self._strips = None
        self._alignment_uuid = alignment.uuid

        if alignment.sample_count == 0:
            self.selection_model.clear()
            self._update_selection_indicator()
            return

        lane_map = build_lane_map(alignment, data_source, sample_order)

        # Y axis = sample lanes.
        ticks = [
            (y, lane_map.name_of_sample[u][-18:])
            for u, y in lane_map.y_of_sample.items()
        ]
        left = self.getAxis('left')
        left.setWidth(130)
        left.setTicks([ticks])
        self.showAxis('bottom', x_axis_visible)
        self.showGrid(x=x_axis_visible, y=False, alpha=0.15)

        strips = EnsembleStripsItem(
            alignment, data_source, lane_map, params, x_of, width_scale,
            stagger,
        )
        self.addItem(strips)
        self._strips = strips

        for item in (self._lane_bands, self._column_bands):   # behind strips
            self.addItem(item, ignoreBounds=True)
        for item in (
            self._hover_outline, self._sel_outline,
            self._anchor_outline, self._focus_outline,
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

        if preserve:
            self.selection_model.remap(self._target_from_key)
        else:
            self.selection_model.clear()
        self._update_selection_indicator()   # the overlays were re-added

    def clear_alignment(self):
        self.clear()
        self._strips = None
        self._alignment_uuid = None
        self.selection_model.clear()
        self._update_selection_indicator()

    # -- selection ----------------------------------------------------------

    def analyte_targets(self, analyte_uuids) -> list[HoverTarget]:
        """Analyte targets for the given AnalyteUUIDs that are on the map."""
        if self._strips is None:
            return []
        return self._strips.analyte_targets(analyte_uuids)

    def set_hover_analytes(self, analyte_uuids: Optional[list]):
        """
        Show the hover outline around these AnalyteUUIDs (e.g. a hovered
        group on the group track); None clears it. Normal map hover
        replaces it again on the next hover event.
        """
        self._hover_outline.clear()
        if analyte_uuids and self._strips is not None:
            self._hover_outline.set_shapes(
                *self._shapes_for(self.analyte_targets(analyte_uuids))
            )

    def select(self, target: Optional[HoverTarget], additive: bool = False):
        """
        A click on `target`: select just it, or focus it if it's already
        selected. `additive` (Ctrl) toggles it in or out of the selection
        instead. Lane (sample) targets and None clear the selection (unless
        additive).
        """
        model = self.selection_model
        if target is None or target.kind not in ('ensemble', 'analyte'):
            if not additive:
                model.clear(anchor=False)
        elif additive:
            model.toggle(target)
        elif model.contains(target):
            model.set_focus(target)
        else:
            model.set_selection([target])

    def reveal(self, kind: str, aid: int, sample_uuid=None) -> bool:
        """Select the target with this key and pan it into view; False if it isn't drawn."""
        target = self._target_from_key((kind, aid, sample_uuid))
        if target is None:
            return False
        self.selection_model.set_selection([target])
        self._ensure_visible(target)
        return True

    def ensemble_target(self, aid: int, sample_uuid) -> Optional[HoverTarget]:
        """The ensemble target for this analyte row + sample, if it's drawn."""
        return self._target_from_key(('ensemble', aid, sample_uuid))

    def _target_from_key(self, key: tuple) -> Optional[HoverTarget]:
        if self._strips is None:
            return None
        kind, aid, sample_uuid = key
        if kind == 'ensemble' and self._strips.rect_for(aid, sample_uuid):
            return self._strips.ensemble_target(aid, sample_uuid)
        if kind == 'analyte' and self._strips.rects_for_analyte(aid):
            return self._strips.analyte_target(aid)
        return None

    # -- hover / click ------------------------------------------------------

    def _on_hover(self, target: Optional[HoverTarget]):
        self._hover_outline.clear()
        if target is not None and self._strips is not None:
            if target.kind != 'sample':  # lanes get no hover outline
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
        outlines = (self._sel_outline, self._anchor_outline, self._focus_outline)
        if self._strips is None:
            for outline in outlines:
                outline.clear()
            self.sigColumnsChanged.emit([])
            return
        model = self.selection_model
        selection = model.selection
        # A lone selected item needs no focus outline (unless a member of it is focused)
        focus = model.focus if len(selection) > 1 or not model.lists(model.focus) else None
        for outline, targets in zip(outlines, (
            selection,
            [model.anchor] if model.anchor else [],
            [focus] if focus else [],
        )):
            outline.set_shapes(*self._shapes_for(targets))

        lanes, columns = self._selection_bands(focus)
        self._lane_bands.set_bands(lanes)
        self._column_bands.set_bands(columns)
        self.sigColumnsChanged.emit(columns)

    def _selection_bands(self, focus: Optional[HoverTarget]) -> tuple[list, list]:
        """
        (lanes, columns) bands: each selected item's sample lane(s) and x
        extent (an analyte: all its members'). `focus`'s column (None: not
        shown apart) is in the focus colour, drawn last; lanes are all grey.
        """
        model, strips = self.selection_model, self._strips
        targets = model.selection
        if focus is not None and not model.lists(focus):
            targets.append(focus)   # a member of a selected analyte
        lanes, columns = [], []
        for target in sorted(targets, key=lambda t: target_key(t) == target_key(focus)):
            color = FOCUS_BAND if target_key(target) == target_key(focus) else SELECTION_BAND
            if target.kind == 'ensemble':
                rect = strips.rect_for(target.analyte_id, target.sample_uuid)
                rects = {target.sample_uuid: rect} if rect is not None else {}
            else:
                rects = strips.rects.get(target.analyte_id, {})
            if not rects:
                continue
            columns.append((
                min(r.left() for r in rects.values()),
                max(r.right() for r in rects.values()),
                color,
            ))
            for sample_uuid in rects:
                y = strips.lane_y(sample_uuid)
                if y is not None:
                    lanes.append((y - 0.5, y + 0.5, SELECTION_BAND))
        return lanes, columns

    # -- keyboard navigation ------------------------------------------------

    def handle_key(self, key: int) -> bool:
        """
        Arrow-key navigation. Focus in a selected analyte: Up/Down move the
        focus through its ensembles (top- / bottom-most from the analyte).
        Otherwise, several items selected: Left/Up and Right/Down move the
        focus to the previous / next of them (in display_order). One:
        select a neighbour of it. Returns True if the key was handled.

        Ensemble: Left/Right -> neighbouring ensemble in the same lane;
                  Up/Down    -> same analyte in the next lane that has it.
        Analyte:  Left/Right -> neighbouring multi-sample analyte.
        """
        model = self.selection_model
        if key == QtCore.Qt.Key_Escape:
            model.clear(anchor=False)
            return True

        if key not in (
            QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
            QtCore.Qt.Key_Up, QtCore.Qt.Key_Down,
        ):
            return False
        current = model.focus
        if self._strips is None or current is None:
            return True

        step = -1 if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Up) else 1
        horizontal = key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right)

        analyte = self._strips.analyte_target(current.analyte_id)
        if not horizontal and model.lists(analyte):
            nxt = (
                self._analyte_member(current, first=step < 0)
                if current.kind == 'analyte'
                else self._step_across_lanes(current, step)
            )
            if nxt is not None:
                model.set_focus(nxt)
                self._ensure_visible(nxt)
            return True
        if not model.lists(current):   # a member of a selected analyte: from the analyte
            current = analyte

        selection = model.selection
        if len(selection) > 1:
            order = display_order(selection)
            i = [target_key(t) for t in order].index(target_key(current)) + step
            nxt = order[min(max(i, 0), len(order) - 1)]
            model.set_focus(nxt)
            self._ensure_visible(nxt)
            return True

        if current.kind == 'ensemble':
            nxt = (
                self._step_in_lane(current, step) if horizontal
                else self._step_across_lanes(current, step)
            )
        else:   # horizontal: Up/Down on a selected analyte were handled above
            nxt = self._step_analyte(current, step)

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
