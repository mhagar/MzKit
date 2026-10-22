"""
Group track for the Alignment Viewer: a thin strip above the feature map,
x-linked to it, drawing each `AnalyteGroup` (core/data_structs/analyte_group.py)
as a tick at every member's x joined by a line from the leftmost to the
rightmost tick, title at the left. Contiguous groups read as bars, broken
groups as combs. Overlapping groups are packed into separate lanes.

Works with any analyte -> x mapping (consensus RT, cluster leaf slot), so
it doesn't depend on the dendrogram. Hovering a group highlights it and
emits its members; clicking emits them for selection.

Promoted from Qt Designer (see gui/resources/AlignmentViewerWindow.ui).
"""
from typing import NamedTuple, Optional, TYPE_CHECKING

import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor, QFont, QPainter, QPicture, QPolygonF

from gui.widgets.alignment_plot.indicators import OutlineIndicator

if TYPE_CHECKING:
    from core.data_structs.analyte_group import AnalyteGroup
    from core.data_structs.uuid_types import AnalyteUUID


# Must match the feature map's lane-axis width so the x axes line up
LEFT_AXIS_WIDTH = 130
# Pixel height of one lane
LANE_PX = 26
# Within a lane (y in lane units, lane i spans [i, i + 1], top down)
LINE_Y = 0.8
TICK_Y = (0.65, 0.95)
LABEL_PT = 7
# Pixel tolerance of the hit test beyond a group's end ticks
HIT_PX = 4.0


def _cosmetic_pen(color: QColor, width: int):
    pen = pg.mkPen(color, width=width)
    pen.setCosmetic(True)
    return pen


def pack_lanes(spans: list[tuple[float, float]], gap: float = 0.0) -> list[int]:
    """
    Lane index per (left, right) span: by left edge, each span goes into the
    first lane whose last right edge + `gap` is strictly left of it.
    """
    lane_ends: list[float] = []
    lanes = [0] * len(spans)
    for i in sorted(range(len(spans)), key=lambda i: spans[i][0]):
        left, right = spans[i]
        for lane, end in enumerate(lane_ends):
            if end + gap < left:
                break
        else:
            lane = len(lane_ends)
            lane_ends.append(right)
        lane_ends[lane] = right
        lanes[i] = lane
    return lanes


class _Row(NamedTuple):
    group: object                 # AnalyteGroup
    uuids: list                   # visible member AnalyteUUIDs
    xs: list[float]               # their x, same order
    n_hidden: int
    lane: int

    @property
    def left(self) -> float:
        return min(self.xs)

    @property
    def right(self) -> float:
        return max(self.xs)

    def shapes(self) -> list[QPolygonF]:
        y0 = self.lane
        line = QPolygonF([
            QPointF(self.left, y0 + LINE_Y), QPointF(self.right, y0 + LINE_Y),
        ])
        ticks = [
            QPolygonF([QPointF(x, y0 + TICK_Y[0]), QPointF(x, y0 + TICK_Y[1])])
            for x in self.xs
        ]
        return [line] + ticks


class GroupTrackItem(pg.GraphicsObject):
    """Every group's ticks + span line in one QPicture, plus hit testing."""

    sigRowHovered = QtCore.pyqtSignal(object)        # row index | None
    sigRowClicked = QtCore.pyqtSignal(object, bool)  # row index | None, additive

    def __init__(self, rows: list[_Row]):
        super().__init__()
        self.rows = rows
        self._hovered: Optional[int] = None
        self._picture = QPicture()

        painter = QPainter(self._picture)
        painter.setPen(_cosmetic_pen(QColor(180, 180, 180, 220), 1))
        for row in rows:
            for shape in row.shapes():
                painter.drawPolyline(shape)
        painter.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self._picture)

    def boundingRect(self):
        rect = self._picture.boundingRect()
        if not rect.isValid():
            return QtCore.QRectF()
        # Whole lanes, so hovering a lane's label area still hits
        rect = QtCore.QRectF(rect)
        return QtCore.QRectF(
            rect.left() - 1, 0, rect.width() + 2, max(r.lane for r in self.rows) + 1,
        )

    def row_at(self, point: QPointF) -> Optional[int]:
        """The group in `point`'s lane whose span contains it (within HIT_PX)."""
        vb = self.getViewBox()
        if vb is None:
            return None
        px, _ = vb.viewPixelSize()
        tol = HIT_PX * px
        lane = int(point.y() // 1)
        for i, row in enumerate(self.rows):
            if row.lane == lane and row.left - tol <= point.x() <= row.right + tol:
                return i
        return None

    def hoverEvent(self, ev):
        row = None if ev.exit else self.row_at(ev.pos())
        if row != self._hovered:
            self._hovered = row
            self.sigRowHovered.emit(row)

    def mouseClickEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        ev.accept()
        additive = bool(ev.modifiers() & QtCore.Qt.ControlModifier)
        self.sigRowClicked.emit(self._hovered, additive)


class AlignmentGroupTrackWidget(pg.PlotWidget):
    """
    Group track above the alignment map. Call `link_to(map_plot_item)` once,
    then `set_groups()` / `clear_groups()`; the viewer shows / hides it via
    `has_visible_groups()`.
    """

    sigGroupHovered = QtCore.pyqtSignal(object)        # list[AnalyteUUID] | None
    sigGroupClicked = QtCore.pyqtSignal(object, bool)  # list[AnalyteUUID], additive

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setBackground(None)
        self.pi: pg.PlotItem = self.getPlotItem()
        self.pi.setMenuEnabled(False)
        self.pi.setMouseEnabled(x=True, y=False)
        self.pi.hideButtons()
        self.pi.hideAxis('bottom')
        self.pi.getViewBox().invertY(True)   # lane 0 at the top

        left = self.pi.getAxis('left')
        left.setWidth(LEFT_AXIS_WIDTH)
        left.setTicks([[]])

        self._rows: list[_Row] = []
        self._labels: list[pg.TextItem] = []
        self._item: Optional[GroupTrackItem] = None
        self._highlight = OutlineIndicator(
            _cosmetic_pen(QColor(255, 255, 255, 255), 2), z_value=100,
        )
        self.pi.sigXRangeChanged.connect(self._update_labels)
        self.pi.getViewBox().sigResized.connect(self._update_labels)
        self._set_lanes(1)

    def contextMenuEvent(self, ev):
        menu = QtWidgets.QMenu(self)
        menu.addAction("Export as .svg", self.export_svg)
        menu.exec_(ev.globalPos())

    def export_svg(self) -> None:
        # Kept on self: the exporter's (non-modal) file dialog dies with it
        self._svg_exporter = pyqtgraph.exporters.SVGExporter(self.pi)
        self._svg_exporter.export()

    def link_to(self, plot_item: pg.PlotItem):
        """Share the x axis with the feature map."""
        self.pi.setXLink(plot_item)

    def has_visible_groups(self) -> bool:
        return bool(self._rows)

    def set_groups(
        self,
        groups: list['AnalyteGroup'],
        x_of_analyte: dict['AnalyteUUID', float],
    ):
        """Draw `groups`; members missing from `x_of_analyte` are skipped."""
        self.clear_groups()
        visible = []
        for group in groups:
            uuids = [u for u in group.analyte_uuids if u in x_of_analyte]
            if uuids:
                visible.append((group, uuids, [float(x_of_analyte[u]) for u in uuids]))
        if not visible:
            return

        lanes = pack_lanes([(min(xs), max(xs)) for _, _, xs in visible])
        self._rows = [
            _Row(group, uuids, xs, len(group.analyte_uuids) - len(uuids), lane)
            for (group, uuids, xs), lane in zip(visible, lanes)
        ]
        self._item = GroupTrackItem(self._rows)
        self._item.sigRowHovered.connect(self._on_row_hovered)
        self._item.sigRowClicked.connect(self._on_row_clicked)
        self.pi.addItem(self._item)
        self.pi.addItem(self._highlight)
        font = QFont()
        font.setPointSize(LABEL_PT)
        for row in self._rows:
            label = pg.TextItem(row.group.title, color=(200, 200, 200), anchor=(0, 1))
            label.setFont(font)
            label.setPos(row.left, row.lane + TICK_Y[0])
            self.pi.addItem(label)
            self._labels.append(label)
        self._set_lanes(max(lanes) + 1)
        self._update_labels()

    def clear_groups(self):
        self.pi.clear()
        self._highlight.clear()
        self._rows = []
        self._labels = []
        self._item = None
        self.setToolTip("")
        self._set_lanes(1)

    def _set_lanes(self, n: int):
        self.pi.setYRange(0, n, padding=0)
        self.setFixedHeight(n * LANE_PX)

    def _update_labels(self, *args):
        # Labels aren't packed (their data width changes with zoom): a title is
        # hidden when it would run into the next group in its lane; it's always in the tooltip.
        if not self._rows:
            return
        px, _ = self.pi.getViewBox().viewPixelSize()
        if px <= 0:
            return
        next_left: dict[int, float] = {}
        room: list[float] = [float('inf')] * len(self._rows)
        for i in sorted(range(len(self._rows)), key=lambda i: -self._rows[i].left):
            row = self._rows[i]
            room[i] = next_left.get(row.lane, float('inf')) - row.left
            next_left[row.lane] = row.left
        for label, space in zip(self._labels, room):
            label.setVisible(label.boundingRect().width() * px <= space)

    # -- hover / click ------------------------------------------------------

    def _on_row_hovered(self, row: Optional[int]):
        if row is None:
            self._highlight.clear()
            self.setToolTip("")
            self.sigGroupHovered.emit(None)
            return
        r = self._rows[row]
        self._highlight.set_shapes(r.shapes(), [])
        tip = f"{r.group.title} ({len(r.uuids)} analytes)"
        if r.n_hidden:
            tip += f"\n{r.n_hidden} members not shown"
        self.setToolTip(tip)
        self.sigGroupHovered.emit(list(r.uuids))

    def _on_row_clicked(self, row: Optional[int], additive: bool):
        if row is None:
            return
        self.sigGroupClicked.emit(list(self._rows[row].uuids), additive)
