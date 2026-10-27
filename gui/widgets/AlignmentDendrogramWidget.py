"""
Dendrogram strip for the Alignment Viewer's cluster modes: draws an
`AnalyteClustering`'s merges (core/cli/cluster_analytes.py) above the
feature map, x-linked to it so leaf slots line up with the strips - or,
oriented 'left', a `SampleClustering`'s (core/cli/cluster_samples.py) left
of the map, y-linked to its lanes.

y = merge distance (1 - similarity), leaves at the bottom (rotated so leaves
are on the right for 'left'); the axis is labelled in similarity. Hovering
a merge highlights its subtree; clicking it emits the subtree's leaves.

Promoted from Qt Designer (see gui/resources/AlignmentViewerWindow.ui).
"""
from typing import Optional

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor, QPainter, QPicture, QPolygonF, QTransform

from gui.widgets.alignment_plot.indicators import BandIndicator, OutlineIndicator

# Must match the feature map's lane-axis width so the x axes line up
LEFT_AXIS_WIDTH = 130
# Fixed y range (distance), padded so the end tick labels aren't clipped
Y_RANGE = (-0.08, 1.08)
SIMILARITY_TICKS = [[(0.0, "1.0"), (0.5, "0.5"), (1.0, "0.0")]]
# Pixel half-height of a merge bar's hit test
BAR_HIT_PX = 5.0


def _cosmetic_pen(color: QColor, width: int):
    pen = pg.mkPen(color, width=width)
    pen.setCosmetic(True)
    return pen


def _node_polyline(node) -> QPolygonF:
    """A merge's U-shape: down to each child, joined at the merge height."""
    return QPolygonF([
        QPointF(node['x_left'], node['h_left']),
        QPointF(node['x_left'], node['height']),
        QPointF(node['x_right'], node['height']),
        QPointF(node['x_right'], node['h_right']),
    ])


class DendrogramItem(pg.GraphicsObject):
    """Every merge in one QPicture, plus merge-bar hit testing."""

    sigNodeHovered = QtCore.pyqtSignal(object)   # node index | None
    sigNodeClicked = QtCore.pyqtSignal(object, bool)   # node index | None, additive

    def __init__(self, nodes: np.ndarray):
        super().__init__()
        self.nodes = nodes
        self._hovered: Optional[int] = None
        self._picture = QPicture()

        painter = QPainter(self._picture)
        painter.setPen(_cosmetic_pen(QColor(180, 180, 180, 200), 1))
        for node in nodes:
            painter.drawPolyline(_node_polyline(node))
        painter.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self._picture)

    def boundingRect(self):
        rect = self._picture.boundingRect()
        return QtCore.QRectF(rect) if rect.isValid() else QtCore.QRectF()

    def node_at(self, point: QPointF) -> Optional[int]:
        """The merge whose bar is nearest `point` (within BAR_HIT_PX)."""
        if self.getViewBox() is None or not len(self.nodes):
            return None
        # One pixel along the height axis, in item coords (works rotated too)
        py = self.pixelLength(pg.Point(0, 1))
        x, y = point.x(), point.y()
        nodes = self.nodes
        dy = np.abs(nodes['height'] - y)
        hits = (
            (nodes['x_left'] <= x) & (x <= nodes['x_right'])
            & (dy <= BAR_HIT_PX * py)
        )
        if not hits.any():
            return None
        idxs = np.flatnonzero(hits)
        return int(idxs[dy[idxs].argmin()])

    def hoverEvent(self, ev):
        node = None if ev.exit else self.node_at(ev.pos())
        if node != self._hovered:
            self._hovered = node
            self.sigNodeHovered.emit(node)

    def mouseClickEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        ev.accept()
        additive = bool(ev.modifiers() & QtCore.Qt.ControlModifier)
        self.sigNodeClicked.emit(self._hovered, additive)


class AlignmentDendrogramWidget(pg.PlotWidget):
    """
    Dendrogram above (or, see `set_orientation`, left of) the alignment
    map. Call `link_to(map_plot_item)` once, then `set_clustering()` /
    `clear_clustering()`.
    """

    sigSubtreeClicked = QtCore.pyqtSignal(object, bool)  # list[leaf uuid], additive
    sigNodeHovered = QtCore.pyqtSignal(object)           # (similarity, n_leaves) | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setBackground(None)
        self.pi: pg.PlotItem = self.getPlotItem()
        self.pi.setMenuEnabled(False)
        self.pi.setMouseEnabled(x=True, y=False)
        self.pi.hideButtons()
        self.pi.hideAxis('bottom')

        left = self.pi.getAxis('left')
        left.setWidth(LEFT_AXIS_WIDTH)
        left.setTicks(SIMILARITY_TICKS)
        left.setLabel("similarity")
        self.pi.setYRange(*Y_RANGE, padding=0)
        self.pi.setLimits(yMin=Y_RANGE[0], yMax=Y_RANGE[1])

        self._orientation = 'top'
        self._leaf_uuids: list = []
        self._clustering = None
        self._item: Optional[DendrogramItem] = None
        self._highlight = OutlineIndicator(
            _cosmetic_pen(QColor(255, 255, 255, 255), 2), z_value=100,
        )
        self._columns = BandIndicator(vertical=True)   # the map's selection

    def contextMenuEvent(self, ev):
        menu = QtWidgets.QMenu(self)
        menu.addAction("Export as .svg", self.export_svg)
        menu.exec_(ev.globalPos())

    def export_svg(self) -> None:
        # Kept on self: the exporter's (non-modal) file dialog dies with it
        self._svg_exporter = pyqtgraph.exporters.SVGExporter(self.pi)
        self._svg_exporter.export()

    def set_orientation(self, orientation: str):
        """
        'top' (default) or 'left': leaves along y (lane order, top down),
        root on the left; the similarity axis goes to the bottom. Call
        before `link_to`.
        """
        self._orientation = orientation
        if orientation != 'left':
            return
        self.pi.setMouseEnabled(x=False, y=True)
        self.pi.hideAxis('left')
        self.pi.showAxis('bottom')
        self.pi.getAxis('bottom').setTicks(SIMILARITY_TICKS)
        self.pi.getViewBox().invertY(True)   # match the map: lane 0 on top
        self.pi.setXRange(-Y_RANGE[1], -Y_RANGE[0], padding=0)
        self.pi.setLimits(
            xMin=-Y_RANGE[1], xMax=-Y_RANGE[0], yMin=None, yMax=None,
        )

    def set_columns(self, bands: list):
        """The map's selection bands: (x0, x1, QColor)s."""
        self._columns.set_bands(bands)

    def show_bottom_axis(self, show: bool):
        """'left' only: mirror the map's bottom axis so the lanes line up."""
        self.pi.showAxis('bottom', show)

    def link_to(self, plot_item: pg.PlotItem):
        """Share the leaf axis with the feature map."""
        if self._orientation == 'left':
            self.pi.setYLink(plot_item)
        else:
            self.pi.setXLink(plot_item)

    def set_clustering(self, clustering, leaf_uuids: Optional[list] = None):
        """
        Draw `clustering.nodes`; `leaf_uuids` (default
        `clustering.analyte_uuids`) are what subtree clicks emit.
        """
        self.clear_clustering()
        self._clustering = clustering
        self._leaf_uuids = (
            clustering.analyte_uuids if leaf_uuids is None else leaf_uuids
        )
        self._item = DendrogramItem(clustering.nodes)
        if self._orientation == 'left':
            # (leaf, height) -> (-height, leaf)
            self._item.setTransform(QTransform(0, 1, -1, 0, 0, 0))
        self._item.sigNodeHovered.connect(self._on_node_hovered)
        self._item.sigNodeClicked.connect(self._on_node_clicked)
        self.pi.addItem(self._item)
        self.pi.addItem(self._highlight)
        self.pi.addItem(self._columns, ignoreBounds=True)
        self._highlight.setTransform(self._item.transform())
        if self._orientation != 'left':
            self.pi.setYRange(*Y_RANGE, padding=0)

    def clear_clustering(self):
        self.pi.clear()
        self._highlight.clear()
        self._clustering = None
        self._leaf_uuids = []
        self._item = None

    # -- hover / click ------------------------------------------------------

    def _subtree(self, node: int) -> np.ndarray:
        """Indices of `node` and every merge beneath it."""
        nodes = self._item.nodes
        n = nodes[node]
        return np.flatnonzero(
            (nodes['lo'] >= n['lo']) & (nodes['hi'] <= n['hi'])
            & (nodes['height'] <= n['height'])
        )

    def _on_node_hovered(self, node: Optional[int]):
        if node is None:
            self._highlight.clear()
            self.sigNodeHovered.emit(None)
            return
        nodes = self._item.nodes
        self._highlight.set_shapes(
            [_node_polyline(nodes[i]) for i in self._subtree(node)], [],
        )
        n = nodes[node]
        self.sigNodeHovered.emit(
            (1.0 - float(n['height']), int(n['hi'] - n['lo'] + 1))
        )

    def _on_node_clicked(self, node: Optional[int], additive: bool):
        if node is None or self._clustering is None:
            return
        n = self._item.nodes[node]
        uuids = self._leaf_uuids[int(n['lo']):int(n['hi']) + 1]
        self.sigSubtreeClicked.emit(list(uuids), additive)
