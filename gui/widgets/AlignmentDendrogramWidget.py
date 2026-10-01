"""
Dendrogram strip for the Alignment Viewer's cluster mode: draws an
`AnalyteClustering`'s merges (core/cli/cluster_analytes.py) above the
feature map, x-linked to it so leaf slots line up with the strips.

y = merge distance (1 - similarity), leaves at the bottom; the axis is
labelled in similarity. Hovering a merge highlights its subtree; clicking
it emits the subtree's analytes.

Promoted from Qt Designer (see gui/resources/AlignmentViewerWindow.ui).
"""
from typing import Optional, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor, QPainter, QPicture, QPolygonF

from gui.widgets.alignment_plot.indicators import OutlineIndicator

if TYPE_CHECKING:
    from core.cli.cluster_analytes import AnalyteClustering


# Must match the feature map's lane-axis width so the x axes line up
LEFT_AXIS_WIDTH = 130
# Fixed y range (distance), padded so the end tick labels aren't clipped
Y_RANGE = (-0.08, 1.08)
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
        vb = self.getViewBox()
        if vb is None or not len(self.nodes):
            return None
        _, py = vb.viewPixelSize()
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
    Dendrogram above the alignment map. Call `link_to(map_plot_item)` once,
    then `set_clustering()` / `clear_clustering()`.
    """

    sigSubtreeClicked = QtCore.pyqtSignal(object, bool)  # list[AnalyteUUID], additive
    sigNodeHovered = QtCore.pyqtSignal(object)           # (similarity, n_analytes) | None

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
        left.setTicks([[(0.0, "1.0"), (0.5, "0.5"), (1.0, "0.0")]])
        left.setLabel("similarity")
        self.pi.setYRange(*Y_RANGE, padding=0)
        self.pi.setLimits(yMin=Y_RANGE[0], yMax=Y_RANGE[1])

        self._clustering: Optional['AnalyteClustering'] = None
        self._item: Optional[DendrogramItem] = None
        self._highlight = OutlineIndicator(
            _cosmetic_pen(QColor(255, 255, 255, 255), 2), z_value=100,
        )

    def link_to(self, plot_item: pg.PlotItem):
        """Share the x axis with the feature map."""
        self.pi.setXLink(plot_item)

    def set_clustering(self, clustering: 'AnalyteClustering'):
        self.clear_clustering()
        self._clustering = clustering
        self._item = DendrogramItem(clustering.nodes)
        self._item.sigNodeHovered.connect(self._on_node_hovered)
        self._item.sigNodeClicked.connect(self._on_node_clicked)
        self.pi.addItem(self._item)
        self.pi.addItem(self._highlight)
        self.pi.setYRange(*Y_RANGE, padding=0)

    def clear_clustering(self):
        self.pi.clear()
        self._highlight.clear()
        self._clustering = None
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
        uuids = self._clustering.analyte_uuids[int(n['lo']):int(n['hi']) + 1]
        self.sigSubtreeClicked.emit(list(uuids), additive)
