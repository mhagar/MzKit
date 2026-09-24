"""
Feature-map plot of an EnsembleAlignment: one strip per Ensemble, laid out
by x (retention time by default) within per-sample lanes; strips of the same
AlignedAnalyte share a colour and a connecting line.

Promoted from Qt Designer (see gui/resources/AlignmentViewerWindow.ui).
The drawing / hit-testing / selection logic lives in gui/widgets/alignment_plot/.
"""
import pyqtgraph as pg
from PyQt5 import QtCore

from gui.widgets.alignment_plot.plot_item import AlignmentPlotItem


class AlignmentPlotWidget(pg.PlotWidget):
    """
    PlotWidget wrapping an AlignmentPlotItem, adding keyboard navigation
    (arrow keys / Esc; see AlignmentPlotItem.handle_key).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, plotItem=AlignmentPlotItem(), **kwargs)
        self.setBackground(None)
        self.pi: AlignmentPlotItem = self.getPlotItem()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def keyPressEvent(self, ev):
        if self.pi.handle_key(ev.key()):
            ev.accept()
            return
        super().keyPressEvent(ev)
