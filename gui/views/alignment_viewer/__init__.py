"""
Alignment Viewer — interactive feature-map for an ``EnsembleAlignment``.

Replaces the old spreadsheet view. Each Ensemble is a strip laid out by
retention time within a per-sample lane; strips of the same AlignedAnalyte
share a colour and a connecting line. Selecting an ensemble, analyte or
sample drives actions into the Ensemble / Sample viewers (via signals the
MainController wires up).
"""
from typing import Optional, TYPE_CHECKING

import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

from gui.views.alignment_viewer.plot_item import AlignmentPlotItem
from gui.views.alignment_viewer.layout import HoverTarget
from gui.views.alignment_viewer.params import RenderParams

if TYPE_CHECKING:
    from core.data_structs import DataRegistry
    from core.data_structs.alignment import EnsembleAlignment


class AlignmentViewer(QtWidgets.QWidget):
    """Feature-map viewer for a single EnsembleAlignment."""

    # Actions requested from the plot, wired up by MainController.
    sigViewEnsembleRequested = QtCore.pyqtSignal(object)  # Ensemble
    sigAddSamplesRequested = QtCore.pyqtSignal(object)     # list[SampleUUID]

    def __init__(self, data_source: 'DataRegistry', parent=None):
        super().__init__(parent)
        self._data_registry = data_source
        self._params = RenderParams()
        self._alignment: Optional['EnsembleAlignment'] = None
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._label = QtWidgets.QLabel("No alignment loaded")
        layout.addWidget(self._label)

        self._plot_item = AlignmentPlotItem()
        self._plot_widget = pg.PlotWidget(plotItem=self._plot_item)
        self._plot_widget.setBackground(None)
        layout.addWidget(self._plot_widget)

        self._plot_item.sigEnsembleActivated.connect(
            self.sigViewEnsembleRequested
        )
        self._plot_item.sigContextRequested.connect(self._show_context_menu)

    # -- public API (kept stable for MainController / SubWindowManager) ------

    def set_alignment(self, alignment: 'EnsembleAlignment'):
        self._alignment = alignment
        self._render(preserve_view=False)

    def _render(self, preserve_view: bool = False):
        """(Re)draw the current alignment with the current RenderParams."""
        if self._alignment is None:
            return
        self._plot_item.set_alignment(
            self._alignment,
            self._data_registry,
            self._params,
            preserve_view=preserve_view,
        )

        alignment = self._alignment
        multi = sum(1 for a in alignment.analytes if len(a.ensemble_map) > 1)
        self._label.setText(
            f"{alignment.analyte_count} analytes across "
            f"{alignment.sample_count} samples "
            f"({multi} matched, "
            f"{alignment.analyte_count - multi} singletons)"
        )

    def reset_for_new_project(self):
        self._alignment = None
        self._plot_item.clear_alignment()
        self._label.setText("No alignment loaded")

    # -- context menu -------------------------------------------------------

    def _show_context_menu(self, target: Optional[HoverTarget], global_pos):
        if target is None:
            return

        menu = QtWidgets.QMenu(self)

        if target.kind == 'ensemble':
            act_open = menu.addAction("Open in Ensemble Viewer")
            act_open.triggered.connect(
                lambda: self.sigViewEnsembleRequested.emit(target.ensemble)
            )
            act_sample = menu.addAction("Add sample to Sample Viewer")
            act_sample.triggered.connect(
                lambda: self.sigAddSamplesRequested.emit([target.sample_uuid])
            )
            if target.analyte and len(target.analyte.ensemble_map) > 1:
                act_all = menu.addAction("Add all aligned samples to Sample Viewer")
                act_all.triggered.connect(
                    lambda: self.sigAddSamplesRequested.emit(
                        list(target.analyte.ensemble_map.keys())
                    )
                )

        elif target.kind == 'analyte':
            act_all = menu.addAction("Add all aligned samples to Sample Viewer")
            act_all.triggered.connect(
                lambda: self.sigAddSamplesRequested.emit(
                    list(target.analyte.ensemble_map.keys())
                )
            )

        elif target.kind == 'sample':
            act_sample = menu.addAction("Add sample to Sample Viewer")
            act_sample.triggered.connect(
                lambda: self.sigAddSamplesRequested.emit([target.sample_uuid])
            )

        if not menu.isEmpty():
            menu.exec_(global_pos)
