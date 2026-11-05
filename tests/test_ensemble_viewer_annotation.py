"""
Offscreen smoke test of the Ensemble Viewer's annotation tabs: its ensemble is
the target, a ranking fills the structure search, selecting a row mirrors it
on the EV's MS2 plot, Accept makes it the ensemble's chosen hypothesis, and
deselecting gives the plot back.
"""
import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
import gui.controllers.main_controller  # noqa: E402,F401  (app import order)

from core.cli.rank_candidates import RankParams, RankResult  # noqa: E402
from tests.test_annotation_panel import _scored  # noqa: E402


def test_ensemble_viewer_ranks_mirrors_and_accepts(data_registry, sample, ensemble, monkeypatch):
    from gui.views.ensemble_viewer import EnsembleViewer

    sample.injection.ensembles[ensemble.uuid] = ensemble
    monkeypatch.setattr('core.utils.glacier.model_available', lambda: True)
    viewer = EnsembleViewer(data_registry)
    viewer.show()
    tabs = viewer.annotation_tabs
    panel = tabs.structure_search
    panel.source_checks = [QtWidgets.QCheckBox('NPAtlas')]
    viewer.set_ensemble(ensemble)

    key = ('ensemble', ensemble.uuid)
    assert panel._target.key == key and panel._target.ensemble is ensemble
    assert np.array_equal(panel.ms2, ensemble.composite_spectrum.ms2)
    assert tabs.editor.label_target.text().startswith(sample.injection.name)

    viewer.tabWidget.setCurrentWidget(panel)
    tabs.set_ranking(key, RankResult([_scored('NPA2', 'CCCO', 2, 0.9)], n_considered=2, adduct='[M+H]+',
                                     adduct_fallback=False, params=RankParams()))
    assert panel.table.rowCount() == 1
    panel.table.selectRow(0)
    assert tabs.mirrored is panel.shown and viewer.ms2_plot.is_mirror()

    panel.accept_btn.click()
    assert ensemble.annotation.chosen.smiles == 'CCCO'
    assert viewer.tabWidget.currentWidget() is tabs.editor_page
    assert tabs.editor.tree.topLevelItem(0).text(1) == "★ npa2"

    tabs.editor.tree.clearSelection()
    assert tabs.mirrored is None and not viewer.ms2_plot.is_mirror()
