"""
Offscreen smoke test of the Alignment Viewer's Search (spectra) tab: Search
asks MainController for a search, the result fills the table, selecting a hit
mirrors its library spectrum, and Accept adds it as the chosen hypothesis.
"""
import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets  # noqa: E402

# The app must exist before the GUI modules load (and in that import order)
_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
import gui.controllers.main_controller  # noqa: E402,F401

from core.cli.search_spectra import IDENTITY, LibraryHit, SearchResult, SpectralSearchParams  # noqa: E402
from core.utils.array_types import to_spec_arr  # noqa: E402


def _hit(spectrum_id, name, smiles, score):
    peaks = to_spec_arr(np.array([50.0, 100.0]), np.array([0.5, 1.0]))
    return LibraryHit(1, 'GNPS', spectrum_id, name, smiles, '', '[M+H]+', 101.0, peaks,
                      score, 0.8, 2, 0.001, [(0, 0), (1, 1)])


def test_search_select_mirror_accept(data_registry, sample, ensemble, monkeypatch):
    from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment
    from gui.views.alignment_viewer import AlignmentViewer

    sample.injection.ensembles[ensemble.uuid] = ensemble
    analyte = AlignedAnalyte(
        ensemble_map={sample.uuid: ensemble.uuid},
        consensus_rt=ensemble.peak_rt, consensus_mz=ensemble.base_mz,
    )
    alignment = EnsembleAlignment(sample_uuids=(sample.uuid,), analytes=[analyte])
    data_registry.register_alignment(alignment)

    viewer = AlignmentViewer(data_registry)
    viewer.show()
    viewer.set_alignment(alignment)
    tabs = viewer.annotation_tabs
    panel = tabs.spectral
    # Independent of the local index
    panel.library_checks = [QtWidgets.QCheckBox('GNPS')]
    panel.library_checks[0].setChecked(True)
    viewer.selection_model.set_selection(viewer.plot_item.analyte_targets([analyte.uuid]))
    viewer.tabsAnnotation.setCurrentWidget(panel)
    assert panel._target.key == (analyte.uuid, None)
    assert [viewer.tabsAnnotation.tabText(i) for i in range(3)] == [
        'Annotation', 'Search (spectra)', 'Search (structures)',
    ]
    panel.ms2 = to_spec_arr(np.array([50.0, 100.0]), np.array([1.0, 0.4]))
    panel._refresh_state()

    requests = []
    viewer.sigSpectralSearchRequested.connect(lambda slot, kwargs: requests.append((slot, kwargs)))
    panel.btnSearch.click()
    (slot, kwargs), = requests
    assert slot == ((analyte.uuid, None), IDENTITY) and kwargs['libraries'] == ['GNPS']

    result = SearchResult([_hit('CCMS2', 'propanol', 'CCCO', 0.95), _hit('CCMS1', 'no structure', '', 0.5)],
                          IDENTITY, 10, SpectralSearchParams())
    tabs.set_spectral_result(slot, result)
    assert panel.table.rowCount() == 2
    assert panel.table.item(0, 2).text() == "0.950" and panel.table.item(0, 3).text() == "0.80 (2)"

    panel.table.selectRow(1)
    assert not panel.btnAccept.isEnabled()   # no SMILES: can't be a hypothesis
    panel.table.selectRow(0)
    assert tabs.mirrored is panel.shown and viewer.plotMS2.is_mirror()

    panel.btnAccept.click()
    annotation = ensemble.annotation
    h = annotation.chosen
    assert (h.smiles, h.origin, h.db_id, h.claim) == ('CCCO', 'GNPS', 'CCMS2', 'structure')
    assert h.scores['cosine'] == 0.95
    assert viewer.tabsAnnotation.currentWidget() is tabs.editor_page

    # Selecting the accepted hypothesis mirrors its library spectrum, not a GLACIER prediction
    requests = []
    viewer.sigRankCandidatesRequested.connect(lambda *a: requests.append(a))
    accepted = panel.shown
    tree = tabs.editor.tree
    tree.clearSelection()
    tree.topLevelItem(0).setSelected(True)
    assert tabs.mirrored is accepted and viewer.plotMS2.is_mirror() and not requests

    # Not cached (e.g. a reopened project): re-read from the index by library + spectrum id
    tabs._library_hits.clear()
    found = []
    monkeypatch.setattr('core.cli.search_spectra.library_hit',
                        lambda ms2, mz, library, spectrum_id, params: found.append((library, spectrum_id)) or accepted)
    tree.clearSelection()
    tree.topLevelItem(0).setSelected(True)
    assert found == [('GNPS', 'CCMS2')] and tabs.mirrored is accepted and not requests
