"""
Session 6 C: the search tools' parameter sheets write the config the searches
read, and the structure search's SMILES / text query modes.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
import gui.controllers.main_controller  # noqa: E402,F401  (app import order)

from core.utils.array_types import to_spec_arr  # noqa: E402
from gui.widgets.annotation_target import AnnotationTarget  # noqa: E402


@pytest.fixture
def user_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('APPDATA' if os.name == 'nt' else 'XDG_CONFIG_HOME', str(tmp_path))
    return tmp_path


def test_spectral_sheet_edits_reach_the_search_params(user_config_dir):
    from core.cli.search_spectra import params_from_config
    from core.utils.config import load_config
    from gui.widgets.search_param_sheet import spectral_sheet

    sheet = spectral_sheet()
    sheet.controls['mz_tolerance'].setValue(0.005)
    sheet.controls['precursor_tolerance'].setValue(0.05)
    sheet.controls['min_matched_peaks'].setValue(5)
    params = params_from_config(load_config())
    assert (params.mz_tolerance, params.precursor_tolerance, params.min_matched_peaks) == (0.005, 0.05, 5)
    # A second sheet (or a reopen) shows what's saved
    assert spectral_sheet().values['min_matched_peaks'] == 5


def test_glacier_sheet_edits_reach_the_rank_params(user_config_dir):
    from core.cli.rank_candidates import rank_params_from_config
    from core.utils.config import load_config
    from gui.widgets.search_param_sheet import glacier_sheet

    sheet = glacier_sheet()
    sheet.controls['instrument'].setCurrentText('QTOF')
    sheet.controls['collision_energy'].setValue(40)
    params = rank_params_from_config(load_config())
    assert (params.instrument, params.collision_energy) == ('QTOF', 40.0)


def _widget(monkeypatch):
    from gui.widgets.structure_search_widget import SMILES, StructureSearchWidget
    monkeypatch.setattr('core.utils.glacier.model_available', lambda: True)
    w = StructureSearchWidget()
    ms2 = to_spec_arr(np.array([50.0, 100.0]), np.array([1.0, 0.4]))
    w.set_target(AnnotationTarget(key='k', ensemble=None, ms2=ms2, precursor_mz=181.07, formula='C6H12O6'))
    requests = []
    w.sigRankRequested.connect(lambda key, kwargs: requests.append((key, kwargs)))
    return w, requests, SMILES


def test_smiles_mode_ranks_the_typed_structure_without_a_database(monkeypatch):
    w, requests, SMILES = _widget(monkeypatch)
    w.source_checks = []   # no compound cache needed
    w.mode_combo.setCurrentIndex(SMILES)
    assert w.query_edit.placeholderText() == "SMILES"
    w.query_edit.setText("not a smiles")
    w.rank_btn.click()
    assert not requests and "Not a valid SMILES" in w.status_label.text()
    w.query_edit.setText("OCC")
    w.rank_btn.click()
    (key, kwargs), = requests
    assert key == 'k' and [(c.smiles, c.source) for c in kwargs['candidates']] == [("OCC", "user")]


def test_text_and_formula_modes_query_the_ticked_sources(monkeypatch):
    from core.cli.rank_candidates import DbCandidate
    w, requests, SMILES = _widget(monkeypatch)
    w.source_checks = []
    w.mode_combo.setCurrentIndex(0)
    w.rank_btn.click()
    assert not requests and "No compound database cache" in w.status_label.text()

    calls = []
    def fake_fetch(sources, **kw):
        calls.append(kw)
        return [DbCandidate('NPAtlas', 'NPA1', 'x', 'CCO', 'C2H6O')], False
    monkeypatch.setattr('gui.widgets.structure_search_widget.fetch_candidates', fake_fetch)
    w.source_checks = [QtWidgets.QCheckBox('NPAtlas')]
    w.source_checks[0].setChecked(True)
    w.mode_combo.setCurrentIndex(w.mode_combo.findText('Organism'))
    w.query_edit.setText("aspergillus")
    w.rank_btn.click()
    assert calls == [dict(text_field='Organism', text='aspergillus')] and len(requests) == 1


def test_query_embeddings_are_shared_and_checked(monkeypatch):
    from pathlib import Path
    from gui.dialogues.LibrariesDialog import LibrariesDialog
    from gui.widgets import spectral_search_widget as sw

    monkeypatch.setattr(sw.spectrum_db, 'read_embeddings_npz', lambda path: (['99'], [np.ones(3)]))
    monkeypatch.setattr(sw.spectrum_db, 'embedders', lambda: ['ms2deepscore', 'extern'])
    monkeypatch.setattr(sw, 'QUERY_EMBEDDINGS', {})
    assert sw.import_query_embeddings(Path("/x/q.extern.npz")) == 'extern'
    assert list(sw.QUERY_EMBEDDINGS['extern']) == ['99']
    with pytest.raises(ValueError, match="No library embeddings for 'other'"):
        sw.import_query_embeddings(Path("/x/q.other.npz"))

    # The dialog announces it, for MainController to refresh every viewer's spectral tab
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *a, **k: ("/x/q.extern.npz", ""))
    dialog = LibrariesDialog()
    imported = []
    dialog.sigEmbeddingsImported.connect(imported.append)
    dialog.embeddings_btn.setEnabled(True)
    dialog.embeddings_btn.click()
    assert imported == ['extern']
