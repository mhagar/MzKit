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
    sheet.controls['precursor_tol_da'].setValue(0.05)
    sheet.controls['min_rel_intensity'].setValue(0.02)
    sheet.controls['min_matched_peaks'].setValue(5)
    params = params_from_config(load_config())
    assert (params.mz_tolerance, params.precursor_tol_da, params.min_rel_intensity, params.min_matched_peaks) \
        == (0.005, 0.05, 0.02, 5)
    # GLACIER builds on the shared [spectral_search] peak matching keys
    from core.cli.rank_candidates import rank_params_from_config
    rank = rank_params_from_config(load_config())
    assert (rank.mz_tolerance, rank.precursor_tol_da, rank.min_rel_intensity) == (0.005, 0.05, 0.02)
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


def test_structure_query_refills_on_a_formula_change_and_mode_switch(monkeypatch):
    from gui.widgets.query_edit import FORMULA, MASS
    w, _, _ = _widget(monkeypatch)
    ms2 = w.ms2
    assert (w.query.mode(), w.query.text()) == (FORMULA, 'C6H12O6')
    w.mode_combo.setCurrentIndex(MASS)   # back to the mass, e.g. to check a doubtful formula
    assert w.query.text() == '180.0627'   # [M+H]+ assumed
    w.mode_combo.setCurrentIndex(FORMULA)
    assert w.query.text() == 'C6H12O6'
    w.query_edit.setText('typed')
    # Same key, same formula (e.g. a hypothesis was added): what's typed stays
    w.set_target(AnnotationTarget(key='k', ensemble=None, ms2=ms2, precursor_mz=181.07, formula='C6H12O6'))
    assert w.query.text() == 'typed'
    # Same key, formula (re)assigned: it overwrites, in Formula mode
    w.mode_combo.setCurrentIndex(MASS)
    w.set_target(AnnotationTarget(key='k', ensemble=None, ms2=ms2, precursor_mz=181.07, formula='C7H16O5'))
    assert (w.query.mode(), w.query.text()) == (FORMULA, 'C7H16O5')
    # No formula: the neutral mass
    w.set_target(AnnotationTarget(key='k2', ensemble=None, ms2=ms2, precursor_mz=181.07))
    assert (w.query.mode(), w.query.text()) == (MASS, '180.0627')


def test_spectral_query_by_mz_or_formula(monkeypatch):
    from core.cli.search_spectra import IDENTITY
    from gui.widgets import spectral_search_widget as sw
    from gui.widgets.query_edit import FORMULA, MASS

    monkeypatch.setattr(sw.spectrum_db, 'cache_ok', lambda *a: True)
    monkeypatch.setattr(sw.spectrum_db, 'embedders', lambda *a: ['ms2deepscore'])
    monkeypatch.setattr(sw.spectrum_db, 'available_libraries', lambda *a: ['GNPS'])
    w = sw.SpectralSearchWidget()
    w.library_checks = [QtWidgets.QCheckBox('GNPS')]
    w.library_checks[0].setChecked(True)
    ms2 = to_spec_arr(np.array([50.0, 100.0]), np.array([1.0, 0.4]))
    w.set_target(AnnotationTarget(key='k', ensemble=None, ms2=ms2, precursor_mz=181.0707, formula='C6H12O6'))
    requests = []
    w.sigSearchRequested.connect(lambda slot, kwargs: requests.append((slot, kwargs)))

    assert (w.query.mode(), w.query.text()) == (FORMULA, 'C6H12O6')
    w.btnSearch.click()
    w.query.combo.setCurrentIndex(MASS)
    assert w.query.text() == '181.0707'
    w.query.edit.setText('203.0526')   # e.g. its [M+Na]+
    w.btnSearch.click()
    (s1, k1), (s2, k2) = requests
    assert s1 == ('k', IDENTITY, ('formula', 'C6H12O6')) and k1['formula'] == 'C6H12O6'
    assert k1['precursor_mz'] == 181.0707   # the target's own, for scoring
    assert s2 == ('k', IDENTITY, ('mz', 203.0526)) and k2['precursor_mz'] == 203.0526 and 'formula' not in k2

    # An edited query doesn't show another query's hits
    from core.cli.search_spectra import SearchResult, SpectralSearchParams
    w.set_result(s2, SearchResult([], IDENTITY, 0, SpectralSearchParams()))
    assert "0 hits" in w.labelStatus.text()
    w.query.edit.setText('203.05')
    assert "0 hits" not in w.labelStatus.text()

    # Analogue: formula is off, the (edited) m/z is what's embedded
    w.query.combo.setCurrentIndex(FORMULA)
    w.comboMode.setCurrentIndex(w.comboMode.findData('ms2deepscore'))
    assert w.query.mode() == MASS and not w.query.combo.model().item(FORMULA).isEnabled()
    w.comboMode.setCurrentIndex(w.comboMode.findData(IDENTITY))
    assert w.query.combo.model().item(FORMULA).isEnabled()
