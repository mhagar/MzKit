"""
core/formula/compound.py: the formula finder's compound search, which sets the
envelope's ion (AdductNetwork) along with the formula; and the finder's Ion combo.
"""
import os

import numpy as np
import pytest
from molmass import Formula
from find_mfs.spectra.ions import ION_VOCAB

from core.formula.compound import ion_query, search_compound

_ION = {ion.label: ion for ion in ION_VOCAB}
_H = Formula('H').monoisotopic_mass
_NA = Formula('Na').monoisotopic_mass
_E = 0.000548579909


def test_ion_query_monomer_passes_through():
    mz, adducts, M = ion_query(204.0866, _ION['[M+Na]+'])
    assert mz == 204.0866 and adducts == [('Na', 1)]
    assert M == pytest.approx(204.0866 + _E - _NA)


def test_ion_query_dimer_searches_M_as_protonated():
    M = 180.0634
    mono = 2 * M + _H - _E                 # [2M+H]+
    mz, adducts, got_M = ion_query(mono, _ION['[2M+H]+'])
    assert got_M == pytest.approx(M, abs=1e-6)
    assert adducts == [('H', 1)] and mz == pytest.approx(M + _H)


def test_ion_query_doubly_charged():
    M = 500.25
    mono = (M + 2 * _H - 2 * _E) / 2       # [M+2H]2+
    _, adducts, got_M = ion_query(mono, _ION['[M+2H]2+'])
    assert got_M == pytest.approx(M, abs=1e-6) and adducts == [('H', 1)]


def _envelope(ensemble, mono_mz=282.17024):
    """Cycloheximide [M+H]+ envelope on the composite scan: (signals, idxs, scan)."""
    scan = ensemble.composite_ms1_scan_num()
    spec = ensemble.get_spectrum(ms_level=1, scan_num=scan)
    idxs = [
        i for i in range(len(spec))
        if mono_mz - 0.01 <= spec['mz'][i] <= mono_mz + 3.1 and spec['intsy'][i] > 0
    ]
    if not idxs or abs(spec['mz'][idxs[0]] - mono_mz) > 0.01:
        pytest.skip("no cycloheximide envelope in the fixture ensemble")
    return [(float(spec['mz'][i]), float(spec['intsy'][i])) for i in idxs], idxs, scan


@pytest.mark.parametrize('ion', [None, '[M+H]+'])
def test_search_compound_sets_envelope_M_and_network(ensemble, ion):
    if ensemble.is_dda:
        pytest.skip("fixture ensemble is DDA")
    signals, idxs, scan = _envelope(ensemble)

    a, net = search_compound(ensemble, signals, idxs, scan, _ION.get(ion))

    assert a.source_uuid == ensemble.uuid and a.scan_num == scan
    assert a.M == pytest.approx(281.16271, abs=0.005) and net.M == pytest.approx(a.M)
    assert idxs[0] in net.base_envelope and a.envelope == net.base_envelope
    assert net.base_ion == '[M+H]+'
    assert net.by == ('auto' if ion is None else 'user')
    assert any(Formula(c.formula_str).formula == 'C15H23NO4' for c in a.candidates)


def test_finder_ion_combo():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    import gui.controllers.main_controller  # noqa: F401  (app import order)
    from PyQt5.QtWidgets import QApplication
    from gui.dialogues.formula_finder import FormulaFinderDialog

    _app = QApplication.instance() or QApplication([])
    dlg = FormulaFinderDialog()
    labels = [dlg.comboIon.itemText(i) for i in range(dlg.comboIon.count())]
    assert labels == ['Unknown'] + [ion.label for ion in ION_VOCAB]
    assert dlg.selected_ion() is None
    dlg.comboIon.setCurrentIndex(labels.index('[2M+Na]+'))
    assert dlg.selected_ion() is _ION['[2M+Na]+']


def test_search_compound_anchored_ion_labels_other_envelopes(ensemble):
    """A picked ion anchors find-mfs's network: M follows from it, and other envelopes get ions too."""
    if ensemble.is_dda:
        pytest.skip("fixture ensemble is DDA")
    signals, idxs, scan = _envelope(ensemble)
    a, net = search_compound(ensemble, signals, idxs, scan, _ION['[M+H]+'])
    assert net.by == 'user' and net.base_ion == '[M+H]+'
    assert net.M == pytest.approx(281.16271, abs=0.005)
    assert len(net.ions) > 1 and net.near_tie_adducts == [('H', 1)]
