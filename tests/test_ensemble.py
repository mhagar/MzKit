"""
Tests for Ensemble class + methods.

The `ensemble` fixture (conftest) is built from gitignored MS data and
skips when that data is absent, so these are integration checks that run
only where the local .mzk fixture is present.
"""
from typing import TYPE_CHECKING

import numpy as np
import pytest

from core.cli.export_ensemble import build_ensemble_export
from core.data_structs.composite_spectrum import CompositeSpectrum

if TYPE_CHECKING:
    from core.data_structs import Ensemble


def test_resolved_precursor_and_charge_defaults(ensemble: 'Ensemble'):
    # This fixture is DIA/MS1-only, so precursor falls back to base_mz
    # and charge to 1 (no user_metadata override).
    assert ensemble.resolved_precursor_mz == pytest.approx(ensemble.base_mz)
    assert ensemble.resolved_charge == 1
    assert ensemble.is_dda is False


def test_get_meta_is_case_insensitive(ensemble: 'Ensemble'):
    ensemble.user_metadata['Adduct'] = '[M+H]+'
    assert ensemble.get_meta('adduct') == '[M+H]+'
    assert ensemble.get_meta('ADDUCT') == '[M+H]+'
    assert ensemble.get_meta('missing') is None


def test_default_ms2_mode_is_acquisition_aware(
        ensemble: 'Ensemble'
):
    # This fixture is DIA, so the default should be 'tallest'
    assert ensemble.default_ms2_mode == 'tallest'

    # mode=None (and the no-arg default) delegate to default_ms2_mode: the
    # resulting spectra are single-scan (plain, no 'freq' field).
    default_spectra = ensemble.get_ms2_spectra()
    none_spectra = ensemble.get_ms2_spectra(mode=None)
    assert len(default_spectra) == len(none_spectra)
    for ps in default_spectra:
        assert 'freq' not in (ps.spectrum.dtype.names or ())


@pytest.mark.parametrize('mode', ['tallest', 'all', 'consensus'])
def test_get_ms2_spectra_modes(ensemble: 'Ensemble', mode):
    spectra = ensemble.get_ms2_spectra(mode=mode)
    # DIA consensus/tallest collapse to at most one spectrum.
    if mode in ('tallest', 'consensus'):
        assert len(spectra) <= 1
    for ps in spectra:
        # Consensus keeps a 'freq' field; tallest/all are plain (mz, intsy).
        names = ps.spectrum.dtype.names
        assert 'mz' in names and 'intsy' in names
        if mode == 'consensus':
            assert 'freq' in names
        assert ps.precursor_mz == pytest.approx(ensemble.resolved_precursor_mz)


def test_get_ms2_spectra_normalize_peaks_to_one(ensemble: 'Ensemble'):
    spectra = ensemble.get_ms2_spectra(mode='consensus', normalize=True)
    for ps in spectra:
        if ps.spectrum.size:
            assert ps.spectrum['intsy'].max() == pytest.approx(1.0)


def test_build_ensemble_export_roundtrips_formats(ensemble: 'Ensemble'):
    export = build_ensemble_export(ensemble, ms2_mode='consensus')
    # MS1 present and normalized to 0-100.
    assert export.ms1_spectrum.size > 0
    assert export.ms1_spectrum['intsy'].max() == pytest.approx(100.0)
    # Formats render without error and carry the expected markers.
    mgf = export.to_mgf_text()
    assert 'BEGIN IONS' in mgf and 'PEPMASS=' in mgf
    assert '>compound' in export.to_sirius_text()
    assert export.to_json_obj()['ms1_spectrum']['mz']


def test_composite_spectrum_dia(ensemble: 'Ensemble'):
    # DIA fixture: MS1 is the apex scan restricted to the ensemble's lanes.
    composite = ensemble.composite_spectrum
    assert isinstance(composite, CompositeSpectrum)

    assert composite.precursor_mz == pytest.approx(ensemble.resolved_precursor_mz)
    assert composite.charge == ensemble.resolved_charge

    # MS1 equals the apex-scan spectrum, and carries real signal.
    apex_ms1 = ensemble.get_spectrum(ms_level=1, scan_num=ensemble.base_scan_num)
    assert composite.ms1.shape == apex_ms1.shape
    np.testing.assert_array_equal(composite.ms1['mz'], apex_ms1['mz'])
    assert composite.ms1['intsy'].max() > 0

    # MS2 is either absent or a plain (mz, intsy) spectrum.
    if composite.ms2 is not None:
        assert {'mz', 'intsy'} <= set(composite.ms2.dtype.names)


def test_composite_spectrum_is_memoized(ensemble: 'Ensemble'):
    # Repeated access returns the identical cached object.
    assert ensemble.composite_spectrum is ensemble.composite_spectrum


def test_composite_spectrum_dda_not_implemented(ensemble: 'Ensemble', monkeypatch):
    # Composite is DIA-only for now; DDA must fail loudly rather than return a
    # half-defined spectrum.
    monkeypatch.setattr(ensemble.injection, 'acquisition_mode', 'dda')
    assert ensemble.is_dda is True
    with pytest.raises(NotImplementedError):
        _ = ensemble.composite_spectrum







