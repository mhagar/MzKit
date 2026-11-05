"""Session 5: a DDA ensemble's representative MS2 is its best-fragmenting precursor's."""
import numpy as np
import pytest

import core.formula.network as network_mod
from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment
from core.data_structs.ensemble import AdductNetwork
from core.data_structs.ensemble.ms2 import MS2Spectrum
from core.data_structs.similarity_matrix import IMPORTED, ImportParams, SimilarityMatrix
from core.formula.network import (
    TIER_METAL, TIER_MONOMER, TIER_MULTICHARGED, TIER_MULTIMER, TIER_UNLABELLED,
    envelope_ions, ion_tier,
)


def test_ion_tier_order():
    assert {ion_tier(x) for x in ('[M+H]+', '[M+NH4]+', '[M+H-H2O]+', '[M]+')} == {TIER_MONOMER}
    assert ion_tier('[M+2H]2+') == TIER_MULTICHARGED
    assert ion_tier('[2M+H]+') == ion_tier('[2M+2H]2+') == TIER_MULTIMER
    assert ion_tier(None) == ion_tier('[M+Foo]+') == TIER_UNLABELLED
    assert {ion_tier(x) for x in ('[M+Na]+', '[M+K]+', '[2M+Na]+')} == {TIER_METAL}
    assert TIER_MONOMER < TIER_MULTICHARGED < TIER_MULTIMER < TIER_UNLABELLED < TIER_METAL


def test_envelope_ions_on_the_fly_and_from_network(ensemble):
    ms1 = ensemble.composite_spectrum.ms1
    base = int(np.argmax(ms1['intsy']))
    assert envelope_ions(ms1)[base] == '[M+H]+'           # cycloheximide, find-mfs's network
    stub = AdductNetwork(M=0.0, base_envelope=[base], ions={base: '[M+Na]+'}, by='user')
    assert envelope_ions(ms1, stub)[base] == '[M+Na]+'    # the stored network wins


def _two_precursors(ensemble, monkeypatch):
    """Force DDA with fake scans fragmenting the tallest lane and a shorter one, >0.5 m/z apart."""
    monkeypatch.setattr(ensemble.injection, 'acquisition_mode', 'dda')
    ms1 = ensemble.get_spectrum(ms_level=1, scan_num=ensemble.composite_ms1_scan_num())
    order = np.argsort(-ms1['intsy'])
    tall = int(order[0])
    short = next(int(i) for i in order[1:]
                 if ms1['intsy'][i] > 0 and abs(ms1['mz'][i] - ms1['mz'][tall]) > 0.6)
    scans = [MS2Spectrum(ms1[:3], float(ms1['mz'][i]), 1, float(ensemble.peak_rt)) for i in (tall, short)]
    monkeypatch.setattr(type(ensemble), '_iter_ms2_scan_spectra', lambda self: scans)
    ensemble._composite = None
    return ms1, tall, short


def test_dda_prefers_non_metal_precursor(ensemble, monkeypatch):
    ms1, tall, short = _two_precursors(ensemble, monkeypatch)
    monkeypatch.setattr(network_mod, 'envelope_ions', lambda *_: {})
    assert ensemble.composite_spectrum.precursor_mz == pytest.approx(ms1['mz'][tall], abs=0.01)

    monkeypatch.setattr(network_mod, 'envelope_ions',
                        lambda *_: {tall: '[M+Na]+', short: '[M+H]+'})
    ensemble._composite = None
    assert ensemble.composite_spectrum.precursor_mz == pytest.approx(ms1['mz'][short], abs=0.01)


def test_network_change_invalidates_ms2_derived_state(data_registry, ensemble, monkeypatch):
    ms1, tall, short = _two_precursors(ensemble, monkeypatch)
    ensemble.injection.ensembles[ensemble.uuid] = ensemble
    sample = data_registry.get_all_samples()[0]
    analyte = AlignedAnalyte(ensemble_map={sample.uuid: ensemble.uuid})
    data_registry.register_alignment(EnsembleAlignment(sample_uuids=(sample.uuid,), analytes=[analyte]))
    analyte.representative_spectrum({sample.uuid: ensemble})
    ids = (str(analyte.uuid),)
    computed = SimilarityMatrix("MS2DS", ids, np.ones((1, 1), np.float32),
                                {'source': 'ms2deepscore', 'params': {}}, ImportParams())
    imported = SimilarityMatrix("ext", ids, np.ones((1, 1), np.float32),
                                {'source': IMPORTED, 'params': {}}, ImportParams())
    for m in (computed, imported):
        data_registry.register_matrix(m)

    # Same precursor either way: nothing goes
    monkeypatch.setattr(network_mod, 'envelope_ions', lambda *_: {})
    data_registry.update_ensemble(ensemble.uuid, adduct_network=None)
    assert analyte._representative is not None and len(data_registry.get_all_matrices()) == 2

    # The network now says the tallest is [M+Na]+: precursor moves, MS2-derived state goes
    monkeypatch.setattr(network_mod, 'envelope_ions', lambda *_: {tall: '[M+Na]+', short: '[M+H]+'})
    net = AdductNetwork(M=0.0, base_envelope=[short], ions={short: '[M+H]+'}, by='user')
    data_registry.update_ensemble(ensemble.uuid, adduct_network=net)
    assert ensemble.composite_spectrum.precursor_mz == pytest.approx(ms1['mz'][short], abs=0.01)
    assert analyte._representative is None
    assert data_registry.get_all_matrices() == [imported]
