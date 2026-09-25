"""
Tests for core/cli/auto_find_mfs — the DIA auto-annotation glue.

find-mfs itself is mocked (a synthetic AnalyteAnnotation): these exercise OUR
integration — result conversion, top-hit auto-pick, adduct-label attachment,
re-run replacement, and the DDA skip — without loading MistNet or depending on
its output. A real DIA `Ensemble` (composite spectrum + cofeature validation)
comes from the committed tests/std_mix.mzk.
"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.cli.main import _load_registry
from core.cli.auto_find_mfs import (
    annotate_ensembles_dia,
    ADDUCT_ANNOT_SOURCE,
)
from core.formula.params import FindMfsParams
from core.data_structs.ensemble import GenericAnnotation


# --- fixtures ---------------------------------------------------------------

@pytest.fixture(scope='module')
def _std_mix_registry():
    p = Path('tests/std_mix.mzk')
    if not p.exists():
        pytest.skip('tests/std_mix.mzk absent')
    return _load_registry(p)


@pytest.fixture
def dia_ensemble(_std_mix_registry):
    """A real DIA ensemble with its mutable annotation state reset."""
    for sample in _std_mix_registry.get_all_samples():
        inj = sample.injection
        if inj and getattr(inj, 'acquisition_mode', None) == 'dia' and inj.ensembles:
            ens = next(iter(inj.ensembles.values()))
            ens.generic_annots = {}
            ens.proposed_formula = None
            ens._composite = None
            return ens
    pytest.skip('no DIA ensemble in std_mix')


def _fake_results(formulas):
    cands = [
        SimpleNamespace(
            formula=SimpleNamespace(formula=f), adduct='H',
            error_ppm=1.0, error_da=0.001, rdbe=4.0,
            mass_loglik=-1.0, iso_loglik=-1.0, chem_logprior=-1.0,
        )
        for f in formulas
    ]
    return SimpleNamespace(
        candidates=cands,
        log_posterior=lambda ms2_weight=1.0: np.full(len(cands), -1.0),
        ms2_loglik=lambda: np.zeros(len(cands)),
        query_params={
            'elements': 'CHNOClBr',
            'max_counts': {'C': float('inf'), 'H': float('inf'), 'N': 3,
                           'O': float('inf'), 'Cl': 4, 'Br': 4},
            'min_counts': {'C': 0, 'H': 0, 'N': 0, 'O': 0, 'Cl': 0, 'Br': 0},
            'halogen_detected': True,
        },
    )


def _install_fake_annotate(monkeypatch, *, formulas=('C6H12O6', 'C5H10O5'),
                           adduct_labels=('[M+H]+', None), mono_idx=(0, 1)):
    """Mock find-mfs so no MistNet load / real search happens."""
    grouped = SimpleNamespace(
        n_groups=len(adduct_labels),
        adduct_label=np.array(adduct_labels, dtype=object),
        mono_idx=np.array(mono_idx),
    )
    res = SimpleNamespace(
        candidates=_fake_results(formulas),
        precursor_mz=180.0634, charge=1, grouped=grouped,
    )
    monkeypatch.setattr('core.cli.auto_find_mfs._get_scorer', lambda mp: object())
    monkeypatch.setattr('find_mfs.annotate_analyte_dia',
                        lambda *a, **k: res, raising=False)
    return res


# --- conversion + top-hit + labels ------------------------------------------

def test_annotate_builds_assignment_and_labels(dia_ensemble, monkeypatch):
    _install_fake_annotate(monkeypatch)

    out = annotate_ensembles_dia([dia_ensemble])

    assert len(out) == 1
    a = out[0]
    assert a.source_uuid == dia_ensemble.uuid
    assert a.chosen_idx == 0                       # top hit auto-selected
    assert a.precursor_mz == pytest.approx(180.0634)
    assert a.charge == 1
    assert [c.formula_str for c in a.candidates] == ['C6H12O6', 'C5H10O5']
    assert a.candidates[0].adduct == 'H'
    # proposed_formula synced to the top hit
    assert dia_ensemble.proposed_formula == 'C6H12O6'

    # Only the labelled group (idx 0) becomes an adduct annotation; None skipped.
    auto = [x for x in dia_ensemble.generic_annots.values()
            if x.source == ADDUCT_ANNOT_SOURCE]
    assert len(auto) == 1
    assert auto[0].text == '[M+H]+'
    assert auto[0].cofeature_idx == 0
    assert auto[0].ms_level == 1
    assert auto[0].scan_num is None                # scan-agnostic (composite)


def test_rerun_replaces_labels_and_spares_user(dia_ensemble, monkeypatch):
    _install_fake_annotate(monkeypatch)

    # A pre-existing manual annotation must survive the auto pass.
    dia_ensemble.add_generic_annot(
        cofeature_idx=1, ms_level=1, text='mine', source='user'
    )

    annotate_ensembles_dia([dia_ensemble])
    first = [x for x in dia_ensemble.generic_annots.values()
             if x.source == ADDUCT_ANNOT_SOURCE]
    annotate_ensembles_dia([dia_ensemble])
    second = [x for x in dia_ensemble.generic_annots.values()
              if x.source == ADDUCT_ANNOT_SOURCE]

    assert len(first) == len(second) == 1          # replaced, not stacked
    user = [x for x in dia_ensemble.generic_annots.values() if x.source == 'user']
    assert len(user) == 1 and user[0].text == 'mine'


def test_dda_ensemble_is_skipped(dia_ensemble, monkeypatch):
    _install_fake_annotate(monkeypatch)
    monkeypatch.setattr(dia_ensemble.injection, 'acquisition_mode', 'dda')
    assert dia_ensemble.is_dda is True

    out = annotate_ensembles_dia([dia_ensemble])
    assert out == []
    assert dia_ensemble.generic_annots == {}       # nothing attached


# --- params -> find-mfs, provenance -> assignment -----------------------------

def test_annotate_forwards_params(dia_ensemble, monkeypatch):
    """Every FindMfsParams setting -- constraints AND scoring weights -- must
    reach find-mfs (the weights used to be dropped on this path)."""
    captured = {}
    res = _install_fake_annotate(monkeypatch)

    def fake(*a, **k):
        captured.update(k)
        return res

    monkeypatch.setattr('find_mfs.annotate_analyte_dia', fake, raising=False)

    params = FindMfsParams(
        max_counts='C*H*N*O*P0S1', halogen_cap='Cl2Br1',
        iso_weight=0.5, chem_weight=2.0, mass_weight=3.0, error_ppm=6.0,
    )
    annotate_ensembles_dia([dia_ensemble], params=params)

    assert captured['max_counts'] == 'C*H*N*O*P0S1'
    assert captured['halogen_cap'] == 'Cl2Br1'
    assert captured['iso_weight'] == 0.5
    assert captured['chem_weight'] == 2.0
    assert captured['mass_weight'] == 3.0
    assert captured['mass_sigma_ppm'] == pytest.approx(2.0)
    assert captured['finder_kwargs']['check_octet'] is True


def test_assignment_records_search_provenance(dia_ensemble, monkeypatch):
    _install_fake_annotate(monkeypatch)
    params = FindMfsParams(halogen_cap='Cl4Br4')

    a = annotate_ensembles_dia([dia_ensemble], params=params)[0]

    assert a.elements == 'CHNOClBr'
    assert a.max_counts == 'C*H*N3O*Cl4Br4'
    assert a.min_counts is None                    # all-zero minimum -> none
    assert a.halogen_detected is True
    assert a.params['halogen_cap'] == 'Cl4Br4'


def test_invalid_params_fail_once_up_front(dia_ensemble, monkeypatch):
    """A bad constraint must raise before the batch starts, not be swallowed
    per ensemble as if it had no resolvable envelope."""
    _install_fake_annotate(monkeypatch)
    with pytest.raises(ValueError, match='halogen_cap'):
        annotate_ensembles_dia(
            [dia_ensemble], params=FindMfsParams(halogen_cap='F2'),
        )


# --- GenericAnnotation.source persistence semantics -------------------------

def test_generic_annotation_source_roundtrips():
    # Mirrors persistence.py: save = asdict(annot); load = GenericAnnotation(**d).
    a = GenericAnnotation(cofeature_idx=1, ms_level=1, text='[M+H]+',
                          source='auto_adduct')
    d = asdict(a)
    assert d['source'] == 'auto_adduct'
    assert GenericAnnotation(**d) == a

    # Old .mzk files predate the field -> default to 'user'.
    d.pop('source')
    assert GenericAnnotation(**d).source == 'user'
