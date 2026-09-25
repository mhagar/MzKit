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
    BatchAnnotationResult,
    EnsembleSelection,
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

    out = annotate_ensembles_dia([dia_ensemble]).assignments

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

    result = annotate_ensembles_dia([dia_ensemble])
    assert result.assignments == []
    assert result.n_dda == 1
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

    a = annotate_ensembles_dia([dia_ensemble], params=params).assignments[0]

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


# --- batch selection + skip reporting ----------------------------------------

def _ens(intsy, dda=False):
    return SimpleNamespace(base_intsy=intsy, is_dda=dda, uuid=int(intsy))


def test_selection_ranks_and_limits():
    ensembles = [_ens(x) for x in (5e4, 3e6, 2e5, 1e3, 8e5)]

    sel, below, beyond = EnsembleSelection().select(ensembles)
    assert [e.base_intsy for e in sel] == [3e6, 8e5, 2e5, 5e4, 1e3]   # no limits
    assert (below, beyond) == (0, 0)

    sel, below, beyond = EnsembleSelection(
        limit_intensity=True, min_base_intsy=1e5,
        limit_count=True, max_ensembles=2,
    ).select(ensembles)
    assert [e.base_intsy for e in sel] == [3e6, 8e5]
    assert (below, beyond) == (2, 1)          # floor applies before the count


def test_selection_values_kept_while_disabled():
    sel = EnsembleSelection(limit_count=False, max_ensembles=1)
    assert len(sel.select([_ens(1.0), _ens(2.0)])[0]) == 2


def test_selection_config_roundtrip():
    from configparser import ConfigParser
    sel = EnsembleSelection(limit_count=True, max_ensembles=75,
                            limit_intensity=True, min_base_intsy=2.5e5)
    cfg = ConfigParser()
    sel.to_config(cfg)
    assert EnsembleSelection.from_config(cfg) == sel
    assert EnsembleSelection.from_config(ConfigParser()) == EnsembleSelection()


def test_batch_reports_every_skip(dia_ensemble, monkeypatch):
    """DDA, below-floor, beyond-count and no-envelope ensembles are each counted,
    and the survivors are annotated most intense first."""
    res = _install_fake_annotate(monkeypatch)
    annotated = []

    def fake(ms1_peaks=None, **k):
        annotated.append(fake.current)
        if fake.current.base_intsy == 4e5:
            raise ValueError("No signal groups in MS1")
        return res

    monkeypatch.setattr('find_mfs.annotate_analyte_dia', fake, raising=False)
    import core.cli.auto_find_mfs as afm
    real_one = afm._annotate_one

    def spy(ensemble, **k):
        fake.current = ensemble
        return real_one(dia_ensemble, **k)       # real conversion, real ensemble

    monkeypatch.setattr(afm, '_annotate_one', spy)

    ensembles = [
        _ens(1e6), _ens(4e5), _ens(9e5), _ens(2e5),   # DIA, ranked 1e6 > 9e5 > 4e5 > 2e5
        _ens(5e4),                                    # below the floor
        _ens(3e6, dda=True),                          # DDA
    ]
    result = annotate_ensembles_dia(
        ensembles,
        selection=EnsembleSelection(limit_intensity=True, min_base_intsy=1e5,
                                    limit_count=True, max_ensembles=3),
    )

    assert [e.base_intsy for e in annotated] == [1e6, 9e5, 4e5]
    assert len(result.assignments) == 2
    assert (result.n_ensembles, result.n_dda, result.n_below_intensity,
            result.n_beyond_count, result.n_no_envelope) == (6, 1, 1, 1, 1)
    assert result.summary() == (
        "Annotated 2 of 6 ensembles; skipped 1 below min. intensity, "
        "1 beyond the N most intense, 1 with no resolvable envelope, "
        "1 DDA (not supported yet)"
    )


def test_cancelled_batch_counts_unreached(dia_ensemble, monkeypatch):
    import threading
    _install_fake_annotate(monkeypatch)
    cancel = threading.Event()
    cancel.set()
    result = annotate_ensembles_dia([dia_ensemble], cancel_event=cancel)
    assert result.assignments == [] and result.n_cancelled == 1
    assert "1 not reached (cancelled)" in result.summary()


def test_summary_mentions_empty_searches():
    from core.data_structs.formula_assignment import FormulaAssignment
    r = BatchAnnotationResult(
        assignments=[FormulaAssignment(source_uuid=1, candidates=[])],
        n_ensembles=1,
    )
    assert r.summary() == "Annotated 1 of 1 ensemble (1 with no candidate formula)"


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
