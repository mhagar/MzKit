"""
Tests for the feature-table + companion-MGF export
(core/cli/export_table.py).

The `_best_ensemble` selection logic is tested with lightweight stubs (no MS
data needed). The MGF/table writing is tested end-to-end against the gitignored
multi-sample alignment fixture, and skipped when that file is absent.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.cli.export_table import (
    _best_ensemble,
    export_feature_mgf,
    export_feature_table_to_file,
)

TEST_ALIGNMENT_MZK = Path(
    '/home/mh/Dropbox/MzKit/tests/test_files/test_alignment.mzk'
)


# --- _best_ensemble (pure, stubbed) ----------------------------------------

def _sample_with_ensembles(ensembles: dict):
    """A stub Sample whose injection.ensembles is the given dict."""
    injection = SimpleNamespace(ensembles=ensembles)
    return SimpleNamespace(injection=injection, name='stub')


def test_best_ensemble_picks_highest_base_intsy():
    ens_a = SimpleNamespace(base_intsy=10.0)
    ens_b = SimpleNamespace(base_intsy=99.0)
    ens_c = SimpleNamespace(base_intsy=50.0)
    samples = {
        1: _sample_with_ensembles({11: ens_a}),
        2: _sample_with_ensembles({22: ens_b}),
        3: _sample_with_ensembles({33: ens_c}),
    }
    analyte = SimpleNamespace(ensemble_map={1: 11, 2: 22, 3: 33})
    assert _best_ensemble(analyte, samples) is ens_b


def test_best_ensemble_skips_missing_samples_and_ensembles():
    ens = SimpleNamespace(base_intsy=5.0)
    samples = {
        1: _sample_with_ensembles({11: ens}),
        2: SimpleNamespace(injection=None, name='no-injection'),
    }
    # sample 3 not in `samples`; sample 2 has no injection; sample 1's
    # ensemble uuid mismatches the map -> falls through to None.
    analyte = SimpleNamespace(ensemble_map={1: 999, 2: 22, 3: 33})
    assert _best_ensemble(analyte, samples) is None


def test_best_ensemble_none_when_empty():
    analyte = SimpleNamespace(ensemble_map={})
    assert _best_ensemble(analyte, {}) is None


# --- integration (needs the alignment fixture) ------------------------------

@pytest.fixture
def aligned():
    if not TEST_ALIGNMENT_MZK.exists():
        pytest.skip(
            f"test data not present: {TEST_ALIGNMENT_MZK} "
            "(MS data is gitignored; place it locally to run this test)"
        )
    from core.utils.persistence import load_project
    samples, alignments, _assignments = load_project(TEST_ALIGNMENT_MZK)
    if not alignments:
        from core.data_structs.alignment import AlignmentParams
        import core.cli.align_ensembles as A
        params = AlignmentParams(
            rt_tolerance=10.0, mz_tolerance=0.01,
            ms1_similarity_threshold=0.7, ms2_similarity_threshold=0.6,
        )
        alignment = A.align_ensembles(samples, params)
    else:
        alignment = alignments[0]
    return samples, alignment


def _samples_by_uuid(samples):
    return {s.uuid: s for s in samples}


def test_export_feature_mgf_consensus(aligned):
    samples, alignment = aligned
    by_uuid = _samples_by_uuid(samples)

    mgf = export_feature_mgf(alignment, by_uuid, mode='consensus')
    assert mgf.strip(), "consensus MGF should not be empty"

    n_begin = mgf.count('BEGIN IONS')
    n_end = mgf.count('END IONS')
    assert n_begin == n_end > 0

    # Every analyte with a detected ensemble contributes exactly one FEATURE_ID.
    n_detectable = sum(
        1 for a in alignment.analytes
        if _best_ensemble(a, by_uuid) is not None
    )
    feature_ids = [
        line for line in mgf.splitlines() if line.startswith('FEATURE_ID=')
    ]
    assert len(set(feature_ids)) == n_detectable


def test_export_feature_mgf_writes_accepted_formulas(aligned):
    """FORMULA tags come from the formulas map (DataRegistry.chosen_formulas),
    only for ensembles that have an accepted formula."""
    samples, alignment = aligned
    by_uuid = _samples_by_uuid(samples)

    first = next(
        e for a in alignment.analytes
        if (e := _best_ensemble(a, by_uuid)) is not None
    )
    mgf = export_feature_mgf(
        alignment, by_uuid, mode='consensus',
        formulas={first.uuid: 'C8H10N4O2'},
    )
    assert mgf.count('FORMULA=C8H10N4O2') >= 1
    assert 'FORMULA=' not in export_feature_mgf(alignment, by_uuid, mode='consensus')


def test_export_feature_mgf_per_sample_has_more_entries(aligned):
    samples, alignment = aligned
    by_uuid = _samples_by_uuid(samples)

    consensus = export_feature_mgf(alignment, by_uuid, mode='consensus')
    per_sample = export_feature_mgf(alignment, by_uuid, mode='per_sample')

    # per-sample explodes each multi-sample analyte into several entries, so it
    # can never have fewer BEGIN IONS blocks than consensus.
    assert per_sample.count('BEGIN IONS') >= consensus.count('BEGIN IONS')
    # SAMPLE tags are only stamped in per-sample mode.
    assert 'SAMPLE=' in per_sample
    assert 'SAMPLE=' not in consensus


def test_export_to_file_writes_sibling_mgf(aligned, tmp_path):
    samples, alignment = aligned
    by_uuid = _samples_by_uuid(samples)
    names = {s.uuid: s.name for s in samples}

    out = tmp_path / 'features.tsv'
    mgf_path = export_feature_table_to_file(
        alignment=alignment,
        samples=by_uuid,
        sample_names=names,
        output=out,
    )

    assert out.exists()
    assert mgf_path == tmp_path / 'features.mgf'
    assert mgf_path.exists()
    assert 'BEGIN IONS' in mgf_path.read_text()


def test_export_to_file_respects_no_mgf(aligned, tmp_path):
    samples, alignment = aligned
    by_uuid = _samples_by_uuid(samples)
    names = {s.uuid: s.name for s in samples}

    out = tmp_path / 'features.tsv'
    mgf_path = export_feature_table_to_file(
        alignment=alignment,
        samples=by_uuid,
        sample_names=names,
        output=out,
        write_mgf=False,
    )

    assert out.exists()
    assert mgf_path is None
    assert not (tmp_path / 'features.mgf').exists()
