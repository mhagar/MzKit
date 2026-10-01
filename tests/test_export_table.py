"""
Tests for the feature-table + companion-MGF/GraphML export
(core/cli/export_table.py, core/cli/export_network.py).

The tables and network are tested with lightweight stubs (no MS data
needed). The MGF/file writing is tested end-to-end against the gitignored
multi-sample alignment fixture, and skipped when that file is absent.
"""
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from core.cli.cluster_analytes import ClusterParams
from core.cli.export_network import build_network_graphml
from core.cli.export_table import (
    export_abundance_table,
    export_feature_mgf,
    export_feature_table_to_file,
    export_formula_table,
)
from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment
from core.utils.array_types import to_spec_arr

TEST_ALIGNMENT_MZK = Path(
    '/home/mh/Dropbox/MzKit/tests/test_files/test_alignment.mzk'
)


# --- tables (pure, stubbed) -------------------------------------------------

def _spec(peaks):
    mz, intsy = zip(*peaks)
    return to_spec_arr(
        mz_arr=np.asarray(mz, dtype=float),
        intsy_arr=np.asarray(intsy, dtype=float),
    )


def _ens(uuid, base_intsy, frags=None, precursor=200.0):
    return SimpleNamespace(
        uuid=uuid,
        base_intsy=base_intsy,
        composite_spectrum=SimpleNamespace(
            ms1=_spec([(precursor, 1.0)]),
            ms2=_spec(frags) if frags else None,
            precursor_mz=precursor,
        ),
    )


def _sample(uuid, name, ensembles):
    return SimpleNamespace(
        uuid=uuid, name=name,
        injection=SimpleNamespace(ensembles={e.uuid: e for e in ensembles}),
    )


@pytest.fixture
def stubbed():
    """
    Five samples. Analyte 0 is in all five (C6H12O6 x3, C7H8 x1, none x1);
    analyte 1 is in samples 1-2 only; analyte 2 is mapped in sample 3 but
    its ensemble is missing (unresolvable).
    """
    s = {
        1: [_ens(11, 100.0), _ens(12, 7.0)],
        2: [_ens(21, 500.0), _ens(22, 9.0)],
        3: [_ens(31, 300.0)],
        4: [_ens(41, 200.0)],
        5: [_ens(51, 50.0)],
    }
    samples = {u: _sample(u, f"S{u}", ens) for u, ens in s.items()}
    analytes = [
        AlignedAnalyte(
            ensemble_map={1: 11, 2: 21, 3: 31, 4: 41, 5: 51},
            consensus_mz=181.07, consensus_rt=60.0,
        ),
        AlignedAnalyte(
            ensemble_map={1: 12, 2: 22}, consensus_mz=93.07, consensus_rt=120.0,
        ),
        AlignedAnalyte(ensemble_map={3: 999}, consensus_rt=200.0),
    ]
    alignment = EnsembleAlignment(
        sample_uuids=tuple(samples), analytes=analytes,
    )
    formulas = {
        11: 'C6H12O6', 21: 'C6H12O6', 31: 'C7H8', 41: 'C6H12O6',
        12: 'C7H8',
    }
    names = {u: smp.name for u, smp in samples.items()}
    return alignment, samples, names, formulas


def _rows(text, sep='\t'):
    return [line.split(sep) for line in text.rstrip('\n').split('\n')]


def test_abundance_table(stubbed):
    alignment, samples, names, _ = stubbed
    header, a0, a1, a2 = _rows(
        export_abundance_table(alignment, samples, names)
    )
    assert header == [
        'analyte_id', 'consensus_mz', 'consensus_rt', 'S1', 'S2', 'S3', 'S4', 'S5',
    ]
    assert a0[3:] == ['100.0', '500.0', '300.0', '200.0', '50.0']
    assert a1[3:] == ['7.0', '9.0', '0', '0', '0']
    # Mapped but unresolvable -> empty, not '0'
    assert a2[3:] == ['0', '0', '', '0', '0']


def test_formula_table(stubbed):
    alignment, samples, names, formulas = stubbed
    header, a0, a1, a2 = _rows(
        export_formula_table(alignment, samples, names, formulas, ',')
        .replace(',', '\t')
    )
    assert header[3:6] == ['consensus_formula', 'count', 'total']
    assert a0[3:] == [
        'C6H12O6', '3', '5', 'C6H12O6', 'C6H12O6', 'C7H8', 'C6H12O6', '0',
    ]
    assert a1[3:] == ['C7H8', '1', '2', 'C7H8', '0', '', '', '']
    assert a2[3:] == ['', '0', '0', '', '', '', '', '']


def test_consensus_formula_tie_goes_to_representative():
    analyte = AlignedAnalyte(ensemble_map={1: 10, 2: 20})
    members = {1: _ens(10, 1.0), 2: _ens(20, 5.0)}   # 20 is tallest
    result = analyte.consensus_formula(members, {10: 'C2H6', 20: 'CH4'})
    assert result.formula == 'CH4'
    assert (result.count, result.total) == (1, 2)


def test_graphml_network():
    frags_a = [(80.0, 30.0), (95.0, 50.0), (120.0, 100.0), (150.0, 40.0)]
    frags_b = [(80.0, 30.0), (95.0, 50.0), (134.0, 100.0), (150.0, 40.0)]
    frags_c = [(61.0, 100.0), (73.0, 60.0), (89.0, 20.0), (201.0, 70.0)]
    ensembles = [
        _ens(10, 1.0, frags_a, 200.0),
        _ens(11, 1.0, frags_c, 250.0),
        _ens(12, 1.0, None, 300.0),          # no MS2 -> singleton
        _ens(13, 1.0, frags_b, 214.0),       # analogue of 10
    ]
    sample = _sample(1, 'S1', ensembles)
    analytes = [
        AlignedAnalyte(ensemble_map={1: e.uuid}, consensus_mz=e.composite_spectrum.precursor_mz)
        for e in ensembles
    ]
    alignment = EnsembleAlignment(sample_uuids=(1,), analytes=analytes)

    text = build_network_graphml(
        alignment, [sample], ClusterParams(), formulas={10: 'C6H6'},
    )
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    graph = ET.fromstring(text).find('g:graph', ns)

    nodes = graph.findall('g:node', ns)
    assert [n.get('id') for n in nodes] == ['0', '1', '2', '3']
    node0 = {d.get('key'): d.text for d in nodes[0].findall('g:data', ns)}
    assert node0['consensus_formula'] == 'C6H6'
    assert node0['has_ms2'] == 'true'

    edges = graph.findall('g:edge', ns)
    assert [(e.get('source'), e.get('target')) for e in edges] == [('0', '3')]
    edge = {d.get('key'): d.text for d in edges[0].findall('g:data', ns)}
    assert float(edge['cosine']) > 0.9
    assert float(edge['mz_delta']) == pytest.approx(14.0)


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


def _representative(analyte, by_uuid):
    return analyte.representative_spectrum(analyte.resolve_members(by_uuid.get))


def test_export_feature_mgf_consensus(aligned):
    samples, alignment = aligned
    by_uuid = _samples_by_uuid(samples)

    mgf = export_feature_mgf(alignment, by_uuid, mode='consensus')
    assert mgf.strip(), "consensus MGF should not be empty"

    n_begin = mgf.count('BEGIN IONS')
    n_end = mgf.count('END IONS')
    assert n_begin == n_end > 0

    # Every analyte with a representative contributes exactly one FEATURE_ID,
    # and at most one MS1 + one MS2 block.
    n_detectable = sum(
        1 for a in alignment.analytes
        if _representative(a, by_uuid) is not None
    )
    feature_ids = [
        line for line in mgf.splitlines() if line.startswith('FEATURE_ID=')
    ]
    assert len(set(feature_ids)) == n_detectable
    assert len(feature_ids) <= 2 * n_detectable


def test_export_feature_mgf_writes_consensus_formula(aligned):
    """In consensus mode FORMULA is the analyte's consensus formula, even if
    the representative ensemble itself has none."""
    samples, alignment = aligned
    by_uuid = _samples_by_uuid(samples)

    analyte = next(
        a for a in alignment.analytes
        if len(a.resolve_members(by_uuid.get)) >= 2
    )
    rep = _representative(analyte, by_uuid)
    other = next(
        e for e in analyte.resolve_members(by_uuid.get).values()
        if e.uuid != rep.ensemble_uuid
    )
    mgf = export_feature_mgf(
        alignment, by_uuid, mode='consensus',
        formulas={other.uuid: 'C8H10N4O2'},
    )
    assert mgf.count('FORMULA=C8H10N4O2') >= 1
    assert 'FORMULA=' not in export_feature_mgf(alignment, by_uuid, mode='consensus')


def test_export_feature_mgf_per_sample_has_more_entries(aligned):
    samples, alignment = aligned
    by_uuid = _samples_by_uuid(samples)

    per_sample = export_feature_mgf(alignment, by_uuid, mode='per_sample')
    n_members = sum(
        len(a.resolve_members(by_uuid.get)) for a in alignment.analytes
    )
    # One FEATURE_ID-stamped entry per resolvable member, at least.
    assert per_sample.count('SAMPLE=') >= n_members
    assert 'SAMPLE=' not in export_feature_mgf(alignment, by_uuid, mode='consensus')


def test_export_to_file_writes_siblings(aligned, tmp_path):
    samples, alignment = aligned
    by_uuid = _samples_by_uuid(samples)
    names = {s.uuid: s.name for s in samples}

    written = export_feature_table_to_file(
        alignment=alignment,
        samples=by_uuid,
        sample_names=names,
        output=tmp_path / 'features.tsv',
    )

    assert written == [
        tmp_path / 'features_abundance.tsv',
        tmp_path / 'features_formulas.tsv',
        tmp_path / 'features.mgf',
        tmp_path / 'features.graphml',
    ]
    assert all(p.exists() for p in written)
    assert 'BEGIN IONS' in (tmp_path / 'features.mgf').read_text()
    ET.parse(tmp_path / 'features.graphml')


def test_export_to_file_respects_no_mgf_no_graphml(aligned, tmp_path):
    samples, alignment = aligned
    by_uuid = _samples_by_uuid(samples)
    names = {s.uuid: s.name for s in samples}

    written = export_feature_table_to_file(
        alignment=alignment,
        samples=by_uuid,
        sample_names=names,
        output=tmp_path / 'features.csv',
        separator=',',
        write_mgf=False,
        write_graphml=False,
    )

    assert written == [
        tmp_path / 'features_abundance.csv',
        tmp_path / 'features_formulas.csv',
    ]
    assert not (tmp_path / 'features.mgf').exists()
