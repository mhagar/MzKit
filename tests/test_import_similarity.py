"""
Tests for imported similarity matrices (core/cli/import_similarity.py).
"""
import numpy as np
import pytest

from core.cli.cluster_analytes import (
    MODIFIED_COSINE, ClusterKey, ClusterParams, analyte_ms2_spectra, cluster_analytes,
    cluster_matrix, similarity_matrix,
)
from core.cli.import_similarity import (
    ImportParams,
    read_similarity_csv,
    transform,
)
from core.data_structs.similarity_matrix import SimilarityMatrix
from core.data_structs.alignment import EnsembleAlignment
from tests.test_cluster_analytes import FRAGS_A, FRAGS_B, FRAGS_C, _build


def _alignment(n):
    alignment, _, analytes = _build(
        [(10.0 * i, 200.0 + i, FRAGS_A) for i in range(n)]
    )
    return alignment, analytes


def _raw(analytes, values, extra_ids=()):
    ids = tuple(str(a.uuid) for a in analytes) + tuple(extra_ids)
    return SimilarityMatrix(
        name='m', ids=ids, values=np.asarray(values, float),
        provenance={'source': 'imported', 'params': {}},
        default_transform=ImportParams(),
    )


NO_FLOOR = ImportParams(floor=0.0)


# --- reading -------------------------------------------------------------------

def test_read_csv_reorders_columns(tmp_path):
    path = tmp_path / 'scores.csv'
    path.write_text(",b,a\na,0.2,1\nb,1,0.2\n")
    raw = read_similarity_csv(path)
    assert raw.ids == ('a', 'b')
    assert np.allclose(raw.values, [[1, 0.2], [0.2, 1]])
    assert raw.name == 'scores'
    assert raw.provenance['source'] == 'imported'
    assert raw.provenance['path'] == str(path)
    assert raw.default_transform == ImportParams()
    assert raw.values.dtype == np.float32


def test_read_tsv_empty_cell_is_nan(tmp_path):
    path = tmp_path / 'scores.tsv'
    path.write_text("\ta\tb\na\t1\t\nb\t0.3\t1\n")
    raw = read_similarity_csv(path)
    assert np.isnan(raw.values[0, 1])


def test_read_preserves_big_int_ids(tmp_path):
    big = str(2**127 + 12345)
    path = tmp_path / 'm.csv'
    path.write_text(f",{big}\n{big},1\n")
    assert read_similarity_csv(path).ids == (big,)


@pytest.mark.parametrize('text', [
    ",a,b\na,1,0\n",               # not square
    ",a,b\na,1,0\nc,0,1\n",        # labels differ
    ",a,b\na,1,x\nb,0,1\n",        # non-numeric
])
def test_read_rejects_malformed(tmp_path, text):
    path = tmp_path / 'bad.csv'
    path.write_text(text)
    with pytest.raises(ValueError):
        read_similarity_csv(path)


# --- transform -----------------------------------------------------------------

def test_matching_unknown_and_uncovered():
    alignment, analytes = _alignment(3)
    raw = _raw(analytes[:2], np.eye(3), extra_ids=['stranger'])
    result = transform(raw, alignment, NO_FLOOR)
    assert [a.uuid for a in result.analytes] == [a.uuid for a in analytes[:2]]
    assert result.report.n_matched == 2
    assert result.report.n_unknown == 1
    assert result.report.n_uncovered == 1


def test_filtered_subset_reuses_matrix():
    alignment, analytes = _alignment(3)
    raw = _raw(analytes, [[1, .9, .1], [.9, 1, .2], [.1, .2, 1]])
    child = EnsembleAlignment(
        sample_uuids=alignment.sample_uuids, analytes=[analytes[0], analytes[2]],
    )
    result = transform(raw, child, NO_FLOOR)
    assert result.report.n_uncovered == 0
    assert np.allclose(result.similarity, [[1, .1], [.1, 1]])


def test_nan_clip_rescale():
    alignment, analytes = _alignment(3)
    raw = _raw(analytes, [
        [1, np.nan, -0.5],
        [np.nan, 1, 0.0],
        [-0.5, 0.0, 1],
    ])
    result = transform(
        raw, alignment, ImportParams(clip_min=-1.0, clip_max=1.0, floor=0.0),
    )
    # NaN -> clip_min -> 0; -0.5 -> 0.25; 0 -> 0.5
    assert result.similarity[0, 1] == 0.0
    assert np.isclose(result.similarity[0, 2], 0.25)
    assert np.isclose(result.similarity[1, 2], 0.5)
    assert result.report.n_nan == 2
    assert result.report.n_clipped == 0


def test_distance_inverted():
    alignment, analytes = _alignment(2)
    raw = _raw(analytes, [[0, 2.0], [2.0, 0]])
    params = ImportParams(is_distance=True, clip_max=8.0, floor=0.0)
    result = transform(raw, alignment, params)
    assert np.isclose(result.similarity[0, 1], 0.75)
    assert np.allclose(np.diag(result.similarity), 1.0)


def test_symmetrize_mean_vs_bidirectional():
    alignment, analytes = _alignment(3)
    raw = _raw(analytes, [
        [1, 0.9, 0.9],
        [0.5, 1, 0.0],
        [0.8, 0.0, 1],
    ])
    mean = transform(raw, alignment, NO_FLOOR)
    assert np.isclose(mean.similarity[0, 1], 0.7)
    assert np.isclose(mean.report.max_asymmetry, 0.4)

    bidir = transform(raw, alignment, ImportParams(
        bidirectional=True, bidirectional_threshold=0.7, floor=0.0,
    ))
    assert bidir.similarity[0, 1] == 0.0       # 0.5 one way
    assert np.isclose(bidir.similarity[0, 2], 0.8)


def test_floor():
    alignment, analytes = _alignment(2)
    raw = _raw(analytes, [[1, 0.55], [0.55, 1]])
    assert transform(raw, alignment, ImportParams(floor=0.6)).similarity[0, 1] == 0
    result = transform(raw, alignment, ImportParams(floor=0.5))
    assert np.isclose(result.similarity[0, 1], 0.55)
    assert result.report.n_pairs_kept == 1


def test_bad_clip_range():
    alignment, analytes = _alignment(2)
    with pytest.raises(ValueError):
        transform(_raw(analytes, np.eye(2)), alignment,
                  ImportParams(clip_min=1.0, clip_max=1.0))


# --- clustering / export -------------------------------------------------------

def test_imported_cosine_matrix_clusters_like_cosine():
    alignment, samples, _ = _build([
        (300.0, 200.0, FRAGS_A),
        (100.0, 250.0, FRAGS_C),
        (500.0, 214.0, FRAGS_B),
        (50.0, 180.0, None),
    ])
    params = ClusterParams(scoring=MODIFIED_COSINE)
    expected = cluster_analytes(alignment, samples, params)

    ms2_analytes, spectra = analyte_ms2_spectra(alignment, samples)
    sim, _ = similarity_matrix(spectra, params)
    raw = _raw(ms2_analytes, sim)
    result = cluster_matrix(alignment, raw, NO_FLOOR, 'average')

    assert result.analyte_uuids == expected.analyte_uuids
    assert np.allclose(result.similarity, expected.similarity)
    assert np.allclose(result.nodes.tolist(), expected.nodes.tolist())
    assert result.n_dropped == 1
    assert result.params == ClusterKey(raw.uuid, NO_FLOOR, 'average')
    assert result.dropped_reason == "not in m"
    assert expected.dropped_reason == "without MS2"


# --- CLI ------------------------------------------------------------------------

def test_cli_import_saves_matrix_into_mzk(tmp_path):
    from core.cli.main import build_parser
    from core.data_structs.data_registry import DataRegistry
    from core.utils.persistence import load_project, save_project

    alignment, _, analytes = _build(
        [(10.0 * i, 200.0 + i, FRAGS_A) for i in range(3)]
    )
    registry = DataRegistry()
    registry.register_alignment(alignment)
    mzk = tmp_path / 'p.mzk'
    save_project(mzk, registry)

    ids = [str(a.uuid) for a in analytes]
    matrix = tmp_path / 'scores.csv'
    matrix.write_text(
        ',' + ','.join(ids) + '\n'
        + ''.join(f"{i},1,0.9,0.1\n" for i in ids)
    )

    def run(path, *extra):
        args = build_parser().parse_args(
            ['import-similarity', str(mzk), str(path), *extra]
        )
        args.func(args)

    run(matrix, '--floor', '0.5')
    run(matrix, '--floor', '0.3')  # re-import replaces, with the new params
    matrices = load_project(mzk).matrices
    assert len(matrices) == 1
    assert matrices[0].ids == tuple(ids)
    assert matrices[0].default_transform.floor == 0.3

    other = tmp_path / 'unrelated.csv'
    other.write_text(",x,y\nx,1,0\ny,0,1\n")
    with pytest.raises(ValueError, match='not saved'):
        run(other)
    assert len(load_project(mzk).matrices) == 1
