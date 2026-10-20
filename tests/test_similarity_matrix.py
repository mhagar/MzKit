"""
Tests for the unified SimilarityMatrix: computed and imported matrices take
the same clustering path, live in the DataRegistry, and persist in .mzk.
"""
import numpy as np

from core.cli.cluster_analytes import (
    MS2DEEPSCORE, ClusterParams, analyte_ms2_spectra, cluster_analytes,
    cluster_from_similarity, compute_similarity_matrix, similarity_matrix,
    similarity_provenance,
)
from core.data_structs import DataRegistry, SimilarityMatrix
from core.data_structs.alignment import EnsembleAlignment
from core.data_structs.similarity_matrix import ImportParams
from core.utils import persistence
from tests.test_cluster_analytes import FRAGS_A, FRAGS_B, FRAGS_C, _build
from tests.test_export_table import aligned  # noqa: F401 (fixture)


def _synthetic():
    return _build([
        (300.0, 200.0, FRAGS_A),
        (100.0, 250.0, FRAGS_C),
        (500.0, 214.0, FRAGS_B),
        (50.0, 180.0, None),
    ])


def _imported(ids):
    return SimilarityMatrix(
        name='simba', ids=tuple(ids), values=np.eye(len(ids), dtype=np.float32),
        provenance={'source': 'imported', 'params': {}, 'path': '/x/simba.csv'},
        default_transform=ImportParams(is_distance=True, floor=0.3),
    )


def test_computed_matrix_shape():
    alignment, samples, analytes = _synthetic()
    matrix = compute_similarity_matrix(alignment, samples, ClusterParams())
    assert matrix.ids == tuple(str(a.uuid) for a in analytes[:3])
    assert matrix.uncovered == (str(analytes[3].uuid),)
    assert matrix.values.dtype == np.float32
    assert matrix.default_transform == ImportParams(floor=0.0)
    assert matrix.provenance == similarity_provenance(ClusterParams())
    assert not matrix.is_imported


def test_new_path_matches_old_on_test_alignment(aligned):  # noqa: F811
    """transform(floor=0) on computed cosine is a no-op: same clustering."""
    samples, alignment = aligned
    params = ClusterParams()
    analytes, spectra = analyte_ms2_spectra(alignment, samples)
    sim, _ = similarity_matrix(spectra, params)
    old = cluster_from_similarity(
        analytes, sim, 'average',
        n_dropped=len(alignment.analytes) - len(analytes), params=None,
    )
    new = cluster_analytes(alignment, samples, params)

    assert new.analyte_uuids == old.analyte_uuids
    assert np.allclose(new.similarity, old.similarity, atol=1e-6)
    # Same dendrogram; ties (identical spectra) may merge in another order
    by_range = lambda c: sorted(c.nodes.tolist(), key=lambda n: (n[5], n[6]))
    assert np.allclose(by_range(new), by_range(old), atol=1e-6)
    assert new.n_dropped == old.n_dropped


def test_registry_lookup_reuse_and_orphans():
    alignment, samples, analytes = _synthetic()
    child = EnsembleAlignment(
        sample_uuids=alignment.sample_uuids, analytes=analytes[:2],
    )
    reg = DataRegistry()
    reg.register_alignment(alignment)
    reg.register_alignment(child)

    computed = compute_similarity_matrix(alignment, samples, ClusterParams())
    imported = _imported([str(analytes[0].uuid), 'stranger'])
    added = []
    reg.subscribe_to_changes(added.append, lambda m: None, change_type='Matrix')
    reg.register_matrix(computed)
    reg.register_matrix(imported)
    assert added == [computed, imported]

    assert reg.matrices_for_analytes([analytes[1].uuid]) == [computed]
    assert reg.matrices_for_analytes([a.uuid for a in child.analytes]) == [computed, imported]

    # Reuse: same provenance, alignment covered by ids ∪ uncovered (incl. a child)
    prov = similarity_provenance(ClusterParams())
    assert reg.find_matrix(prov, alignment) is computed
    assert reg.find_matrix(prov, child) is computed
    assert reg.find_matrix(similarity_provenance(ClusterParams(tolerance=0.05)), alignment) is None
    assert reg.find_matrix(similarity_provenance(ClusterParams(scoring=MS2DEEPSCORE)), alignment) is None

    # Removing the parent keeps both (the child still shares ids)...
    reg.remove_alignment(alignment.uuid)
    assert reg.get_all_matrices() == [computed, imported]
    # ...and a matrix sharing no id with what's left is collected
    reg.register_alignment(EnsembleAlignment(
        sample_uuids=alignment.sample_uuids, analytes=[analytes[2]],
    ))
    reg.remove_alignment(child.uuid)
    assert reg.get_all_matrices() == [computed]

    reg.clear()
    assert reg.get_all_matrices() == []


def test_matrix_roundtrip(tmp_path):
    alignment, samples, analytes = _synthetic()
    computed = compute_similarity_matrix(alignment, samples, ClusterParams())
    imported = _imported([str(a.uuid) for a in analytes])
    imported.values[0, 1] = np.nan

    reg = DataRegistry()
    reg.register_matrix(computed)
    reg.register_matrix(imported)
    path = tmp_path / 'matrices.mzk'
    persistence.save_project(filepath=path, data_registry=reg)
    loaded = {m.uuid: m for m in persistence.load_project(filepath=path).matrices}

    assert set(loaded) == {computed.uuid, imported.uuid}
    for original in (computed, imported):
        m = loaded[original.uuid]
        assert m.name == original.name
        assert m.ids == original.ids
        assert m.uncovered == original.uncovered
        assert m.provenance == original.provenance
        assert m.default_transform == original.default_transform
        assert isinstance(m.default_transform, ImportParams)
        assert m.values.dtype == original.values.dtype
        assert np.array_equal(m.values, original.values, equal_nan=True)
