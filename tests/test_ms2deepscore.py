"""
Tests for the NumPy MS2DeepScore port (core/utils/ms2deepscore.py) against
torch reference outputs in ms2deepscore_parity.npz (written by
tools/ms2deepscore/export_weights.py). Skipped when models/ has no weights.
"""
from pathlib import Path

import numpy as np
import pytest

from core.cli.cluster_analytes import MS2DEEPSCORE, ClusterParams, cluster_analytes
from core.utils import ms2deepscore
from tests.test_cluster_analytes import FRAGS_A, FRAGS_B, FRAGS_C, _build

FIXTURE = Path(__file__).parent / "ms2deepscore_parity.npz"
needs_model = pytest.mark.skipif(not ms2deepscore.model_available(), reason="MS2DeepScore weights not in models/")


def _fixture_spectra(f):
    o = f["offsets"]
    return [
        (f["mz"][o[i]:o[i + 1]], f["intensities"][o[i]:o[i + 1]],
         {"precursor_mz": float(f["precursor_mz"][i]), "ionmode": str(f["ionmode"][i])})
        for i in range(len(o) - 1)
    ]


@needs_model
def test_embeddings_match_torch():
    with np.load(FIXTURE) as f:
        spectra, expected = _fixture_spectra(f), f["embeddings"]
    np.testing.assert_allclose(ms2deepscore.load_model().embed(spectra), expected, atol=1e-5)


@pytest.mark.skipif(not ms2deepscore.EVALUATOR_PATH.exists(), reason="evaluator weights not in models/")
def test_evaluator_matches_torch():
    with np.load(FIXTURE) as f:
        embeddings, expected = f["embeddings"], f["predicted_mse"]
    evaluator = ms2deepscore.EmbeddingEvaluatorNumpy(ms2deepscore.EVALUATOR_PATH)
    np.testing.assert_allclose(evaluator.predict_mse(embeddings), expected, atol=1e-5)


@needs_model
def test_clustering_by_ms2deepscore():
    alignment, samples, _ = _build([
        (300.0, 200.0, FRAGS_A),
        (100.0, 250.0, FRAGS_C),
        (500.0, 214.0, FRAGS_B),
    ])
    result = cluster_analytes(alignment, samples, ClusterParams(scoring=MS2DEEPSCORE))
    sim = result.similarity
    assert sim.shape == (3, 3)
    assert np.allclose(sim, sim.T)
    assert np.all((sim >= 0) & (sim <= 1))
    assert np.allclose(np.diag(sim), 1.0)


def test_missing_weights_message(monkeypatch, tmp_path):
    monkeypatch.setattr(ms2deepscore, "MODEL_PATH", tmp_path / "ms2deepscore_model.npz")
    ms2deepscore.load_model.cache_clear()
    with pytest.raises(FileNotFoundError, match=ms2deepscore.WEIGHTS_URL):
        ms2deepscore.load_model()
    ms2deepscore.load_model.cache_clear()


@needs_model
def test_network_export_by_ms2deepscore():
    from core.cli.cluster_analytes import compute_similarity_matrix
    from core.cli.export_network import build_matrix_network_graphml
    alignment, samples, _ = _build([(300.0, 200.0, FRAGS_A), (500.0, 214.0, FRAGS_B)])
    matrix = compute_similarity_matrix(alignment, samples, ClusterParams(scoring=MS2DEEPSCORE))
    graphml = build_matrix_network_graphml(alignment, samples, matrix, min_edge_score=0.01)
    assert 'attr.name="similarity"' in graphml
    assert "<edge " in graphml
