# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ms2deepscore==2.11.0",
#     "torch>=2.4",
#     "onnxruntime",  # imported unconditionally by ms2deepscore 2.11
#     "scipy",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""
Dump the official MS2DeepScore PyTorch model to an .npz for ms2ds_numpy.py,
then check the NumPy embeddings/scores against torch.

    uv run --script export_weights.py [--mgf spectra.mgf]

The torch environment is pinned by export_weights.py.lock (`uv lock --script`).
To move to a new model release: bump ZENODO / the ms2deepscore pin, re-lock, re-run.
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from matchms.filtering import normalize_intensities
from matchms.importing import load_from_mgf
from ms2deepscore.models import load_embedding_evaluator, load_model

from ms2ds_numpy import EmbeddingEvaluatorNumpy, MS2DeepScoreNumpy, similarity_matrix

HERE = Path(__file__).parent
ZENODO = "https://zenodo.org/records/17826815/files"  # dual-mode model, CC-BY-4.0
FILES = ("ms2deepscore_model.pt", "embedding_evaluator.pt")
TEST_MGF = HERE / "ms2deepscore/tests/resources/pesticides_processed.mgf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=HERE / "models", help="download/output directory")
    parser.add_argument("--mgf", type=Path, default=TEST_MGF, help="spectra for the parity check")
    args = parser.parse_args()

    args.models.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if not (args.models / name).exists():
            print(f"Downloading {ZENODO}/{name} ...")
            urllib.request.urlretrieve(f"{ZENODO}/{name}", args.models / name)
    model_npz = args.models / "ms2deepscore_model.npz"
    evaluator_npz = args.models / "embedding_evaluator.npz"

    model = load_model(args.models / "ms2deepscore_model.pt", allow_legacy=True).cpu().eval()
    enc = model.encoder
    layers = [layer[0] for layer in enc.dense_layers] + [enc.embedding_layer[0]]
    arrays = {}
    for i, lin in enumerate(layers):
        arrays[f"w{i}t"] = lin.weight.detach().numpy().T.copy()  # (in, out)
        arrays[f"b{i}"] = lin.bias.detach().numpy()
    np.savez(model_npz, settings=json.dumps(model.model_settings.get_dict()), n_layers=len(layers), **arrays)
    print(f"Wrote {model_npz} ({model_npz.stat().st_size / 1e6:.0f} MB, {len(layers)} layers)")

    evaluator = load_embedding_evaluator(args.models / "embedding_evaluator.pt", allow_legacy=True).cpu().eval()
    state = {k: v.numpy() for k, v in evaluator.state_dict().items() if not k.endswith("num_batches_tracked")}
    np.savez(evaluator_npz, settings=json.dumps(vars(evaluator.settings), default=str), **state)
    print(f"Wrote {evaluator_npz} ({len(state)} tensors)")

    # Parity check
    spectra = [normalize_intensities(s) for s in load_from_mgf(str(args.mgf))]
    emb_torch = model.compute_embedding_array(spectra, progress_bar=False)
    emb_np = MS2DeepScoreNumpy(model_npz).embed((s.peaks.mz, s.peaks.intensities, s.metadata) for s in spectra)
    scores_torch = similarity_matrix(emb_torch, emb_torch)
    scores_np = similarity_matrix(emb_np, emb_np)

    emb_err = np.abs(emb_torch - emb_np).max()
    score_err = np.abs(scores_torch - scores_np).max()
    off_diag = scores_np[~np.eye(len(spectra), dtype=bool)]
    print(f"{len(spectra)} spectra: max |embedding diff| = {emb_err:.2e}, max |score diff| = {score_err:.2e}")
    print(f"off-diagonal scores: min {off_diag.min():.3f}, max {off_diag.max():.3f}, "
          f"{(off_diag < 0).mean():.1%} negative")
    mse_torch = evaluator.compute_embedding_evaluations(emb_torch, device="cpu")
    mse_np = EmbeddingEvaluatorNumpy(evaluator_npz).predict_mse(emb_np)
    mse_err = np.abs(mse_torch - mse_np).max()
    print(f"evaluator: max |predicted MSE diff| = {mse_err:.2e}; predicted RMSE range "
          f"{np.sqrt(mse_np.clip(0)).min():.3f}-{np.sqrt(mse_np.clip(0)).max():.3f}")
    assert score_err < 1e-4, "NumPy port disagrees with torch"
    assert mse_err < 1e-4, "NumPy evaluator disagrees with torch"
    print("PARITY OK")


if __name__ == "__main__":
    main()
