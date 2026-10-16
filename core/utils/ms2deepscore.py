"""
NumPy/SciPy-only MS2DeepScore inference (no torch, no onnxruntime).

Weights live in `<project root>/models/` as .npz files written by
`tools/ms2deepscore/export_weights.py`, which dumps the official PyTorch
models and checks this module against them.

A spectrum is `(mz, intensities, metadata)`, where metadata is a dict holding
whatever fields the model's `additional_metadata` needs (for the released
dual-mode model: "ionmode" = "positive"/"negative" and "precursor_mz").
"""
from functools import lru_cache
import json

import numpy as np
from scipy import sparse

from core.utils.config import get_project_root

MODEL_PATH = get_project_root() / "models" / "ms2deepscore_model.npz"
EVALUATOR_PATH = get_project_root() / "models" / "embedding_evaluator.npz"
WEIGHTS_URL = "https://vault.sfu.ca/index.php/s/JLFRp3Njry5LFkM"


def model_available() -> bool:
    return MODEL_PATH.exists()


def missing_model_message() -> str:
    return (
        f"MS2DeepScore weights not found at {MODEL_PATH}.\n"
        f"Download ms2deepscore_model.npz from {WEIGHTS_URL} "
        f"and put it in {MODEL_PATH.parent}."
    )


@lru_cache(maxsize=1)
def load_model() -> 'MS2DeepScoreNumpy':
    """The bundled MS2DeepScore model, loaded once (~400 MB)."""
    if not model_available():
        raise FileNotFoundError(missing_model_message())
    return MS2DeepScoreNumpy(MODEL_PATH)


class MS2DeepScoreNumpy:
    def __init__(self, npz_path):
        with np.load(npz_path) as f:
            self.settings = json.loads(str(f["settings"]))
            n_layers = int(f["n_layers"])
            self.weights = [(f[f"w{i}t"], f[f"b{i}"]) for i in range(n_layers)]

        s = self.settings
        if s["train_binning_layer"] or s["activation_function"] != "relu":
            raise ValueError("Only plain ReLU encoders (no trained binning layer) are supported.")
        self.n_bins = int((s["max_mz"] - s["min_mz"]) / s["mz_bin_width"])
        self.n_meta = len(s["additional_metadata"])

    def embed(self, spectra, batch_size: int = 1024) -> np.ndarray:
        """(n_spectra, embedding_dim) float32 embeddings."""
        spectra = list(spectra)
        out = [self._embed_batch(spectra[i:i + batch_size]) for i in range(0, len(spectra), batch_size)]
        return np.concatenate(out) if out else np.zeros((0, self.settings["embedding_dim"]), np.float32)

    def _embed_batch(self, spectra) -> np.ndarray:
        meta = np.array([self._metadata_vector(m) for _, _, m in spectra], np.float32).reshape(len(spectra), self.n_meta)
        binned = sparse.vstack([self._bin(mz, intsy) for mz, intsy, _ in spectra], format="csr")

        # Encoder input is cat([metadata, binned]); split the first matmul so the spectrum part stays sparse
        w0t, b0 = self.weights[0]
        x = meta @ w0t[:self.n_meta] + binned @ w0t[self.n_meta:] + b0
        for wt, b in self.weights[1:]:
            x = np.maximum(x, 0) @ wt + b
        return np.tanh(x).astype(np.float32)

    def _bin(
            self,
            mz,
            intensities
    ) -> sparse.csr_matrix:
        """
        Max intensity**scaling per bin,
        intensities max-normalised first
        (idempotent if already normalised).
        """
        s = self.settings
        mz = np.asarray(mz, float)
        intsy = np.asarray(intensities, float)
        keep = (mz >= s["min_mz"]) & (mz < s["max_mz"])
        top = intsy.max(initial=0.0)

        row = np.zeros(self.n_bins, np.float32)
        if top > 0:
            idx = ((mz[keep] - s["min_mz"]) / s["mz_bin_width"]).astype(np.int64)
            np.maximum.at(row, idx, (intsy[keep] / top) ** s["intensity_scaling"])
        return sparse.csr_matrix(row)

    def _metadata_vector(
            self,
            metadata: dict,
    ) -> list[float]:
        vec = []
        for kind, p in self.settings["additional_metadata"]:
            value = metadata.get(p["metadata_field"])
            if value is None:
                raise ValueError(f"Spectrum is missing metadata field {p['metadata_field']!r}")
            if kind == "StandardScaler":
                std = p["standard_deviation"]
                vec.append((value - p["mean"]) / std if std else value - p["mean"])
            elif kind == "CategoricalToBinary":
                ones, zeros = (v if isinstance(v, list) else [v] for v in (p["entries_becoming_one"], p["entries_becoming_zero"]))
                if value not in ones + zeros:
                    raise ValueError(f"{p['metadata_field']} should be one of {ones + zeros}, not {value!r}")
                vec.append(float(value in ones))
            elif kind == "OneHotEncoder":
                vec.append(float(value == p["entries_becoming_one"]))
            else:
                raise ValueError(f"Unsupported metadata feature {kind}")
        return vec


class EmbeddingEvaluatorNumpy:
    """
    MS2DeepScore's embedding evaluator: an InceptionTime-style 1D CNN that reads a
    500-d embedding as a signal and predicts the MSE of that spectrum's MS2DeepScore
    scores vs. true Tanimoto. ms2deepscore reports sqrt(max(mse, 0)) as predicted RMSE.

    Architecture (depth, kernel sizes, residuals) is read off the exported state dict.
    """

    def __init__(self, npz_path):
        with np.load(npz_path) as f:
            self.p = {k: f[k] for k in f.files if k != "settings"}
        self.depth = 1 + max(int(k.split(".")[2]) for k in self.p if k.startswith("inception_block.inception_modules."))

    def predict_mse(self, embeddings: np.ndarray, batch_size: int = 64) -> np.ndarray:
        embeddings = np.asarray(embeddings, np.float32)
        return np.concatenate([self._forward(embeddings[i:i + batch_size, None, :])
                               for i in range(0, len(embeddings), batch_size)] or [np.zeros(0, np.float32)])

    def _forward(self, x):
        residual = x
        for d in range(self.depth):
            x = self._module(x, f"inception_block.inception_modules.{d}.")
            if d % 3 == 2:  # residual connection every 3rd module
                s = f"inception_block.shortcuts.{d // 3}."
                x = np.maximum(x + (self._conv(residual, s) if s + "running_mean" not in self.p else self._bn(residual, s)), 0)
                residual = x
        x = x.mean(axis=2)  # global average pool
        return (x @ self.p["fc.weight"].T + self.p["fc.bias"]).ravel()

    def _module(self, x, m):
        bottleneck = self._conv(x, m + "bottleneck.") if m + "bottleneck.weight" in self.p else x
        convs = [self._conv(bottleneck, f"{m}convs.{i}.") for i in range(3)]
        padded = np.pad(x, ((0, 0), (0, 0), (1, 1)), constant_values=-np.inf)
        pooled = np.maximum(np.maximum(padded[:, :, :-2], padded[:, :, 1:-1]), padded[:, :, 2:])  # MaxPool1d(3, 1, 1)
        out = np.concatenate(convs + [self._conv(pooled, m + "maxconvpool.1.")], axis=1)
        return np.maximum(self._bn(out, m + "bn."), 0)

    def _conv(self, x, name):
        """Conv1d with padding='same' (odd kernels) as a single im2col GEMM."""
        w = self.p[name + "weight"]  # (out, in, K)
        n, c, length = x.shape
        k = w.shape[2]
        xp = np.pad(x, ((0, 0), (0, 0), (k // 2, k // 2)))
        cols = np.lib.stride_tricks.sliding_window_view(xp, k, axis=2)  # (N, C, L, K) view
        cols = cols.transpose(0, 2, 1, 3).reshape(n * length, c * k)  # copy: (N*L, C*K)
        out = cols @ np.ascontiguousarray(w.reshape(w.shape[0], -1).T)  # (N*L, out)
        if name + "bias" in self.p:
            out += self.p[name + "bias"]
        return np.ascontiguousarray(out.reshape(n, length, -1).transpose(0, 2, 1))

    def _bn(self, x, name, eps=1e-5):
        p = self.p
        scale = p[name + "weight"] / np.sqrt(p[name + "running_var"] + eps)
        return (x - p[name + "running_mean"][:, None]) * scale[:, None] + p[name + "bias"][:, None]


def similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between embedding rows; zero-norm rows score 0 (as in ms2deepscore).
    """
    def unit(e):
        n = np.linalg.norm(e, axis=1, keepdims=True)
        return np.divide(e, n, out=np.zeros_like(e), where=n > 0)
    return unit(emb_a) @ unit(emb_b).T
