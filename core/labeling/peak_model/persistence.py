"""Load and save NumPy peak-model weights."""
from functools import lru_cache
from pathlib import Path

import numpy as np


def default_model_path() -> Path:
    from core.utils.config import get_project_root

    return get_project_root() / "models" / "peak_model.npz"


@lru_cache(maxsize=4)
def load_model(path: str) -> dict:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def save_model(model: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **model)
    load_model.cache_clear()
