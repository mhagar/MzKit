"""
A labelled, square analyte similarity matrix.

Can either be computed in-house (MS2 modified cosine, MS2DeepScore)
or imported from file.

Stored raw so it can be re-transformed (`core.cli.import_similarity.transform`)
and re-clustered (`core.cli.cluster_analytes.cluster_matrix`)

ids are analyte uuid strings, which survive
filtering, so a parent alignment's matrix serves its filtered children.
"""
from dataclasses import dataclass, field
from typing import NamedTuple
import uuid

import numpy as np


IMPORTED = 'imported'


class ImportParams(NamedTuple):
    """
    How a raw matrix maps to a [0, 1] similarity (see `transform`).

    is_distance: values are distances (larger = less similar).
    clip_min / clip_max: values are clipped to this range, then rescaled
        to [0, 1]. For distances, clip_max is the "fully unrelated"
        distance.
    bidirectional: keep a pair only if both directions score at least
        `bidirectional_threshold` (after rescaling), taking the smaller;
        otherwise the two directions are averaged.
    floor: (symmetrized) scores below this become 0, i.e. unrelated.
    """
    is_distance: bool = False
    clip_min: float = 0.0
    clip_max: float = 1.0
    bidirectional: bool = False
    bidirectional_threshold: float = 0.7
    floor: float = 0.6


def initial_transform(source: str) -> ImportParams:
    """
    Computed scores are already similarities in [0, 1], so no floor.
    """
    return ImportParams() if source == IMPORTED else ImportParams(floor=0.0)


@dataclass(eq=False)
class SimilarityMatrix:
    """
    name: display name ("MS2 modified cosine", "MS2DeepScore", file stem).
    ids: analyte uuid strings labelling rows/columns.
    values: raw (untransformed) scores, float32.
    provenance: {'source': 'modified_cosine' | 'ms2deepscore' | 'imported',
        'params': {...} (scoring params; {} if none), 'path': str (imported)}.
    default_transform: the transform to cluster by - initially
        `initial_transform(source)`, then whatever the user last chose.
    uncovered: analytes deliberately absent (computed: no MS2).

    Hashes by identity; `uuid` identifies it in cache keys.
    """
    name: str
    ids: tuple[str, ...]
    values: np.ndarray
    provenance: dict
    default_transform: ImportParams
    uncovered: tuple[str, ...] = ()
    uuid: int = field(default_factory=lambda: uuid.uuid4().int)

    @property
    def is_imported(self) -> bool:
        return self.provenance.get('source') == IMPORTED
