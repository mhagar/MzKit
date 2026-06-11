"""
Cross-sample alignment domain types.

An ``EnsembleAlignment`` groups Ensembles that represent the same chemical
entity across multiple Samples. It is the cross-sample analogue of an
Ensemble (which groups coeluting ions within a single sample).

These are *data* types only — the alignment *algorithm* that produces them
lives in ``core/cli/align_ensembles.py``. Keeping the dataclasses here (next
to Sample, Injection, Ensemble) is what lets ``data_registry`` and
``persistence`` depend on them without reaching into ``core/cli``.
"""
from dataclasses import dataclass, field
from typing import NamedTuple, TYPE_CHECKING

import uuid as _uuid

if TYPE_CHECKING:
    from core.data_structs import SampleUUID, EnsembleUUID


class AlignmentParams(NamedTuple):
    """
    Parameters for cross-sample ensemble alignment.

    rt_tolerance: Maximum RT difference (seconds) between
        ensembles to be considered candidates.
    mz_tolerance: Maximum m/z difference for pairing peaks
        when computing spectral cosine similarity.
    ms1_similarity_threshold: Minimum MS1 cosine similarity
        to consider a match.
    ms2_similarity_threshold: Minimum MS2 cosine similarity
        to consider a match (only used when both ensembles
        have MS2 data).
    ms1_weight: Weight for MS1 similarity in combined score.
    ms2_weight: Weight for MS2 similarity in combined score.
    """
    rt_tolerance: float = 10.0
    mz_tolerance: float = 0.01
    ms1_similarity_threshold: float = 0.7
    ms2_similarity_threshold: float = 0.6
    ms1_weight: float = 0.5
    ms2_weight: float = 0.5


@dataclass
class AlignedAnalyte:
    """
    One chemical entity tracked across multiple samples.

    Maps SampleUUID -> EnsembleUUID for each sample where
    this analyte was detected.
    """
    ensemble_map: dict['SampleUUID', 'EnsembleUUID'] = field(
        default_factory=dict
    )
    consensus_rt: float = 0.0
    consensus_mz: float = 0.0


@dataclass
class EnsembleAlignment:
    """
    Result of aligning ensembles across a set of samples.

    Immutable after creation — adding a new sample requires
    performing a new alignment.
    """
    sample_uuids: tuple['SampleUUID', ...]
    analytes: list[AlignedAnalyte] = field(default_factory=list)
    parameters: AlignmentParams = field(default_factory=AlignmentParams)
    uuid: int = field(default_factory=lambda: _uuid.uuid4().int)
    name: str = ""

    @property
    def sample_count(self) -> int:
        return len(self.sample_uuids)

    @property
    def analyte_count(self) -> int:
        return len(self.analytes)
