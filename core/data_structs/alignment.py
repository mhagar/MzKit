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
from typing import Callable, Mapping, NamedTuple, Optional, TYPE_CHECKING

import uuid as _uuid

if TYPE_CHECKING:
    from core.data_structs import (
        SampleUUID, EnsembleUUID, AnalyteUUID, AlignmentUUID, Ensemble, Sample,
    )
    from core.data_structs.composite_spectrum import CompositeSpectrum


class RepresentativeSpectrum(NamedTuple):
    """
    An AlignedAnalyte's representative (MS1, MS2) pair, plus which member
    ensemble it was taken from.
    """
    composite: 'CompositeSpectrum'
    sample_uuid: 'SampleUUID'
    ensemble_uuid: 'EnsembleUUID'

    @property
    def base_mz(self) -> float:
        """m/z of the most intense peak in the representative MS1."""
        ms1 = self.composite.ms1
        return float(ms1['mz'][ms1['intsy'].argmax()]) if ms1.size else 0.0


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
    uuid: 'AnalyteUUID' = field(default_factory=lambda: _uuid.uuid4().int)

    # Not persisted; see `representative_spectrum`
    _representative: Optional[RepresentativeSpectrum] = field(
        default=None, init=False, repr=False, compare=False,
    )

    def resolve_member(
        self,
        sample_uuid: 'SampleUUID',
        get_sample: Callable[['SampleUUID'], Optional['Sample']],
    ) -> Optional['Ensemble']:
        """
        The concrete Ensemble this analyte points at in `sample_uuid`, or
        None if it isn't present / can't be resolved.

        :param get_sample: sample lookup, e.g. `DataRegistry.get_sample`
        """
        ens_uuid = self.ensemble_map.get(sample_uuid)
        if ens_uuid is None:
            return None
        sample = get_sample(sample_uuid)
        if not sample or not sample.injection:
            return None
        return sample.injection.ensembles.get(ens_uuid)

    def resolve_members(
        self,
        get_sample: Callable[['SampleUUID'], Optional['Sample']],
    ) -> dict['SampleUUID', 'Ensemble']:
        """This analyte's resolvable member ensembles, keyed by sample uuid."""
        out = {}
        for sample_uuid in self.ensemble_map:
            ensemble = self.resolve_member(sample_uuid, get_sample)
            if ensemble is not None:
                out[sample_uuid] = ensemble
        return out

    def representative_spectrum(
        self,
        ensembles: Mapping['SampleUUID', 'Ensemble'],
    ) -> Optional[RepresentativeSpectrum]:
        """
        The analyte's representative (MS1, MS2) pair: the composite
        spectrum of its tallest member ensemble (by base intensity).
        Saturation is handled within each ensemble's composite spectrum.

        :param ensembles: this analyte's resolved member ensembles, keyed
            by sample uuid (members that can't be resolved may be omitted).
        :return: None when no member ensemble is given.

        Cached; the analyte's membership is immutable, so the cache never
        goes stale.
        """
        if self._representative is not None:
            return self._representative

        members = [
            (sample_uuid, ens) for sample_uuid, ens in ensembles.items()
            if sample_uuid in self.ensemble_map
        ]
        if not members:
            return None

        sample_uuid, ens = max(members, key=lambda m: m[1].base_intsy)
        self._representative = RepresentativeSpectrum(
            composite=ens.composite_spectrum,
            sample_uuid=sample_uuid,
            ensemble_uuid=ens.uuid,
        )
        return self._representative


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
    uuid: 'AlignmentUUID' = field(default_factory=lambda: _uuid.uuid4().int)
    name: str = ""

    # Lazily built by `get_analyte`; analytes are fixed after creation
    _analytes_by_uuid: Optional[dict['AnalyteUUID', AlignedAnalyte]] = field(
        default=None, init=False, repr=False, compare=False,
    )

    @property
    def sample_count(self) -> int:
        return len(self.sample_uuids)

    @property
    def analyte_count(self) -> int:
        return len(self.analytes)

    def get_analyte(
        self,
        uuid: 'AnalyteUUID',
    ) -> Optional[AlignedAnalyte]:
        if self._analytes_by_uuid is None:
            self._analytes_by_uuid = {a.uuid: a for a in self.analytes}
        return self._analytes_by_uuid.get(uuid)
