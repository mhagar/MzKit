"""
Central data registry for performing UUID <-> object look-ups.

Dependent on PyQt5 for signals/slots, but should not otherwise involve
any Qt.
"""

from PyQt5 import QtCore

from core.interfaces.data_sources import SampleDataSource

from typing import Optional, TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from core.data_structs import (
        Sample,
        SampleUUID,
        InjectionUUID,
        FingerprintUUID,
        AlignmentUUID,
    )
    from core.data_structs.alignment import EnsembleAlignment

class DataRegistry(
    QtCore.QObject,
    # SampleDataSource,
):
    """
    Central data registry. Stores Samples,
     and performs UUID <-> object look-ups

    # TODO: Implement global Ensemble searching here
    """
    sigSampleAdded = QtCore.pyqtSignal(
        object  # Sample
    )
    sigSampleRemoved = QtCore.pyqtSignal(
        object  # Sample
    )
    sigSampleUpdated = QtCore.pyqtSignal(
        object  # Sample
    )
    sigAlignmentAdded = QtCore.pyqtSignal(
        object  # EnsembleAlignment
    )
    sigAlignmentRemoved = QtCore.pyqtSignal(
        object  # EnsembleAlignment
    )

    def __init__(self):
        self._samples: dict['SampleUUID', 'Sample'] = {}
        self._sample_name_to_uuid: dict[str, 'SampleUUID'] = {}
        self._alignments: dict['AlignmentUUID', 'EnsembleAlignment'] = {}
        super().__init__()

    def subscribe_to_changes(
        self,
        addition_callback,
        removal_callback,
        update_callback=None,
        change_type: Literal['Sample', 'Alignment'] = 'Sample',
    ):
        match change_type:
            case 'Sample':
                self.sigSampleAdded.connect(
                    addition_callback
                )

                self.sigSampleRemoved.connect(
                    removal_callback
                )

                if update_callback:
                    self.sigSampleUpdated.connect(
                        update_callback
                    )

            case 'Alignment':
                self.sigAlignmentAdded.connect(
                    addition_callback
                )

                self.sigAlignmentRemoved.connect(
                    removal_callback
                )

    def register_sample(
        self,
        sample: 'Sample',
    ):
        """
        Registers a Sample into the database.

        If there already exists a Sample with the same name, but
        with compatible contents (i.e. one has MS data and the other
        has fingerprints), merges the two.
        :param sample:
        :return:
        """
        self.validate_new_sample(sample)

        matched_sample_uuid: Optional['SampleUUID'] = self.match_samplename(
            sample.name
        )

        if matched_sample_uuid:
            # Sample with same name already exists
            self.merge_samples(
                source=sample,
                destination_uuid=matched_sample_uuid
            )

        else:
            # Brand new sample
            self._samples[sample.uuid] = sample
            self._sample_name_to_uuid[sample.name] = sample.uuid
            self.sigSampleAdded.emit(
                sample
            )

    def register_samples(
        self,
        samples: list['Sample'],
    ):
        for sample in samples:
            self.register_sample(sample)

    def get_sample(
        self,
        uuid: 'SampleUUID',
    ) -> Optional['Sample']:
        return self._samples.get(uuid)

    def get_all_samples(self) -> list['Sample']:
        return list(self._samples.values())

    def get_all_sample_uuids(self) -> list['SampleUUID']:
        return list(self._samples.keys())

    def remove_sample(
        self,
        uuid: 'SampleUUID',
    ):
        if uuid in self._samples:
            sample_to_remove = self._samples[uuid]
            self.sigSampleRemoved.emit(
                sample_to_remove
            )
            del self._samples[uuid]
            del self._sample_name_to_uuid[sample_to_remove.name]

            return

        raise ValueError(
            f"No sample with UUID {uuid} registered"
        )

    def validate_new_sample(
        self,
        sample: 'Sample'
    ):
        """
        Validates a sample for registration. This check must
        succeed before registering a sample.

        Currently enforces uuid uniqueness and that a Sample carries at
        least one of (Injection, Fingerprint). Fingerprint cross-sample
        consistency checks (matching array size / descriptors / metadata
        fields) are not yet ported from the old fingerprint model.

        :param sample:
        :return:
        """
        if sample.uuid in self._samples:
            raise ValueError(
                f"Sample {sample.name} is already registered with uuid: "
                f"{self._samples[sample.uuid].uuid}"
            )

        if not sample.injection and not sample.fingerprint:
            raise ValueError(
                f"Sample must have either a Fingerprint or Injection assigned"
                f"before being registered. Exception caused by sample: {sample}"
            )

    def match_samplename(
        self,
        name: str,
    ) -> Optional['SampleUUID']:
        """
        Given a name, checks whether there already exists
        a Sample with the same name.

        This is used to merge imported fingerprints and MS data

        :param name:
        :return:
        """
        if name in self._sample_name_to_uuid:
            return self._sample_name_to_uuid[name]

        return None

    def merge_samples(
        self,
        source: 'Sample',
        destination_uuid: 'SampleUUID',
    ) -> None:
        """
        Given a source sample and a destination UUID, merges
        the two.

        This is used if two samples have identical names, but one
        only has MS data, and the other has only fingerprint data.
        """
        destination = self.get_sample(destination_uuid)

        if source.name != destination.name:
            raise ValueError(
                f"Source and destination samples have different names: "
                f"{source.name} vs {destination.name}"
            )

        if not _merge_is_valid(source, destination):
            raise ValueError(
                f"Incompatible merge attempted:\n"
                f"Source: {source}\n"
                f"Destination: {destination}"
            )

        # Merge the fingerprint/injection, whichever is appropriate
        if not destination.fingerprint:
            destination.set_fingerprint(
                source.fingerprint
            )
            self.sigSampleUpdated.emit(destination)

        if not destination.injection:
            destination.set_injection(
                source.injection
            )
            self.sigSampleUpdated.emit(destination)

        # Merge the metadata attribute
        destination.metadata.update(source.metadata)

    def clear(self) -> None:
        """
        Empties the registry of all Samples and Alignments, emitting the
        corresponding removal signals so that every subscribed model/view
        resets to an empty state.

        The DataRegistry instance itself is preserved (only its contents
        are dropped), so all existing references and signal/slot
        connections held by controllers, models and views remain valid.

        Alignments are removed first, since they reference Samples.
        """
        for uuid in self.get_all_alignment_uuids():
            self.remove_alignment(uuid)

        for uuid in self.get_all_sample_uuids():
            self.remove_sample(uuid)

    def sample_count(self) -> int:
        return len(self._samples)

    def notify_sample_updated(
        self,
        uuid: 'SampleUUID',
    ):
        """
        Emits sigSampleUpdated
        """
        self.sigSampleUpdated.emit(
            self.get_sample(uuid)
        )

    def update_sample_metadata(
        self,
        uuid: 'SampleUUID',
        metadata: dict[str, any],
    ):
        """
        Given a UUID and a metadata dictionary, updates the
        sample's "metadata" attribute and signals that it was
        changed.
        """
        sample = self.get_sample(uuid)

        for key, value in metadata.items():
            sample.metadata[key] = value

        self.sigSampleUpdated.emit(sample)

    # --- Alignment methods ---

    def register_alignment(
        self,
        alignment: 'EnsembleAlignment',
    ):
        self._alignments[alignment.uuid] = alignment
        self.sigAlignmentAdded.emit(alignment)

    def remove_alignment(
        self,
        uuid: 'AlignmentUUID',
    ):
        if uuid in self._alignments:
            alignment = self._alignments[uuid]
            self.sigAlignmentRemoved.emit(alignment)
            del self._alignments[uuid]

    def get_alignment(
        self,
        uuid: 'AlignmentUUID',
    ) -> Optional['EnsembleAlignment']:
        return self._alignments.get(uuid)

    def get_all_alignment_uuids(
        self,
    ) -> list['AlignmentUUID']:
        return list(self._alignments.keys())

    def alignment_count(self) -> int:
        return len(self._alignments)

def _merge_is_valid(
    source: 'Sample',
    destination: 'Sample',
) -> bool:
    source_pattern = (
        source.fingerprint is not None,
        source.injection is not None,
    )

    destination_pattern = (
        destination.fingerprint is not None,
        destination.injection is not None,
    )

    match (source_pattern, destination_pattern):
        case (
        ( (True, False),
          (False, True) )
        |
        ( (False, True),
          (True, False) )
        ):
            return True

        case _:
            return False


