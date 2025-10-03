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
        AnalyteTable,
        SampleUUID,
        InjectionUUID,
        FingerprintUUID,
        AnalyteTableUUID,
        AnalyteID,
    )

class DataRegistry(
    QtCore.QObject,
    # SampleDataSource,
):
    """
    Central data registry. Stores Samples,
     and performs UUID <-> object look-ups
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
    sigAnalyteTableAdded = QtCore.pyqtSignal(
        object  # Sample
    )
    sigAnalyteTableRemoved = QtCore.pyqtSignal(
        object  # Sample
    )
    sigAnalyteTableUpdated = QtCore.pyqtSignal(
        object  # Sample
    )

    def __init__(self):
        self._samples: dict['SampleUUID', 'Sample'] = {}
        self._sample_name_to_uuid: dict[str, 'SampleUUID'] = {}
        self._analyte_tables: dict['AnalyteTableUUID', 'AnalyteTable'] = {}
        super().__init__()

    def subscribe_to_changes(
        self,
        addition_callback,
        removal_callback,
        update_callback,
        change_type: Literal['Sample', 'AnalyteTable'],
    ):
        match change_type:
            case 'Sample':
                self.sigSampleAdded.connect(
                    addition_callback
                )

                self.sigSampleRemoved.connect(
                    removal_callback
                )

                self.sigSampleUpdated.connect(
                    update_callback
                )

            case 'AnalyteTable':
                self.sigAnalyteTableAdded.connect(
                    addition_callback
                )

                self.sigAnalyteTableRemoved.connect(
                    removal_callback
                )

                self.sigAnalyteTableUpdated.connect(
                    update_callback
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

         # TODO: Re-implement this code from fingerprint_list_model.py
        # # Used to enforce all same size
        # if not self._fingerprint_array_size:
        #     self._fingerprint_array_size = fingerprint.array.size
        #
        # # Used to enforce all same metadata fields
        # if not self._fingerprint_metadata_fields:
        #     self._fingerprint_metadata_fields = {
        #         x for x in fingerprint.metadata.keys()
        #     }
        #
        # # Used to enforce all same descriptors
        # if not self._fingerprint_descriptors:
        #     self._fingerprint_descriptors = set(fingerprint.descriptors)



        # if fingerprint.samplename in self._fingerprint_names:
        #     raise Exception(
        #         f"Fingerprint with sample name {fingerprint.samplename} "
        #         f"already exists"
        #     )
        #
        # fingerprint_keys = {x for x in fingerprint.metadata.keys()}
        # if fingerprint_keys != self._fingerprint_metadata_fields:
        #     raise Exception(
        #         f"Fingerprint {fingerprint.samplename} has different metadata "
        #         f"fields than ones already loaded.\n"
        #         f"Fingerprint {fingerprint.samplename}: {fingerprint.metadata.keys()}\n"
        #         f"Others: {self._fingerprint_metadata_fields}"
        #     )
        #
        # fingerprint_descriptors = set(fingerprint.descriptors)
        # if fingerprint_descriptors != self._fingerprint_descriptors:
        #     raise Exception(
        #         f"Fingerprint {fingerprint.samplename} has different descriptors "
        #         f"fields than ones already loaded.\n"
        #         f"Fingerprint {fingerprint.samplename}: {fingerprint.descriptors}\n"
        #         f"Others: {self._fingerprint_descriptors}"
        #     )
        #
        #
        # if self._fingerprint_array_size:
        #     if fingerprint.array.size != self._fingerprint_array_size:
        #         raise Exception(
        #             f"Fingerprint {fingerprint.samplename} has "
        #             f"{fingerprint.array.size} descriptors, whereas other "
        #             f"fingerprints have {self._fingerprint_array_size}."
        #         )

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

        if not destination.injection:
            destination.set_injection(
                source.injection
            )

        # Merge the metadata attribute
        for key, value in source.metadata:
            destination.metadata[key] = value

    def sample_count(self) -> int:
        return len(self._samples)

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

    def register_analyte_table(
        self,
        analyte_table: 'AnalyteTable'
    ):
        self._analyte_tables[analyte_table.uuid] = analyte_table
        self.sigAnalyteTableAdded.emit(analyte_table)

    def remove_analyte_table(
        self,
        uuid: 'AnalyteTableUUID',
    ):
        if uuid in self._analyte_tables:
            analyte_table_to_remove = self._analyte_tables[uuid]
            self.sigAnalyteTableRemoved.emit(
                analyte_table_to_remove
            )
            del self._analyte_tables[uuid]

            return

    def get_analyte_table(
        self,
        uuid: 'AnalyteTableUUID',
    ) -> 'AnalyteTable':
        return self._analyte_tables[uuid]


    def get_all_analyte_table_uuids(
        self,
    ) -> list['AnalyteTableUUID']:
        return list(self._analyte_tables.keys())


    def analyte_table_count(self) -> int:
        return len(self._analyte_tables)

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


