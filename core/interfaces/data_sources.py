from typing import Protocol, Optional, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import pandas as pd
    from core.data_structs import (
        Sample,
        AnalyteTable,
        Analyte,
        SampleUUID,
        InjectionUUID,
        FingerprintUUID,
        AnalyteTableUUID,
        AnalyteID,
    )

class SampleDataSource(Protocol):
    def get_sample(
        self,
        uuid: 'SampleUUID',
    ) -> Optional['Sample']:
        pass

    def get_all_samples(self) -> list['Sample']:
        pass

    def get_all_sample_uuids(self) -> list['SampleUUID']:
        pass

    def sample_count(self) -> int:
        pass

    def subscribe_to_changes(
        self,
        addition_callback: Callable[['Sample'], ...],
        removal_callback: Callable[['Sample'], ...],
        update_callback: Callable[['Sample'], ...],
        change_type = 'Sample',
    ):
        """
        Subscribe to notification when sample registry changes.
        :param change_type: Should be set to 'Sample' to subscribe to
        Sample signale
        :param addition_callback: Function to call when a
        sample is added
        :param removal_callback: Function to call when a
        sample is removed
        :param update_callback: Function to call when a
        sample is updated/changed
        """
        pass


class AnalyteTableSource(Protocol):
    def get_analyte_table(
        self,
        uuid: 'AnalyteTableUUID',
    ) -> 'AnalyteTable':
        pass

    def get_all_analyte_tables(
        self,
    ) -> list['AnalyteTable']:
        pass

    def get_all_analyte_table_uuids(
        self,
    ) -> list['AnalyteTableUUID']:
        pass

    def analyte_table_count(self) -> int:
        pass

    def subscribe_to_changes(
        self,
        addition_callback: Callable[['AnalyteTable'], ...],
        removal_callback: Callable[['AnalyteTable'], ...],
        update_callback: Callable[['AnalyteTable'], ...],
        change_type='AnalyteTable',
    ):
        """
        Subscribe to notification when analyte table registry changes.
        :param change_type: Should be set to 'AnalyteTable' to subscribe to
        analyte table signals
        :param addition_callback: Function to call when a
        analyte table is added
        :param removal_callback: Function to call when a
        analyte table is removed
        :param update_callback: Function to call when a
        analyte table is updated/changed
        """
        pass


class AnalyteSource(Protocol):
    # @property
    # def uuid(self) -> 'AnalyteTableUUID':
    #     pass

    def get_analyte(
        self,
        analyte_id: 'AnalyteID',
    ) -> Optional['Analyte']:
        pass

    def get_analyte_ids(self) -> list['AnalyteID']:
        pass

    def get_analytes_by_sample_uuid(
        self,
        sample_uuid: 'SampleUUID'
    ) -> list['Analyte']:
        pass

    def get_analytes_by_sample_name(
        self,
        sample_name: str,
    ) -> list['Analyte']:
        pass

    def get_sample_names(self) -> list[str]:
        pass

    def analyte_count(
        self,
    ) -> int:
        pass

    def sample_count(
        self,
    ) -> int:
        pass

    def get_mz(
        self,
        analyte_id: 'AnalyteID',
    ) -> float:
        pass

    def get_rt(
        self,
        analyte_id: 'AnalyteID',
    ) -> float:
        pass

    def get_intsy(
        self,
        analyte_id: 'AnalyteID',
        sample_name: str,
    ) -> Optional[float]:
        pass

    def get_metadata_fields(self) -> list[str]:
        pass

    def get_metadata(
        self,
        analyte_id: 'AnalyteID'
    ) -> dict[str, any]:
        pass




