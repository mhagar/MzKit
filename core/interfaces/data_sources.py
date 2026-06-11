from typing import Protocol, Optional, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import pandas as pd
    from core.data_structs import (
        Sample,
        SampleUUID,
        InjectionUUID,
        FingerprintUUID,
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




