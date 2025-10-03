"""
Dataclass containing activity fingerprint data
"""
import numpy as np

from dataclasses import dataclass, field
from pathlib import Path
import uuid

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.data_structs import FingerprintUUID


@dataclass
class Fingerprint:
    """
    Dataclass containing activity fingerprint data.

    array: a 1D array containing floats representing assay data
    descriptors: a list (of the same length) naming each assay
    uuid: A unique 128-bit integer generated upon initializing this class
    injection_uuid: Assigned as an Injection object's UUID if the fingerprint
        is 'linked' (i.e. is deemed to correspond to the same sample)
    metadata: A dictionary containing other information about the fingerprint
    that the user can import
    """
    array: np.ndarray[float]
    descriptors: list[str]
    uuid: 'FingerprintUUID' = field(default_factory=lambda: uuid.uuid4().int)

    def __post_init__(self):
        """
        Some error checks
        :return:
        """
        if self.array.ndim != 1:
            raise ValueError(
                f"array must be 1-dimensional. "
                f"Given: {self.array.ndim}"
            )

        if len(self.array) != len(self.descriptors):
            raise ValueError(
                f"array and descriptors must be same length. \n"
                f"(Given {len(self.array)} vs {len(self.descriptors)})"
            )

    def __repr__(self):
        return (f"Fingerprint("
                f"uuid={self.uuid}, "
                f"{len(self.descriptors)} descriptors"
                f")")


@dataclass
class FingerprintImportParams:
    """
    :param csv_path: Path leading to .csv file
    :param sample_names: List of sample names the user would like to import.
                        If empty list is given, will import all samples
    :param descriptors: List of descriptors the user would like to import
                        If empty list is given, will import all descriptors
    :param samples_in_rows: Whether the .csv file has each row corresponding to a sample.
        (False means each row is a *descriptor*)
    """
    csv_path: Path
    sample_names: list[str]
    descriptors: list[str]
    samples_in_rows: bool

