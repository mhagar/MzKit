"""
Dataclass storing *Analyte* information
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.data_structs import AnalyteID


@dataclass
class Analyte:
    id: 'AnalyteID'
    mz: float
    rt: float
    intsys: dict[str, float]
    metadata: dict[str, any] = field(
        default_factory=dict
    )
    srpnt_array: np.ndarray[...,] = field(
        default_factory = lambda: np.array([])
    )

    def get_sample_names(self) -> list[str]:
        return list( self.intsys.keys() )


    def set_metadata(
        self,
        metadata: dict[str, any],
    ):
        for key, value in metadata.items():
            self.metadata[key] = value


    def set_srpnt_array(
        self,
        array: np.ndarray[...,],
    ):
        self.srpnt_array = array