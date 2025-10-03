"""
Script for little peak overlays that can be drawn on SampleWidgets
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Literal

import numpy as np
import pyqtgraph as pg

if TYPE_CHECKING:
    from core.data_structs import (
        Analyte, AnalyteID,
        AnalyteTableUUID,
    )

@dataclass
class PeakOverlay:
    analyte_id: 'AnalyteID'
    analyte_table_uuid: 'AnalyteTableUUID'
    peak_arr: np.ndarray
    color_idx: int
    visible: bool = True
    graphic_item: Optional[pg.GraphicsObject] = None

    

