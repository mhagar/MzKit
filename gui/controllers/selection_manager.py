from typing import TYPE_CHECKING

from PyQt5 import QtCore

if TYPE_CHECKING:
    from core.data_structs import (
        SampleUUID,
    )


class SelectionManager(QtCore.QObject):
    # Different selection types
    sigSampleSelectionChanged = QtCore.pyqtSignal(
        list, # list[SampleUUID]
    )

    def __init__(self):
        super().__init__()

        self.current_samples: list['SampleUUID'] = []
