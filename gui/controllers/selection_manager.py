from typing import TYPE_CHECKING

from PyQt5 import QtCore

if TYPE_CHECKING:
    from core.data_structs import (
        AnalyteTableUUID,
        AnalyteID,
        SampleUUID,
    )


class SelectionManager(QtCore.QObject):
    # Different selection types
    sigAnalyteSelectionChanged = QtCore.pyqtSignal(
        dict, # dict['AnalyteTableUUID',list['AnalyteID']]
    )
    sigSampleSelectionChanged = QtCore.pyqtSignal(
        list, # list[SampleUUID]
    )

    def __init__(self):
        super().__init__()

        self.current_analytes: dict['AnalyteTableUUID',list['AnalyteID']] = {}

        self.current_samples: list['SampleUUID'] = []


    def on_analyte_selection(
        self,
        analyte_table_uuid: 'AnalyteTableUUID',
        analyte_ids: list['AnalyteID']
    ):
        self.current_analytes[analyte_table_uuid] = analyte_ids
        self.sigAnalyteSelectionChanged.emit(
            self.current_analytes
        )
