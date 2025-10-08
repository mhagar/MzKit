from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QVariant,
)
from PyQt5.QtGui import (
    QStandardItem,
    QStandardItemModel,
    QBrush,
)
import pyqtgraph as pg

from gui.resources.AnalyteTableViewerWindow import Ui_Form

from typing import Optional, Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from core.data_structs import (
        Sample, SampleUUID,
        AnalyteTable, AnalyteTableUUID,
        Analyte, AnalyteID
    )
    from core.interfaces.data_sources import (
        SampleDataSource,
        AnalyteSource,
    )


class AnalyteTableViewerWindow(
    QtWidgets.QWidget,
    Ui_Form,
):
    """
    Window housing an analyte table viewer
    """
    sigAnalytesSelected = QtCore.pyqtSignal(
        object, # AnalyteTableUUID
        list,  # list[AnalyteID]
    )

    def __init__(
        self,
        data_source: 'SampleDataSource',
    ):
        super().__init__()
        self.setupUi(self)

        self.sample_data_source = data_source

        # Main table view
        self.table_model = AnalyteTableModel(
            sample_data_source=data_source
        )
        self.viewTableAnalytes.setModel(
            self.table_model
        )

        # Selection handling
        self.selection_timer = QtCore.QTimer() # Used to batch selection sigs
        self.selection_timer.setSingleShot(True)
        self.selection_timer.timeout.connect(
            self._emit_final_selection,
        )

        self.viewTableAnalytes.selectionModel().selectionChanged.connect(
            self._on_analyte_selection_changed,
        )

    def set_analyte_table(
        self,
        analyte_table: 'AnalyteSource'
    ):
        self.table_model.setAnalyteTable(analyte_table)


    def _on_analyte_selection_changed(
        self,
    ):
        # Reset timer on each selection change
        self.selection_timer.stop()
        self.selection_timer.start(150)  # 150 ms delay


    def _emit_final_selection(self):
        rows = [
            idx.row() for idx in self.viewTableAnalytes.selectionModel().selectedRows()
        ]
        self.sigAnalytesSelected.emit(
            self.table_model.get_analyte_table_uuid(),
            rows,
        )


# Model for organizing the contents of the
# AnalyteTable currently being viewed
class AnalyteTableModel(
    QAbstractTableModel,
):
    def __init__(
        self,
        sample_data_source: 'SampleDataSource',
        colormap: str = 'CET-D1A',
        parent=None,
    ):
        super().__init__(parent)
        self.sample_data_source = sample_data_source

        self._analyte_table: Optional['AnalyteSource'] = None

        self._row_to_analyte_id: dict[int, 'AnalyteID'] = {}
        self._col_to_sample_name: dict[int, str] = {}
        self._col_to_metadata_field: dict[int, str] = {}

        self.colormap = pg.colormap.get(colormap)
        print(
            "TODO: Using default colormap range, expose to user"
        )


    def setAnalyteTable(
        self,
        analyte_table: 'AnalyteSource'
    ):
        self.beginResetModel()
        self._row_to_analyte_id = {}
        self._col_to_sample_name = {}
        self._col_to_metadata_field = {}

        self._analyte_table: 'AnalyteSource' = analyte_table

        # Enumerate rows <-> analyte IDs
        for row, analyte_id in enumerate(
            self._analyte_table.get_analyte_ids()
        ):
            self._row_to_analyte_id[row] = analyte_id


        # Enumerate columns <-> sample name or metadata field
        for idx, metadata_field in enumerate(
            self._analyte_table.get_metadata_fields()
        ):
            # i.e. metafield_1 should be col 3, metafield_2 -> col 4, etc
            self._col_to_metadata_field[idx + 2] = metadata_field

        num_metadata_fields = self.metadataFieldCount()
        for idx, sample_name in enumerate(
            self._analyte_table.get_sample_names()
        ):
            # i.e. samplename_1 should be (num_metafields) + (mz and rt columns)
            self._col_to_sample_name[idx + 2 + num_metadata_fields] = sample_name

        self.endResetModel()

    def get_analyte_table_uuid(self) -> 'AnalyteTableUUID':
        if self._analyte_table:
            return self._analyte_table.uuid

    def rowCount(
        self,
        parent=QModelIndex(),
    ):
        if self._analyte_table:
            return self._analyte_table.analyte_count()
        return 0


    def columnCount(
        self,
        parent=QModelIndex(),
    ):
        if self._analyte_table:
            return (
                self.sampleCount() +
                self.metadataFieldCount() +
                2  # For m/z and rt
            )

        return 0


    def sampleCount(
        self,
    ) -> int:
        return self._analyte_table.sample_count()


    def metadataFieldCount(
        self,
    ) -> int:
        return len(  self._analyte_table.get_metadata_fields() )


    def col_to_type(
        self,
        col: int,
    ) -> Literal['m/z', 'rt', 'sample', 'metadata']:
        """
        Given a column index, returns a string indicating
        what kind of data should be in the column
        """
        match col:
            case 0:
                return 'm/z'
            case 1:
                return 'rt'
            case _:
                if col < self.metadataFieldCount() + 2:
                    return 'metadata'
                return 'sample'



    def data(
        self,
        index: QModelIndex,
        role=Qt.DisplayRole,
    ):
        if not index.isValid():
            return QVariant()

        row = index.row()
        col = index.column()

        analyte_id: 'AnalyteID' = self._row_to_analyte_id[row]

        match role:
            case Qt.DisplayRole:
                match self.col_to_type(col):
                    case 'm/z':
                        return self._analyte_table.get_mz(
                            analyte_id=analyte_id
                        )

                    case 'rt':
                        return self._analyte_table.get_rt(
                            analyte_id=analyte_id
                        )

                    case 'metadata':
                        metadata_field = self._col_to_metadata_field[col]
                        return str(
                            self._analyte_table.get_analyte(
                                analyte_id=analyte_id
                            ).metadata.get(metadata_field)
                        )

                    case 'sample':
                        sample_name = self._col_to_sample_name[col]
                        return str(
                            self._analyte_table.get_intsy(
                            sample_name=sample_name,
                            analyte_id=analyte_id
                           )
                        )


    def headerData(
        self,
        section: int,
        orientation,
        role=Qt.DisplayRole,
    ):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            match self.col_to_type(section):
                case 'm/z':
                    return 'm/z'

                case 'rt':
                    return 'rt'

                case 'metadata':
                    metadata_field = self._col_to_metadata_field[section]
                    return str(metadata_field)

                case 'sample':
                    sample_name = self._col_to_sample_name[section]
                    return str(sample_name)


        elif role == Qt.DisplayRole and orientation == Qt.Vertical:
            return str( self._row_to_analyte_id[section] )

        return QVariant()


# Delegate for painting the cells according to intensity
class AnalyteDelegate(
    QtWidgets.QStyledItemDelegate,
):
    pass
