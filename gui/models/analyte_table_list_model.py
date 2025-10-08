"""
Qt model for representing samples in the data registry
"""
from PyQt5.QtCore import (
    QAbstractListModel,
    QModelIndex,
    Qt,
)


from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from core.interfaces.data_sources import AnalyteTableSource
    from core.data_structs import (
        AnalyteTable,
        AnalyteTableUUID,
    )


class AnalyteTableListModel(QAbstractListModel):
    """
    Model representing AnalyteTable s in DataRegistry
    """
    def __init__(
        self,
        registry: 'AnalyteTableSource',
        parent=None,
    ):
        super().__init__(parent)
        self.registry: 'AnalyteTableSource' = registry
        self._analyte_table_uuids: list['AnalyteTableUUID'] = (
            self.registry.get_all_analyte_table_uuids()
        )

        self.registry.subscribe_to_changes(
            change_type='AnalyteTable',
            addition_callback=self.onAnalyteTableAdded,
            removal_callback=self.onAnalyteTableRemoved,
            update_callback=self.onAnalyteTableChanged,
        )


    def data(
        self,
        index: QModelIndex,
        role=Qt.DisplayRole,
    ):
        analyte_table = self.getAnalyteTableAtIndex(index)
        if not analyte_table:
            # Invalid index
            return

        match role:
            case Qt.DisplayRole:
                # TODO: Implement table name?
                return f"...{str(analyte_table.uuid)[-5:]}"

            case Qt.DecorationRole:
                return get_data_type_icon(analyte_table)

            case Qt.ToolTipRole:
                return get_analyte_table_tooltip(analyte_table)


    def rowCount(
        self,
        parent=QModelIndex(),
    ):
        return self.registry.analyte_table_count()


    def getAnalyteTableAtIndex(
        self,
        index: QModelIndex,
    ) -> Optional['AnalyteTable']:

        if ( not index.isValid()
            or index.row() >= self.rowCount()):
            return None

        uuid = self._analyte_table_uuids[index.row()]
        return self.registry.get_analyte_table(uuid)


    def getAllAnalyteTables(
        self,
    ) -> list['AnalyteTable']:
        return self.registry.get_all_analyte_tables()


    def getAnalyteTableByUUID(
        self,
        uuid: 'AnalyteTableUUID',
    ) -> Optional['AnalyteTable']:
        return self.registry.get_analyte_table(uuid)


    def onAnalyteTableAdded(
        self,
        analyte_table: 'AnalyteTable',
    ):
        """
        Update Qt model to reflect registry changes.
        Inserts analyte table at end
        """
        row = self.rowCount()
        self.beginInsertRows(
            QModelIndex(),
            row,
            row,
        )

        self._analyte_table_uuids.append(
            analyte_table.uuid,
        )
        self.endInsertRows()


    def onAnalyteTableRemoved(
        self,
        analyte_table: 'AnalyteTable',
    ):
        """
        Update Qt model to reflect registry changes
        """
        try:
            row = self._analyte_table_uuids.index(
                analyte_table.uuid
            )
            self.beginRemoveRows(
                QModelIndex(),
                row,
                row,
            )
            self._analyte_table_uuids.pop(
                row,
            )
            self.endRemoveRows()
        except ValueError:
            # UUID not found; possibly already removed
            pass

    def onAnalyteTableChanged(
        self,
        analyte_table: 'AnalyteTable'
    ):
        return


def get_data_type_icon(analyte_table: 'AnalyteTable'):
    # TODO: implement this, kinda important for UX
    pass


def get_analyte_table_tooltip(analyte_table: 'AnalyteTable') -> str:
    return ""