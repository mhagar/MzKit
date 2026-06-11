"""
Qt model for representing EnsembleAlignments in the data registry
"""
from PyQt5.QtCore import (
    QAbstractListModel,
    QModelIndex,
    Qt,
)

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from core.data_structs import DataRegistry, AlignmentUUID
    from core.data_structs.alignment import EnsembleAlignment


class AlignmentListModel(QAbstractListModel):
    """
    Model representing EnsembleAlignments in DataRegistry
    """
    UuidRole = Qt.UserRole + 1

    def __init__(
        self,
        registry: 'DataRegistry',
        parent=None,
    ):
        super().__init__(parent)
        self.registry = registry
        self._alignment_uuids: list['AlignmentUUID'] = (
            self.registry.get_all_alignment_uuids()
        )

        self.registry.subscribe_to_changes(
            change_type='Alignment',
            addition_callback=self.on_alignment_added,
            removal_callback=self.on_alignment_removed,
        )

    def data(
        self,
        index: QModelIndex,
        role=Qt.DisplayRole,
    ):
        alignment = self._get_alignment_at_index(index)
        if not alignment:
            return

        match role:
            case Qt.DisplayRole:
                label = alignment.name or f"Alignment ...{str(alignment.uuid)[-5:]}"
                return f"{label} ({alignment.analyte_count} analytes, {alignment.sample_count} samples)"

            case Qt.ToolTipRole:
                multi = sum(
                    1 for a in alignment.analytes
                    if len(a.ensemble_map) > 1
                )
                return (
                    f"Analytes: {alignment.analyte_count}\n"
                    f"Matched: {multi}\n"
                    f"Singletons: {alignment.analyte_count - multi}\n"
                    f"Samples: {alignment.sample_count}"
                )

            case self.UuidRole:
                return alignment.uuid

    def rowCount(
        self,
        parent=QModelIndex(),
    ):
        return self.registry.alignment_count()

    def _get_alignment_at_index(
        self,
        index: QModelIndex,
    ) -> Optional['EnsembleAlignment']:
        if (not index.isValid()
            or index.row() >= self.rowCount()):
            return None

        uuid = self._alignment_uuids[index.row()]
        return self.registry.get_alignment(uuid)

    def on_alignment_added(
        self,
        alignment: 'EnsembleAlignment',
    ):
        row = self.rowCount()
        self.beginInsertRows(QModelIndex(), row, row)
        self._alignment_uuids.append(alignment.uuid)
        self.endInsertRows()

    def on_alignment_removed(
        self,
        alignment: 'EnsembleAlignment',
    ):
        try:
            row = self._alignment_uuids.index(alignment.uuid)
            self.beginRemoveRows(QModelIndex(), row, row)
            self._alignment_uuids.pop(row)
            self.endRemoveRows()
        except ValueError:
            pass
