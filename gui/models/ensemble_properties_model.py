"""
Qt model for displaying Ensemble properties in a key-value table
"""
from PyQt5.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    Qt,
)

from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structs import Ensemble


class RowType(Enum):
    """Identifies the type of each row for edit logic"""
    READONLY = auto()
    EDITABLE = auto()
    SEPARATOR = auto()
    METADATA = auto()


class EnsemblePropertiesModel(QAbstractTableModel):
    """
    2-column table model for Ensemble properties.
    Column 0: Property name
    Column 1: Value
    """
    # Fixed rows (before user metadata)
    FIXED_ROWS = [
        # (label, row_type)
        ("Sample Name", RowType.READONLY),
        ("MS1 Cofeatures", RowType.READONLY),
        ("MS2 Cofeatures", RowType.READONLY),
        ("Base m/z", RowType.READONLY),
        ("Peak RT (s)", RowType.READONLY),
        ("", RowType.SEPARATOR),
        ("Proposed Formula", RowType.EDITABLE),
        ("Identity", RowType.EDITABLE),
        ("", RowType.SEPARATOR),
    ]

    def __init__(
        self,
        ensemble: 'Ensemble',
        sample_name: str,
        parent=None,
    ):
        super().__init__(parent)
        self._ensemble: 'Ensemble' = ensemble
        self._sample_name: str = sample_name

    @property
    def ensemble(self) -> 'Ensemble':
        return self._ensemble

    @property
    def sample_name(self) -> str:
        return self._sample_name

    def set_ensemble(
        self,
        ensemble: 'Ensemble',
        sample_name: str,
    ):
        """
        Update the model with a new ensemble
        """
        self.beginResetModel()
        self._ensemble = ensemble
        self._sample_name = sample_name
        self.endResetModel()

    def refresh(self):
        """
        Force refresh of the entire model
        """
        self.beginResetModel()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        if parent.isValid():
            return 0
        # Fixed rows + metadata entries
        return len(self.FIXED_ROWS) + len(self._ensemble.user_metadata)

    def columnCount(self, parent=QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return 2

    def _get_row_info(self, row: int) -> tuple[str, RowType, Optional[str]]:
        """
        Returns (label, row_type, metadata_key_or_none) for a given row.
        """
        if row < len(self.FIXED_ROWS):
            label, row_type = self.FIXED_ROWS[row]
            return label, row_type, None
        else:
            # Metadata row
            meta_idx = row - len(self.FIXED_ROWS)
            meta_keys = list(self._ensemble.user_metadata.keys())
            if meta_idx < len(meta_keys):
                key = meta_keys[meta_idx]
                return key, RowType.METADATA, key
            return "", RowType.SEPARATOR, None

    def _get_readonly_value(self, row: int) -> str:
        """Get the value for a read-only property row"""
        match row:
            case 0:  # Sample Name
                return self._sample_name
            case 1:  # MS1 Cofeatures
                return str(len(self._ensemble.ms1_cofeatures))
            case 2:  # MS2 Cofeatures
                return str(len(self._ensemble.ms2_cofeatures))
            case 3:  # Base m/z
                return f"{self._ensemble.base_mz:.5f}"
            case 4:  # Peak RT
                return f"{self._ensemble.peak_rt:.2f}"
            case _:
                return ""

    def _get_editable_value(self, row: int) -> str:
        """
        Get the value for an editable property row
        """
        match row:
            case 6:  # Proposed Formula
                return self._ensemble.proposed_formula or ""
            case 7:  # Identity
                return self._ensemble.identity or ""
            case _:
                return ""

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()
        label, row_type, meta_key = self._get_row_info(row)

        if role == Qt.DisplayRole or role == Qt.EditRole:
            if col == 0:
                # Property name column
                return label

            # Value column
            match row_type:
                case RowType.READONLY:
                    return self._get_readonly_value(row)
                case RowType.EDITABLE:
                    return self._get_editable_value(row)
                case RowType.SEPARATOR:
                    return "─" * 20
                case RowType.METADATA:
                    return self._ensemble.user_metadata.get(meta_key, "")

        elif role == Qt.TextAlignmentRole:
            if row_type == RowType.SEPARATOR:
                return Qt.AlignCenter
            return Qt.AlignLeft | Qt.AlignVCenter

        elif role == Qt.ForegroundRole:
            if row_type == RowType.READONLY:
                from PyQt5.QtGui import QColor
                return QColor(128, 128, 128)  # Gray for read-only

        return None

    def setData(self, index: QModelIndex, value, role=Qt.EditRole) -> bool:
        if not index.isValid() or role != Qt.EditRole:
            return False

        row = index.row()
        col = index.column()
        label, row_type, meta_key = self._get_row_info(row)

        # Only value column is editable
        if col != 1:
            return False

        # Only editable and metadata rows can be edited
        if row_type == RowType.EDITABLE:
            match row:
                case 6:  # Proposed Formula
                    self._ensemble.proposed_formula = value if value else None
                case 7:  # Identity
                    self._ensemble.identity = value if value else None
                case _:
                    return False

            self.dataChanged.emit(index, index, [Qt.DisplayRole])
            return True

        elif row_type == RowType.METADATA and meta_key is not None:
            self._ensemble.user_metadata[meta_key] = value
            self.dataChanged.emit(index, index, [Qt.DisplayRole])
            return True

        return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            return Qt.NoItemFlags

        row = index.row()
        col = index.column()
        label, row_type, meta_key = self._get_row_info(row)

        base_flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable

        # Only value column (col 1) can be editable
        if col == 1 and row_type in (RowType.EDITABLE, RowType.METADATA):
            return base_flags | Qt.ItemIsEditable

        # Property name column for metadata rows is also editable (to rename keys)
        if col == 0 and row_type == RowType.METADATA:
            return base_flags | Qt.ItemIsEditable

        return base_flags

    def headerData(self, section: int, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return ["Property", "Value"][section]

        return None

    def add_metadata_field(self, key: str = None) -> int:
        """
        Add a new metadata field. Returns the row index of the new field.
        """
        if key is None:
            # Generate unique key
            existing = set(self._ensemble.user_metadata.keys())
            i = 0
            while f"field_{i}" in existing:
                i += 1
            key = f"field_{i}"

        new_row = self.rowCount()
        self.beginInsertRows(QModelIndex(), new_row, new_row)
        self._ensemble.user_metadata[key] = ""
        self.endInsertRows()
        return new_row

    def remove_metadata_field(self, row: int) -> bool:
        """
        Remove a metadata field by row index.
        Returns True if successful.
        """
        label, row_type, meta_key = self._get_row_info(row)

        if row_type != RowType.METADATA or meta_key is None:
            return False

        self.beginRemoveRows(QModelIndex(), row, row)
        del self._ensemble.user_metadata[meta_key]
        self.endRemoveRows()
        return True

    def is_metadata_row(self, row: int) -> bool:
        """Check if a row is a metadata row"""
        _, row_type, _ = self._get_row_info(row)
        return row_type == RowType.METADATA

    def rename_metadata_key(self, row: int, new_key: str) -> bool:
        """
        Rename a metadata key. Returns True if successful.
        """
        label, row_type, old_key = self._get_row_info(row)

        if row_type != RowType.METADATA or old_key is None:
            return False

        if new_key == old_key:
            return True

        if new_key in self._ensemble.user_metadata:
            return False  # Key already exists

        # Move value to new key
        value = self._ensemble.user_metadata.pop(old_key)
        self._ensemble.user_metadata[new_key] = value

        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [Qt.DisplayRole])
        return True
