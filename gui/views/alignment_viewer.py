"""
Viewer for EnsembleAlignment results.

Displays a table where rows = AlignedAnalytes and columns =
consensus m/z, consensus RT, then one column per sample showing
the matched ensemble's base intensity (or blank if absent).
"""
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
)

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from core.data_structs.alignment import EnsembleAlignment, AlignedAnalyte
    from core.data_structs import DataRegistry, SampleUUID, Ensemble


class AlignmentTableModel(QAbstractTableModel):
    """
    Table model for an EnsembleAlignment.

    Fixed columns: m/z, RT, # samples
    Dynamic columns: one per sample (intensity or blank)
    """
    FIXED_COLUMNS = ['m/z', 'RT (s)', '# Samples']
    N_FIXED = len(FIXED_COLUMNS)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._alignment: Optional['EnsembleAlignment'] = None
        self._data_registry: Optional['DataRegistry'] = None
        self._sample_names: list[str] = []
        self._sample_uuids: list['SampleUUID'] = []

        # Cache: (row, sample_uuid) -> ensemble or None
        self._ensemble_cache: dict[tuple[int, 'SampleUUID'], Optional['Ensemble']] = {}

    def set_alignment(
        self,
        alignment: 'EnsembleAlignment',
        data_registry: 'DataRegistry',
    ):
        self.beginResetModel()

        self._alignment = alignment
        self._data_registry = data_registry
        self._ensemble_cache.clear()

        # Resolve sample names in alignment order
        self._sample_uuids = list(alignment.sample_uuids)
        self._sample_names = []
        for uuid in self._sample_uuids:
            sample = data_registry.get_sample(uuid)
            self._sample_names.append(
                sample.name if sample else f"...{str(uuid)[-5:]}"
            )

        self.endResetModel()

    def reset(self):
        """
        Clear the model back to an empty (no-alignment) state.
        """
        self.beginResetModel()
        self._alignment = None
        self._data_registry = None
        self._sample_names = []
        self._sample_uuids = []
        self._ensemble_cache.clear()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        if not self._alignment:
            return 0
        return self._alignment.analyte_count

    def columnCount(self, parent=QModelIndex()):
        if not self._alignment:
            return 0
        return self.N_FIXED + len(self._sample_uuids)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            if section < self.N_FIXED:
                return self.FIXED_COLUMNS[section]
            sample_idx = section - self.N_FIXED
            return self._sample_names[sample_idx]

        if orientation == Qt.Vertical:
            return str(section + 1)

        return None

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid() or not self._alignment:
            return None

        row = index.row()
        col = index.column()
        analyte = self._alignment.analytes[row]

        if role == Qt.DisplayRole:
            return self._display_data(row, col, analyte)

        if role == Qt.TextAlignmentRole:
            return Qt.AlignRight | Qt.AlignVCenter

        return None

    def _display_data(
        self,
        row: int,
        col: int,
        analyte: 'AlignedAnalyte',
    ):
        match col:
            case 0:
                return f"{analyte.consensus_mz:.4f}"
            case 1:
                return f"{analyte.consensus_rt:.1f}"
            case 2:
                return str(len(analyte.ensemble_map))

        # Sample column
        sample_idx = col - self.N_FIXED
        sample_uuid = self._sample_uuids[sample_idx]

        ensemble = self._get_ensemble(row, sample_uuid, analyte)
        if ensemble is None:
            return ""

        return f"{ensemble.base_intsy:.0f}"

    def _get_ensemble(
        self,
        row: int,
        sample_uuid: 'SampleUUID',
        analyte: 'AlignedAnalyte',
    ) -> Optional['Ensemble']:
        cache_key = (row, sample_uuid)
        if cache_key in self._ensemble_cache:
            return self._ensemble_cache[cache_key]

        ensemble = None
        if sample_uuid in analyte.ensemble_map:
            ens_uuid = analyte.ensemble_map[sample_uuid]
            sample = self._data_registry.get_sample(sample_uuid)
            if sample and sample.injection:
                ensemble = sample.injection.ensembles.get(ens_uuid)

        self._ensemble_cache[cache_key] = ensemble
        return ensemble


class AlignmentViewer(QtWidgets.QWidget):
    """
    Widget for viewing an EnsembleAlignment as a table.
    """
    def __init__(
        self,
        data_source: 'DataRegistry',
        parent=None,
    ):
        super().__init__(parent)
        self._data_registry = data_source
        self._model = AlignmentTableModel()

        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header label
        self._label = QtWidgets.QLabel("No alignment loaded")
        layout.addWidget(self._label)

        # Table view
        self._table_view = QtWidgets.QTableView()
        self._table_view.setModel(self._model)
        self._table_view.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self._table_view.setAlternatingRowColors(True)
        self._table_view.setSortingEnabled(True)
        self._table_view.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table_view)

    def set_alignment(
        self,
        alignment: 'EnsembleAlignment',
    ):
        self._model.set_alignment(alignment, self._data_registry)

        multi = sum(
            1 for a in alignment.analytes
            if len(a.ensemble_map) > 1
        )
        self._label.setText(
            f"{alignment.analyte_count} analytes across "
            f"{alignment.sample_count} samples "
            f"({multi} matched, "
            f"{alignment.analyte_count - multi} singletons)"
        )

    def reset_for_new_project(self):
        """
        Clear any displayed alignment when the workspace is reset.
        """
        self._model.reset()
        self._label.setText("No alignment loaded")
