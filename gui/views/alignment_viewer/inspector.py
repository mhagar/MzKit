"""
The AlignmentViewer's Inspector side panel: a stack of pages, one per
selection mode (nothing / ensemble / analyte / compare), each backed by a
read-only table model.
"""
from typing import Any, Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt

if TYPE_CHECKING:
    from core.cli.align_ensembles import PairScore
    from core.data_structs import Ensemble, SampleUUID
    from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment
    from gui.views.alignment_viewer.context import AlignmentContext, ItemSpectra


class ReadOnlyTableModel(QAbstractTableModel):
    """A plain headers + rows table. Row payloads are kept for lookups."""

    def __init__(self, headers: list[str], parent=None):
        super().__init__(parent)
        self._headers = headers
        self._rows: list[list[Any]] = []
        self._payloads: list[Any] = []

    def set_rows(self, rows: list[list[Any]], payloads: Optional[list] = None):
        self.beginResetModel()
        self._rows = rows
        self._payloads = payloads if payloads is not None else [None] * len(rows)
        self.endResetModel()

    def payload(self, row: int) -> Any:
        return self._payloads[row] if 0 <= row < len(self._payloads) else None

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        value = self._rows[index.row()][index.column()]
        return "" if value is None else str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]
        return None


def _fmt_intsy(value: float) -> str:
    return f"{value:.2e}"


def _fmt_opt(value: Optional[float], fmt: str) -> str:
    return "n/a" if value is None else format(value, fmt)


class Inspector(QtCore.QObject):
    """
    Populates the Inspector widgets generated from the .ui (passed in by
    the viewer) for the current selection.
    """
    # Emitted when the user picks a compare-table row: the row's index
    # into the selection (never the anchor, row 0).
    sigCompareRowSelected = QtCore.pyqtSignal(int)
    # Emitted when a member ensemble row is double-clicked.
    sigMemberActivated = QtCore.pyqtSignal(object)  # Ensemble

    PAGE_EMPTY, PAGE_ENSEMBLE, PAGE_ANALYTE, PAGE_COMPARE = range(4)

    def __init__(
        self,
        title_label: QtWidgets.QLabel,
        stack: QtWidgets.QStackedWidget,
        summary_label: QtWidgets.QLabel,
        table_ensemble: QtWidgets.QTableView,
        table_analyte_props: QtWidgets.QTableView,
        table_analyte_members: QtWidgets.QTableView,
        table_compare: QtWidgets.QTableView,
    ):
        super().__init__()
        self.title_label = title_label
        self.stack = stack
        self.summary_label = summary_label
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.ensemble_model = ReadOnlyTableModel(["Property", "Value"])
        self.analyte_props_model = ReadOnlyTableModel(["Property", "Value"])
        self.members_model = ReadOnlyTableModel(
            ["Sample", "RT (s)", "Base m/z", "Intensity", "Formula"]
        )
        self.compare_model = ReadOnlyTableModel(
            ["Item", "ΔRT (s)", "Δ ppm", "MS1 cos", "MS2 cos",
             "MS1 matched", "MS2 matched"]
        )

        for table, model in (
            (table_ensemble, self.ensemble_model),
            (table_analyte_props, self.analyte_props_model),
            (table_analyte_members, self.members_model),
            (table_compare, self.compare_model),
        ):
            table.setModel(model)
            table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            table.horizontalHeader().setStretchLastSection(True)
            table.verticalHeader().setVisible(False)
            table.verticalHeader().setStretchLastSection(False)

        self.table_compare = table_compare
        table_compare.selectionModel().currentRowChanged.connect(
            self._on_compare_row_changed
        )
        table_analyte_members.doubleClicked.connect(
            lambda index: self._emit_member(index.row())
        )

    # -- pages --------------------------------------------------------------

    def show_summary(self, alignment: Optional['EnsembleAlignment']):
        self.stack.setCurrentIndex(self.PAGE_EMPTY)
        if alignment is None:
            self.title_label.setText("<b>No alignment loaded</b>")
            self.summary_label.setText("")
            return

        multi = sum(1 for a in alignment.analytes if len(a.ensemble_map) > 1)
        p = alignment.parameters
        self.title_label.setText(f"<b>{alignment.name or 'Alignment'}</b>")
        self.summary_label.setText(
            f"{alignment.analyte_count} analytes across "
            f"{alignment.sample_count} samples\n"
            f"{multi} matched, {alignment.analyte_count - multi} singletons\n\n"
            f"Parameters:\n"
            f"  RT tolerance: {p.rt_tolerance} s\n"
            f"  m/z tolerance: {p.mz_tolerance}\n"
            f"  MS1 / MS2 thresholds: "
            f"{p.ms1_similarity_threshold} / {p.ms2_similarity_threshold}\n"
            f"  MS1 / MS2 weights: {p.ms1_weight} / {p.ms2_weight}\n\n"
            f"Click a strip or line to inspect it; "
            f"Ctrl+click to compare several."
        )

    def show_ensemble(
        self,
        ctx: 'AlignmentContext',
        sample_uuid: 'SampleUUID',
        ensemble: 'Ensemble',
        analyte: Optional['AlignedAnalyte'],
    ):
        self.stack.setCurrentIndex(self.PAGE_ENSEMBLE)
        name = ctx.sample_name(sample_uuid)
        self.title_label.setText(f"<b>Ensemble</b> · {name}")

        rows = [
            ["Sample", name],
            ["Base m/z", f"{ensemble.base_mz:.4f}"],
            ["Base intensity", _fmt_intsy(ensemble.base_intsy)],
            ["Peak RT (s)", f"{ensemble.peak_rt:.2f}"],
            ["MS1 cofeatures", len(ensemble.ms1_cofeatures)],
            ["MS2 cofeatures", len(ensemble.ms2_cofeatures)],
            ["Formula", ctx.formula(ensemble)],
            ["Identity", ensemble.identity or ""],
        ]
        if analyte is not None:
            rows.append([
                "Aligned in",
                f"{len(analyte.ensemble_map)}/{ctx.alignment.sample_count} samples",
            ])
        rows.extend([k, v] for k, v in ensemble.user_metadata.items())
        self.ensemble_model.set_rows(rows)

    def show_analyte(
        self,
        ctx: 'AlignmentContext',
        analyte: 'AlignedAnalyte',
    ):
        self.stack.setCurrentIndex(self.PAGE_ANALYTE)
        members = ctx.members(analyte)
        rep = ctx.representative(analyte)
        n, total = len(analyte.ensemble_map), ctx.alignment.sample_count
        self.title_label.setText(f"<b>Analyte</b> · {n}/{total} samples")

        agreement = ctx.formula_agreement(analyte)
        props = [
            ["Consensus RT (s)", f"{analyte.consensus_rt:.2f}"],
            ["Representative base m/z",
             f"{rep.base_mz:.4f}" if rep else "n/a"],
            ["Mean base m/z", f"{analyte.consensus_mz:.4f}"],
            ["Detected in", f"{n}/{total} samples"],
            ["Representative from",
             ctx.sample_name(rep.sample_uuid) if rep else "n/a"],
            ["Formula",
             f"{agreement[0]} ({agreement[1]}/{agreement[2]})" if agreement else ""],
        ]
        self.analyte_props_model.set_rows(props)

        ordered = sorted(members.items(), key=lambda kv: ctx.sample_name(kv[0]))
        self.members_model.set_rows(
            [
                [
                    ctx.sample_name(sample_uuid),
                    f"{ens.peak_rt:.2f}",
                    f"{ens.base_mz:.4f}",
                    _fmt_intsy(ens.base_intsy),
                    ctx.formula(ens),
                ]
                for sample_uuid, ens in ordered
            ],
            payloads=[ens for _, ens in ordered],
        )

    def show_compare(
        self,
        items: list['ItemSpectra'],
        scores: list[Optional['PairScore']],
        current_row: int,
    ):
        """
        :param items: selected items; items[0] is the anchor.
        :param scores: scores[i] is items[i] vs the anchor (None for the anchor).
        :param current_row: the row whose pair is on the mirror plots.
        """
        self.stack.setCurrentIndex(self.PAGE_COMPARE)
        self.title_label.setText(f"<b>Compare</b> · {len(items)} items")

        anchor = items[0]
        rows = [[f"{anchor.label} (anchor)", "—", "—", "—", "—", "—", "—"]]
        for item, score in zip(items[1:], scores[1:]):
            d_ppm = (
                1e6 * (item.base_mz - anchor.base_mz) / anchor.base_mz
                if anchor.base_mz else None
            )
            rows.append([
                item.label,
                f"{item.rt - anchor.rt:+.2f}",
                _fmt_opt(d_ppm, "+.1f"),
                f"{score.ms1:.3f}",
                _fmt_opt(score.ms2, ".3f"),
                len(score.ms1_matches),
                len(score.ms2_matches) if score.ms2 is not None else "n/a",
            ])

        blocker = QtCore.QSignalBlocker(self.table_compare.selectionModel())
        self.compare_model.set_rows(rows)
        self.table_compare.selectRow(current_row)
        del blocker
        self.table_compare.resizeColumnsToContents()

    # -- signals ------------------------------------------------------------

    def _on_compare_row_changed(self, current: QModelIndex, _previous):
        if current.isValid() and current.row() > 0:
            self.sigCompareRowSelected.emit(current.row())

    def _emit_member(self, row: int):
        ensemble = self.members_model.payload(row)
        if ensemble is not None:
            self.sigMemberActivated.emit(ensemble)
