"""
The AlignmentViewer's Inspector side panel: a stack of pages (the alignment
summary while nothing is selected, else a tree of the analytes the selection
touches, each with its member ensembles), plus an always-visible analyte
editor (name / SMILES) for the focused item's analyte.

The tree mirrors the viewer's SelectionModel: picking a row asks to focus
it; its context menu copies rows, expands / collapses them and sets / clears
the anchor.
"""
from typing import Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QModelIndex, Qt

from core.formula import delta_formula
from core.utils.natural_sort import natural_sort_key
from gui.views.alignment_viewer.group_dialog import show_smiles
from gui.widgets.alignment_plot.layout import (
    ANCHOR_COLOR, CURRENT_COLOR, HoverTarget,
)
from gui.widgets.alignment_plot.selection import target_key

if TYPE_CHECKING:
    from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment
    from gui.views.alignment_viewer.context import AlignmentContext


def _copy_selection(table: QtWidgets.QAbstractItemView):
    """Copy the selected cells as TSV (a column -> one value per line)."""
    indexes = table.selectionModel().selectedIndexes()
    if not indexes:
        return
    # (top-level row, child row or -1): display order in tables and in
    # the (two-level) compare tree
    rows: dict[tuple[int, int], dict[int, str]] = {}
    for index in indexes:
        parent = index.parent()
        key = (parent.row(), index.row()) if parent.isValid() else (index.row(), -1)
        rows.setdefault(key, {})[index.column()] = index.data() or ""
    cols = sorted({c for row in rows.values() for c in row})
    QtWidgets.QApplication.clipboard().setText('\n'.join(
        '\t'.join(rows[r].get(c, "") for c in cols) for r in sorted(rows)
    ))


def _copy_formulae(tree: QtWidgets.QTreeView):
    """Copy the selected rows' formulae (FORMULA_COLUMN), one per line, in display order."""
    def order(index):
        parent = index.parent()
        return (parent.row(), index.row()) if parent.isValid() else (index.row(), -1)

    formulae = [
        (index.data() or "").rstrip('?')   # '?': an unconfirmed pick
        for index in sorted(tree.selectionModel().selectedRows(FORMULA_COLUMN), key=order)
    ]
    QtWidgets.QApplication.clipboard().setText('\n'.join(f for f in formulae if f))


def _add_copy_action(view: QtWidgets.QAbstractItemView):
    copy = QtWidgets.QAction(view)
    copy.setShortcut(QtGui.QKeySequence.Copy)
    copy.setShortcutContext(Qt.WidgetShortcut)
    copy.triggered.connect(lambda _: _copy_selection(view))
    view.addAction(copy)


# Selection tree: the HoverTarget behind a row, on its first cell (None on
# analyte header rows)
ITEM_ROLE = Qt.UserRole
FORMULA_COLUMN = 2


def _fmt_intsy(value: float) -> str:
    return f"{value:.2e}"


def _fmt_opt(value: Optional[float], fmt: str) -> str:
    return "n/a" if value is None else format(value, fmt)


def _cells(values: list) -> list[QtGui.QStandardItem]:
    cells = [QtGui.QStandardItem(str(v)) for v in values]
    for cell in cells:
        cell.setEditable(False)
    return cells


class Inspector(QtCore.QObject):
    """
    Populates the Inspector widgets generated from the .ui (passed in by
    the viewer) for the current selection.
    """
    # Emitted when the user picks a selection-tree row: focus its target.
    sigFocusRequested = QtCore.pyqtSignal(object)  # HoverTarget
    # From the tree's context menu: make a target the anchor (None clears it).
    sigAnchorRequested = QtCore.pyqtSignal(object)  # HoverTarget | None
    # Emitted when an ensemble's tree row is double-clicked (or opened from
    # the tree's context menu).
    sigEnsembleActivated = QtCore.pyqtSignal(object)  # Ensemble
    # Emitted when the focused analyte's name / SMILES is edited:
    # (AnalyteUUID, field, new value)
    sigAnnotationEdited = QtCore.pyqtSignal(object, str, str)

    def __init__(
        self,
        title_label: QtWidgets.QLabel,
        stack: QtWidgets.QStackedWidget,
        summary_label: QtWidgets.QLabel,
        edit_name: QtWidgets.QLineEdit,
        edit_smiles: QtWidgets.QLineEdit,
        tree_compare: QtWidgets.QTreeView,
        svg_structure,
    ):
        super().__init__()
        self.title_label = title_label
        self.stack = stack
        self.summary_label = summary_label
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.compare_model = QtGui.QStandardItemModel()

        self._focus_uuid = None
        self.edit_name, self.edit_smiles = edit_name, edit_smiles
        for edit, field in ((edit_name, 'name'), (edit_smiles, 'smiles')):
            edit.setEnabled(False)
            edit.editingFinished.connect(
                lambda edit=edit, field=field: self._focus_uuid is not None
                and self.sigAnnotationEdited.emit(
                    self._focus_uuid, field, edit.text().strip()
                )
            )
        self.svg_structure = svg_structure
        svg_structure.hide()

        self.tree_compare = tree_compare
        tree_compare.setModel(self.compare_model)
        tree_compare.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # Whole rows, several at once (for copying)
        tree_compare.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        tree_compare.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        tree_compare.header().setStretchLastSection(True)
        _add_copy_action(tree_compare)
        # Target key -> its row's first cell and label (for marks / current)
        self._item_cells: dict[tuple, tuple[QtGui.QStandardItem, str]] = {}
        self._focus: Optional['HoverTarget'] = None
        self._anchor: Optional['HoverTarget'] = None
        self._listed: set = set()
        tree_compare.selectionModel().currentRowChanged.connect(
            self._on_tree_row_changed
        )
        tree_compare.doubleClicked.connect(
            lambda index: self._activate(self._target_at(index))
        )
        tree_compare.setContextMenuPolicy(Qt.CustomContextMenu)
        tree_compare.customContextMenuRequested.connect(self._show_tree_menu)

    # -- pages --------------------------------------------------------------

    def show_summary(self, alignment: Optional['EnsembleAlignment']):
        self.stack.setCurrentWidget(self.summary_label.parentWidget())
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
            f"Alignment Parameters:\n"
            f"  RT tolerance: {p.rt_tolerance} s\n"
            f"  m/z tolerance: {p.mz_tolerance}\n"
            f"  MS1 / MS2 thresholds: "
            f"{p.ms1_similarity_threshold} / {p.ms2_similarity_threshold}\n"
            f"  MS1 / MS2 weights: {p.ms1_weight} / {p.ms2_weight}\n\n"
            f"Click a strip or line to inspect it; "
            f"Ctrl+click to select several."
        )

    def show_focus(
        self,
        ctx: Optional['AlignmentContext'],
        analyte: Optional['AlignedAnalyte'],
    ):
        """Fill the annotation editor for `analyte` (None clears it)."""
        self._focus_uuid = None if analyte is None else analyte.uuid
        annotation = (
            None if ctx is None or analyte is None
            else ctx.data_source.get_annotation(analyte.uuid)
        )
        smiles = annotation.smiles if annotation else ""
        self.edit_name.setText(annotation.name if annotation else "")
        self.edit_smiles.setText(smiles)
        for edit in (self.edit_name, self.edit_smiles):
            edit.setEnabled(self._focus_uuid is not None)
        show_smiles(self.svg_structure, smiles)
        self.svg_structure.setVisible(bool(smiles))

    def show_selection(
        self,
        ctx: 'AlignmentContext',
        targets: list['HoverTarget'],
        focus: Optional['HoverTarget'],
        anchor: Optional['HoverTarget'],
        similarities: dict,
    ):
        """
        :param similarities: AnalyteUUID -> similarity to the anchor (an
            analyte), from the clustering; shown for analyte rows.

        One row per selected analyte, with all its member ensembles under
        it; collapsed, except the focused ensemble's analyte. An anchor not
        in the tree gets a disabled row on top. With an anchor, columns
        comparing each item to it are added.
        """
        self.stack.setCurrentWidget(self.tree_compare.parentWidget())
        n = len(targets)
        self.title_label.setText(f"<b>Selection</b> · {n} item{'' if n == 1 else 's'}")
        self._anchor = anchor
        anchor_key = target_key(anchor)
        anchor_item = None if anchor is None else ctx.item_spectra(anchor)

        model = self.compare_model
        # Rebuilding moves the current row about: not the user picking one
        blocker = QtCore.QSignalBlocker(self.tree_compare.selectionModel())
        model.clear()
        headers = ["Item", "Intensity", "Formula", "Count"]
        if anchor is not None:
            headers += ["Similarity", "Δm/z", "ΔFmla"]
        model.setHorizontalHeaderLabels(headers)
        self._item_cells = {}

        def item_row(target: 'HoverTarget') -> list[QtGui.QStandardItem]:
            item = ctx.item_spectra(target)  # None: an analyte without spectra
            label = (
                item.label if item is not None
                else ctx.analyte_label(target.analyte) if target.kind == 'analyte'
                else ctx.sample_name(target.sample_uuid)
            )
            values = [
                label,
                _fmt_intsy(target.ensemble.base_intsy) if target.ensemble else "",
                item.formula_label if item else "",
                item.formula_count if item else "",
            ]
            if anchor is not None:
                if target_key(target) == anchor_key:
                    values += ["—", "—", "—"]
                else:
                    similarity = (
                        similarities.get(target.analyte.uuid)
                        if target.kind == 'analyte' else None
                    )
                    both = item is not None and anchor_item is not None
                    values += [
                        _fmt_opt(similarity, ".3f"),
                        f"{item.base_mz - anchor_item.base_mz:+.4f}" if both else "n/a",
                        delta_formula(anchor_item.formula, item.formula) if both else "",
                    ]
            row = _cells(values)
            row[0].setData(target, ITEM_ROLE)
            self._item_cells[target_key(target)] = (row[0], label)
            return row

        self._listed = {target_key(t) for t in targets}
        focused_aid = focus.analyte_id if focus and focus.kind == 'ensemble' else None

        rows = []   # (top-level row, expanded)
        for head in targets:
            aid, analyte = head.analyte_id, head.analyte
            parent = item_row(head)
            members = sorted(
                ctx.members(analyte).items(),
                key=lambda kv: natural_sort_key(ctx.sample_name(kv[0])),
            )
            for sample_uuid, ensemble in members:
                member = HoverTarget('ensemble', aid, sample_uuid, ensemble, analyte)
                parent[0].appendRow(item_row(member))
            rows.append((parent, aid == focused_aid))

        if anchor is not None and anchor_key not in self._item_cells:
            pinned = item_row(anchor)
            for cell in pinned:
                cell.setEnabled(False)
            rows.insert(0, (pinned, False))

        for parent, _ in rows:
            model.appendRow(parent)
        for parent, expanded in rows:
            self.tree_compare.setExpanded(parent[0].index(), expanded)
        del blocker
        self.set_focus(focus)
        for column in range(model.columnCount()):
            self.tree_compare.resizeColumnToContents(column)

    def set_focus(self, focus: Optional['HoverTarget']):
        """Mark the focused row and make it current, silently."""
        self._focus = focus
        self._set_marks()
        found = self._item_cells.get(target_key(focus))
        current = self._target_at(self.tree_compare.currentIndex())
        if found is not None and target_key(current) == target_key(focus):
            return   # picked in the tree: keep the user's row selection (copying)
        blocker = QtCore.QSignalBlocker(self.tree_compare.selectionModel())
        if found is None:
            self.tree_compare.clearSelection()
        else:
            self.tree_compare.setCurrentIndex(found[0].index())
            self.tree_compare.scrollTo(found[0].index())   # expands its parent
        del blocker
        # The view repaints selections on the (blocked) selection signals
        self.tree_compare.viewport().update()

    def _set_marks(self):
        """● the anchor's label, and the focus's within a multi-selection."""
        marks = {target_key(self._anchor): ANCHOR_COLOR}
        # A lone listed item needs no mark, unless one of its members is focused
        focus_key = target_key(self._focus)
        if len(self._listed) > 1 or focus_key not in self._listed:
            marks.setdefault(focus_key, CURRENT_COLOR)
        for key, (cell, label) in self._item_cells.items():
            color = marks.get(key)
            cell.setText(f"{label} ●" if color else label)
            # None clears the role (theme colour); an empty QBrush paints black
            cell.setData(QtGui.QBrush(color) if color else None, Qt.ForegroundRole)

    @staticmethod
    def _target_at(index: QModelIndex) -> Optional['HoverTarget']:
        """The target behind a tree row (None for a header row)."""
        if not index.isValid():
            return None
        return index.sibling(index.row(), 0).data(ITEM_ROLE)

    # -- signals ------------------------------------------------------------

    def _on_tree_row_changed(self, current: QModelIndex, _previous):
        target = self._target_at(current)
        if target is not None:
            self.sigFocusRequested.emit(target)

    def _activate(self, target: Optional['HoverTarget']):
        if target is not None and target.kind == 'ensemble' and target.ensemble:
            self.sigEnsembleActivated.emit(target.ensemble)

    def _show_tree_menu(self, pos: QtCore.QPoint):
        target = self._target_at(self.tree_compare.indexAt(pos))
        menu = QtWidgets.QMenu(self.tree_compare)
        if target is not None and target_key(target) != target_key(self._anchor):
            menu.addAction("Set as anchor").triggered.connect(
                lambda: self.sigAnchorRequested.emit(target)
            )
        if self._anchor is not None:
            menu.addAction("Clear anchor").triggered.connect(
                lambda: self.sigAnchorRequested.emit(None)
            )
        if target is not None and target.kind == 'ensemble':
            menu.addAction("Open in Ensemble Viewer").triggered.connect(
                lambda: self._activate(target)
            )
        if not menu.isEmpty():
            menu.addSeparator()
        tree = self.tree_compare
        menu.addAction("Copy", lambda: _copy_selection(tree))
        menu.addAction("Copy formulae", lambda: _copy_formulae(tree))
        menu.addAction("Expand all", tree.expandAll)
        menu.addAction("Collapse all", tree.collapseAll)
        menu.exec_(tree.viewport().mapToGlobal(pos))
