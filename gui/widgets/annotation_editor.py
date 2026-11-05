"""
The Annotation tab: one Ensemble's EnsembleAnnotation, the host's
AnnotationTarget's (in the Alignment Viewer: the focused member's, or the
representative member's when an analyte is focused; the header says which, and
what the other members say). Its name (the chosen
hypothesis's unless typed over), notes, and one row per hypothesis (★ the
chosen structure). The last row takes a typed / pasted SMILES as a new user
hypothesis. A structure whose formula isn't that ensemble's is drawn red.

Edits go out as a whole new annotation (sigAnnotationChanged) for the host
to set on the ensemble; the registry's change signal re-shows it here.
Selecting a row asks the viewer to mirror its GLACIER prediction
(sigHypothesisSelected).
"""
import dataclasses
from functools import lru_cache
from typing import Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtGui, QtWidgets
from rdkit import Chem

from core.data_structs.ensemble import USER, EnsembleAnnotation, Hypothesis
from core.formula.manual import canonical_formula
from core.utils.glacier import neutral_formula
from core.utils.formula_formatting import format_assignment_formula_text
from gui.widgets.structure_search_widget import show_fragment
from gui.widgets.StructureWidget import NAME_ROLE, StructureDelegate

if TYPE_CHECKING:
    from core.cli.rank_candidates import ScoredCandidate
    from core.data_structs.ensemble import Ensemble
    from gui.widgets.annotation_target import AnnotationTarget
    from gui.widgets.StructureWidget import StructureView

STRUCTURE, NAME, ORIGIN = range(3)
ROW_HEIGHT = 48
# On a row's Name cell: its index into `hypotheses` (None: the add row), and its bare name
IDX_ROLE = QtCore.Qt.UserRole + 1
RAW_ROLE = QtCore.Qt.UserRole + 2
ADD_TEXT = "Add SMILES…"
MISMATCH = "SMILES does not match formula"


@lru_cache(maxsize=256)
def matches_formula(smiles: str, formula: str) -> bool:
    """False only if both parse and differ."""
    mol = Chem.MolFromSmiles(smiles)
    try:
        return mol is None or canonical_formula(neutral_formula(mol)) == canonical_formula(formula)
    except ValueError:
        return True


def _parses(smiles: str) -> bool:
    return bool(smiles) and Chem.MolFromSmiles(smiles) is not None


class _NameDelegate(QtWidgets.QStyledItemDelegate):
    """Edits the bare name (no ★), and the add row opens empty."""

    def setEditorData(self, editor, index):
        editor.setText(index.data(RAW_ROLE) or "")
        if index.data(IDX_ROLE) is None:
            editor.setPlaceholderText("SMILES")


class AnnotationEditor(QtCore.QObject):
    sigAnnotationChanged = QtCore.pyqtSignal(object, object)   # EnsembleUUID, EnsembleAnnotation (None: clear)
    sigHypothesisSelected = QtCore.pyqtSignal(object)          # Hypothesis, or None
    sigApplyToAllRequested = QtCore.pyqtSignal(object)         # Hypothesis

    def __init__(
        self,
        edit_name: QtWidgets.QLineEdit,
        label_target: QtWidgets.QLabel,
        edit_notes: QtWidgets.QLineEdit,
        tree: QtWidgets.QTreeWidget,
        structure: 'StructureView',
    ):
        super().__init__()
        self.edit_name, self.label_target, self.edit_notes = edit_name, label_target, edit_notes
        self.tree, self.structure = tree, structure
        self._target: Optional['Ensemble'] = None   # whose annotation this edits
        self._key = None   # the AnnotationTarget's key, like the Search tabs'
        self._formula: Optional[str] = None
        self._n_members = 0   # > 1: offer "Apply to all aligned ensembles"
        self._selected: Optional[Hypothesis] = None   # as last announced
        self._keep: Optional[str] = None   # SMILES to select once the edit re-shows the tree

        edit_name.editingFinished.connect(lambda: self._set(name=edit_name.text().strip()))
        edit_notes.editingFinished.connect(lambda: self._set(notes=edit_notes.text().strip()))

        tree.setItemDelegateForColumn(STRUCTURE, StructureDelegate(tree, extend_menu=self._extend_menu))
        tree.setItemDelegateForColumn(NAME, _NameDelegate(tree))
        tree.setColumnWidth(STRUCTURE, ROW_HEIGHT * 3 // 2)
        tree.setColumnWidth(NAME, 140)
        tree.itemChanged.connect(self._on_item_changed)
        tree.itemSelectionChanged.connect(self._on_selected)

        structure.enlarge_on_hover = False   # it's already the big view
        self.set_target(None)

    # -- focus --------------------------------------------------------------

    def set_target(self, target: Optional['AnnotationTarget']):
        """
        Follow the host's focus (None clears): `target.ensemble`'s annotation,
        checked against `target.formula`. A new focus drops the selected row.
        """
        key = None if target is None else target.key
        if key != self._key:   # the host redraws the MS2 plot: no mirror left to follow
            self.tree.clearSelection()
            self._selected = None
        self._key = key
        self._target = None if target is None else target.ensemble
        # Checked against the ensemble's own formula (the target's is a query prefill)
        self._formula = None if self._target is None else (
            format_assignment_formula_text(self._target.formula_assignment).rstrip('?') or None
        )
        self._n_members = 0 if target is None else len(target.members)
        self.label_target.setText("" if target is None else target.header)
        self.label_target.setToolTip("" if target is None else target.header_tip)
        self._refresh()

    @property
    def target_uuid(self):
        """The EnsembleUUID being edited (None: nothing focused)."""
        return None if self._target is None else self._target.uuid

    @property
    def annotation(self) -> EnsembleAnnotation:
        existing = None if self._target is None else self._target.annotation
        return existing or EnsembleAnnotation()

    def _refresh(self):
        ann, enabled = self.annotation, self._target is not None
        for w in (self.edit_name, self.edit_notes, self.tree):
            w.setEnabled(enabled)
        self.edit_name.setText(ann.name)
        self.edit_name.setPlaceholderText((ann.chosen.name if ann.chosen else "") or "Name...")
        self.edit_notes.setText(ann.notes)

        keep = self._keep or (self._selected and self._selected.smiles)
        self._keep = None
        with QtCore.QSignalBlocker(self.tree):
            self.tree.clear()
            if enabled:
                for idx, h in enumerate(ann.hypotheses):
                    self.tree.addTopLevelItem(self._row(idx, h, idx == ann.chosen_idx))
                add = QtWidgets.QTreeWidgetItem(["", ADD_TEXT])
                add.setData(NAME, RAW_ROLE, "")
                add.setForeground(NAME, self.tree.palette().brush(QtGui.QPalette.Disabled, QtGui.QPalette.Text))
                add.setFlags(add.flags() | QtCore.Qt.ItemIsEditable)
                self.tree.addTopLevelItem(add)
            # Keep the selected structure selected (an edit re-shows the tree)
            row = next((i for i, h in enumerate(ann.hypotheses) if keep and h.smiles == keep), None)
            if row is not None:
                self.tree.topLevelItem(row).setSelected(True)
        self._on_selected()

    def _row(self, idx: int, h: Hypothesis, chosen: bool) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem(["", f"★ {h.name}" if chosen else h.name, h.origin])
        item.setSizeHint(STRUCTURE, QtCore.QSize(ROW_HEIGHT * 3 // 2, ROW_HEIGHT))
        item.setData(STRUCTURE, QtCore.Qt.UserRole, h.smiles)
        item.setData(STRUCTURE, NAME_ROLE, h.name)
        item.setData(NAME, IDX_ROLE, idx)
        item.setData(NAME, RAW_ROLE, h.name)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        if chosen:
            font = item.font(NAME)
            font.setBold(True)
            item.setFont(NAME, font)
        tips = [f"{k}: {v}" for k, v in h.scores.items()]
        if self._formula and not matches_formula(h.smiles, self._formula):
            tips.insert(0, MISMATCH)
            for col in range(self.tree.columnCount()):
                item.setForeground(col, QtGui.QColor('red'))
        for col in range(self.tree.columnCount()):
            item.setToolTip(col, "\n".join(tips))
        return item

    # -- edits --------------------------------------------------------------

    def _emit(self, new: EnsembleAnnotation):
        if new != self.annotation:
            self.sigAnnotationChanged.emit(self._target.uuid, None if new.is_empty else new)

    def _set(self, **changes):
        if self._target is not None:
            self._emit(dataclasses.replace(self.annotation, **changes))

    def _with(self, hypotheses: list, chosen_idx: Optional[int]):
        self._emit(dataclasses.replace(self.annotation, hypotheses=tuple(hypotheses), chosen_idx=chosen_idx))

    def _valid(self, smiles: str) -> bool:
        if _parses(smiles):
            return True
        QtWidgets.QMessageBox.warning(self.tree, "Invalid SMILES", f"Not a valid SMILES: {smiles!r}")
        return False

    def add_hypothesis(self, h: Hypothesis):
        """Add `h` (or find it already there), choose it and select it."""
        if self._target is not None:
            self._keep = h.smiles
            self._emit(self.annotation.with_hypothesis(h))

    def _on_item_changed(self, item: QtWidgets.QTreeWidgetItem, col: int):
        if col != NAME:
            return
        idx, text = item.data(NAME, IDX_ROLE), item.text(NAME).strip()
        ann = self.annotation
        if idx is None:   # the add row
            if text and text != ADD_TEXT and self._valid(text):
                h = Hypothesis(text, origin=USER)
                self._keep = text
                # A first structure is the ensemble's
                self._with([*ann.hypotheses, h], len(ann.hypotheses) if ann.chosen_idx is None else ann.chosen_idx)
                return
        else:
            hyps = list(ann.hypotheses)
            hyps[idx] = dataclasses.replace(hyps[idx], name=text)
            self._with(hyps, ann.chosen_idx)
            return
        self._refresh()

    def _extend_menu(self, menu: QtWidgets.QMenu, index: QtCore.QModelIndex):
        idx = index.sibling(index.row(), NAME).data(IDX_ROLE)
        if idx is None:
            return
        chosen = idx == self.annotation.chosen_idx
        menu.addSeparator()
        menu.addAction("Unchoose" if chosen else "Choose as structure", lambda: self._choose(None if chosen else idx))
        menu.addAction("Edit SMILES…", lambda: self._edit_smiles(idx))
        menu.addAction("Delete", lambda: self._delete(idx))
        if self._n_members > 1:
            menu.addSeparator()
            menu.addAction("Apply to all aligned ensembles",
                           lambda: self.sigApplyToAllRequested.emit(self.annotation.hypotheses[idx]))

    def _choose(self, idx: Optional[int]):
        self._with(list(self.annotation.hypotheses), idx)

    def _edit_smiles(self, idx: int):
        ann = self.annotation
        old = ann.hypotheses[idx]
        smiles, ok = QtWidgets.QInputDialog.getText(self.tree, "Edit SMILES", "SMILES:", text=old.smiles)
        smiles = smiles.strip()
        if not ok or smiles == old.smiles or not self._valid(smiles):
            return
        hyps = list(ann.hypotheses)
        if old.origin == USER:
            hyps[idx] = dataclasses.replace(old, smiles=smiles, scores={})
            self._keep = smiles
            self._with(hyps, ann.chosen_idx)
        else:   # a database's structure stays as it was: the edit is the user's
            self._keep = smiles
            hyps.append(Hypothesis(smiles, old.name, origin=USER))
            self._with(hyps, len(hyps) - 1 if idx == ann.chosen_idx else ann.chosen_idx)

    def _delete(self, idx: int):
        ann = self.annotation
        hyps = list(ann.hypotheses)
        del hyps[idx]
        chosen = ann.chosen_idx
        chosen = None if chosen == idx else chosen - 1 if chosen is not None and chosen > idx else chosen
        self._with(hyps, chosen)

    # -- selection ----------------------------------------------------------

    def selected(self) -> Optional[Hypothesis]:
        items = self.tree.selectedItems()
        idx = items[0].data(NAME, IDX_ROLE) if items else None
        return None if idx is None else self.annotation.hypotheses[idx]

    def _on_selected(self):
        h = self.selected()
        shown = h or self.annotation.chosen
        self.structure.set_structure(shown.smiles if shown else "", shown.name if shown else "")
        self.structure.set_caption("")
        self.structure.setVisible(shown is not None)
        changed = (h and h.smiles) != (self._selected and self._selected.smiles)
        self._selected = h
        if changed:
            self.sigHypothesisSelected.emit(h)

    def reselect(self):
        """Re-announce the selected hypothesis (its tab came back to the front)."""
        self.sigHypothesisSelected.emit(self._selected)

    def highlight_fragment(self, scored: Optional['ScoredCandidate'], peak_idx: Optional[int]):
        """The mirrored prediction's predicted peak `peak_idx` lights its fragment."""
        if scored is not None:
            show_fragment(self.structure, scored, peak_idx)
