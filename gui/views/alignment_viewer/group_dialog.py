"""
Dialog creating or editing an `EnsembleGroup` (title, notes, SMILES with a
live structure preview) for the Alignment Viewer. A selected analyte stands
for all of its member ensembles.

Exact-match rule (`match_groups`): grouping a selection whose membership
equals an existing group's edits that group instead; if groups merely
overlap the selection, they're listed so the user can pick "Edit existing"
(one of them; its membership is left as is) or "Create new".
"""
from typing import Iterable, Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtWidgets

from core.data_structs.ensemble_group import EnsembleGroup
from core.utils.render_molecule import smiles_to_svg
from gui.resources.AnalyteGroupDialog import Ui_Dialog

if TYPE_CHECKING:
    from core.data_structs.uuid_types import EnsembleUUID
    from gui.views.alignment_viewer.data_source import AlignmentViewerDataSource


def match_groups(
    ensemble_uuids: Iterable['EnsembleUUID'],
    groups: Iterable[EnsembleGroup],
) -> tuple[Optional[EnsembleGroup], list[EnsembleGroup]]:
    """
    (the group whose membership equals the selection, or None;
    the groups sharing members with it - empty when there is an exact match).
    """
    selection = set(ensemble_uuids)
    groups = list(groups)
    for group in groups:
        if set(group.ensemble_uuids) == selection:
            return group, []
    return None, [g for g in groups if not selection.isdisjoint(g.ensemble_uuids)]


def jsonable(value):
    """NamedTuples (e.g. a ClusterKey and its ImportParams) -> nested dicts."""
    if hasattr(value, '_asdict'):
        return {k: jsonable(v) for k, v in value._asdict().items()}
    return value


_SVG_SCALE = 2.0  # ACS1996 bonds are 14.4 px: too small on screen
_SVG_MAX_HEIGHT = 400


def show_smiles(svg_widget, smiles: str) -> list[str]:
    """
    Render `smiles` into a QSvgWidget (cleared when empty).
    :return: warnings, or the parse error, to show the user.
    """
    if not smiles.strip():
        svg_widget.load(QtCore.QByteArray())
        return []
    try:
        svg, warnings = smiles_to_svg(smiles)
    except ValueError as e:
        svg_widget.load(QtCore.QByteArray())
        return [str(e)]
    svg_widget.load(QtCore.QByteArray(svg.encode()))
    # Fixed bond length: height = natural (ACS1996) size x _SVG_SCALE; aspect
    # kept, so a too-narrow widget shrinks the drawing instead of stretching it
    renderer = svg_widget.renderer()
    renderer.setAspectRatioMode(QtCore.Qt.KeepAspectRatio)
    svg_widget.setFixedHeight(
        min(round(renderer.defaultSize().height() * _SVG_SCALE), _SVG_MAX_HEIGHT)
    )
    return warnings


class EnsembleGroupDialog(QtWidgets.QDialog, Ui_Dialog):
    """
    :param ensemble_uuids: the selection to group (ignored when `group` is given).
    :param provenance: stored on a newly created group.
    :param group: edit this group directly (no matching).
    """

    def __init__(
        self,
        data_source: 'AlignmentViewerDataSource',
        ensemble_uuids: list['EnsembleUUID'],
        provenance: Optional[dict] = None,
        group: Optional[EnsembleGroup] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.data_source = data_source
        self._ensemble_uuids = list(dict.fromkeys(ensemble_uuids))
        self._provenance = provenance

        overlapping: list[EnsembleGroup] = []
        if group is None:
            group, overlapping = match_groups(
                self._ensemble_uuids,
                data_source.groups_for_ensembles(self._ensemble_uuids),
            )
        self._group = group     # edited in place on accept, else None
        self._overlapping = overlapping

        n = len(self._ensemble_uuids) if group is None else len(group.ensemble_uuids)
        self.setWindowTitle(
            f"{'Edit' if group is not None else 'New'} Group "
            f"({n} ensemble{'' if n == 1 else 's'})"
        )
        self.labelSmilesWarning.setWordWrap(True)
        self.labelSmilesWarning.setStyleSheet("color: #c33")

        for widget in (self.label_2, self.listOverlapping,
                       self.radioEditExisting, self.radioCreateNew):
            widget.setVisible(bool(overlapping))
        for g in overlapping:
            self.listOverlapping.addItem(
                f"{g.title or '(untitled)'} - {len(g.ensemble_uuids)} members"
            )
        self.radioCreateNew.setChecked(True)

        self.editSmiles.textChanged.connect(self._update_preview)
        self.listOverlapping.currentRowChanged.connect(self._on_overlap_picked)
        self.radioEditExisting.toggled.connect(self._on_mode_toggled)
        self.btnSnapMfs.clicked.connect(self._run_snap_mfs)
        from core.cli.snap_mfs import METRICS, MIN_FAMILY_SIZE, TANIMOTO_CUTOFF
        for key, label in METRICS.items():
            self.comboSnapMfsMetric.addItem(label, key)
        self.spinSnapMfsCutoff.setValue(TANIMOTO_CUTOFF)
        self.spinSnapMfsMinSize.setMinimum(1)
        self.spinSnapMfsMinSize.setValue(MIN_FAMILY_SIZE)
        from core.utils import compound_db
        if compound_db.cache_ok():
            self.comboDatabase.addItems(compound_db.available_sources())

        self._load(group)

    def _load(self, group: Optional[EnsembleGroup]):
        self.editTitle.setText(group.title if group else "")
        self.editNotes.setPlainText(group.notes if group else "")
        self.editSmiles.setText(group.smiles if group else "")
        self._update_preview()

    def _picked(self) -> Optional[EnsembleGroup]:
        row = self.listOverlapping.currentRow()
        return self._overlapping[row] if 0 <= row < len(self._overlapping) else None

    def _on_overlap_picked(self, _row: int):
        if not self.radioEditExisting.isChecked():
            self.radioEditExisting.setChecked(True)   # loads via _on_mode_toggled
        else:
            self._load(self._picked())

    def _on_mode_toggled(self, edit_existing: bool):
        if edit_existing and self.listOverlapping.currentRow() < 0 and self._overlapping:
            self.listOverlapping.setCurrentRow(0)
        self._load(self._picked() if edit_existing else None)

    def _update_preview(self, *_):
        warnings = show_smiles(self.svgPreview, self.editSmiles.text())
        self.labelSmilesWarning.setText("\n".join(warnings))
        self.labelSmilesWarning.setVisible(bool(warnings))

    def _member_formulas(self) -> list[str]:
        """The member ensembles' formulae (accepted, else top candidate), deduplicated."""
        from core.utils.formula_formatting import format_assignment_formula_text
        group = self._group or (
            self._picked() if self.radioEditExisting.isChecked() else None
        )
        uuids = group.ensemble_uuids if group is not None else self._ensemble_uuids
        ensembles = (self.data_source.get_ensemble(u) for u in uuids)
        formulas = (
            format_assignment_formula_text(e.formula_assignment).rstrip('?')
            for e in ensembles if e is not None
        )
        return list(dict.fromkeys(f for f in formulas if f))

    def _run_snap_mfs(self):
        """Testing hook: run SNAP-MFs on the members' formulae, print to stdout (blocks)."""
        from core.cli.snap_mfs import format_families, snap_mfs
        from core.utils import compound_db
        if not compound_db.cache_ok():
            QtWidgets.QMessageBox.warning(
                self, "No compound database",
                "Build (or rebuild) the cache first: Tools > Libraries & databases.",
            )
            return
        metric = self.comboSnapMfsMetric.currentData()
        cutoff = self.spinSnapMfsCutoff.value()
        min_size = self.spinSnapMfsMinSize.value()
        source = self.comboDatabase.currentText()
        formulas = self._member_formulas()
        print(f"SNAP-MFs on {len(formulas)} member formulae: {sorted(formulas)}")
        if not formulas:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            print(format_families(snap_mfs(
                formulas, source, cutoff, min_size, metric=metric, log=print,
            ), formulas))
        except (OSError, ValueError) as e:
            print(f"SNAP-MFs failed: {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def accept(self):
        smiles = self.editSmiles.text().strip()
        if smiles:
            try:
                smiles_to_svg(smiles)
            except ValueError as e:
                QtWidgets.QMessageBox.warning(self, "Invalid SMILES", str(e))
                return

        group = self._group
        if group is None and self.radioEditExisting.isChecked():
            group = self._picked()
        title = self.editTitle.text().strip()
        notes = self.editNotes.toPlainText().strip()
        if group is None:
            self.data_source.register_group(EnsembleGroup(
                ensemble_uuids=self._ensemble_uuids, title=title, notes=notes,
                smiles=smiles, provenance=self._provenance,
            ))
        else:
            group.title, group.notes, group.smiles = title, notes, smiles
            self.data_source.notify_group_updated(group)
        super().accept()
