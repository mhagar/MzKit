"""
Dialog creating or editing an `AnalyteGroup` (title, notes, SMILES with a
live structure preview) for the Alignment Viewer.

Exact-match rule (`match_groups`): grouping a selection whose membership
equals an existing group's edits that group instead; if groups merely
overlap the selection, they're listed so the user can pick "Edit existing"
(one of them; its membership is left as is) or "Create new".
"""
from typing import Iterable, Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtWidgets

from core.data_structs.analyte_group import AnalyteGroup
from core.utils.render_molecule import smiles_to_svg
from gui.resources.AnalyteGroupDialog import Ui_Dialog

if TYPE_CHECKING:
    from core.data_structs.uuid_types import AnalyteUUID
    from gui.views.alignment_viewer.data_source import AlignmentViewerDataSource


def match_groups(
    analyte_uuids: Iterable['AnalyteUUID'],
    groups: Iterable[AnalyteGroup],
) -> tuple[Optional[AnalyteGroup], list[AnalyteGroup]]:
    """
    (the group whose membership equals the selection, or None;
    the groups sharing members with it - empty when there is an exact match).
    """
    selection = set(analyte_uuids)
    groups = list(groups)
    for group in groups:
        if set(group.analyte_uuids) == selection:
            return group, []
    return None, [g for g in groups if not selection.isdisjoint(g.analyte_uuids)]


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


class AnalyteGroupDialog(QtWidgets.QDialog, Ui_Dialog):
    """
    :param analyte_uuids: the selection to group (ignored when `group` is given).
    :param provenance: stored on a newly created group.
    :param group: edit this group directly (no matching).
    """

    def __init__(
        self,
        data_source: 'AlignmentViewerDataSource',
        analyte_uuids: list['AnalyteUUID'],
        provenance: Optional[dict] = None,
        group: Optional[AnalyteGroup] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.data_source = data_source
        self._analyte_uuids = list(dict.fromkeys(analyte_uuids))
        self._provenance = provenance

        overlapping: list[AnalyteGroup] = []
        if group is None:
            group, overlapping = match_groups(
                self._analyte_uuids,
                data_source.groups_for_analytes(self._analyte_uuids),
            )
        self._group = group     # edited in place on accept, else None
        self._overlapping = overlapping

        n = len(self._analyte_uuids) if group is None else len(group.analyte_uuids)
        self.setWindowTitle(
            f"{'Edit' if group is not None else 'New'} Analyte Group "
            f"({n} analyte{'' if n == 1 else 's'})"
        )
        self.labelSmilesWarning.setWordWrap(True)
        self.labelSmilesWarning.setStyleSheet("color: #c33")

        for widget in (self.label_2, self.listOverlapping,
                       self.radioEditExisting, self.radioCreateNew):
            widget.setVisible(bool(overlapping))
        for g in overlapping:
            self.listOverlapping.addItem(
                f"{g.title or '(untitled)'} - {len(g.analyte_uuids)} members"
            )
        self.radioCreateNew.setChecked(True)

        self.editSmiles.textChanged.connect(self._update_preview)
        self.listOverlapping.currentRowChanged.connect(self._on_overlap_picked)
        self.radioEditExisting.toggled.connect(self._on_mode_toggled)

        self._load(group)

    def _load(self, group: Optional[AnalyteGroup]):
        self.editTitle.setText(group.title if group else "")
        self.editNotes.setPlainText(group.notes if group else "")
        self.editSmiles.setText(group.smiles if group else "")
        self._update_preview()

    def _picked(self) -> Optional[AnalyteGroup]:
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
            self.data_source.register_group(AnalyteGroup(
                analyte_uuids=self._analyte_uuids, title=title, notes=notes,
                smiles=smiles, provenance=self._provenance,
            ))
        else:
            group.title, group.notes, group.smiles = title, notes, smiles
            self.data_source.notify_group_updated(group)
        super().accept()
