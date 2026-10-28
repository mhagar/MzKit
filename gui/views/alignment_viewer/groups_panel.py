"""
The Inspector's groups area: a combo of every `AnalyteGroup` the selected
analytes belong to, the chosen group's structure, and edit / delete / membership
buttons. Every edit goes through the registry (`notify_group_updated`).
"""
from typing import Iterable, Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtWidgets

from gui.views.alignment_viewer.group_dialog import AnalyteGroupDialog, show_smiles

if TYPE_CHECKING:
    from core.data_structs.analyte_group import AnalyteGroup
    from core.data_structs.uuid_types import AnalyteUUID
    from gui.views.alignment_viewer.data_source import AlignmentViewerDataSource
    from gui.widgets.alignment_plot.layout import HoverTarget


def selection_analyte_uuids(targets: Iterable['HoverTarget']) -> list['AnalyteUUID']:
    """AnalyteUUIDs behind plot targets (ensembles -> their analyte), deduped, in order."""
    found = {}
    for t in targets:
        if t.kind in ('ensemble', 'analyte') and t.analyte is not None:
            found.setdefault(t.analyte.uuid, None)
    return list(found)


def members_in_view(group: 'AnalyteGroup', in_view: set) -> str:
    """'n of m': members present in the shown alignment / all members."""
    n = sum(1 for u in group.analyte_uuids if u in in_view)
    return f"{n} of {len(group.analyte_uuids)}"


class GroupsPanel(QtCore.QObject):

    def __init__(
        self,
        data_source: 'AlignmentViewerDataSource',
        combo: QtWidgets.QComboBox,
        svg,
        btn_edit: QtWidgets.QToolButton,
        btn_delete: QtWidgets.QToolButton,
        btn_add: QtWidgets.QToolButton,
        btn_remove: QtWidgets.QToolButton,
    ):
        super().__init__()
        self.data_source = data_source
        self.combo, self.svg = combo, svg
        self.btn_edit, self.btn_delete = btn_edit, btn_delete
        self.btn_add, self.btn_remove = btn_add, btn_remove
        self._selection: list['AnalyteUUID'] = []
        self._in_view: set = set()

        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed,
        )
        combo.currentIndexChanged.connect(self._update_current)

        for button, text, tip, slot in (
            (btn_edit, "Edit", "Edit the group's title, notes and structure", self._edit),
            (btn_delete, "Del", "Delete the group", self._delete),
            (btn_add, "Add", "Add the selected analytes to the group", self._add),
            (btn_remove, "Remove",
             "Remove the selected analytes from the group "
             "(removing the last member deletes it)", self._remove),
        ):
            button.setText(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
        self.refresh(set(), [])

    def refresh(self, in_view: set, selection: list['AnalyteUUID']):
        """
        :param in_view: the shown alignment's AnalyteUUIDs.
        :param selection: the selected AnalyteUUIDs.
        """
        current = self.current_group()
        self._in_view, self._selection = in_view, selection
        groups = self.data_source.groups_for_analytes(selection) if selection else []
        with QtCore.QSignalBlocker(self.combo):
            self.combo.clear()
            for g in groups:
                self.combo.addItem(
                    f"{g.title or '(untitled)'} ({members_in_view(g, in_view)})", g,
                )
                self.combo.setItemData(
                    self.combo.count() - 1, g.notes, QtCore.Qt.ToolTipRole,
                )
            if groups:
                uuids = [g.uuid for g in groups]
                self.combo.setCurrentIndex(
                    uuids.index(current.uuid) if current and current.uuid in uuids else 0
                )
        self.combo.setEnabled(bool(groups))
        self._update_current()

    def current_group(self) -> Optional['AnalyteGroup']:
        return self.combo.currentData()

    def _update_current(self, *_):
        group = self.current_group()
        smiles = group.smiles if group else ""
        show_smiles(self.svg, smiles)
        self.svg.setVisible(bool(smiles.strip()))
        self.btn_edit.setEnabled(group is not None)
        self.btn_delete.setEnabled(group is not None)
        members = set(group.analyte_uuids) if group else set()
        self.btn_add.setEnabled(group is not None and not members.issuperset(self._selection))
        self.btn_remove.setEnabled(not members.isdisjoint(self._selection))

    # -- actions ------------------------------------------------------------

    def _edit(self, *_):
        group = self.current_group()
        if group is not None:
            AnalyteGroupDialog(
                self.data_source, [], group=group, parent=self.combo.window(),
            ).exec_()

    def _delete(self):
        group = self.current_group()
        if group is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self.combo.window(), "Delete group",
            f"Delete the group '{group.title or '(untitled)'}' "
            f"({len(group.analyte_uuids)} members)?",
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self.data_source.remove_group(group.uuid)

    def _add(self):
        group = self.current_group()
        if group is None:
            return
        group.analyte_uuids = list(dict.fromkeys(group.analyte_uuids + self._selection))
        self.data_source.notify_group_updated(group)

    def _remove(self):
        group = self.current_group()
        if group is None:
            return
        drop = set(self._selection)
        group.analyte_uuids = [u for u in group.analyte_uuids if u not in drop]
        if group.analyte_uuids:
            self.data_source.notify_group_updated(group)
        else:
            self.data_source.remove_group(group.uuid)
