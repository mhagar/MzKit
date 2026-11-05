"""
The Inspector's groups area: a combo of every `EnsembleGroup` the selection
belongs to, and edit / delete / membership buttons. A selected analyte stands
for all of its member ensembles. Every edit goes through the registry
(`notify_group_updated`).
"""
from typing import Iterable, Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtWidgets

from gui.views.alignment_viewer.group_dialog import EnsembleGroupDialog

if TYPE_CHECKING:
    from core.data_structs.ensemble_group import EnsembleGroup
    from core.data_structs.uuid_types import EnsembleUUID
    from gui.views.alignment_viewer.data_source import AlignmentViewerDataSource
    from gui.widgets.alignment_plot.layout import HoverTarget


def selection_ensemble_uuids(targets: Iterable['HoverTarget']) -> list['EnsembleUUID']:
    """
    EnsembleUUIDs behind plot targets (an analyte -> all its members, an
    ensemble -> itself), deduped, in order.
    """
    found = {}
    for t in targets:
        if t.analyte is None:
            continue
        if t.kind == 'analyte':
            uuids = t.analyte.ensemble_map.values()
        elif t.kind == 'ensemble':
            uuids = [t.analyte.ensemble_map.get(t.sample_uuid)]
        else:
            continue
        for u in uuids:
            if u is not None:
                found.setdefault(u, None)
    return list(found)


def members_in_view(group: 'EnsembleGroup', in_view: set) -> str:
    """'n of m': member ensembles in the shown alignment / all members."""
    n = sum(1 for u in group.ensemble_uuids if u in in_view)
    return f"{n} of {len(group.ensemble_uuids)}"


class GroupsPanel(QtCore.QObject):

    def __init__(
        self,
        data_source: 'AlignmentViewerDataSource',
        combo: QtWidgets.QComboBox,
        btn_edit: QtWidgets.QToolButton,
        btn_delete: QtWidgets.QToolButton,
        btn_add: QtWidgets.QToolButton,
        btn_remove: QtWidgets.QToolButton,
    ):
        super().__init__()
        self.data_source = data_source
        self.combo = combo
        self.btn_edit, self.btn_delete = btn_edit, btn_delete
        self.btn_add, self.btn_remove = btn_add, btn_remove
        self._selection: list['EnsembleUUID'] = []
        self._in_view: set = set()

        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed,
        )
        combo.currentIndexChanged.connect(self._update_current)

        for button, text, tip, slot in (
            (btn_edit, "Edit", "Edit the group's title, notes and structure", self._edit),
            (btn_delete, "Del", "Delete the group", self._delete),
            (btn_add, "Add", "Add the selection (an analyte: all its members) to the group", self._add),
            (btn_remove, "Remove",
             "Remove the selection (an analyte: all its members) from the group "
             "(removing the last member deletes it)", self._remove),
        ):
            button.setText(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
        self.refresh(set(), [])

    def refresh(self, in_view: set, selection: list['EnsembleUUID']):
        """
        :param in_view: the shown alignment's member EnsembleUUIDs.
        :param selection: the selected EnsembleUUIDs (see selection_ensemble_uuids).
        """
        current = self.current_group()
        self._in_view, self._selection = in_view, selection
        groups = self.data_source.groups_for_ensembles(selection) if selection else []
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

    def current_group(self) -> Optional['EnsembleGroup']:
        return self.combo.currentData()

    def _update_current(self, *_):
        group = self.current_group()
        self.btn_edit.setEnabled(group is not None)
        self.btn_delete.setEnabled(group is not None)
        members = set(group.ensemble_uuids) if group else set()
        self.btn_add.setEnabled(group is not None and not members.issuperset(self._selection))
        self.btn_remove.setEnabled(not members.isdisjoint(self._selection))

    # -- actions ------------------------------------------------------------

    def _edit(self, *_):
        group = self.current_group()
        if group is not None:
            EnsembleGroupDialog(
                self.data_source, [], group=group, parent=self.combo.window(),
            ).exec_()

    def _delete(self):
        group = self.current_group()
        if group is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self.combo.window(), "Delete group",
            f"Delete the group '{group.title or '(untitled)'}' "
            f"({len(group.ensemble_uuids)} members)?",
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self.data_source.remove_group(group.uuid)

    def _add(self):
        group = self.current_group()
        if group is None:
            return
        group.ensemble_uuids = list(dict.fromkeys(group.ensemble_uuids + self._selection))
        self.data_source.notify_group_updated(group)

    def _remove(self):
        group = self.current_group()
        if group is None:
            return
        drop = set(self._selection)
        group.ensemble_uuids = [u for u in group.ensemble_uuids if u not in drop]
        if group.ensemble_uuids:
            self.data_source.notify_group_updated(group)
        else:
            self.data_source.remove_group(group.uuid)
