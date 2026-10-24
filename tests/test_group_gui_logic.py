"""
Analyte-group GUI logic: the dialog's exact / overlapping / new decision,
plot selection -> AnalyteUUIDs, "n of m" counting, provenance dicts, and an
offscreen smoke test of the AlignmentViewer's group track + panel.
"""
import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets  # noqa: E402

# The app must exist before the GUI modules load (and in that import order)
_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
import gui.controllers.main_controller  # noqa: E402,F401

from core.cli.cluster_analytes import ClusterKey  # noqa: E402
from core.data_structs.analyte_group import AnalyteGroup  # noqa: E402
from core.data_structs.similarity_matrix import ImportParams  # noqa: E402
from gui.views.alignment_viewer.group_dialog import jsonable, match_groups  # noqa: E402
from gui.views.alignment_viewer.groups_panel import (  # noqa: E402
    members_in_view, selection_analyte_uuids,
)
from gui.widgets.alignment_plot.layout import HoverTarget  # noqa: E402


def test_exact_match_wins_regardless_of_order():
    a, b = AnalyteGroup([1, 2, 3]), AnalyteGroup([3, 2, 1])
    other = AnalyteGroup([3, 4])
    exact, overlapping = match_groups([2, 3, 1, 1], [other, b, a])
    assert exact is b and overlapping == []


def test_partial_overlaps_listed_disjoint_excluded():
    a, b, c = AnalyteGroup([1, 2, 3]), AnalyteGroup([3, 9]), AnalyteGroup([7])
    exact, overlapping = match_groups([1, 2], [a, b, c])
    assert exact is None and overlapping == [a]
    assert match_groups([5], [a, b, c]) == (None, [])


def test_selection_to_analyte_uuids():
    an1, an2 = SimpleNamespace(uuid=11), SimpleNamespace(uuid=22)
    targets = [
        HoverTarget('ensemble', 0, sample_uuid=1, analyte=an2),
        HoverTarget('sample', sample_uuid=1),
        HoverTarget('analyte', 1, analyte=an1),
        HoverTarget('ensemble', 1, sample_uuid=2, analyte=an1),
        HoverTarget('ensemble', 2, sample_uuid=2, analyte=None),
    ]
    assert selection_analyte_uuids(targets) == [22, 11]


def test_members_in_view():
    assert members_in_view(AnalyteGroup([1, 2, 3]), {1, 3, 8}) == "2 of 3"


def test_clustering_provenance_is_json_safe():
    key = ClusterKey(12345, ImportParams(), 'average')
    d = jsonable(key)
    assert d['linkage_method'] == 'average'
    assert d['transform'] == ImportParams()._asdict()
    json.dumps(d)


# -- offscreen smoke test ------------------------------------------------------

def test_viewer_shows_group_track_and_panel(data_registry, sample, ensemble, monkeypatch):
    from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment
    from gui.views.alignment_viewer import AlignmentViewer

    sample.injection.ensembles[ensemble.uuid] = ensemble
    analyte = AlignedAnalyte(
        ensemble_map={sample.uuid: ensemble.uuid},
        consensus_rt=ensemble.peak_rt, consensus_mz=ensemble.base_mz,
    )
    alignment = EnsembleAlignment(sample_uuids=(sample.uuid,), analytes=[analyte])
    data_registry.register_alignment(alignment)

    viewer = AlignmentViewer(data_registry)
    viewer.show()
    viewer.set_alignment(alignment)
    assert viewer.plotGroupTrack.isHidden()

    group = AnalyteGroup([analyte.uuid, 999], title="Psammaplins", notes="a\nb")
    data_registry.register_group(group)
    assert not viewer.plotGroupTrack.isHidden()

    viewer.selection_model.set_selection(
        viewer.plot_item.analyte_targets([analyte.uuid])
    )
    combo = viewer.comboGroups
    assert combo.count() == 1
    assert combo.itemText(0) == "Psammaplins (1 of 2)"
    assert viewer.btnGroupSelected.isEnabled()
    # A lone analyte: the tree shows it, its member (selected with it) collapsed under it
    tree = viewer.inspector.compare_model
    assert tree.rowCount() == 1 and tree.item(0).isEnabled()
    assert tree.item(0).rowCount() == 1 and tree.item(0).child(0).isEnabled()
    assert not viewer.treeCompare.isExpanded(tree.item(0).index())

    # Removing the last in-view member via the panel keeps the group (999 remains)
    viewer.groups_panel._remove()
    assert group.analyte_uuids == [999]
    assert viewer.plotGroupTrack.isHidden()
    assert combo.count() == 0

    # An ensemble: under its (disabled: not selected) analyte, expanded
    target = viewer.plot_item.ensemble_target(0, sample.uuid)
    viewer.selection_model.set_selection([target])
    assert not tree.item(0).isEnabled()
    assert tree.item(0).child(0).text() == sample.name
    assert tree.item(0).child(0).isEnabled()
    assert viewer.treeCompare.isExpanded(tree.item(0).index())

    # Inline Name / SMILES edits land in the registry and the panel
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[1]),
    )

    def edit(line_edit, text):
        line_edit.setText(text)
        line_edit.editingFinished.emit()
        _app.processEvents()

    edit(viewer.editAnalyteName, " Psammaplin A ")
    edit(viewer.editAnalyteSmiles, "not a smiles")
    assert data_registry.get_annotation(analyte.uuid).name == "Psammaplin A"
    assert data_registry.get_annotation(analyte.uuid).smiles == ""
    assert warnings == ["Invalid SMILES"]
    assert viewer.editAnalyteName.text() == "Psammaplin A"
    assert viewer.editAnalyteSmiles.text() == ""          # reverted

    edit(viewer.editAnalyteSmiles, "CCO")
    assert data_registry.get_annotation(analyte.uuid).smiles == "CCO"
    assert not viewer.svgAnalyteStructure.isHidden()

    # Clearing both fields drops the annotation
    edit(viewer.editAnalyteName, "")
    edit(viewer.editAnalyteSmiles, "")
    assert data_registry.get_annotation(analyte.uuid) is None
    assert viewer.svgAnalyteStructure.isHidden()

    viewer.selection_model.set_selection([])
    assert not viewer.editAnalyteName.isEnabled()
    assert viewer.stackedWidget.currentWidget() is viewer.labelAlignmentSummary.parentWidget()
    viewer.close()


def test_viewer_selection_focus_and_anchor(data_registry, sample, ensemble):
    """Plot clicks, arrow keys, tree rows and the anchor all go through the SelectionModel."""
    from PyQt5.QtCore import Qt
    from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment
    from gui.views.alignment_viewer import AlignmentViewer
    from gui.widgets.alignment_plot.selection import target_key

    sample.injection.ensembles[ensemble.uuid] = ensemble
    analyte = AlignedAnalyte(
        ensemble_map={sample.uuid: ensemble.uuid},
        consensus_rt=ensemble.peak_rt, consensus_mz=ensemble.base_mz,
    )
    alignment = EnsembleAlignment(sample_uuids=(sample.uuid,), analytes=[analyte])
    data_registry.register_alignment(alignment)
    viewer = AlignmentViewer(data_registry)
    viewer.show()
    viewer.set_alignment(alignment)

    plot, model = viewer.plot_item, viewer.selection_model
    (ta,) = plot.analyte_targets([analyte.uuid])
    te = plot.ensemble_target(0, sample.uuid)

    def keys(targets):
        return [target_key(t) for t in targets]

    # Ctrl+clicking a member of a selected analyte only focuses it
    plot.select(ta)
    plot.select(te, additive=True)
    assert keys(model.selection) == keys([ta]) and target_key(model.focus) == target_key(te)

    plot.select(None)
    plot.select(te)
    plot.select(ta, additive=True)
    assert keys(model.selection) == keys([te, ta])
    assert target_key(model.focus) == target_key(ta)

    # A plain click on a selected item only focuses it
    plot.select(te)
    assert len(model.selection) == 2 and target_key(model.focus) == target_key(te)

    # Several selected: arrows move the focus (tree order), clamped at the ends
    plot.handle_key(Qt.Key_Left)
    assert target_key(model.focus) == target_key(ta)
    plot.handle_key(Qt.Key_Left)
    assert target_key(model.focus) == target_key(ta)
    plot.handle_key(Qt.Key_Right)
    assert target_key(model.focus) == target_key(te)
    assert len(model.selection) == 2

    # The tree: the analyte's row with its ensemble under it; picking a row focuses
    tree, inspector = viewer.treeCompare, viewer.inspector
    analyte_row = inspector.compare_model.item(0)
    assert analyte_row.child(0).text() == f"● {sample.name}"   # focus, marked
    tree.setCurrentIndex(analyte_row.index())
    assert target_key(model.focus) == target_key(ta)

    # A lone analyte's member is selected with it: the tree row focuses it
    model.set_selection([ta])
    analyte_row = inspector.compare_model.item(0)
    assert analyte_row.child(0).isEnabled()
    tree.setCurrentIndex(analyte_row.child(0).index())
    assert target_key(model.focus) == target_key(te) and len(model.selection) == 1
    assert analyte_row.child(0).text() == f"● {sample.name}"   # focus, marked

    # Up/Down step the focus through a selected analyte's ensembles
    model.set_selection([ta])
    columns = []
    plot.sigColumnsChanged.connect(columns.append)
    plot.handle_key(Qt.Key_Down)
    assert target_key(model.focus) == target_key(te) and len(model.selection) == 1
    plot.handle_key(Qt.Key_Down)      # the last one: stays
    assert target_key(model.focus) == target_key(te)
    # Its column is banded in the focus colour (lanes: grey), also on the dendrogram / track
    from gui.widgets.alignment_plot.plot_item import FOCUS_BAND, SELECTION_BAND
    assert [c for *_, c in plot._lane_bands._bands] == [SELECTION_BAND]   # never focus
    assert [c for *_, c in plot._column_bands._bands] == [SELECTION_BAND, FOCUS_BAND]
    assert columns[-1][-1][2] == FOCUS_BAND
    assert viewer.plotGroupTrack._columns._bands
    model.set_selection([te, ta], focus=ta)
    assert inspector.compare_model.columnCount() == 4

    # The anchor leaves the selection alone and adds the comparison columns
    model.set_anchor(te)
    assert len(model.selection) == 2
    assert inspector.compare_model.columnCount() == 7

    # Deselecting the focus moves it; the anchor stays, marked in its member row
    plot.select(ta, additive=True)
    assert keys(model.selection) == keys([te]) and target_key(model.focus) == target_key(te)
    model.set_selection([ta])
    member = inspector.compare_model.item(0).child(0)
    assert target_key(model.anchor) == target_key(te)
    assert member.isEnabled() and member.text() == f"● {sample.name}"

    # Esc clears the selection, not the anchor; a new alignment clears both
    plot.handle_key(Qt.Key_Escape)
    assert model.selection == [] and model.focus is None
    assert model.anchor is not None
    viewer.set_alignment(alignment)
    assert model.anchor is None
    viewer.close()


def test_dialog_overlap_choice_and_create():
    from core.data_structs.data_registry import DataRegistry
    from gui.views.alignment_viewer.group_dialog import AnalyteGroupDialog

    reg = DataRegistry()
    existing = AnalyteGroup([1, 2, 3], title="Old")
    reg.register_group(existing)

    dialog = AnalyteGroupDialog(reg, [2, 4, 4], {'alignment_uuid': 7, 'clustering': None})
    assert not dialog.listOverlapping.isHidden()
    dialog.listOverlapping.setCurrentRow(0)          # picks "Edit existing"
    assert dialog.radioEditExisting.isChecked()
    assert dialog.editTitle.text() == "Old"
    dialog.radioCreateNew.setChecked(True)
    dialog.editTitle.setText("New")
    dialog.editSmiles.setText("not a smiles")
    assert not dialog.labelSmilesWarning.isHidden()
    dialog.editSmiles.setText("CCO")
    assert dialog.labelSmilesWarning.isHidden()
    dialog.accept()

    new = [g for g in reg.get_all_groups() if g is not existing]
    assert len(new) == 1 and new[0].analyte_uuids == [2, 4]
    assert new[0].title == "New" and new[0].smiles == "CCO"
    assert new[0].provenance == {'alignment_uuid': 7, 'clustering': None}
    assert existing.title == "Old"

    # Exact match opens the existing group, overlap widgets hidden
    dialog = AnalyteGroupDialog(reg, [3, 2, 1])
    assert dialog.listOverlapping.isHidden() and dialog.editTitle.text() == "Old"
    dialog.editTitle.setText("Renamed")
    dialog.accept()
    assert existing.title == "Renamed" and len(reg.get_all_groups()) == 2


def test_selection_tree_groups_items_under_analytes():
    """One row per analyte the selection touches, all its members under it:
    unselected rows disabled; analytes with a selected ensemble expanded.
    An anchor not in the tree is pinned on top."""
    import numpy as np
    from PyQt5 import QtSvg
    from core.utils.array_types import to_spec_arr
    from gui.views.alignment_viewer.context import ItemSpectra
    from gui.views.alignment_viewer.inspector import Inspector

    a, b, c, d = (SimpleNamespace(uuid=u) for u in (1 << 100, 2, 3, 4))   # big ints, like real UUIDs
    members = {
        id(a): {'s2': SimpleNamespace(base_intsy=2e5), 's1': SimpleNamespace(base_intsy=1e5),
                's3': SimpleNamespace(base_intsy=3e5)},
        id(b): {'s1': SimpleNamespace(base_intsy=1.0)},
        id(c): {'s1': SimpleNamespace(base_intsy=1.0)},
        id(d): {'s9': SimpleNamespace(base_intsy=1.0)},
    }

    def ens(aid, analyte, sample_uuid):
        return HoverTarget('ensemble', aid, sample_uuid, members[id(analyte)][sample_uuid], analyte)

    e0, e1 = ens(0, a, 's1'), ens(0, a, 's2')
    tb = HoverTarget('analyte', 1, analyte=b)
    e2 = ens(2, c, 's1')
    spec = to_spec_arr(np.array([100.0]), np.array([1.0]))
    ctx = SimpleNamespace(
        item_spectra=lambda t: ItemSpectra(
            label=f"{t.analyte_id}/{t.sample_uuid}", ms1=spec, ms2=None,
            rt=0.0, base_mz=100.0,
        ),
        members=lambda analyte: members[id(analyte)],
        sample_name=str,
        formula_agreement=lambda analyte: None,
        analyte_label=lambda analyte: f"analyte {analyte.uuid}",
    )

    pane = QtWidgets.QWidget()   # the stack's pane, as in the .ui
    stack = QtWidgets.QStackedWidget(pane)
    summary_page, compare_page = QtWidgets.QWidget(), QtWidgets.QWidget()
    stack.addWidget(summary_page)
    stack.addWidget(compare_page)
    summary = QtWidgets.QLabel(summary_page)
    tree = QtWidgets.QTreeView(compare_page)
    inspector = Inspector(
        QtWidgets.QLabel(), stack, summary, QtWidgets.QLineEdit(), QtWidgets.QLineEdit(), tree,
        QtSvg.QSvgWidget(),
    )
    inspector.show_selection(ctx, [e0, tb, e1, e2], focus=e1, anchor=e0, similarities={})

    model = inspector.compare_model
    top = [model.item(r) for r in range(model.rowCount())]
    assert [t.text() for t in top] == [f"analyte {1 << 100}", "1/None", "analyte 3"]
    children = [top[0].child(r) for r in range(top[0].rowCount())]
    assert [ch.text() for ch in children] == [
        "● 0/s1", "● 0/s2", "0/s3",   # anchor + focus are marked; by sample name
    ]
    assert [ch.isEnabled() for ch in children] == [True, True, False]
    assert model.item(0).child(2, 1).text() == "3.00e+05"   # intensity
    assert not top[0].isEnabled() and top[1].isEnabled()
    assert [tree.isExpanded(t.index()) for t in top] == [True, False, True]
    assert top[1].rowCount() == 1 and top[1].child(0).isEnabled()  # selected with tb
    assert tree.currentIndex() == children[1].index()
    assert model.columnCount() == 7

    focused = []
    inspector.sigFocusRequested.connect(focused.append)
    tree.setCurrentIndex(top[2].child(0).index())
    assert focused == [e2]

    # No anchor: no comparison columns; one outside the tree is pinned
    inspector.show_selection(ctx, [e0, e1], focus=e0, anchor=None, similarities={})
    assert model.columnCount() == 4
    anchor = ens(3, d, 's9')
    inspector.show_selection(ctx, [e0], focus=e0, anchor=anchor, similarities={})
    assert model.item(0).text() == "● 3/s9" and not model.item(0).isEnabled()
    assert model.item(1).child(0).text() == "0/s1"   # a lone item: no focus mark
