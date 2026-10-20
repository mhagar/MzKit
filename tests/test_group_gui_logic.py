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

def test_viewer_shows_group_track_and_panel(data_registry, sample, ensemble):
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

    viewer.plot_item.set_selection(viewer.plot_item.analyte_targets([analyte.uuid]))
    model = viewer.groups_panel.model
    assert model.rowCount() == 1
    assert [model.index(0, c).data() for c in range(3)] == ["Psammaplins", "1 of 2", "a"]
    assert viewer.btnGroupSelected.isEnabled()

    # Removing the last in-view member via the panel keeps the group (999 remains)
    viewer.groups_panel._remove()
    assert group.analyte_uuids == [999]
    assert viewer.plotGroupTrack.isHidden()
    assert model.rowCount() == 0
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
