"""
Group track: greedy lane packing, and a smoke test of the widget.
"""
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets  # noqa: E402

from gui.widgets.AlignmentGroupTrackWidget import pack_lanes  # noqa: E402


def test_disjoint_spans_share_one_lane():
    assert pack_lanes([(0, 1), (2, 3), (4, 5)]) == [0, 0, 0]


def test_overlapping_spans_get_separate_lanes():
    assert pack_lanes([(0, 5), (3, 8)]) == [0, 1]


def test_order_is_by_left_edge():
    assert pack_lanes([(4, 5), (0, 10), (2, 3)]) == [1, 0, 1]


def test_nested_span_and_reuse_of_upper_lane():
    # (2, 3) nests inside (0, 10); (5, 6) fits after it in lane 1
    assert pack_lanes([(0, 10), (2, 3), (5, 6)]) == [0, 1, 1]


def test_touching_and_gap():
    assert pack_lanes([(0, 1), (1, 2)]) == [0, 1]           # touching isn't disjoint
    assert pack_lanes([(0, 1), (1.5, 2)]) == [0, 0]
    assert pack_lanes([(0, 1), (1.5, 2)], gap=1.0) == [0, 1]


def test_empty():
    assert pack_lanes([]) == []


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import gui.controllers.main_controller  # noqa: F401  (app import order)
    return app


def test_widget_packs_overlapping_groups(qapp):
    from gui.widgets.AlignmentGroupTrackWidget import (
        AlignmentGroupTrackWidget, LANE_PX,
    )
    track = AlignmentGroupTrackWidget()
    x_of = {1: 0.0, 2: 5.0, 3: 3.0, 4: 8.0}
    groups = [
        SimpleNamespace(uuid=10, title="A", analyte_uuids=[1, 2]),
        SimpleNamespace(uuid=11, title="B", analyte_uuids=[3, 4, 99]),
        SimpleNamespace(uuid=12, title="hidden", analyte_uuids=[98]),
    ]
    track.set_groups(groups, x_of)
    assert track.has_visible_groups()
    assert [r.lane for r in track._rows] == [0, 1]
    assert track._rows[1].n_hidden == 1
    assert track.height() == 2 * LANE_PX

    hovered = []
    track.sigGroupHovered.connect(hovered.append)
    track._on_row_hovered(1)
    assert hovered == [[3, 4]]
    assert "1 members not shown" in track.toolTip()

    track.clear_groups()
    assert not track.has_visible_groups()
    track.set_groups(groups[2:], x_of)
    assert not track.has_visible_groups()
