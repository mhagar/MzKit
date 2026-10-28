"""The alignment map's SelectionModel: selection, focus and anchor rules."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from gui.widgets.alignment_plot.layout import HoverTarget  # noqa: E402
from gui.widgets.alignment_plot.selection import (  # noqa: E402
    SelectionModel, display_order, target_key,
)

A = A0 = HoverTarget('analyte', 0)
A1 = HoverTarget('analyte', 1)
A2 = HoverTarget('analyte', 2)
E0 = HoverTarget('ensemble', 0, 's1')
E1 = HoverTarget('ensemble', 1, 's1')
E2 = HoverTarget('ensemble', 2, 's1')


def keys(targets):
    return [target_key(t) for t in targets]


def recording_model():
    model, events = SelectionModel(), []
    model.sigSelectionChanged.connect(lambda t: events.append(('selection', keys(t))))
    model.sigFocusChanged.connect(lambda t: events.append(('focus', target_key(t))))
    model.sigAnchorChanged.connect(lambda t: events.append(('anchor', target_key(t))))
    return model, events


def test_focus_follows_selection():
    model, events = recording_model()
    model.set_selection([E0, E1, E0])           # analytes listed, last focused
    assert keys(model.selection) == keys([A0, A1])
    assert model.focus == E0
    assert events == [('selection', keys([A0, A1])), ('focus', target_key(E0))]

    model.set_focus(E2)                          # not selected: ignored
    assert model.focus == E0

    model.toggle(E2)                             # its analyte added, E2 focused
    assert keys(model.selection) == keys([A0, A1, A2]) and model.focus == E2
    model.toggle(E0)                             # its analyte removed; focus unaffected
    assert keys(model.selection) == keys([A1, A2]) and model.focus == E2
    model.toggle(E2)                             # focus removed: falls to the last
    assert keys(model.selection) == keys([A1]) and model.focus == A1

    model.clear()
    assert model.selection == [] and model.focus is None


def test_signals_only_on_change():
    model, events = recording_model()
    model.set_selection([E0, E1])
    events.clear()
    model.set_selection([E0, E1], focus=E1)
    model.set_focus(E1)
    model.set_anchor(None)
    assert events == []
    model.set_focus(E0)
    assert events == [('focus', target_key(E0))]


def test_anchor_is_independent_of_the_selection():
    model, _ = recording_model()
    model.set_selection([E0])
    model.set_anchor(E1)                         # needn't be selected
    model.set_selection([E2])
    model.clear(anchor=False)
    assert model.anchor == E1
    model.clear()
    assert model.anchor is None


def test_remap_swaps_targets_and_drops_missing():
    model, events = recording_model()
    model.set_selection([E0, E1], focus=E1)
    model.set_anchor(E2)
    events.clear()

    fresh = {target_key(E0): E0._replace(ensemble='new')}
    fresh[target_key(A0)] = A0._replace(analyte='new')
    model.remap(fresh.get)                       # A1, E1 (focus) and E2 (anchor) gone
    assert model.selection == [fresh[target_key(A0)]]
    assert model.focus == fresh[target_key(A0)] and model.anchor is None
    assert [e[0] for e in events] == ['selection', 'focus', 'anchor']


def test_display_order_groups_by_analyte_analyte_first():
    e0b = HoverTarget('ensemble', 0, 's2')
    assert keys(display_order([E0, E1, A, e0b])) == keys([A, E0, e0b, E1])


def test_selected_analyte_covers_its_ensembles():
    model, _ = recording_model()
    model.set_selection([A0, E1])
    assert model.contains(E0) and not model.lists(E0)
    model.set_focus(E0)                          # a member: focusable
    assert model.focus == E0
    model.toggle(E0)                             # a member toggles its analyte out
    assert keys(model.selection) == keys([A1]) and model.focus == A1
    model.set_selection([A0], focus=E0)
    assert model.focus == E0


def test_merge_intervals():
    from gui.widgets.alignment_plot.indicators import merge_intervals
    assert merge_intervals([(3, 4), (0, 1), (0.5, 2), (2, 2.5)]) == [(0, 2.5), (3, 4)]
