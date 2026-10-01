"""
Ensemble peak-label decluttering in ChromPlotItem: overlapping labels
collapse into glyphs (tallest apex wins), hover / selection behaviour.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import gui.controllers.main_controller  # noqa: F401  (app import order)
    return app


def _peak(center: float, height: float) -> np.ndarray:
    rt = np.linspace(center - 8, center + 8, 41)
    arr = np.zeros(rt.size, dtype=[('rt', 'f8'), ('intsy', 'f8')])
    arr['rt'], arr['intsy'] = rt, height * np.exp(-0.5 * ((rt - center) / 2.5) ** 2)
    return arr


@pytest.fixture
def plot(qapp):
    from gui.widgets.ChromPlotWidget import ChromPlotWidget
    widget = ChromPlotWidget()
    widget.resize(1000, 400)
    widget.show()
    pi = widget.pi
    # Three crowded, similar-height peaks (labels must collide; labels only
    # collide with labels, so heights must be close) + one far away
    for uuid, center, height in [
        (1, 100.0, 5.0e6), (2, 103.0, 4.6e6), (3, 106.0, 4.8e6), (4, 500.0, 2e6),
    ]:
        pi.addPeak(_peak(center, height), uuid, color='y',
                   html_label=f"<b>C<sub>{uuid}0</sub>H<sub>8</sub>O</b>([M+H]+)<br>1.0δ")
    pi.autoScale()
    pi.vb.setYRange(0, 1e7, padding=0)   # headroom so labels fit
    qapp.processEvents()
    pi._layout_peak_labels()
    yield pi
    widget.close()


def _shown(pi):
    return {u for u, label in pi._peak_labels.items() if label.isVisible()}


def _glyphs(pi):
    return set(pi._label_glyphs.data['data'])


def test_crowded_labels_collapse_tallest_wins(plot):
    shown = _shown(plot)
    assert 1 in shown and 4 in shown          # tallest of the cluster + lone peak
    assert _glyphs(plot) == {2, 3}
    rects = [plot._peak_labels[u].sceneBoundingRect() for u in shown]
    assert not any(a.intersects(b) for i, a in enumerate(rects) for b in rects[i + 1:])


def test_zooming_in_expands_labels(plot):
    plot.vb.setXRange(99, 108, padding=0)
    plot._layout_peak_labels()
    assert len(_shown(plot)) > 1


def test_hover_expands_without_reflow_and_restores(plot):
    plot.set_peak_hover_state(2)
    assert 2 in _shown(plot) and 2 not in _glyphs(plot)
    assert 1 in _shown(plot)                  # not displaced
    assert plot._peak_labels[2].zValue() > plot._peak_labels[1].zValue()

    plot.set_peak_hover_state(None)
    assert 2 not in _shown(plot) and 2 in _glyphs(plot)


def test_selected_label_wins_collisions(plot):
    plot.set_peak_selected(3)
    assert 3 in _shown(plot)
    assert 1 not in _shown(plot)


def test_label_poking_out_of_view_collapses(plot):
    plot.vb.setYRange(0, 5.05e6, padding=0)   # no room above the tallest
    plot._layout_peak_labels()
    assert 1 not in _shown(plot) and 1 in _glyphs(plot)


def test_clear_peaks_clears_glyphs(plot):
    plot.clearPeaks()
    assert len(plot._label_glyphs.data) == 0
    assert not plot._peak_labels
