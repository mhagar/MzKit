"""
ProcessController completion delivery.

Regression: processes finishing close together lost their completion calls
(the first one's cleanup removed the others before the poll loop reported
them), so e.g. find-mfs chained after a multi-sample extraction never ran.
"""
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets  # noqa: E402

from core.controllers.ProcessController import ProcessController  # noqa: E402

TASKS = str(Path(__file__).with_name("_process_tasks.py"))


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _pump(qapp, until, timeout=5.0):
    t0 = time.time()
    while not until() and time.time() - t0 < timeout:
        qapp.processEvents()
        time.sleep(0.01)


def test_simultaneous_completions_are_all_delivered(qapp):
    pc = ProcessController(main_controller=None)
    done = []
    for _ in range(4):
        pc.start_process(TASKS, "sleep_then_return", {"delay": 0.2},
                         on_completion_func=done.append)

    _pump(qapp, lambda: len(done) == 4)

    assert done == [0.2] * 4
    assert pc.last_known_status == {} and pc.running_processes == {}


def test_process_started_from_a_completion_runs(qapp):
    """The extraction-batch pattern: the last completion chains a new process."""
    pc = ProcessController(main_controller=None)
    state = {"remaining": 3}
    chained = []

    def on_one(_result):
        state["remaining"] -= 1
        if state["remaining"] == 0:
            pc.start_process(TASKS, "sleep_then_return", {"delay": 0.05},
                             on_completion_func=chained.append)

    for _ in range(3):
        pc.start_process(TASKS, "sleep_then_return", {"delay": 0.2},
                         on_completion_func=on_one)

    _pump(qapp, lambda: chained)
    assert chained == [0.05]


def test_failed_process_completes_with_none(qapp):
    pc = ProcessController(main_controller=None)
    done = []
    pc.start_process(TASKS, "fail", {}, on_completion_func=done.append)
    _pump(qapp, lambda: done)
    assert done == [None]
