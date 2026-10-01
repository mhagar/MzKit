"""Tiny ProcessRunner-compatible tasks for tests/test_process_controller.py."""
import time


def sleep_then_return(delay, progress_callback=None, cancel_event=None):
    time.sleep(delay)
    return delay


def fail(progress_callback=None, cancel_event=None):
    raise RuntimeError("boom")


def report_progress(progress_callback=None, cancel_event=None):
    for pct in (25.0, 75.0):
        if progress_callback is not None:
            progress_callback(pct, f"step {pct:.0f}")
        time.sleep(0.15)
    return "done"
