"""Tiny ProcessRunner-compatible tasks for tests/test_process_controller.py."""
import time


def sleep_then_return(delay, progress_callback=None, cancel_event=None):
    time.sleep(delay)
    return delay


def fail(progress_callback=None, cancel_event=None):
    raise RuntimeError("boom")
