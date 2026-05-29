"""Tests for viz_authoring.load_aware — the system-load-aware pacing
helpers used by background workers."""
from __future__ import annotations

import multiprocessing
import threading
import time
from unittest.mock import patch

import pytest

from viz_authoring.load_aware import (
    LOAD_THRESHOLD,
    LOAD_CHECK_INTERVAL,
    is_loaded,
    sleep_if_loaded,
)


class TestDefaults:
    """The defaults exist; documented values."""

    def test_load_threshold_default(self):
        # 6.0 is the documented extraction value (~1.5x typical core
        # count on a desktop). If this changes, the per-worker
        # IDLE_CHECK_INTERVAL choices should be re-audited too.
        assert LOAD_THRESHOLD == 6.0

    def test_load_check_interval_default(self):
        assert LOAD_CHECK_INTERVAL == 10.0


class TestIsLoaded:
    """is_loaded reads os.getloadavg()[0] and compares to threshold."""

    def test_below_threshold(self):
        with patch('viz_authoring.load_aware.os.getloadavg',
                   return_value=(0.5, 0.5, 0.5)):
            assert is_loaded() is False

    def test_above_threshold(self):
        with patch('viz_authoring.load_aware.os.getloadavg',
                   return_value=(10.0, 8.0, 6.0)):
            assert is_loaded() is True

    def test_at_threshold(self):
        # > not >= — exactly threshold is not loaded
        with patch('viz_authoring.load_aware.os.getloadavg',
                   return_value=(LOAD_THRESHOLD, 1.0, 1.0)):
            assert is_loaded() is False

    def test_uses_one_minute_average(self):
        # First element only — 5min and 15min should not affect result
        with patch('viz_authoring.load_aware.os.getloadavg',
                   return_value=(0.5, 100.0, 100.0)):
            assert is_loaded() is False

    def test_custom_threshold(self):
        with patch('viz_authoring.load_aware.os.getloadavg',
                   return_value=(3.0, 3.0, 3.0)):
            assert is_loaded(threshold=2.0) is True
            assert is_loaded(threshold=4.0) is False


class TestSleepIfLoaded:
    """sleep_if_loaded combines is_loaded with stop_event.wait."""

    def test_not_loaded_returns_false_immediately(self):
        stop = multiprocessing.Event()
        with patch('viz_authoring.load_aware.os.getloadavg',
                   return_value=(0.5, 0.5, 0.5)):
            t0 = time.perf_counter()
            result = sleep_if_loaded(stop)
            elapsed = time.perf_counter() - t0
        assert result is False
        assert elapsed < 0.01  # no wait happened

    def test_loaded_returns_true_after_wait(self):
        # Use a tiny check_interval so the test runs fast
        stop = multiprocessing.Event()
        with patch('viz_authoring.load_aware.os.getloadavg',
                   return_value=(100.0, 100.0, 100.0)):
            t0 = time.perf_counter()
            result = sleep_if_loaded(stop, check_interval=0.05)
            elapsed = time.perf_counter() - t0
        assert result is True
        assert elapsed >= 0.04  # actually waited

    def test_loaded_wait_interrupted_by_stop_event(self):
        # The wait is interruptible — setting stop_event mid-wait
        # should let sleep_if_loaded return promptly even with a long
        # check_interval. Critical because graceful shutdown depends
        # on this.
        stop = multiprocessing.Event()
        # Set stop_event after 50ms
        threading.Timer(0.05, stop.set).start()
        with patch('viz_authoring.load_aware.os.getloadavg',
                   return_value=(100.0, 100.0, 100.0)):
            t0 = time.perf_counter()
            result = sleep_if_loaded(stop, check_interval=5.0)
            elapsed = time.perf_counter() - t0
        # Returned True (was loaded) and didn't wait the full 5s
        assert result is True
        assert elapsed < 1.0
