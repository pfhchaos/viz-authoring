"""System-load-aware pacing for background workers.

Visualization-author pattern: a background worker (genome scorer,
transition computer, training job, anything periodic) does its work
while the user isn't actively using the system. Each iteration
consults the 1-minute load average; if the box is busy, the worker
sleeps for a beat and rechecks. Keeps the foreground visualization
smooth even with batch CPU work running in the background.

Defaults (6.0 threshold, 10s check interval) come from extraction
from flame-sheep, where multiple workers had independently arrived at
the same numbers. Override per-worker via the parameters if needed.

This is the application-level half of the load-aware pattern. The
companion piece — declarative worker contracts with priority levels,
external-load detection, etc. — is the eventual GPU scheduler design
(planned for viz_authoring once the Vulkan transition provides the
primitives).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as MpEvent

# 1-minute load average above which workers should pause and recheck
# later. Set to roughly 1.5× the number of cores on a typical desktop:
# normal interactive load sits well below; a sustained spike pushes
# above. Workers yield to keep the wallpaper smooth.
LOAD_THRESHOLD = 6.0

# Seconds to wait before rechecking after a loaded-state hit. Long
# enough that brief CPU spikes (a compile, a tab switch) don't make
# the worker thrash; short enough that workers resume promptly once
# the spike passes.
LOAD_CHECK_INTERVAL = 10.0


def is_loaded(threshold: float = LOAD_THRESHOLD) -> bool:
    """True if the 1-minute load average exceeds `threshold`."""
    return os.getloadavg()[0] > threshold


def sleep_if_loaded(stop_event: 'MpEvent',
                    threshold: float = LOAD_THRESHOLD,
                    check_interval: float = LOAD_CHECK_INTERVAL) -> bool:
    """Pace one iteration of a load-aware worker loop.

    If the system is loaded, sleep `check_interval` seconds
    (interruptible by `stop_event`) and return True — caller should
    `continue` and try again. Otherwise return False immediately —
    caller proceeds to do work.

    Pattern at the top of a worker's main loop:

        while not stop_event.is_set():
            if sleep_if_loaded(stop_event):
                continue
            ...do one unit of work...
    """
    if is_loaded(threshold):
        stop_event.wait(check_interval)
        return True
    return False
