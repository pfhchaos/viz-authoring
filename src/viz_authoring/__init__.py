"""viz_authoring — framework primitives for building audio-reactive
Wayland wallpaper visualizations.

This package is the extraction of the generic infrastructure that
flame-sheep built up for its own use, factored out so third-party (or
future you) visualization authors can build new wallpaper-style
visualizations against the same foundation without reaching into
flame-specific code.

What's here:
  - `GpuContext` + `Viewport` + `GpuRingTimer` + shader helpers — wraps
    moderngl with shader compilation, buffer/texture allocation,
    fullscreen-quad rendering, diagnostic GPU timers, debug overlays,
    framebuffer→PNG snapshots.
  - `load_aware` — system-load-aware pacing utilities for background
    workers (`is_loaded`, `sleep_if_loaded`).
  - `worker_bootstrap` — standard subprocess setup for background
    workers (logging reset, nice, SQLite connection).

What's NOT here yet (and where they live until they move):
  - Worker patterns (GpuWorker / CpuWorkerPool / ScheduledTask) — gated
    on Vulkan transition, which is the layer that gives GPU-aware
    scheduling its real primitives.
  - Audio response/easing helpers (currently in `flame_sheep_audio.response`)
  - Wayland session/window plumbing (currently in `flame_sheep.rendering.window`)

The intended consumer pattern: import primitives from `viz_authoring`,
import audio data from `flame_sheep_audio`, build your visualization
on top. `flame_sheep` itself is one such consumer; `flame_sheep.debug/`
is the reference implementation showing how to wire things up.

See `docs/reorg_plan.md` Stage 9 in the parent flame-sheep repo for
the extraction's context and the planned future contents.
"""
from __future__ import annotations

from .gpu_context import (
    Viewport,
    GpuContext,
    GpuRingTimer,
    GPU_TIMING_ENABLED,
    _resolve_includes,
    _bind_default_framebuffer,
)
from .load_aware import (
    LOAD_THRESHOLD,
    LOAD_CHECK_INTERVAL,
    is_loaded,
    sleep_if_loaded,
)
from .worker_bootstrap import init_worker_subprocess

__all__ = [
    # GpuContext + helpers
    'Viewport',
    'GpuContext',
    'GpuRingTimer',
    'GPU_TIMING_ENABLED',
    '_resolve_includes',
    '_bind_default_framebuffer',
    # load_aware
    'LOAD_THRESHOLD',
    'LOAD_CHECK_INTERVAL',
    'is_loaded',
    'sleep_if_loaded',
    # worker_bootstrap
    'init_worker_subprocess',
]
