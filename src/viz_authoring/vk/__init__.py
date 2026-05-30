"""Vulkan graphics + WSI primitives for visualization authors.

Complements the compute-only path in wallpaper_ml — this subpackage
handles the half Vulkan needs for actual rendering: graphics
pipelines, swapchains, presentation. Initial scaffolding (Phase 0)
contains a triangle demo that risk-validates Vulkan-on-Wayland
support on the target hardware. Production primitives (instance,
device, swapchain, render pass builders) grow out of that demo as
the flame renderer migrates off moderngl.
"""
