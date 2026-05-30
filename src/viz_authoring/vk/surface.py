"""GLFW window + Vulkan surface integration.

Owns the GLFW init / window / surface lifecycle. Production wallpaper
needs Wayland layer-shell instead (no decorations, fullscreen, anchored
to outputs) — a future LayerShellSurface will sit alongside this one
implementing the same interface. For Phase 1 / Phase 2 development,
GLFW gets us to "render in a windowed sandbox" quickly.
"""
from __future__ import annotations

import ctypes

import glfw
import vulkan as vk


class WindowSurface:
    """A windowed Vulkan surface via GLFW.

    Usage:
        ws = WindowSurface(800, 600, title='my-app')
        ctx = VkContext(ws.required_instance_extensions())
        ws.create_surface(ctx)
        ctx.select_device(surface=ws.surface)
        ...
        while not ws.should_close():
            glfw.poll_events()
            ...
        ws.cleanup()

    Lifecycle is split (init / create_surface / cleanup) so the caller
    can interleave GLFW window creation with instance/device setup —
    the instance needs to know which extensions the surface requires,
    and the device needs the surface to query present support.
    """

    def __init__(self, width: int, height: int, title: str = 'viz_authoring'):
        if not glfw.init():
            raise RuntimeError('glfw init failed')
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        # Resizing requires swapchain recreation; the Phase 1 demo doesn't
        # bother. Disable until production wallpaper paths need it.
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(width, height, title, None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError('glfw create_window failed')
        self.surface = None
        self._instance = None

    @staticmethod
    def required_instance_extensions() -> list[str]:
        """Instance extensions the surface needs (platform-specific —
        GLFW returns VK_KHR_surface + e.g. VK_KHR_wayland_surface)."""
        return list(glfw.get_required_instance_extensions() or [])

    def create_surface(self, ctx) -> int:
        """Create the VkSurfaceKHR via GLFW's cross-platform helper.
        Stores it on the WindowSurface and returns the handle."""
        surface_handle = ctypes.c_uint64(0)
        result = glfw.create_window_surface(
            ctx.instance, self.window, None,
            ctypes.byref(surface_handle))
        if result != 0:
            raise RuntimeError(
                f'glfwCreateWindowSurface failed with VkResult={result}')
        self.surface = surface_handle.value
        self._instance = ctx.instance
        return self.surface

    def framebuffer_size(self) -> tuple[int, int]:
        """Current framebuffer size (pixels — may differ from window size
        on hi-DPI). Used when the surface reports a sentinel extent and
        the client has to pick the swapchain dimensions."""
        return glfw.get_framebuffer_size(self.window)

    def should_close(self) -> bool:
        return glfw.window_should_close(self.window)

    def destroy_surface(self):
        """Destroy the VkSurfaceKHR. Must be called while the instance is
        still alive (split from cleanup() so the caller can interleave
        surface teardown with device + instance teardown correctly)."""
        if self.surface is not None and self._instance is not None:
            destroy = vk.vkGetInstanceProcAddr(
                self._instance, 'vkDestroySurfaceKHR')
            destroy(self._instance, self.surface, None)
            self.surface = None

    def cleanup(self):
        """Destroy the surface + window + terminate GLFW. Idempotent.

        If the instance has already been destroyed before this call, the
        surface destroy is skipped — call destroy_surface() explicitly
        before instance teardown to avoid leaking the surface handle."""
        self.destroy_surface()
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()
