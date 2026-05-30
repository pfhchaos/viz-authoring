"""LayerShellSurface — wlr-layer-shell background surface for wallpaper mode.

Mirror of WindowSurface's interface but uses pywayland + wlr-layer-shell
instead of GLFW. The compositor-side wl_surface is wrapped as a
`background` layer surface (renders beneath all other windows); the
Vulkan-side VkSurfaceKHR is built via vkCreateWaylandSurfaceKHR.

Single-output for Phase 4; multi-output (one LayerShellSurface per
wl_output) lands in Phase 6 once the single-output path is solid.

Reuses the wlr-layer-shell-v1 protocol bindings from
flame_sheep/protocol/. Doesn't depend on the GL renderer's
WallpaperSession — just on the protocol consumer.

Usage:
    ls = LayerShellSurface(output_name='DP-3')
    ctx = VkContext(LayerShellSurface.required_instance_extensions())
    ls.create_surface(ctx)
    ctx.select_device(surface=ls.surface)
    swapchain = Swapchain(ctx, ls.surface, ls.framebuffer_size())
    ...
    while not ls.should_close():
        ls.dispatch_events()      # process Wayland queue
        # render frame ...
"""
from __future__ import annotations

import logging
import signal
import time
import sys
from pathlib import Path

import cffi
import vulkan as vk

log = logging.getLogger(__name__)

# Importing from the flame_sheep package for the protocol bindings.
# Long-term, the wlr-layer-shell protocol consumer probably belongs in
# viz_authoring directly so the package doesn't reach back into
# flame_sheep — but for Phase 4 reusing the existing generated
# protocol code keeps scope down.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pywayland.client import Display  # noqa: E402
from pywayland.protocol.wayland import WlCompositor, WlOutput  # noqa: E402
from flame_sheep.protocol.wlr_layer_shell_unstable_v1.zwlr_layer_shell_v1 import ZwlrLayerShellV1  # noqa: E402
from flame_sheep.protocol.wlr_layer_shell_unstable_v1.zwlr_layer_surface_v1 import ZwlrLayerSurfaceV1  # noqa: E402

_FFI = cffi.FFI()


def _ptr_value(cdata) -> int:
    """Extract the underlying C pointer value from a cffi cdata object.
    pywayland stores its native handles as `._ptr` cffi cdata; Vulkan's
    wayland-surface-create-info needs the same pointer as a uintptr_t."""
    return int(_FFI.cast('uintptr_t', cdata))


class LayerShellSurface:
    """Wallpaper-mode layer-shell surface, mirror of WindowSurface."""

    @staticmethod
    def required_instance_extensions() -> list[str]:
        """Wayland WSI instance extensions — same as what GLFW reports
        for a windowed Wayland app. Vulkan's VK_KHR_wayland_surface
        works for layer surfaces too; the protocol that backed the
        wl_surface doesn't matter to Vulkan."""
        return ['VK_KHR_surface', 'VK_KHR_wayland_surface']

    @staticmethod
    def list_outputs() -> list[str]:
        """Names of all active Wayland outputs. Standalone — connects
        + lists + disconnects, no LayerShellSurface needed."""
        display = Display()
        display.connect()
        registry = display.get_registry()
        outputs = []

        def _on_global(reg, name, interface, version):
            if interface == WlOutput.name:
                outputs.append(reg.bind(name, WlOutput, min(version, 4)))

        registry.dispatcher['global'] = _on_global
        display.roundtrip()

        names = {}
        for out in outputs:
            out.dispatcher['name'] = (
                lambda output, oname, _o=out: names.update({id(_o): oname})
            )
        display.roundtrip()
        display.disconnect()
        return list(names.values())

    def __init__(self, output_name: str | None = None,
                 namespace: str = 'flame-sheep-vk',
                 install_sigint: bool = True):
        """Connect to Wayland and bind the compositor + layer-shell +
        target wl_output globals. `output_name=None` picks the first
        output the compositor advertises."""
        self.namespace = namespace
        self.width = 0
        self.height = 0
        self._closed = False
        self._configured = False
        self.surface = None        # VkSurfaceKHR (uint64)
        self._instance = None

        self._display = Display()
        self._display.connect()

        self._compositor = None
        self._layer_shell = None
        self._outputs: dict[str, object] = {}

        registry = self._display.get_registry()
        registry.dispatcher['global'] = self._on_global
        # Two roundtrips: first collects globals + binds them, second
        # collects output names that arrive via wl_output.name events.
        self._display.roundtrip()
        self._display.roundtrip()

        if self._compositor is None:
            raise RuntimeError('No wl_compositor in registry')
        if self._layer_shell is None:
            raise RuntimeError(
                'No zwlr_layer_shell_v1 — needs sway/wlroots compositor')
        if not self._outputs:
            raise RuntimeError('No wl_outputs in registry')

        if output_name is None:
            output_name = next(iter(self._outputs))
            log.info(f'no output specified, using {output_name!r}')
        if output_name not in self._outputs:
            raise RuntimeError(
                f'Output {output_name!r} not found. '
                f'Available: {list(self._outputs)}')
        self._output_name = output_name
        self._wl_output = self._outputs[output_name]

        # Install Ctrl+C handler — sets _closed so the main loop can exit
        # cleanly (otherwise pywayland's blocking calls don't get
        # interrupted and the wallpaper hangs on shutdown). Multi-output
        # cases should install at the session level so all surfaces
        # observe the close — pass install_sigint=False on the per-
        # surface instances and the multi-output owner handles SIGINT
        # itself.
        if install_sigint:
            signal.signal(signal.SIGINT, lambda *_: self._on_signal())

        # Per-surface state, filled in by create_surface().
        self._wl_surface = None
        self._layer_surface = None

    def _on_global(self, registry, name, interface, version):
        if interface == WlCompositor.name:
            self._compositor = registry.bind(
                name, WlCompositor, min(version, 5))
        elif interface == ZwlrLayerShellV1.name:
            self._layer_shell = registry.bind(
                name, ZwlrLayerShellV1, min(version, 4))
        elif interface == WlOutput.name:
            out = registry.bind(name, WlOutput, min(version, 4))
            # Cache by name once the compositor sends it
            def _on_name(output, oname, _out=out):
                self._outputs[oname] = _out
            out.dispatcher['name'] = _on_name

    def _on_signal(self):
        log.info('SIGINT — marking surface for close')
        self._closed = True

    def create_surface(self, ctx) -> int:
        """Create the wl_surface, wrap it as a background layer, wait
        for the configure event, then build the VkSurfaceKHR.

        After this returns, self.surface is the Vulkan handle and
        self.width/self.height are the configured pixel size."""
        if self._wl_surface is not None:
            raise RuntimeError('create_surface called twice')

        self._wl_surface = self._compositor.create_surface()

        # Wrap as background layer surface. Anchor all 4 edges +
        # set_size(0,0) tells the compositor "fill the whole output."
        # exclusive_zone=-1 means we accept being under top-layer
        # surfaces (status bar, etc.) — standard wallpaper behavior.
        layer = ZwlrLayerShellV1.layer.background.value
        self._layer_surface = self._layer_shell.get_layer_surface(
            self._wl_surface, self._wl_output, layer, self.namespace)

        anchor_all = (ZwlrLayerSurfaceV1.anchor.top.value
                       | ZwlrLayerSurfaceV1.anchor.bottom.value
                       | ZwlrLayerSurfaceV1.anchor.left.value
                       | ZwlrLayerSurfaceV1.anchor.right.value)
        self._layer_surface.set_anchor(anchor_all)
        self._layer_surface.set_size(0, 0)
        self._layer_surface.set_exclusive_zone(-1)
        self._layer_surface.set_keyboard_interactivity(0)

        self._layer_surface.dispatcher['configure'] = self._on_configure
        self._layer_surface.dispatcher['closed'] = (
            lambda ls: setattr(self, '_closed', True))

        self._wl_surface.commit()
        self._display.roundtrip()

        # Wait for the compositor to ACK with a configure event giving
        # the actual surface size. 5s deadline so we fail loudly rather
        # than hang forever.
        deadline = time.monotonic() + 5.0
        while not self._configured:
            self._display.roundtrip()
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f'Timed out waiting for layer-shell configure on '
                    f'{self._output_name}')
            time.sleep(0.005)
        log.info(f'configured {self.width}x{self.height} on {self._output_name}')

        # Build the VkSurfaceKHR. Pywayland and vulkan-py each carry their
        # own CFFI instance — cdata from one isn't directly assignable
        # to the other's struct fields ('different ffi instances' check).
        # Round-trip the pointer value through a Python int, then cast
        # in vulkan-py's FFI to the expected type.
        wl_disp_int = _ptr_value(self._display._ptr)
        wl_surf_int = _ptr_value(self._wl_surface._ptr)
        vk_disp = vk.ffi.cast('struct wl_display*', wl_disp_int)
        vk_surf = vk.ffi.cast('struct wl_surface*', wl_surf_int)
        create_info = vk.VkWaylandSurfaceCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
            display=vk_disp,
            surface=vk_surf,
        )
        # vulkan-py exposes the WSI entry points through vkGetInstanceProcAddr.
        create_fn = vk.vkGetInstanceProcAddr(
            ctx.instance, 'vkCreateWaylandSurfaceKHR')
        self.surface = create_fn(ctx.instance, create_info, None)
        self._instance = ctx.instance
        return self.surface

    def _on_configure(self, layer_surface, serial: int,
                       width: int, height: int):
        if width > 0:
            self.width = width
        if height > 0:
            self.height = height
        layer_surface.ack_configure(serial)
        self._wl_surface.commit()
        self._configured = True

    # --- WindowSurface-equivalent interface so the rest of the stack
    # plugs in unchanged ---------------------------------------------

    def framebuffer_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def should_close(self) -> bool:
        return self._closed

    def dispatch_events(self, blocking: bool = False) -> None:
        """Pump the Wayland event queue. Call once per frame in the
        main loop so configure / closed events get delivered.

        pywayland's `dispatch(block=False)` returns 0 immediately if the
        queue is empty; with block=True it waits for the FD. Both also
        flush queued protocol messages to the compositor, so we don't
        need a separate flush() call."""
        self._display.dispatch(block=blocking)

    def destroy_surface(self):
        if self.surface is not None and self._instance is not None:
            destroy = vk.vkGetInstanceProcAddr(
                self._instance, 'vkDestroySurfaceKHR')
            destroy(self._instance, self.surface, None)
            self.surface = None

    def cleanup(self):
        """Tear down the layer-shell surface + disconnect from Wayland.
        destroy_surface() should have already run."""
        self.destroy_surface()
        if self._layer_surface is not None:
            self._layer_surface.destroy()
            self._layer_surface = None
        if self._wl_surface is not None:
            self._wl_surface.destroy()
            self._wl_surface = None
        if self._display is not None:
            self._display.disconnect()
            self._display = None
