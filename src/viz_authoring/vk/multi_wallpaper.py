"""Multi-monitor wallpaper — one continuous flame spanning N outputs.

The Phase 6 layout: one shared ChaosGame writes one shared histogram;
N LayerShellSurfaces each render their viewport slice of the canvas
into their own swapchain via the same GraphicsPipeline (dynamic
viewport). Outputs are stacked horizontally in the order specified —
each takes a `(viewport_x, viewport_y, viewport_w, viewport_h)` push-
constant rectangle of the virtual canvas.

What's NOT here yet:
  - Compositor-aware output placement (we assume left-to-right stacking;
    sway lets you configure arbitrary positions, ignored here).
  - Hot-plug (outputs added/removed mid-run).
  - Per-output framerate independence (we render all outputs every frame).
  - Cross-output parallel graphics submission (we serialize).
All three are straightforward extensions once the basic span works.

Run:
    python -m viz_authoring.vk.multi_wallpaper                 # all outputs
    python -m viz_authoring.vk.multi_wallpaper --outputs DP-2 DP-3 DP-4
    python -m viz_authoring.vk.multi_wallpaper --genome-id 2505
"""
from __future__ import annotations

import argparse
import signal
import struct
import sys
import time
from pathlib import Path

import cffi
import numpy as np
import vulkan as vk

from .chaos_game import ChaosGame
from .context import VkContext
from .image import create_image_rgba8, upload_image_rgba8, create_linear_sampler
from .layer_shell import LayerShellSurface
from .pipeline import GraphicsPipeline, make_color_attachment_render_pass
from .swapchain import Swapchain
from .flame_demo import _fire_palette, _sierpinski_genome
from .wallpaper_demo import _genome_from_catalog

SHADER_DIR = Path(__file__).parent / 'shaders'


class _OutputBundle:
    """Per-output resources — surface + swapchain + framebuffers + sync
    + pre-recorded command buffers indexed by swapchain image."""
    __slots__ = ('name', 'surface', 'swapchain', 'image_available',
                 'render_finished', 'in_flight', 'command_buffers',
                 'viewport_x', 'viewport_y', 'viewport_w', 'viewport_h')


class MultiMonitorWallpaper:
    CANVAS_W = 1920    # virtual canvas resolution — independent of monitor sizes
    CANVAS_H = 1080
    N_WALKERS = 65536
    GAMMA = 2.0

    # Canvas density: render at half the densest monitor's PPI. Compute
    # cost grows ~ppmm², so 0.5× is a meaningful saver vs full density,
    # and the chaos-game's stochastic sampling means full density mostly
    # just buys extra noise per pixel anyway. Match the GL renderer's
    # choice for consistency.
    RENDER_SCALE = 0.5

    def __init__(self, output_names: list[str],
                 genome: dict | None = None,
                 palette: np.ndarray | None = None,
                 zoom: tuple[float, float] = (1.0, 1.0),
                 rotation: float = 0.0,
                 center: tuple[float, float] = (0.0, 0.0)):
        if not output_names:
            raise ValueError('at least one output required')
        self._zoom = zoom
        self._rotation = rotation
        self._center = center
        self._closed = False

        signal.signal(signal.SIGINT, lambda *_: setattr(self, '_closed', True))

        # 0. Discover physical layout (mm) — reuse the GL renderer's
        # output-layout helper (xdg-output + swaymsg fallback, rotation-
        # aware). Gives us {name: {x, y, w, h, ppi, phys_w_mm, phys_h_mm}}.
        # Without this, viewports are pixel-based and a flame feature
        # is a different physical size on each monitor (= lines don't
        # match across boundaries on screens with different DPIs).
        from flame_sheep.rendering.surface import _get_output_layout
        layout = _get_output_layout()
        missing = [n for n in output_names if n not in layout]
        if missing:
            raise RuntimeError(
                f'outputs not in layout: {missing} '
                f'(have {list(layout)})')
        self._layout = {n: layout[n] for n in output_names}
        max_ppi = max(g['ppi'] for g in self._layout.values())
        self.canvas_ppmm = max_ppi / 25.4 * self.RENDER_SCALE

        # 1. Create all LayerShellSurfaces. Each opens its own Wayland
        # Display — wasteful but simpler than sharing one.
        self.outputs: list[_OutputBundle] = []
        for name in output_names:
            ob = _OutputBundle()
            ob.name = name
            ob.surface = LayerShellSurface(output_name=name,
                                            install_sigint=False)
            self.outputs.append(ob)

        # 2. One VkContext — instance + device shared across all surfaces.
        # Pick the device using the FIRST output's surface for present
        # support (all surfaces on the same compositor + same GPU will
        # accept the same physical device).
        self.ctx = VkContext(
            instance_extensions=LayerShellSurface.required_instance_extensions(),
            app_name='flame-sheep multi-wallpaper',
        )
        for ob in self.outputs:
            ob.surface.create_surface(self.ctx)
        self.ctx.select_device(surface=self.outputs[0].surface.surface)

        # 3. Per-output swapchain + framebuffer chain. Render pass is
        # shared across outputs (we assume identical surface format; on
        # the project's hardware all outputs report BGRA8_SRGB).
        first_sc = Swapchain(
            self.ctx, self.outputs[0].surface.surface,
            requested_extent=self.outputs[0].surface.framebuffer_size(),
        )
        self.render_pass = make_color_attachment_render_pass(
            self.ctx, first_sc.format)
        first_sc.build_framebuffers(self.render_pass)
        self.outputs[0].swapchain = first_sc

        for ob in self.outputs[1:]:
            sc = Swapchain(self.ctx, ob.surface.surface,
                            requested_extent=ob.surface.framebuffer_size())
            if sc.format != first_sc.format:
                raise NotImplementedError(
                    f'surface format mismatch: output {ob.name} has '
                    f'{sc.format} but {self.outputs[0].name} has '
                    f'{first_sc.format} — per-output render passes needed')
            sc.build_framebuffers(self.render_pass)
            ob.swapchain = sc

        # 4. Compute viewport rectangles in PHYSICAL UNITS (mm), then
        # project into canvas pixels at canvas_ppmm. This is what makes
        # lines align across monitors with different pixel densities:
        # a feature that's 10mm wide on DP-3's panel occupies the same
        # number of canvas pixels as 10mm on DP-2, regardless of the
        # monitors' pixel counts. Horizontal layout: stack in supplied
        # order. Vertical: center each output on the tallest's center.
        #
        # The compositor-reported pixel position (layout[n]['x']) is
        # NOT used directly — we accumulate physical widths in supplied
        # order. Caller is expected to pass outputs in physical L→R
        # order (or query via swaymsg / xdg-output and sort by x).
        cursor_mm = 0.0
        phys_x_mm: dict[str, float] = {}
        for ob in self.outputs:
            phys_x_mm[ob.name] = cursor_mm
            cursor_mm += self._layout[ob.name]['phys_w_mm']
        total_w_mm = cursor_mm
        max_h_mm = max(g['phys_h_mm'] for g in self._layout.values())

        for ob in self.outputs:
            g = self._layout[ob.name]
            # Center each output vertically on the tallest's center
            y_offset_mm = (max_h_mm - g['phys_h_mm']) / 2.0
            ob.viewport_x = int(phys_x_mm[ob.name] * self.canvas_ppmm)
            ob.viewport_y = int(y_offset_mm * self.canvas_ppmm)
            ob.viewport_w = int(g['phys_w_mm'] * self.canvas_ppmm)
            ob.viewport_h = int(g['phys_h_mm'] * self.canvas_ppmm)

        self.virtual_w = int(total_w_mm * self.canvas_ppmm)
        self.virtual_h = int(max_h_mm * self.canvas_ppmm)
        total_w = self.virtual_w  # alias for the existing references below
        total_h = self.virtual_h

        # 5. Shared chaos game + palette + tonemap pipeline.
        # Canvas size matches the virtual desktop so each viewport pixel
        # = one canvas pixel — no resampling in the tonemap fragment.
        self.chaos = ChaosGame(self.ctx, total_w, total_h,
                                n_walkers=self.N_WALKERS)
        self.chaos.set_genome(**(genome or _sierpinski_genome()))
        self.chaos.reset_walkers(seed=0)

        self.palette_image = create_image_rgba8(self.ctx, 256, 1)
        upload_image_rgba8(
            self.ctx, self.palette_image,
            palette if palette is not None else _fire_palette())
        self.palette_sampler = create_linear_sampler(self.ctx)

        # Dynamic viewport — one pipeline, N viewports baked into per-
        # output command buffers.
        self.tonemap = GraphicsPipeline(
            self.ctx, self.render_pass,
            SHADER_DIR / 'tonemap.vert',
            SHADER_DIR / 'tonemap.frag',
            extent=self.outputs[0].swapchain.extent,  # ignored — dynamic
            storage_buffers=[self.chaos.histogram, self.chaos.max_buf],
            sampled_images=[(self.palette_image, self.palette_sampler)],
            push_constant_size=28,
            dynamic_viewport=True,
        )

        # 6. Sync primitives + pre-recorded command buffers per output.
        self._record_per_output_command_buffers()

    def _record_per_output_command_buffers(self):
        sem_create = vk.VkSemaphoreCreateInfo()
        fence_create = vk.VkFenceCreateInfo(
            flags=vk.VK_FENCE_CREATE_SIGNALED_BIT)
        ffi = cffi.FFI()

        for ob in self.outputs:
            ob.image_available = vk.vkCreateSemaphore(
                self.ctx.device, sem_create, None)
            ob.render_finished = vk.vkCreateSemaphore(
                self.ctx.device, sem_create, None)
            ob.in_flight = vk.vkCreateFence(
                self.ctx.device, fence_create, None)
            ob.command_buffers = self.ctx.allocate_command_buffers(
                len(ob.swapchain.framebuffers))

            # Build the push-constant blob once per output (stable values:
            # canvas size, this viewport, gamma).
            push = struct.pack(
                '6if',
                self.virtual_w, self.virtual_h,
                ob.viewport_x, ob.viewport_y,
                ob.viewport_w, ob.viewport_h,
                self.GAMMA)
            pc_ptr = ffi.new('char[]', push)

            # Dynamic viewport + scissor — set per recording, baked into
            # the command buffer that gets replayed.
            viewport = vk.VkViewport(
                x=0.0, y=0.0,
                width=float(ob.swapchain.extent.width),
                height=float(ob.swapchain.extent.height),
                minDepth=0.0, maxDepth=1.0,
            )
            scissor = vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0),
                extent=ob.swapchain.extent)

            clear = vk.VkClearValue(
                color=vk.VkClearColorValue(float32=[0, 0, 0, 1]))
            for i, cb in enumerate(ob.command_buffers):
                vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo())
                vk.vkCmdSetViewport(cb, 0, 1, [viewport])
                vk.vkCmdSetScissor(cb, 0, 1, [scissor])
                rp_begin = vk.VkRenderPassBeginInfo(
                    renderPass=self.render_pass,
                    framebuffer=ob.swapchain.framebuffers[i],
                    renderArea=vk.VkRect2D(
                        offset=vk.VkOffset2D(x=0, y=0),
                        extent=ob.swapchain.extent),
                    clearValueCount=1, pClearValues=[clear],
                )
                vk.vkCmdBeginRenderPass(cb, rp_begin,
                                          vk.VK_SUBPASS_CONTENTS_INLINE)
                vk.vkCmdBindPipeline(
                    cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                    self.tonemap.pipeline)
                vk.vkCmdBindDescriptorSets(
                    cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                    self.tonemap.layout, 0, 1,
                    [self.tonemap.descriptor_set], 0, None)
                vk.vkCmdPushConstants(
                    cb, self.tonemap.layout,
                    vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, len(push), ffi.cast('void*', pc_ptr))
                vk.vkCmdDraw(cb, 6, 1, 0, 0)
                vk.vkCmdEndRenderPass(cb)
                vk.vkEndCommandBuffer(cb)

    def _draw_frame(self):
        # Compute once into the shared histogram.
        self.chaos.clear_histogram(decay=0.0)
        self.chaos.render_frame(iterations=80, zoom=self._zoom,
                                  rotation=self._rotation,
                                  center=self._center)
        self.chaos.reduce_max_hits()

        # Serial graphics submission per output. Each output renders its
        # viewport slice from the same (now-written) histogram, presents
        # to its own swapchain.
        for ob in self.outputs:
            vk.vkWaitForFences(self.ctx.device, 1, [ob.in_flight],
                                vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
            vk.vkResetFences(self.ctx.device, 1, [ob.in_flight])
            image_index = ob.swapchain.acquire_next_image(ob.image_available)

            submit = vk.VkSubmitInfo(
                waitSemaphoreCount=1,
                pWaitSemaphores=[ob.image_available],
                pWaitDstStageMask=[
                    vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
                commandBufferCount=1,
                pCommandBuffers=[ob.command_buffers[image_index]],
                signalSemaphoreCount=1,
                pSignalSemaphores=[ob.render_finished],
            )
            vk.vkQueueSubmit(self.ctx.graphics_queue, 1, [submit],
                              ob.in_flight)
            ob.swapchain.present(image_index, ob.render_finished)

    def run(self, max_frames: int | None = None):
        frame = 0
        t0 = time.monotonic()
        try:
            while not self._closed and not any(
                    ob.surface.should_close() for ob in self.outputs):
                for ob in self.outputs:
                    ob.surface.dispatch_events(blocking=False)
                self._draw_frame()
                frame += 1
                if max_frames is not None and frame >= max_frames:
                    break
        finally:
            vk.vkDeviceWaitIdle(self.ctx.device)
            dt = time.monotonic() - t0
            if frame > 0:
                print(f'rendered {frame} frames in {dt:.2f}s '
                      f'({frame/dt:.1f} fps × {len(self.outputs)} outputs)')

    def cleanup(self):
        if self.ctx.device is not None:
            vk.vkDeviceWaitIdle(self.ctx.device)
            for ob in self.outputs:
                for s in (ob.image_available, ob.render_finished):
                    vk.vkDestroySemaphore(self.ctx.device, s, None)
                vk.vkDestroyFence(self.ctx.device, ob.in_flight, None)
                ob.swapchain.cleanup()
            self.tonemap.cleanup()
            self.palette_image.destroy()
            self.palette_sampler.destroy()
            self.chaos.cleanup()
            vk.vkDestroyRenderPass(self.ctx.device, self.render_pass, None)
        for ob in self.outputs:
            ob.surface.destroy_surface()
        self.ctx.cleanup()
        for ob in self.outputs:
            ob.surface.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description='Multi-monitor Vulkan flame wallpaper')
    parser.add_argument('--outputs', nargs='+', default=None,
                        help='wl_output names (default: all available)')
    parser.add_argument('--list-outputs', action='store_true')
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--genome-id', type=int, default=None,
                        help='catalog genome (default: hardcoded Sierpinski)')
    args = parser.parse_args()

    if args.list_outputs:
        for name in LayerShellSurface.list_outputs():
            print(name)
        return

    if args.outputs:
        outputs = args.outputs   # honor explicit order from CLI
    else:
        # Auto-order by physical x so the user doesn't have to figure
        # out their compositor's advertisement order. _get_output_layout
        # has the x positions (compositor's logical coords) already —
        # we sort by them and the resulting list goes left-to-right.
        from flame_sheep.rendering.surface import _get_output_layout
        layout = _get_output_layout()
        outputs = sorted(layout.keys(), key=lambda n: layout[n]['x'])
    print(f'rendering across {len(outputs)} output(s): {outputs}')

    extra = {}
    if args.genome_id is not None:
        kwargs, palette, zoom, rotation, center = _genome_from_catalog(
            args.genome_id)
        extra = dict(genome=kwargs, palette=palette, zoom=zoom,
                      rotation=rotation, center=center)

    demo = MultiMonitorWallpaper(outputs, **extra)
    print(f'virtual canvas: {demo.virtual_w}x{demo.virtual_h} '
          f'(@ {demo.canvas_ppmm:.2f} px/mm)')
    for ob in demo.outputs:
        g = demo._layout[ob.name]
        print(f'  {ob.name}: panel {g["phys_w_mm"]:.0f}x{g["phys_h_mm"]:.0f}mm '
              f'({g["ppi"]:.0f} PPI), surface {ob.surface.width}x{ob.surface.height}, '
              f'viewport ({ob.viewport_x},{ob.viewport_y}) '
              f'{ob.viewport_w}x{ob.viewport_h}')
    print('Ctrl+C to exit')
    try:
        demo.run(max_frames=args.max_frames)
    finally:
        demo.cleanup()


if __name__ == '__main__':
    main()
