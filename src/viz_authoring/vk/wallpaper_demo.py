"""Wallpaper-mode flame demo — same chaos game + tonemap as flame_demo,
but rendered to a wlr-layer-shell BACKGROUND surface instead of a GLFW
window. End state: a flame fractal painted under all your other windows.

This is the Phase 4 milestone — the production wallpaper-mode surface
path works end-to-end on Vulkan.

Run on sway/wlroots (won't work on Gnome / KDE — they don't expose
layer-shell):

    python -m viz_authoring.vk.wallpaper_demo
    python -m viz_authoring.vk.wallpaper_demo --output DP-3
    python -m viz_authoring.vk.wallpaper_demo --list-outputs

Ctrl+C to exit cleanly (the surface installs a SIGINT handler).
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import vulkan as vk

from .chaos_game import ChaosGame, MAX_TRANSFORMS, MAX_ACTIVE_VARS, SLOT_SIZE
from .context import VkContext
from .image import create_image_rgba8, upload_image_rgba8, create_linear_sampler
from .layer_shell import LayerShellSurface
from .pipeline import GraphicsPipeline, make_color_attachment_render_pass
from .swapchain import Swapchain
from .flame_demo import _fire_palette, _sierpinski_genome

SHADER_DIR = Path(__file__).parent / 'shaders'


def _genome_from_catalog(genome_id: int):
    """Load a real genome from flame_sheep's catalog and shape it into
    ChaosGame.set_genome() kwargs + a palette as (1, 256, 4) uint8.

    Returns (set_genome_kwargs, palette_rgba8, zoom_tuple, rotation, center).
    Caller uses zoom/rotation/center on each render_frame() call —
    they're per-frame uniforms, not part of set_genome state."""
    from flame_sheep.storage.library import Library
    lib = Library()
    g = lib.load_genome(genome_id)
    arrays = g.to_gpu_arrays()
    # ChaosGame.set_genome expects 3D shape; catalog returns flattened (T, 80).
    av = arrays['active_vars'].reshape(7, 8, 10)
    pv = arrays['pre_active_vars'].reshape(7, 8, 10)
    kwargs = dict(
        affines=arrays['affines'],
        post_affines=arrays['post_affines'],
        active_vars=av, pre_vars=pv,
        colors=arrays['colors'],
        color_speeds=arrays['color_speeds'],
        weights=arrays['weights'],
        n_transforms=len(g.transforms),
        has_final_xform=arrays['has_final_xform'],
    )
    # Palette: catalog gives (256, 3) float32 in [0,1]; image uploader
    # wants (1, 256, 4) uint8 with alpha=255.
    pal_rgb = (np.clip(g.palette, 0.0, 1.0) * 255.0).astype(np.uint8)
    pal_rgba = np.zeros((1, 256, 4), dtype=np.uint8)
    pal_rgba[0, :, :3] = pal_rgb
    pal_rgba[0, :, 3] = 255
    return kwargs, pal_rgba, (g.zoom, g.zoom), g.rotation, tuple(g.center)


class WallpaperDemo:
    CANVAS_W = 512
    CANVAS_H = 512
    N_WALKERS = 65536
    GAMMA = 2.0

    def __init__(self, output_name: str | None = None,
                 genome: dict | None = None,
                 palette: np.ndarray | None = None,
                 zoom: tuple[float, float] = (1.0, 1.0),
                 rotation: float = 0.0,
                 center: tuple[float, float] = (0.0, 0.0)):
        # Per-frame chaos game uniforms stashed for the render loop.
        self._zoom = zoom
        self._rotation = rotation
        self._center = center
        # Layer-shell surface first — it talks to the compositor and
        # negotiates a size, which we need before building the swapchain.
        self.window = LayerShellSurface(output_name=output_name)
        self.ctx = VkContext(
            instance_extensions=LayerShellSurface.required_instance_extensions(),
            app_name='flame-sheep wallpaper',
        )
        self.window.create_surface(self.ctx)
        self.ctx.select_device(surface=self.window.surface)

        self.swapchain = Swapchain(
            self.ctx, self.window.surface,
            requested_extent=self.window.framebuffer_size(),
        )
        self.render_pass = make_color_attachment_render_pass(
            self.ctx, self.swapchain.format)
        self.swapchain.build_framebuffers(self.render_pass)

        # Chaos game with the supplied (or default Sierpinski) genome.
        self.chaos = ChaosGame(self.ctx, self.CANVAS_W, self.CANVAS_H,
                                n_walkers=self.N_WALKERS)
        self.chaos.set_genome(**(genome or _sierpinski_genome()))
        self.chaos.reset_walkers(seed=0)

        self.palette_image = create_image_rgba8(self.ctx, 256, 1)
        upload_image_rgba8(
            self.ctx, self.palette_image,
            palette if palette is not None else _fire_palette())
        self.palette_sampler = create_linear_sampler(self.ctx)

        self.tonemap = GraphicsPipeline(
            self.ctx, self.render_pass,
            SHADER_DIR / 'tonemap.vert',
            SHADER_DIR / 'tonemap.frag',
            extent=self.swapchain.extent,
            storage_buffers=[self.chaos.histogram, self.chaos.max_buf],
            sampled_images=[(self.palette_image, self.palette_sampler)],
            push_constant_size=12,
        )
        self._record_command_buffers()
        self._create_sync_objects()

    def _record_command_buffers(self):
        # Identical to flame_demo's recording (full-screen quad +
        # bound descriptor set + push constants).
        import cffi
        ffi = cffi.FFI()
        self.command_buffers = self.ctx.allocate_command_buffers(
            len(self.swapchain.framebuffers))
        push = struct.pack('2if', self.CANVAS_W, self.CANVAS_H, self.GAMMA)
        pc_ptr = ffi.new('char[]', push)
        clear = vk.VkClearValue(
            color=vk.VkClearColorValue(float32=[0.0, 0.0, 0.0, 1.0]))
        for i, cb in enumerate(self.command_buffers):
            vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo())
            rp_begin = vk.VkRenderPassBeginInfo(
                renderPass=self.render_pass,
                framebuffer=self.swapchain.framebuffers[i],
                renderArea=vk.VkRect2D(
                    offset=vk.VkOffset2D(x=0, y=0),
                    extent=self.swapchain.extent),
                clearValueCount=1, pClearValues=[clear],
            )
            vk.vkCmdBeginRenderPass(cb, rp_begin,
                                      vk.VK_SUBPASS_CONTENTS_INLINE)
            vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
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

    def _create_sync_objects(self):
        sem_create = vk.VkSemaphoreCreateInfo()
        self.image_available = vk.vkCreateSemaphore(
            self.ctx.device, sem_create, None)
        self.render_finished = vk.vkCreateSemaphore(
            self.ctx.device, sem_create, None)
        fence_create = vk.VkFenceCreateInfo(
            flags=vk.VK_FENCE_CREATE_SIGNALED_BIT)
        self.in_flight = vk.vkCreateFence(self.ctx.device, fence_create, None)

    def _draw_frame(self):
        self.chaos.clear_histogram(decay=0.0)
        self.chaos.render_frame(iterations=80, zoom=self._zoom,
                                  rotation=self._rotation,
                                  center=self._center)
        self.chaos.reduce_max_hits()

        vk.vkWaitForFences(self.ctx.device, 1, [self.in_flight],
                            vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
        vk.vkResetFences(self.ctx.device, 1, [self.in_flight])
        image_index = self.swapchain.acquire_next_image(self.image_available)

        submit = vk.VkSubmitInfo(
            waitSemaphoreCount=1, pWaitSemaphores=[self.image_available],
            pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffers[image_index]],
            signalSemaphoreCount=1, pSignalSemaphores=[self.render_finished],
        )
        vk.vkQueueSubmit(self.ctx.graphics_queue, 1, [submit], self.in_flight)
        self.swapchain.present(image_index, self.render_finished)

    def run(self, max_frames: int | None = None):
        frame = 0
        t0 = time.monotonic()
        try:
            while not self.window.should_close():
                # Pump Wayland events so configure / closed get delivered.
                self.window.dispatch_events(blocking=False)
                self._draw_frame()
                frame += 1
                if max_frames is not None and frame >= max_frames:
                    break
        finally:
            vk.vkDeviceWaitIdle(self.ctx.device)
            dt = time.monotonic() - t0
            if frame > 0:
                print(f'rendered {frame} frames in {dt:.2f}s '
                      f'({frame/dt:.1f} fps)')

    def cleanup(self):
        if hasattr(self, 'ctx') and self.ctx.device is not None:
            vk.vkDeviceWaitIdle(self.ctx.device)
            for s in ('image_available', 'render_finished'):
                if hasattr(self, s):
                    vk.vkDestroySemaphore(self.ctx.device, getattr(self, s), None)
            if hasattr(self, 'in_flight'):
                vk.vkDestroyFence(self.ctx.device, self.in_flight, None)
            if hasattr(self, 'tonemap'):
                self.tonemap.cleanup()
            if hasattr(self, 'palette_image'):
                self.palette_image.destroy()
            if hasattr(self, 'palette_sampler'):
                self.palette_sampler.destroy()
            if hasattr(self, 'chaos'):
                self.chaos.cleanup()
            if hasattr(self, 'render_pass'):
                vk.vkDestroyRenderPass(self.ctx.device, self.render_pass, None)
            if hasattr(self, 'swapchain'):
                self.swapchain.cleanup()
        if hasattr(self, 'window'):
            self.window.destroy_surface()
        if hasattr(self, 'ctx'):
            self.ctx.cleanup()
        if hasattr(self, 'window'):
            self.window.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description='Vulkan flame fractal wallpaper (wlr-layer-shell)')
    parser.add_argument('--output', default=None,
                        help='wl_output name (default: first available)')
    parser.add_argument('--list-outputs', action='store_true',
                        help='List available wl_output names + exit')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Render N frames + exit (for headless smoke)')
    parser.add_argument('--genome-id', type=int, default=None,
                        help='Load a genome from the flame_sheep catalog '
                             '(default: hardcoded Sierpinski + fire palette)')
    args = parser.parse_args()

    if args.list_outputs:
        for name in LayerShellSurface.list_outputs():
            print(name)
        return

    extra = {}
    if args.genome_id is not None:
        kwargs, palette, zoom, rotation, center = _genome_from_catalog(
            args.genome_id)
        extra = dict(genome=kwargs, palette=palette, zoom=zoom,
                      rotation=rotation, center=center)
        print(f'loaded genome {args.genome_id} '
              f'(zoom={zoom[0]:.3f}, rotation={rotation:.3f})')
    demo = WallpaperDemo(output_name=args.output, **extra)
    print(f'device: {demo.ctx.device_name}')
    print(f'output {demo.window._output_name}: '
          f'{demo.swapchain.extent.width}x{demo.swapchain.extent.height}, '
          f'{len(demo.swapchain.images)} swapchain images')
    print('rendering wallpaper... Ctrl+C to exit')
    try:
        demo.run(max_frames=args.max_frames)
    finally:
        demo.cleanup()


if __name__ == '__main__':
    main()
