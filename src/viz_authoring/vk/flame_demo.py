"""End-to-end Vulkan flame demo — chaos game compute → tonemap → window.

The Phase 3 milestone: ChaosGame produces a histogram via compute
shaders, a tonemap graphics pass reads that histogram + max_buf into
the swapchain image, GLFW presents the result. Sierpinski-triangle
genome by default so the rendered output is visually verifiable.

Per-frame loop:
  1. clear_histogram() with decay (temporal smoothing across frames)
  2. render_frame() — one chaos game pass (compute)
  3. reduce_max_hits() — compute pass for tonemap brightness divisor
  4. record graphics command buffer: bind tonemap pipeline, bind
     descriptor set (histogram + max_buf), push constants, draw 6
     vertices for the fullscreen quad
  5. submit + present

Sync: each compute dispatch goes through ctx.dispatch_compute which
already waits on a fence (synchronous one-shot). So by the time we
submit the graphics work, the histogram writes are visible. Could
be more efficient (overlap compute and graphics), but correctness
first; perf in Phase 4.

Run:
    python -m viz_authoring.vk.flame_demo
"""
from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np
import vulkan as vk

from .chaos_game import ChaosGame, MAX_TRANSFORMS, MAX_ACTIVE_VARS, SLOT_SIZE
from .context import VkContext
from .pipeline import GraphicsPipeline, make_color_attachment_render_pass
from .surface import WindowSurface
from .swapchain import Swapchain

SHADER_DIR = Path(__file__).parent / 'shaders'


def _sierpinski_genome():
    """Three-transform Sierpinski triangle genome — arrays sized for
    ChaosGame.set_genome. Verifiable by eye in the rendered output."""
    T = MAX_TRANSFORMS + 1
    affines = np.zeros((T, 6), dtype=np.float32)
    affines[0] = [0.5, 0, -0.5,  0, 0.5, -0.5]
    affines[1] = [0.5, 0,  0.5,  0, 0.5, -0.5]
    affines[2] = [0.5, 0,  0.0,  0, 0.5,  0.5]
    post_affines = np.tile([1, 0, 0, 0, 1, 0], (T, 1)).astype(np.float32)
    active_vars = np.full((T, MAX_ACTIVE_VARS, SLOT_SIZE), -1.0, dtype=np.float32)
    for t in range(3):
        active_vars[t, 0, 0:2] = [0, 1]  # linear variation, weight 1
    pre_vars = np.full((T, MAX_ACTIVE_VARS, SLOT_SIZE), -1.0, dtype=np.float32)
    colors = np.array([0.1, 0.5, 0.9, 0, 0, 0, 0], dtype=np.float32)
    color_speeds = np.array([0.5]*3 + [0]*4, dtype=np.float32)
    weights = np.array([1/3, 1/3, 1/3, 0, 0, 0], dtype=np.float32)
    return dict(
        affines=affines, post_affines=post_affines,
        active_vars=active_vars, pre_vars=pre_vars,
        colors=colors, color_speeds=color_speeds, weights=weights,
        n_transforms=3, has_final_xform=False,
    )


class FlameDemo:
    WIDTH = 800
    HEIGHT = 800
    CANVAS_W = 512   # histogram resolution — independent of window size
    CANVAS_H = 512
    N_WALKERS = 65536
    GAMMA = 2.0      # tonemap exponent

    def __init__(self, genome: dict | None = None):
        # GLFW + Vulkan plumbing (matches triangle_demo.py).
        self.window = WindowSurface(self.WIDTH, self.HEIGHT,
                                     title='flame-sheep vulkan flame')
        self.ctx = VkContext(
            instance_extensions=WindowSurface.required_instance_extensions(),
            app_name='flame-sheep flame',
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

        # Chaos game with the supplied genome
        self.chaos = ChaosGame(self.ctx, self.CANVAS_W, self.CANVAS_H,
                                n_walkers=self.N_WALKERS)
        self.chaos.set_genome(**(genome or _sierpinski_genome()))
        self.chaos.reset_walkers(seed=0)

        # Tonemap graphics pipeline — reads histogram + max_buf.
        self.tonemap = GraphicsPipeline(
            self.ctx, self.render_pass,
            SHADER_DIR / 'tonemap.vert',
            SHADER_DIR / 'tonemap.frag',
            extent=self.swapchain.extent,
            storage_buffers=[self.chaos.histogram, self.chaos.max_buf],
            push_constant_size=12,
        )

        self._record_command_buffers()
        self._create_sync_objects()

    def _record_command_buffers(self):
        self.command_buffers = self.ctx.allocate_command_buffers(
            len(self.swapchain.framebuffers))

        import cffi
        ffi = cffi.FFI()
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
        # Compute side first — chaos game + reduce_max. Both are
        # synchronous via dispatch_compute's internal fence-wait, so
        # by the time we return the histogram + max_buf are settled
        # and the graphics queue can safely read them.
        # Heavy decay so the latest frame dominates the visible output
        # — we're not accumulating across frames in the demo, each frame
        # builds from scratch but the walker state persists so it stays
        # on the attractor.
        self.chaos.clear_histogram(decay=0.0)
        self.chaos.render_frame(iterations=80)
        self.chaos.reduce_max_hits()

        # Graphics side: standard acquire/submit/present.
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
        import glfw
        frame = 0
        t0 = time.monotonic()
        try:
            while not self.window.should_close():
                glfw.poll_events()
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
    demo = FlameDemo()
    print(f'device: {demo.ctx.device_name}')
    print(f'window: {demo.swapchain.extent.width}x'
          f'{demo.swapchain.extent.height}, '
          f'canvas: {demo.CANVAS_W}x{demo.CANVAS_H}, '
          f'{demo.N_WALKERS} walkers')
    print('rendering flame... close the window to exit')
    try:
        demo.run()
    finally:
        demo.cleanup()


if __name__ == '__main__':
    main()
