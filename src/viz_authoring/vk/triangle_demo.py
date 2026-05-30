"""Vulkan-on-Wayland triangle demo — exercises the viz_authoring.vk
primitives end-to-end.

After Phase 1's extraction (context / surface / swapchain / pipeline),
this demo is the smallest thing that uses all of them. If it ever
breaks, the change has probably broken the foundation for the flame
renderer port. Keep this passing.

Run:
    python -m viz_authoring.vk.triangle_demo
"""
from __future__ import annotations

import time
from pathlib import Path

import vulkan as vk

from .context import VkContext
from .surface import WindowSurface
from .swapchain import Swapchain
from .pipeline import GraphicsPipeline, make_color_attachment_render_pass

SHADER_DIR = Path(__file__).parent / 'shaders'


class TriangleDemo:
    WIDTH = 800
    HEIGHT = 600

    def __init__(self):
        self.window = WindowSurface(self.WIDTH, self.HEIGHT,
                                     title='flame-sheep vulkan triangle')
        self.ctx = VkContext(
            instance_extensions=WindowSurface.required_instance_extensions(),
            app_name='flame-sheep triangle',
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
        self.pipeline = GraphicsPipeline(
            self.ctx, self.render_pass,
            SHADER_DIR / 'triangle.vert',
            SHADER_DIR / 'triangle.frag',
            extent=self.swapchain.extent,
        )
        self._record_command_buffers()
        self._create_sync_objects()

    def _record_command_buffers(self):
        self.command_buffers = self.ctx.allocate_command_buffers(
            len(self.swapchain.framebuffers))
        clear = vk.VkClearValue(
            color=vk.VkClearColorValue(float32=[0.05, 0.05, 0.10, 1.0]))
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
                                   self.pipeline.pipeline)
            vk.vkCmdDraw(cb, 3, 1, 0, 0)
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
        vk.vkWaitForFences(self.ctx.device, 1, [self.in_flight], vk.VK_TRUE,
                            0xFFFFFFFFFFFFFFFF)
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
        # Order matters: device-owned objects → surface (needs instance)
        # → device + instance (ctx) → GLFW window. If the order is
        # wrong, vkGetInstanceProcAddr blows up with VUID-...-instance-
        # parameter because the instance has already been destroyed.
        if hasattr(self, 'ctx') and self.ctx.device is not None:
            vk.vkDeviceWaitIdle(self.ctx.device)
            for s in ('image_available', 'render_finished'):
                if hasattr(self, s):
                    vk.vkDestroySemaphore(self.ctx.device, getattr(self, s), None)
            if hasattr(self, 'in_flight'):
                vk.vkDestroyFence(self.ctx.device, self.in_flight, None)
            if hasattr(self, 'pipeline'):
                self.pipeline.cleanup()
            if hasattr(self, 'render_pass'):
                vk.vkDestroyRenderPass(self.ctx.device, self.render_pass, None)
            if hasattr(self, 'swapchain'):
                self.swapchain.cleanup()
        if hasattr(self, 'window'):
            self.window.destroy_surface()  # while instance still alive
        if hasattr(self, 'ctx'):
            self.ctx.cleanup()
        if hasattr(self, 'window'):
            self.window.cleanup()  # surface already gone; just GLFW


def main():
    demo = TriangleDemo()
    print(f'device: {demo.ctx.device_name}')
    print(f'swapchain: {demo.swapchain.extent.width}x'
          f'{demo.swapchain.extent.height}, '
          f'{len(demo.swapchain.images)} images, '
          f'format={demo.swapchain.format}')
    print('rendering... close the window to exit')
    try:
        demo.run()
    finally:
        demo.cleanup()


if __name__ == '__main__':
    main()
