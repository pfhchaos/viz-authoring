"""Swapchain + image views + framebuffers — owns the per-image rendering
targets that a Surface presents.

Recreate-on-resize is hard-coded out for Phase 1 (windows are
non-resizable, wallpaper is fullscreen): once the swapchain is built,
its extent / format are stable for the lifetime of the surface.
"""
from __future__ import annotations

import vulkan as vk

# Sentinel value Vulkan returns for currentExtent on platforms where the
# client picks the surface size (Wayland, primarily). When seen, the
# caller substitutes its desired framebuffer size, clamped to min/max.
_CLIENT_PICKS_EXTENT = 0xFFFFFFFF


class Swapchain:
    """Managed swapchain — pick format + extent + present mode, create
    images + views, build framebuffers against a given render pass."""

    def __init__(self, ctx, surface_handle: int, requested_extent: tuple[int, int],
                 prefer_format: int = vk.VK_FORMAT_B8G8R8A8_SRGB,
                 prefer_present_mode: int = vk.VK_PRESENT_MODE_MAILBOX_KHR):
        """ctx: VkContext (must have device + present queue already).
        surface_handle: VkSurfaceKHR.
        requested_extent: (w, h) — used when the surface reports
            currentExtent as the client-picks-size sentinel.
        prefer_format / prefer_present_mode: best-effort; falls back to
            the first available format / FIFO (always supported) if the
            preferred is missing.
        """
        self.ctx = ctx
        self.surface = surface_handle
        self._build(requested_extent, prefer_format, prefer_present_mode)

    def _build(self, requested_extent, prefer_format, prefer_present_mode):
        get_caps = vk.vkGetInstanceProcAddr(
            self.ctx.instance, 'vkGetPhysicalDeviceSurfaceCapabilitiesKHR')
        get_formats = vk.vkGetInstanceProcAddr(
            self.ctx.instance, 'vkGetPhysicalDeviceSurfaceFormatsKHR')
        get_present_modes = vk.vkGetInstanceProcAddr(
            self.ctx.instance, 'vkGetPhysicalDeviceSurfacePresentModesKHR')

        caps = get_caps(self.ctx.physical_device, self.surface)
        # vulkan-py returns generators — materialize so we can search.
        formats = list(get_formats(self.ctx.physical_device, self.surface))
        present_modes = list(get_present_modes(
            self.ctx.physical_device, self.surface))

        # --- format -----------------------------------------------------
        chosen_format = formats[0]
        for f in formats:
            if (f.format == prefer_format
                    and f.colorSpace == vk.VK_COLORSPACE_SRGB_NONLINEAR_KHR):
                chosen_format = f
                break
        self.format = chosen_format.format

        # --- extent -----------------------------------------------------
        if (caps.currentExtent.width == _CLIENT_PICKS_EXTENT
                or caps.currentExtent.height == _CLIENT_PICKS_EXTENT):
            rw, rh = requested_extent
            w = max(caps.minImageExtent.width,
                    min(caps.maxImageExtent.width, rw))
            h = max(caps.minImageExtent.height,
                    min(caps.maxImageExtent.height, rh))
            self.extent = vk.VkExtent2D(width=w, height=h)
        else:
            self.extent = caps.currentExtent

        # --- present mode -----------------------------------------------
        # FIFO is mandated by the spec — safe fallback.
        present_mode = vk.VK_PRESENT_MODE_FIFO_KHR
        if prefer_present_mode in present_modes:
            present_mode = prefer_present_mode

        # --- image count + sharing --------------------------------------
        image_count = caps.minImageCount + 1
        if caps.maxImageCount > 0 and image_count > caps.maxImageCount:
            image_count = caps.maxImageCount

        if self.ctx.graphics_queue_family == self.ctx.present_queue_family:
            sharing_mode = vk.VK_SHARING_MODE_EXCLUSIVE
            qf_indices = []
        else:
            sharing_mode = vk.VK_SHARING_MODE_CONCURRENT
            qf_indices = [self.ctx.graphics_queue_family,
                           self.ctx.present_queue_family]

        sc_create = vk.VkSwapchainCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            surface=self.surface,
            minImageCount=image_count,
            imageFormat=chosen_format.format,
            imageColorSpace=chosen_format.colorSpace,
            imageExtent=self.extent,
            imageArrayLayers=1,
            imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            imageSharingMode=sharing_mode,
            queueFamilyIndexCount=len(qf_indices),
            pQueueFamilyIndices=qf_indices,
            preTransform=caps.currentTransform,
            compositeAlpha=vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=present_mode,
            clipped=vk.VK_TRUE,
        )

        create_sc = vk.vkGetDeviceProcAddr(
            self.ctx.device, 'vkCreateSwapchainKHR')
        self.swapchain = create_sc(self.ctx.device, sc_create, None)

        get_images = vk.vkGetDeviceProcAddr(
            self.ctx.device, 'vkGetSwapchainImagesKHR')
        self.images = list(get_images(self.ctx.device, self.swapchain))

        # --- image views ------------------------------------------------
        self.image_views = []
        for img in self.images:
            view_create = vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=img,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=self.format,
                components=vk.VkComponentMapping(
                    r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                ),
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1,
                    baseArrayLayer=0, layerCount=1,
                ),
            )
            self.image_views.append(
                vk.vkCreateImageView(self.ctx.device, view_create, None))

        self.framebuffers = []  # built by build_framebuffers()

    def build_framebuffers(self, render_pass):
        """One framebuffer per swapchain image, all wrapping `render_pass`."""
        if self.framebuffers:
            raise RuntimeError(
                'framebuffers already built; destroy + rebuild not yet wired')
        for view in self.image_views:
            fb_create = vk.VkFramebufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass=render_pass,
                attachmentCount=1, pAttachments=[view],
                width=self.extent.width, height=self.extent.height,
                layers=1,
            )
            self.framebuffers.append(
                vk.vkCreateFramebuffer(self.ctx.device, fb_create, None))

    def acquire_next_image(self, signal_semaphore,
                           timeout: int = 0xFFFFFFFFFFFFFFFF) -> int:
        """Returns the swapchain image index for the next frame."""
        acquire = vk.vkGetDeviceProcAddr(
            self.ctx.device, 'vkAcquireNextImageKHR')
        return acquire(self.ctx.device, self.swapchain, timeout,
                        signal_semaphore, vk.VK_NULL_HANDLE)

    def present(self, image_index: int, wait_semaphore):
        """Hand the image to the presentation engine. wait_semaphore is
        signalled by the render submission this present depends on."""
        present_info = vk.VkPresentInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1, pWaitSemaphores=[wait_semaphore],
            swapchainCount=1, pSwapchains=[self.swapchain],
            pImageIndices=[image_index],
        )
        present = vk.vkGetDeviceProcAddr(
            self.ctx.device, 'vkQueuePresentKHR')
        present(self.ctx.present_queue, present_info)

    def cleanup(self):
        for fb in self.framebuffers:
            vk.vkDestroyFramebuffer(self.ctx.device, fb, None)
        self.framebuffers = []
        for view in self.image_views:
            vk.vkDestroyImageView(self.ctx.device, view, None)
        self.image_views = []
        if hasattr(self, 'swapchain') and self.swapchain is not None:
            destroy = vk.vkGetDeviceProcAddr(
                self.ctx.device, 'vkDestroySwapchainKHR')
            destroy(self.ctx.device, self.swapchain, None)
            self.swapchain = None
