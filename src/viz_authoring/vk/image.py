"""VkImage + VkSampler wrappers — minimal subset for fragment-stage
texture sampling.

What's here: 2D RGBA8 images created HOST_VISIBLE | HOST_COHERENT so
we can upload pixel data without a staging-buffer detour. Same
simplification wallpaper_ml made for its compute buffers — fine for
the kilobyte-sized palette textures we need, would need rework for
hi-res sampled inputs (audio FFT spectrograms etc.) where a
DEVICE_LOCAL + staging path performs better.

What's not here yet:
  - Mipmaps, multi-array, multi-sample images
  - Cube maps, 3D textures
  - Format other than RGBA8
  - Asynchronous upload via staging
These show up when needed; not in scope for Phase 3.5 palette work.
"""
from __future__ import annotations

import cffi

import numpy as np
import vulkan as vk

from .buffer import find_memory_type

_FFI = cffi.FFI()


class VkImage:
    """A 2D image + its backing device memory + a default 2D image view.

    The view is created with the same format as the image and standard
    color-aspect / single-mip / single-layer subresource range. Most
    fragment-shader sampling consumers want exactly this view; a future
    advanced consumer can build its own view from self.image.
    """
    __slots__ = ('image', 'memory', 'view', 'format', 'width', 'height',
                 '_device')

    def __init__(self, image, memory, view, fmt, width, height, device):
        self.image = image
        self.memory = memory
        self.view = view
        self.format = fmt
        self.width = width
        self.height = height
        self._device = device

    def destroy(self):
        if self.image is None:
            return
        vk.vkDestroyImageView(self._device, self.view, None)
        vk.vkDestroyImage(self._device, self.image, None)
        vk.vkFreeMemory(self._device, self.memory, None)
        self.image = None
        self.memory = None
        self.view = None


def create_color_attachment_image(ctx, width: int, height: int) -> VkImage:
    """DEVICE_LOCAL RGBA8 image suitable for being a render target AND
    being sampled by a later pass. Layout starts in UNDEFINED — the
    render pass transitions it on first use. Final layout (after the
    render pass that writes it) should be SHADER_READ_ONLY_OPTIMAL so
    the next pipeline can sample it without an extra barrier.

    Differs from create_image_rgba8 (host-visible, sampled-only) in two
    ways:
      - Usage flags include COLOR_ATTACHMENT_BIT so the image can be a
        framebuffer attachment
      - Memory is DEVICE_LOCAL (GPU-only) — host can't read/write, but
        bandwidth from the GPU's perspective is way higher. Used here
        for intermediate render targets that never touch the CPU.
    """
    img_create = vk.VkImageCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        imageType=vk.VK_IMAGE_TYPE_2D,
        format=vk.VK_FORMAT_R8G8B8A8_UNORM,
        extent=vk.VkExtent3D(width=width, height=height, depth=1),
        mipLevels=1, arrayLayers=1,
        samples=vk.VK_SAMPLE_COUNT_1_BIT,
        tiling=vk.VK_IMAGE_TILING_OPTIMAL,
        usage=(vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
               | vk.VK_IMAGE_USAGE_SAMPLED_BIT),
        sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
    )
    image = vk.vkCreateImage(ctx.device, img_create, None)
    reqs = vk.vkGetImageMemoryRequirements(ctx.device, image)
    mem_type = find_memory_type(
        ctx.mem_props, reqs.memoryTypeBits,
        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    ai = vk.VkMemoryAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=reqs.size, memoryTypeIndex=mem_type,
    )
    memory = vk.vkAllocateMemory(ctx.device, ai, None)
    vk.vkBindImageMemory(ctx.device, image, memory, 0)

    view_create = vk.VkImageViewCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        image=image,
        viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
        format=vk.VK_FORMAT_R8G8B8A8_UNORM,
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
    view = vk.vkCreateImageView(ctx.device, view_create, None)
    return VkImage(image, memory, view,
                    vk.VK_FORMAT_R8G8B8A8_UNORM, width, height, ctx.device)


def create_image_rgba8(ctx, width: int, height: int) -> VkImage:
    """Create a HOST_VISIBLE RGBA8 image suitable for fragment-stage
    sampling. Layout starts in GENERAL — the host can write to it
    directly and the shader can sample it without an extra transition.
    Trades the optimal SHADER_READ_ONLY_OPTIMAL layout's potential
    perf win for upload simplicity, which is the right call for tiny
    palette textures."""
    img_create = vk.VkImageCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        imageType=vk.VK_IMAGE_TYPE_2D,
        format=vk.VK_FORMAT_R8G8B8A8_UNORM,
        extent=vk.VkExtent3D(width=width, height=height, depth=1),
        mipLevels=1, arrayLayers=1,
        samples=vk.VK_SAMPLE_COUNT_1_BIT,
        tiling=vk.VK_IMAGE_TILING_LINEAR,
        usage=vk.VK_IMAGE_USAGE_SAMPLED_BIT,
        sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        initialLayout=vk.VK_IMAGE_LAYOUT_PREINITIALIZED,
    )
    image = vk.vkCreateImage(ctx.device, img_create, None)
    reqs = vk.vkGetImageMemoryRequirements(ctx.device, image)
    mem_type = find_memory_type(
        ctx.mem_props, reqs.memoryTypeBits,
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )
    ai = vk.VkMemoryAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=reqs.size, memoryTypeIndex=mem_type,
    )
    memory = vk.vkAllocateMemory(ctx.device, ai, None)
    vk.vkBindImageMemory(ctx.device, image, memory, 0)

    view_create = vk.VkImageViewCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        image=image,
        viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
        format=vk.VK_FORMAT_R8G8B8A8_UNORM,
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
    view = vk.vkCreateImageView(ctx.device, view_create, None)

    # The image starts in PREINITIALIZED — host writes are visible to
    # the GPU as long as the memory is HOST_COHERENT. We need to
    # transition to GENERAL for the shader to sample. Layout
    # transitions normally need a one-time command buffer + barrier;
    # for the tiny palette case we do it inline here so the caller
    # doesn't have to think about it.
    _transition_image_layout(
        ctx, image,
        old_layout=vk.VK_IMAGE_LAYOUT_PREINITIALIZED,
        new_layout=vk.VK_IMAGE_LAYOUT_GENERAL)

    return VkImage(image, memory, view,
                    vk.VK_FORMAT_R8G8B8A8_UNORM, width, height, ctx.device)


def _transition_image_layout(ctx, image, *, old_layout: int, new_layout: int):
    """One-shot synchronous image layout transition. Allocates a command
    buffer, records a pipeline barrier, submits + waits."""
    cb = ctx.allocate_command_buffers(1)[0]
    vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo(
        sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    ))
    barrier = vk.VkImageMemoryBarrier(
        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        oldLayout=old_layout, newLayout=new_layout,
        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
        image=image,
        subresourceRange=vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1,
            baseArrayLayer=0, layerCount=1,
        ),
        srcAccessMask=vk.VK_ACCESS_HOST_WRITE_BIT,
        dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
    )
    vk.vkCmdPipelineBarrier(
        cb,
        vk.VK_PIPELINE_STAGE_HOST_BIT,
        vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, None, 0, None, 1, [barrier])
    vk.vkEndCommandBuffer(cb)
    submit = vk.VkSubmitInfo(
        sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        commandBufferCount=1, pCommandBuffers=[cb])
    fence = vk.vkCreateFence(ctx.device,
        vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO),
        None)
    vk.vkQueueSubmit(ctx.graphics_queue, 1, [submit], fence)
    vk.vkWaitForFences(ctx.device, 1, [fence], vk.VK_TRUE,
                        0xFFFFFFFFFFFFFFFF)
    vk.vkDestroyFence(ctx.device, fence, None)
    vk.vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, [cb])


def upload_image_rgba8(ctx, img: VkImage, rgba: np.ndarray):
    """Copy host RGBA8 pixel data into the image's memory. `rgba` must
    be shape (H, W, 4) uint8."""
    if rgba.shape != (img.height, img.width, 4) or rgba.dtype != np.uint8:
        raise ValueError(
            f'rgba must be ({img.height},{img.width},4) uint8, '
            f'got shape={rgba.shape} dtype={rgba.dtype}')
    # Query the row pitch so we copy with the right stride. Linear-
    # tiling images can have padding between rows; on most desktop
    # GPUs the row pitch equals width*4 but we shouldn't assume.
    sub = vk.VkImageSubresource(
        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
        mipLevel=0, arrayLayer=0)
    layout = vk.vkGetImageSubresourceLayout(ctx.device, img.image, sub)
    pitch = layout.rowPitch
    ptr = vk.vkMapMemory(ctx.device, img.memory, 0, layout.size, 0)
    data_bytes = rgba.tobytes()
    if pitch == img.width * 4:
        _FFI.memmove(ptr, data_bytes, len(data_bytes))
    else:
        # Per-row copy when the GPU added padding
        row_bytes = img.width * 4
        for y in range(img.height):
            src_off = y * row_bytes
            row_ptr = _FFI.cast('char*', ptr) + y * pitch
            _FFI.memmove(row_ptr, data_bytes[src_off:src_off + row_bytes],
                          row_bytes)
    vk.vkUnmapMemory(ctx.device, img.memory)


class VkSampler:
    """Wraps a VkSampler. One sampler typically gets reused across many
    images — the (image, sampler) pair is what a COMBINED_IMAGE_SAMPLER
    descriptor binds, but sampler state (filter modes, address modes)
    is the shader's concern, not the image's."""

    def __init__(self, sampler, device):
        self.sampler = sampler
        self._device = device

    def destroy(self):
        if self.sampler is not None:
            vk.vkDestroySampler(self._device, self.sampler, None)
            self.sampler = None


def create_linear_sampler(ctx) -> VkSampler:
    """Standard bilinear-filtered, clamp-to-edge sampler. Fits the
    palette-lookup case (1D color ramp sampled in fragment shader)
    and most other "look up a texture without surprises" use cases."""
    create = vk.VkSamplerCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        magFilter=vk.VK_FILTER_LINEAR,
        minFilter=vk.VK_FILTER_LINEAR,
        mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_LINEAR,
        addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        anisotropyEnable=vk.VK_FALSE,
        maxAnisotropy=1.0,
        borderColor=vk.VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
        unnormalizedCoordinates=vk.VK_FALSE,
        compareEnable=vk.VK_FALSE,
        compareOp=vk.VK_COMPARE_OP_ALWAYS,
        mipLodBias=0.0, minLod=0.0, maxLod=0.0,
    )
    sampler = vk.vkCreateSampler(ctx.device, create, None)
    return VkSampler(sampler, ctx.device)
