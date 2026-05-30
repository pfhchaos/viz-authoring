"""GPU buffer management — wraps VkBuffer + VkDeviceMemory pair.

Mirrors the buffer subset of wallpaper_ml's VkCompute. The flame
renderer uses storage buffers (SSBOs) for histograms + walker state;
uniform buffers for genome params; staging buffers when uploading
from host memory. This module handles all three via the same VkBuffer
wrapper differentiated by `usage`.

Buffers are HOST_VISIBLE | HOST_COHERENT so we can upload/download
from numpy without explicit memory barriers — fine for the
wallpaper-sized data (a few MB) and simpler than the staging-buffer
pattern. If a particular buffer becomes a bottleneck, we can wrap
DEVICE_LOCAL versions of it later.
"""
from __future__ import annotations

import cffi

import numpy as np
import vulkan as vk

_FFI = cffi.FFI()


# Usage strings → Vulkan usage flags. The full set isn't enumerated;
# add entries when a new use case shows up.
_USAGE_FLAGS = {
    'storage': vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    'uniform': vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    'transfer_src': vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    'transfer_dst': vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    # Combined common case: storage buffer you can also transfer to/from.
    'storage_xfer': (vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                     | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                     | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT),
}


class VkBuffer:
    """A Vulkan buffer + its backing device memory.

    Created by VkContext.create_buffer. Holds raw handles plus the size
    so upload/download can shape numpy arrays correctly. Destroy via
    destroy() — also called automatically by VkContext.cleanup() for
    any buffers still alive at teardown.
    """
    __slots__ = ('buffer', 'memory', 'size', '_device')

    def __init__(self, buffer, memory, size, device):
        self.buffer = buffer
        self.memory = memory
        self.size = size
        self._device = device

    def destroy(self):
        if self.buffer is None:
            return
        vk.vkDestroyBuffer(self._device, self.buffer, None)
        vk.vkFreeMemory(self._device, self.memory, None)
        self.buffer = None
        self.memory = None


def find_memory_type(mem_props, type_filter: int, properties: int) -> int:
    """Pick the first memory type that satisfies `type_filter`'s mask
    AND has all the requested property flags. Raises if no suitable type
    exists (which would mean the GPU doesn't support the requested
    combination — rare but signals a bug worth surfacing loudly)."""
    for i in range(mem_props.memoryTypeCount):
        if (type_filter & (1 << i)) and \
           (mem_props.memoryTypes[i].propertyFlags & properties) == properties:
            return i
    raise RuntimeError(
        f'No memory type with filter={type_filter:#x} '
        f'props={properties:#x}')


def create_buffer(device, mem_props, size: int, usage: str = 'storage'
                  ) -> VkBuffer:
    """Allocate a HOST_VISIBLE | HOST_COHERENT buffer of `size` bytes.
    Pure function (no context state) so it's testable in isolation.

    See VkContext.create_buffer for the context-aware variant that
    tracks lifetime."""
    if usage not in _USAGE_FLAGS:
        raise ValueError(
            f'Unknown buffer usage {usage!r}. '
            f'Known: {sorted(_USAGE_FLAGS)}')
    usage_flags = _USAGE_FLAGS[usage]

    bc = vk.VkBufferCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=size, usage=usage_flags,
        sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
    )
    buf_handle = vk.vkCreateBuffer(device, bc, None)
    reqs = vk.vkGetBufferMemoryRequirements(device, buf_handle)
    mem_type = find_memory_type(
        mem_props, reqs.memoryTypeBits,
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )
    ai = vk.VkMemoryAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=reqs.size, memoryTypeIndex=mem_type,
    )
    mem = vk.vkAllocateMemory(device, ai, None)
    vk.vkBindBufferMemory(device, buf_handle, mem, 0)
    return VkBuffer(buf_handle, mem, size, device)


def upload(device, buf: VkBuffer, data: np.ndarray | bytes) -> None:
    """Copy host bytes into the buffer's memory. Buffer must be
    HOST_VISIBLE — true for everything create_buffer() produces.

    Uses cffi.FFI.memmove since vulkan-py's vkMapMemory returns a cffi
    pointer (not a raw integer); ctypes can't address it directly.
    Pattern matches wallpaper_ml.VkCompute.upload."""
    if isinstance(data, np.ndarray):
        data = data.tobytes()
    if len(data) > buf.size:
        raise ValueError(
            f'upload size {len(data)} exceeds buffer capacity {buf.size}')
    ptr = vk.vkMapMemory(device, buf.memory, 0, len(data), 0)
    _FFI.memmove(ptr, data, len(data))
    vk.vkUnmapMemory(device, buf.memory)


def download(device, buf: VkBuffer, dtype=np.float32,
             count: int | None = None) -> np.ndarray:
    """Read buffer contents back into a numpy array of `dtype`.
    If `count` is None, reads the whole buffer worth of `dtype`."""
    n_bytes = (buf.size if count is None
               else count * np.dtype(dtype).itemsize)
    ptr = vk.vkMapMemory(device, buf.memory, 0, n_bytes, 0)
    # vkMapMemory returns a cffi buffer object; bytes() materializes it.
    raw = bytes(ptr)[:n_bytes]
    vk.vkUnmapMemory(device, buf.memory)
    arr = np.frombuffer(raw, dtype=dtype)
    if count is not None:
        arr = arr[:count]
    return arr.copy()


def zero(device, buf: VkBuffer) -> None:
    """Fill the buffer with zeros (host-side, via upload)."""
    upload(device, buf, b'\x00' * buf.size)
