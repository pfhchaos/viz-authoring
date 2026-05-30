"""VkContext — instance + physical device + logical device + queues + command pool.

Long-lived state shared by everything else in the rendering stack.
Created once per application; Surface, Swapchain, Pipeline, buffer
allocations all hold a reference to it.

Supports both graphics + compute via a single queue family — Intel
Arc and most discrete GPUs expose a queue family with both
GRAPHICS_BIT and COMPUTE_BIT (and TRANSFER_BIT), so we don't need
separate compute queues. The graphics_queue handles compute dispatches
too. If a future hardware setup splits them, add a compute_queue_family
+ separate queue selection path.

Separate from wallpaper_ml's VkCompute (used in standalone training
processes). The flame renderer process uses this VkContext for
everything; the training tools keep their own.
"""
from __future__ import annotations

import vulkan as vk

from . import buffer as _buffer


# Device extensions we always request. VK_KHR_swapchain is needed for
# rendering to a surface; without it the device can't present.
REQUIRED_DEVICE_EXTENSIONS = ['VK_KHR_swapchain']


class VkContext:
    """Vulkan instance + device + queues. Holds the long-lived state.

    Instance extensions are determined by the caller (the surface code
    knows the WSI extensions it needs — VkContext just enables what it's
    asked to enable).
    """

    def __init__(self, instance_extensions: list[str],
                 app_name: str = 'viz_authoring',
                 picker=None):
        """instance_extensions: list of instance extensions to enable.
        Typically includes VK_KHR_surface + a platform surface extension
        (VK_KHR_wayland_surface / VK_KHR_xlib_surface / ...).

        picker: optional callable(physical_device, surface_or_None) -> bool
        for selecting which physical device to use when multiple are
        present. If None, picks the first device that supports graphics
        + the required extensions. For surface-aware picking (graphics +
        present support), use VkContext.create_with_surface() after the
        surface exists.
        """
        self.instance = self._create_instance(instance_extensions, app_name)
        self.physical_device = None  # set by select_device()
        self.device = None
        self.graphics_queue = None
        self.graphics_queue_family = None
        # populated by select_device():
        self._picker = picker

    @staticmethod
    def _create_instance(extensions: list[str], app_name: str):
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=app_name,
            applicationVersion=vk.VK_MAKE_VERSION(0, 1, 0),
            pEngineName='viz_authoring',
            engineVersion=vk.VK_MAKE_VERSION(0, 1, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
            enabledLayerCount=0,
        )
        return vk.vkCreateInstance(create_info, None)

    def select_device(self, surface: int | None = None,
                       present_queue_family: int | None = None):
        """Pick a physical device and create the logical device.

        If `surface` is given, requires a queue family that supports
        present on that surface; also returns the present queue family
        index via the `present_queue_family` attribute. If `surface` is
        None, no present queue is required (compute-only or offscreen).

        After this returns, the context has:
            self.physical_device, self.device, self.graphics_queue,
            self.graphics_queue_family, self.present_queue (if surface),
            self.present_queue_family (if surface), self.device_name.
        """
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not devices:
            raise RuntimeError('No Vulkan devices found')

        get_surface_support = None
        if surface is not None:
            get_surface_support = vk.vkGetInstanceProcAddr(
                self.instance, 'vkGetPhysicalDeviceSurfaceSupportKHR')

        for dev in devices:
            queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(dev)
            graphics_idx = None
            present_idx = None
            # We require GRAPHICS+COMPUTE in the same family — Intel/
            # AMD/Nvidia all expose at least one such family. Saves
            # the cross-queue synchronization complexity of running
            # compute on a separate queue.
            for i, qf in enumerate(queue_families):
                has_graphics = bool(qf.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT)
                has_compute = bool(qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT)
                if has_graphics and has_compute and graphics_idx is None:
                    graphics_idx = i
                if surface is not None:
                    supports_present = vk.ffi.new('VkBool32*')
                    get_surface_support(dev, i, surface, supports_present)
                    if supports_present[0] and present_idx is None:
                        present_idx = i

            if graphics_idx is None:
                continue
            if surface is not None and present_idx is None:
                continue

            # Check required device extensions
            dev_exts = vk.vkEnumerateDeviceExtensionProperties(dev, None)
            ext_names = {e.extensionName for e in dev_exts}
            if not all(ext in ext_names for ext in REQUIRED_DEVICE_EXTENSIONS):
                continue

            # Optional caller filter
            if self._picker is not None and not self._picker(dev, surface):
                continue

            self.physical_device = dev
            self.graphics_queue_family = graphics_idx
            self.present_queue_family = present_idx
            self.mem_props = vk.vkGetPhysicalDeviceMemoryProperties(dev)
            props = vk.vkGetPhysicalDeviceProperties(dev)
            self.device_name = props.deviceName
            self._create_device()
            self._create_command_pool()
            return

        raise RuntimeError(
            'No Vulkan device with required capabilities '
            f'(graphics{"+present" if surface is not None else ""} + '
            f'extensions {REQUIRED_DEVICE_EXTENSIONS})')

    def _create_device(self):
        families = {self.graphics_queue_family}
        if self.present_queue_family is not None:
            families.add(self.present_queue_family)
        queue_create_infos = [
            vk.VkDeviceQueueCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=fam,
                queueCount=1,
                pQueuePriorities=[1.0],
            )
            for fam in sorted(families)
        ]
        device_create = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=len(queue_create_infos),
            pQueueCreateInfos=queue_create_infos,
            enabledExtensionCount=len(REQUIRED_DEVICE_EXTENSIONS),
            ppEnabledExtensionNames=REQUIRED_DEVICE_EXTENSIONS,
            enabledLayerCount=0,
        )
        self.device = vk.vkCreateDevice(
            self.physical_device, device_create, None)
        self.graphics_queue = vk.vkGetDeviceQueue(
            self.device, self.graphics_queue_family, 0)
        if self.present_queue_family is not None:
            self.present_queue = vk.vkGetDeviceQueue(
                self.device, self.present_queue_family, 0)
        else:
            self.present_queue = None

    def _create_command_pool(self):
        # Pool flags: RESET_COMMAND_BUFFER_BIT so we can re-record
        # individual buffers without recreating the pool. Matches
        # wallpaper_ml's VkCompute pattern.
        cp_create = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.graphics_queue_family,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        )
        self.command_pool = vk.vkCreateCommandPool(
            self.device, cp_create, None)

    # -----------------------------------------------------------------
    # Helpers commonly needed by code that builds on this context
    # -----------------------------------------------------------------

    def allocate_command_buffers(self, count: int):
        """Allocate `count` primary command buffers from the pool."""
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=count,
        )
        return vk.vkAllocateCommandBuffers(self.device, alloc_info)

    # ----- Buffer management (delegates to viz_authoring.vk.buffer) -----
    #
    # Context-aware wrappers around the pure-functional create_buffer/
    # upload/download helpers — tracks live buffers so cleanup() can
    # auto-destroy stragglers and so the caller doesn't have to thread
    # the device handle through every call.

    def create_buffer(self, size: int, usage: str = 'storage') -> _buffer.VkBuffer:
        """Allocate a HOST_VISIBLE | HOST_COHERENT buffer. Tracked for
        auto-cleanup at context teardown."""
        if not hasattr(self, '_live_buffers'):
            self._live_buffers = []
        buf = _buffer.create_buffer(self.device, self.mem_props, size, usage)
        self._live_buffers.append(buf)
        return buf

    def upload(self, buf: _buffer.VkBuffer, data) -> None:
        _buffer.upload(self.device, buf, data)

    def download(self, buf: _buffer.VkBuffer, dtype=None, count=None):
        import numpy as np
        if dtype is None:
            dtype = np.float32
        return _buffer.download(self.device, buf, dtype, count)

    def zero_buffer(self, buf: _buffer.VkBuffer) -> None:
        _buffer.zero(self.device, buf)

    # ----- Compute dispatch -----

    def dispatch_compute(self, pipeline, groups_x: int,
                          groups_y: int = 1, groups_z: int = 1,
                          push_constants: bytes | None = None):
        """One-shot compute dispatch: record, submit, wait. Synchronous.

        For per-frame command-buffer-reused patterns, build the command
        buffer directly via allocate_command_buffers — this helper is
        for the common "fire-and-forget compute step" case (matches
        wallpaper_ml.VkCompute.dispatch's shape so callers can move
        between the two)."""
        import cffi
        ffi = cffi.FFI()

        cb = self.allocate_command_buffers(1)[0]
        vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        ))
        vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipeline.pipeline)
        vk.vkCmdBindDescriptorSets(
            cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout, 0, 1, [pipeline.descriptor_set], 0, None)
        if push_constants is not None:
            # vulkan-py wants a typed pointer for the push-constant
            # blob — pass a freshly-allocated char[] + cast to void*.
            pc_ptr = ffi.new('char[]', push_constants)
            vk.vkCmdPushConstants(
                cb, pipeline.layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0, len(push_constants),
                ffi.cast('void*', pc_ptr))
        vk.vkCmdDispatch(cb, groups_x, groups_y, groups_z)
        vk.vkEndCommandBuffer(cb)

        submit = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1, pCommandBuffers=[cb],
        )
        fence = vk.vkCreateFence(
            self.device, vk.VkFenceCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO), None)
        vk.vkQueueSubmit(self.graphics_queue, 1, [submit], fence)
        vk.vkWaitForFences(self.device, 1, [fence], vk.VK_TRUE,
                            0xFFFFFFFFFFFFFFFF)
        vk.vkDestroyFence(self.device, fence, None)
        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [cb])

    def cleanup(self):
        """Destroy device + instance. Idempotent."""
        if self.device is not None:
            vk.vkDeviceWaitIdle(self.device)
            # Auto-destroy any live buffers the caller forgot.
            for buf in getattr(self, '_live_buffers', []):
                buf.destroy()
            self._live_buffers = []
            if hasattr(self, 'command_pool'):
                vk.vkDestroyCommandPool(self.device, self.command_pool, None)
            vk.vkDestroyDevice(self.device, None)
            self.device = None
        if self.instance is not None:
            vk.vkDestroyInstance(self.instance, None)
            self.instance = None
