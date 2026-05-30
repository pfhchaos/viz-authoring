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

import logging
import os
from pathlib import Path

import vulkan as vk

from . import buffer as _buffer

log = logging.getLogger(__name__)


# Device extensions we always request. VK_KHR_swapchain is needed for
# rendering to a surface; without it the device can't present.
REQUIRED_DEVICE_EXTENSIONS = ['VK_KHR_swapchain']


class VkContext:
    """Vulkan instance + device + queues. Holds the long-lived state.

    Instance extensions are determined by the caller (the surface code
    knows the WSI extensions it needs — VkContext just enables what it's
    asked to enable).
    """

    # Default cache path. Putting it under XDG_CACHE_HOME means a system
    # wipe of caches automatically clears stale data, and it doesn't
    # pollute the user's home dir. The "vulkan" subdir gives wallpaper_ml
    # (or other future Vulkan apps) a sibling spot to live without
    # stepping on each other.
    DEFAULT_PIPELINE_CACHE_PATH = (
        Path(os.environ.get('XDG_CACHE_HOME',
                             Path.home() / '.cache'))
        / 'flame-sheep' / 'vulkan' / 'pipeline_cache.bin'
    )

    def __init__(self, instance_extensions: list[str],
                 app_name: str = 'viz_authoring',
                 picker=None,
                 pipeline_cache_path: Path | None | str = 'default',
                 queue_priority: int | None = None):
        """queue_priority: optional VK_QUEUE_GLOBAL_PRIORITY_* value
        (LOW/MEDIUM/HIGH/REALTIME). When set, requires
        VK_KHR_global_priority on the device; the queue is created with
        that priority for cross-process arbitration via the DRM scheduler.
        Default None = use the driver's default priority (typically
        MEDIUM-ish).
        """
        """instance_extensions: list of instance extensions to enable.
        Typically includes VK_KHR_surface + a platform surface extension
        (VK_KHR_wayland_surface / VK_KHR_xlib_surface / ...).

        picker: optional callable(physical_device, surface_or_None) -> bool
        for selecting which physical device to use when multiple are
        present. If None, picks the first device that supports graphics
        + the required extensions. For surface-aware picking (graphics +
        present support), use VkContext.create_with_surface() after the
        surface exists.

        pipeline_cache_path: where to load/save the Vulkan pipeline
        cache. 'default' uses DEFAULT_PIPELINE_CACHE_PATH. None disables
        the cache entirely (useful for tests or when you want a clean
        compile every time). A Path uses that exact location.
        """
        self.instance = self._create_instance(instance_extensions, app_name)
        self.physical_device = None  # set by select_device()
        self.device = None
        self.graphics_queue = None
        self.graphics_queue_family = None
        self.pipeline_cache = None
        self.queue_priority = queue_priority
        if pipeline_cache_path == 'default':
            self._cache_path: Path | None = self.DEFAULT_PIPELINE_CACHE_PATH
        elif pipeline_cache_path is None:
            self._cache_path = None
        else:
            self._cache_path = Path(pipeline_cache_path)
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
            self._create_pipeline_cache()
            return

        raise RuntimeError(
            'No Vulkan device with required capabilities '
            f'(graphics{"+present" if surface is not None else ""} + '
            f'extensions {REQUIRED_DEVICE_EXTENSIONS})')

    def _create_device(self):
        families = {self.graphics_queue_family}
        if self.present_queue_family is not None:
            families.add(self.present_queue_family)

        # Optional pNext: global priority for cross-process DRM
        # scheduler arbitration. Only attached when caller opted in
        # via queue_priority — otherwise driver picks the default.
        gp_pnext = None
        if self.queue_priority is not None:
            gp_pnext = vk.VkDeviceQueueGlobalPriorityCreateInfoKHR(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
                globalPriority=self.queue_priority,
            )

        queue_create_infos = [
            vk.VkDeviceQueueCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                pNext=gp_pnext,
                queueFamilyIndex=fam,
                queueCount=1,
                pQueuePriorities=[1.0],
            )
            for fam in sorted(families)
        ]

        device_extensions = list(REQUIRED_DEVICE_EXTENSIONS)
        if self.queue_priority is not None:
            device_extensions.append(
                vk.VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME)

        device_create = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=len(queue_create_infos),
            pQueueCreateInfos=queue_create_infos,
            enabledExtensionCount=len(device_extensions),
            ppEnabledExtensionNames=device_extensions,
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

    def _create_pipeline_cache(self):
        """Build the VkPipelineCache, seeding from the on-disk file if
        present. The driver uses cached data to skip back-end pipeline
        compilation — the same shader → pipeline build that takes ~ms
        on first run drops to near-zero on subsequent runs.

        Cache data is GPU/driver-version specific. The driver tags it
        with a header that includes vendor+device IDs; if those don't
        match, vkCreatePipelineCache silently ignores the bad data.
        Stale-after-driver-upgrade is automatic.
        """
        if self._cache_path is None:
            self.pipeline_cache = vk.VK_NULL_HANDLE
            return
        initial_data = b''
        if self._cache_path.exists():
            try:
                initial_data = self._cache_path.read_bytes()
                log.debug(f'loaded pipeline cache: {len(initial_data)} bytes')
            except OSError as e:
                log.warning(f'failed to read {self._cache_path}: {e}')
        if initial_data:
            # vulkan-py's high-level wrapper rejects bytes for pInitialData
            # (expects a void* it can cast); allocate a cffi buffer with
            # the bytes copied in and pass that as the initial data ptr.
            data_buf = vk.ffi.new('uint8_t[]', initial_data)
            create = vk.VkPipelineCacheCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
                initialDataSize=len(initial_data),
                pInitialData=vk.ffi.cast('void*', data_buf),
            )
        else:
            create = vk.VkPipelineCacheCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
                initialDataSize=0,
            )
        self.pipeline_cache = vk.vkCreatePipelineCache(
            self.device, create, None)

    def save_pipeline_cache(self) -> bool:
        """Serialize the current cache to disk. Called from cleanup(),
        but also exposed so a long-running app can checkpoint manually
        without tearing down. Returns True on success.

        vulkan-py 1.3.275's high-level binding doesn't wrap
        vkGetPipelineCacheData; reach into vulkan.lib for the raw
        cffi function pointer and call the loader's two-step
        "query size, then read into buffer" protocol directly."""
        if self._cache_path is None or self.pipeline_cache in (
                None, vk.VK_NULL_HANDLE):
            return False
        try:
            import vulkan
            get_fn = vulkan.lib.vkGetPipelineCacheData
            # Two-call pattern: pass a size pointer + NULL data → driver
            # writes the required size; second call with that size gets
            # the bytes.
            size_p = vk.ffi.new('size_t*')
            get_fn(self.device, self.pipeline_cache, size_p, vk.ffi.NULL)
            size = int(size_p[0])
            if size == 0:
                return False
            buf = vk.ffi.new('uint8_t[]', size)
            get_fn(self.device, self.pipeline_cache, size_p,
                    vk.ffi.cast('void*', buf))
            data = bytes(vk.ffi.buffer(buf, size))
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(
                self._cache_path.suffix + '.tmp')
            tmp.write_bytes(data)
            tmp.replace(self._cache_path)
            log.debug(f'saved pipeline cache: {size} bytes to '
                       f'{self._cache_path}')
            return True
        except Exception as e:
            log.warning(f'failed to save pipeline cache: {e}')
            return False

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
            # Save the pipeline cache to disk BEFORE destroying the
            # cache object or device (vkGetPipelineCacheData needs both).
            self.save_pipeline_cache()
            if self.pipeline_cache not in (None, vk.VK_NULL_HANDLE):
                vk.vkDestroyPipelineCache(
                    self.device, self.pipeline_cache, None)
                self.pipeline_cache = None
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
