"""VkContext — instance + physical device + logical device + queues + command pool.

Long-lived state shared by everything else in the rendering stack.
Created once per application; Surface, Swapchain, Pipeline objects
all hold a reference to it.

Why graphics + present queues but not compute queue (for now): the
flame renderer will need compute too, but routing wallpaper_ml's
existing VkCompute and the new graphics path through ONE shared
instance/device is a bigger refactor (touches wallpaper_ml's instance
creation and lifecycle). Phase 1 keeps them separate; Phase N
unifies if/when the duplication starts hurting.
"""
from __future__ import annotations

import vulkan as vk


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
            for i, qf in enumerate(queue_families):
                if qf.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
                    if graphics_idx is None:
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

    def cleanup(self):
        """Destroy device + instance. Idempotent."""
        if self.device is not None:
            vk.vkDeviceWaitIdle(self.device)
            if hasattr(self, 'command_pool'):
                vk.vkDestroyCommandPool(self.device, self.command_pool, None)
            vk.vkDestroyDevice(self.device, None)
            self.device = None
        if self.instance is not None:
            vk.vkDestroyInstance(self.instance, None)
            self.instance = None
