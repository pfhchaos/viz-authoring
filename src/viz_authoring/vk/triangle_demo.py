"""Vulkan-on-Wayland triangle demo — Phase 0 risk validation.

Walks the full Vulkan WSI rendering loop end-to-end to confirm the
target hardware (Intel Arc A770 + Mesa Xe + Wayland compositor)
supports the graphics + presentation path we'll need to migrate the
flame renderer off moderngl. Compute already works (wallpaper_ml has
been shipping for weeks); this verifies the graphics + WSI half.

Layout choices intentionally simple:
  - No vertex buffer — vertices are baked into the vertex shader,
    indexed by gl_VertexIndex. Keeps the demo focused on lifecycle.
  - Single in-flight frame, no double-buffering pipeline. Adds a
    presentation stall but the goal here is correctness, not framerate.
  - GLFW for the window + surface, even though production will need
    layer-shell for the wallpaper. GLFW gets us to a triangle in
    minutes; the surface-creation half can be swapped later.
  - Cleanup is best-effort on exit — Wayland tears the surface down
    anyway and the demo is bounded.

Run as a module:
    python -m viz_authoring.vk.triangle_demo

When this works, Phase 1 starts extracting the reusable primitives
(instance, device picker, swapchain manager, render-pass builder,
pipeline builder) into proper modules.
"""
from __future__ import annotations

import ctypes
import subprocess
import tempfile
import time
from pathlib import Path

import glfw
import vulkan as vk

SHADER_DIR = Path(__file__).parent / 'shaders'


# ---------------------------------------------------------------------------
# Shader compilation — same glslc → SPIR-V path wallpaper_ml uses.
# Keeping this inline rather than depending on wallpaper_ml's compile_shader
# since this is supposed to stand on its own.
# ---------------------------------------------------------------------------

def compile_shader(source_path: Path, stage: str) -> bytes:
    """GLSL → SPIR-V binary. stage in {'vertex', 'fragment', 'compute', ...}."""
    with tempfile.NamedTemporaryFile(suffix='.spv', delete=False) as spv:
        spv_path = spv.name
    try:
        result = subprocess.run(
            ['glslc', f'-fshader-stage={stage}', str(source_path),
             '-o', spv_path],
            capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f'glslc failed for {source_path}:\n{result.stderr}')
        return Path(spv_path).read_bytes()
    finally:
        Path(spv_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main demo class — encapsulates the full Vulkan + WSI lifecycle.
# Single class rather than split modules so the lifecycle ordering is
# visible in one file. Splitting happens in Phase 1 when the patterns
# are confirmed.
# ---------------------------------------------------------------------------

class TriangleDemo:
    WIDTH = 800
    HEIGHT = 600

    def __init__(self):
        # GLFW window — give us a Wayland surface to render into.
        if not glfw.init():
            raise RuntimeError('glfw init failed')
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)  # no OpenGL context
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)    # skip swapchain rebuild
        self.window = glfw.create_window(
            self.WIDTH, self.HEIGHT, 'flame-sheep vulkan triangle', None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError('glfw create_window failed')

        self._create_instance()
        self._create_surface()
        self._select_physical_device()
        self._create_logical_device()
        self._create_swapchain()
        self._create_image_views()
        self._create_render_pass()
        self._create_graphics_pipeline()
        self._create_framebuffers()
        self._create_command_pool()
        self._create_command_buffers()
        self._create_sync_objects()

    # -----------------------------------------------------------------
    # Lifecycle steps in render-time order
    # -----------------------------------------------------------------

    def _create_instance(self):
        # GLFW tells us which instance extensions we need for surface
        # creation on this platform — on Wayland that's VK_KHR_surface
        # + VK_KHR_wayland_surface.
        glfw_exts = glfw.get_required_instance_extensions() or []
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName='flame-sheep triangle',
            applicationVersion=vk.VK_MAKE_VERSION(0, 1, 0),
            pEngineName='viz_authoring',
            engineVersion=vk.VK_MAKE_VERSION(0, 1, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledExtensionCount=len(glfw_exts),
            ppEnabledExtensionNames=glfw_exts,
            enabledLayerCount=0,
        )
        self.instance = vk.vkCreateInstance(create_info, None)

    def _create_surface(self):
        # GLFW returns the VkSurfaceKHR handle as a uint64 — we wrap it
        # in the vulkan-py opaque-handle type for downstream use.
        surface_handle = ctypes.c_uint64(0)
        # glfwCreateWindowSurface is the canonical cross-platform path —
        # the python-glfw wrapper exposes it under create_window_surface.
        result = glfw.create_window_surface(
            self.instance, self.window, None,
            ctypes.byref(surface_handle))
        if result != 0:
            raise RuntimeError(
                f'glfwCreateWindowSurface failed with VkResult={result}')
        self.surface = surface_handle.value

    def _select_physical_device(self):
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not devices:
            raise RuntimeError('No Vulkan devices found')

        # Need a queue family with BOTH graphics and surface-presentation
        # support (often the same family on integrated GPUs and Arc).
        # Also need the device to support the swapchain extension.
        get_surface_support = vk.vkGetInstanceProcAddr(
            self.instance, 'vkGetPhysicalDeviceSurfaceSupportKHR')

        for dev in devices:
            queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(dev)
            graphics_idx = None
            present_idx = None
            for i, qf in enumerate(queue_families):
                if qf.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
                    graphics_idx = i
                supports_present = vk.ffi.new('VkBool32*')
                get_surface_support(dev, i, self.surface, supports_present)
                if supports_present[0]:
                    present_idx = i
                if graphics_idx is not None and present_idx is not None:
                    break

            if graphics_idx is None or present_idx is None:
                continue

            # Check VK_KHR_swapchain extension support
            dev_exts = vk.vkEnumerateDeviceExtensionProperties(dev, None)
            ext_names = {e.extensionName for e in dev_exts}
            if 'VK_KHR_swapchain' not in ext_names:
                continue

            self.physical_device = dev
            self.graphics_queue_family = graphics_idx
            self.present_queue_family = present_idx
            props = vk.vkGetPhysicalDeviceProperties(dev)
            self.device_name = props.deviceName
            return

        raise RuntimeError(
            'No Vulkan device with graphics + present + swapchain support')

    def _create_logical_device(self):
        # One queue from each of (graphics, present) — usually the same
        # family, in which case create one queue not two.
        unique_families = list({self.graphics_queue_family,
                                 self.present_queue_family})
        queue_create_infos = [
            vk.VkDeviceQueueCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=fam,
                queueCount=1,
                pQueuePriorities=[1.0],
            )
            for fam in unique_families
        ]
        device_create = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=len(queue_create_infos),
            pQueueCreateInfos=queue_create_infos,
            enabledExtensionCount=1,
            ppEnabledExtensionNames=['VK_KHR_swapchain'],
            enabledLayerCount=0,
        )
        self.device = vk.vkCreateDevice(
            self.physical_device, device_create, None)
        self.graphics_queue = vk.vkGetDeviceQueue(
            self.device, self.graphics_queue_family, 0)
        self.present_queue = vk.vkGetDeviceQueue(
            self.device, self.present_queue_family, 0)

    def _create_swapchain(self):
        # Surface format / present mode / extent are queried via
        # KHR_surface instance-level functions, fetched dynamically.
        get_caps = vk.vkGetInstanceProcAddr(
            self.instance, 'vkGetPhysicalDeviceSurfaceCapabilitiesKHR')
        get_formats = vk.vkGetInstanceProcAddr(
            self.instance, 'vkGetPhysicalDeviceSurfaceFormatsKHR')
        get_present_modes = vk.vkGetInstanceProcAddr(
            self.instance, 'vkGetPhysicalDeviceSurfacePresentModesKHR')

        caps = get_caps(self.physical_device, self.surface)
        # vulkan-py returns these as generators — materialize to lists so
        # we can subscript / search them.
        formats = list(get_formats(self.physical_device, self.surface))
        present_modes = list(get_present_modes(
            self.physical_device, self.surface))

        # Pick BGRA8 SRGB if available — common, well-supported on Arc.
        chosen_format = formats[0]
        for f in formats:
            if (f.format == vk.VK_FORMAT_B8G8R8A8_SRGB
                    and f.colorSpace == vk.VK_COLORSPACE_SRGB_NONLINEAR_KHR):
                chosen_format = f
                break
        self.swapchain_format = chosen_format.format

        # On Wayland, the compositor doesn't pick the surface size — the
        # client does. Vulkan signals this by setting currentExtent to
        # 0xFFFFFFFF in both dimensions, in which case we substitute our
        # framebuffer size (clamped to min/max).
        WL_SENTINEL = 0xFFFFFFFF
        if (caps.currentExtent.width == WL_SENTINEL
                or caps.currentExtent.height == WL_SENTINEL):
            fb_w, fb_h = glfw.get_framebuffer_size(self.window)
            w = max(caps.minImageExtent.width,
                    min(caps.maxImageExtent.width, fb_w))
            h = max(caps.minImageExtent.height,
                    min(caps.maxImageExtent.height, fb_h))
            self.swapchain_extent = vk.VkExtent2D(width=w, height=h)
        else:
            self.swapchain_extent = caps.currentExtent

        # Prefer mailbox (low-latency triple-buffer); fall back to FIFO
        # which is mandated to be present.
        present_mode = vk.VK_PRESENT_MODE_FIFO_KHR
        if vk.VK_PRESENT_MODE_MAILBOX_KHR in present_modes:
            present_mode = vk.VK_PRESENT_MODE_MAILBOX_KHR

        # +1 image over minImageCount avoids waiting on the driver to
        # finish presenting before we can acquire the next image.
        image_count = caps.minImageCount + 1
        if caps.maxImageCount > 0 and image_count > caps.maxImageCount:
            image_count = caps.maxImageCount

        # Sharing mode: EXCLUSIVE if graphics and present are the same
        # family (almost always on Arc + Mesa); CONCURRENT otherwise to
        # avoid manual queue ownership transfers.
        if self.graphics_queue_family == self.present_queue_family:
            sharing_mode = vk.VK_SHARING_MODE_EXCLUSIVE
            queue_family_indices = []
        else:
            sharing_mode = vk.VK_SHARING_MODE_CONCURRENT
            queue_family_indices = [self.graphics_queue_family,
                                     self.present_queue_family]

        sc_create = vk.VkSwapchainCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            surface=self.surface,
            minImageCount=image_count,
            imageFormat=chosen_format.format,
            imageColorSpace=chosen_format.colorSpace,
            imageExtent=self.swapchain_extent,
            imageArrayLayers=1,
            imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            imageSharingMode=sharing_mode,
            queueFamilyIndexCount=len(queue_family_indices),
            pQueueFamilyIndices=queue_family_indices,
            preTransform=caps.currentTransform,
            compositeAlpha=vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=present_mode,
            clipped=vk.VK_TRUE,
        )
        create_swapchain = vk.vkGetDeviceProcAddr(
            self.device, 'vkCreateSwapchainKHR')
        self.swapchain = create_swapchain(self.device, sc_create, None)

        get_images = vk.vkGetDeviceProcAddr(
            self.device, 'vkGetSwapchainImagesKHR')
        self.swapchain_images = get_images(self.device, self.swapchain)

    def _create_image_views(self):
        self.image_views = []
        for img in self.swapchain_images:
            view_create = vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=img,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=self.swapchain_format,
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
                vk.vkCreateImageView(self.device, view_create, None))

    def _create_render_pass(self):
        # Single color attachment: clear at start, store at end. Goes
        # straight to PRESENT_SRC_KHR layout so we can hand it to the
        # swapchain after the render pass without an explicit transition.
        color_attachment = vk.VkAttachmentDescription(
            format=self.swapchain_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )
        color_ref = vk.VkAttachmentReference(
            attachment=0,
            layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[color_ref],
        )
        # Subpass dependency that makes the image acquire happen-before
        # the render pass's color-attachment-output stage. Without this,
        # we can read the image before the swapchain is done with it.
        dependency = vk.VkSubpassDependency(
            srcSubpass=vk.VK_SUBPASS_EXTERNAL,
            dstSubpass=0,
            srcStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask=0,
            dstStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        )
        rp_create = vk.VkRenderPassCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=1, pAttachments=[color_attachment],
            subpassCount=1, pSubpasses=[subpass],
            dependencyCount=1, pDependencies=[dependency],
        )
        self.render_pass = vk.vkCreateRenderPass(
            self.device, rp_create, None)

    def _create_graphics_pipeline(self):
        vert_spv = compile_shader(SHADER_DIR / 'triangle.vert', 'vertex')
        frag_spv = compile_shader(SHADER_DIR / 'triangle.frag', 'fragment')
        vert_module = vk.vkCreateShaderModule(self.device,
            vk.VkShaderModuleCreateInfo(codeSize=len(vert_spv),
                                          pCode=vert_spv), None)
        frag_module = vk.vkCreateShaderModule(self.device,
            vk.VkShaderModuleCreateInfo(codeSize=len(frag_spv),
                                          pCode=frag_spv), None)

        stages = [
            vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=vert_module, pName='main'),
            vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=frag_module, pName='main'),
        ]

        # No vertex buffer — vertices come from gl_VertexIndex in shader.
        vertex_input = vk.VkPipelineVertexInputStateCreateInfo(
            vertexBindingDescriptionCount=0,
            vertexAttributeDescriptionCount=0,
        )
        input_assembly = vk.VkPipelineInputAssemblyStateCreateInfo(
            topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=vk.VK_FALSE,
        )
        viewport = vk.VkViewport(
            x=0.0, y=0.0,
            width=float(self.swapchain_extent.width),
            height=float(self.swapchain_extent.height),
            minDepth=0.0, maxDepth=1.0,
        )
        scissor = vk.VkRect2D(
            offset=vk.VkOffset2D(x=0, y=0),
            extent=self.swapchain_extent,
        )
        viewport_state = vk.VkPipelineViewportStateCreateInfo(
            viewportCount=1, pViewports=[viewport],
            scissorCount=1, pScissors=[scissor],
        )
        rasterizer = vk.VkPipelineRasterizationStateCreateInfo(
            depthClampEnable=vk.VK_FALSE,
            rasterizerDiscardEnable=vk.VK_FALSE,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            lineWidth=1.0,
            cullMode=vk.VK_CULL_MODE_NONE,
            frontFace=vk.VK_FRONT_FACE_CLOCKWISE,
            depthBiasEnable=vk.VK_FALSE,
        )
        multisampling = vk.VkPipelineMultisampleStateCreateInfo(
            sampleShadingEnable=vk.VK_FALSE,
            rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
        )
        color_blend_attachment = vk.VkPipelineColorBlendAttachmentState(
            colorWriteMask=(vk.VK_COLOR_COMPONENT_R_BIT
                            | vk.VK_COLOR_COMPONENT_G_BIT
                            | vk.VK_COLOR_COMPONENT_B_BIT
                            | vk.VK_COLOR_COMPONENT_A_BIT),
            blendEnable=vk.VK_FALSE,
        )
        color_blending = vk.VkPipelineColorBlendStateCreateInfo(
            logicOpEnable=vk.VK_FALSE,
            attachmentCount=1, pAttachments=[color_blend_attachment],
        )

        pl_create = vk.VkPipelineLayoutCreateInfo(
            setLayoutCount=0, pushConstantRangeCount=0)
        self.pipeline_layout = vk.vkCreatePipelineLayout(
            self.device, pl_create, None)

        gp_create = vk.VkGraphicsPipelineCreateInfo(
            stageCount=2, pStages=stages,
            pVertexInputState=vertex_input,
            pInputAssemblyState=input_assembly,
            pViewportState=viewport_state,
            pRasterizationState=rasterizer,
            pMultisampleState=multisampling,
            pColorBlendState=color_blending,
            layout=self.pipeline_layout,
            renderPass=self.render_pass,
            subpass=0,
        )
        self.pipeline = vk.vkCreateGraphicsPipelines(
            self.device, vk.VK_NULL_HANDLE, 1, [gp_create], None)[0]

        # Modules can be destroyed once the pipeline is built.
        vk.vkDestroyShaderModule(self.device, vert_module, None)
        vk.vkDestroyShaderModule(self.device, frag_module, None)

    def _create_framebuffers(self):
        self.framebuffers = []
        for view in self.image_views:
            fb_create = vk.VkFramebufferCreateInfo(
                renderPass=self.render_pass,
                attachmentCount=1, pAttachments=[view],
                width=self.swapchain_extent.width,
                height=self.swapchain_extent.height,
                layers=1,
            )
            self.framebuffers.append(
                vk.vkCreateFramebuffer(self.device, fb_create, None))

    def _create_command_pool(self):
        cp_create = vk.VkCommandPoolCreateInfo(
            queueFamilyIndex=self.graphics_queue_family,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        )
        self.command_pool = vk.vkCreateCommandPool(
            self.device, cp_create, None)

    def _create_command_buffers(self):
        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(self.framebuffers),
        )
        self.command_buffers = vk.vkAllocateCommandBuffers(
            self.device, alloc_info)

        for i, cb in enumerate(self.command_buffers):
            begin = vk.VkCommandBufferBeginInfo()
            vk.vkBeginCommandBuffer(cb, begin)
            clear = vk.VkClearValue(
                color=vk.VkClearColorValue(float32=[0.05, 0.05, 0.10, 1.0]))
            rp_begin = vk.VkRenderPassBeginInfo(
                renderPass=self.render_pass,
                framebuffer=self.framebuffers[i],
                renderArea=vk.VkRect2D(
                    offset=vk.VkOffset2D(x=0, y=0),
                    extent=self.swapchain_extent),
                clearValueCount=1, pClearValues=[clear],
            )
            vk.vkCmdBeginRenderPass(cb, rp_begin,
                                      vk.VK_SUBPASS_CONTENTS_INLINE)
            vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                                   self.pipeline)
            vk.vkCmdDraw(cb, 3, 1, 0, 0)
            vk.vkCmdEndRenderPass(cb)
            vk.vkEndCommandBuffer(cb)

    def _create_sync_objects(self):
        # One semaphore pair + fence per concurrent frame. We only ever
        # have one in flight so single semaphores suffice.
        sem_create = vk.VkSemaphoreCreateInfo()
        self.image_available = vk.vkCreateSemaphore(self.device, sem_create, None)
        self.render_finished = vk.vkCreateSemaphore(self.device, sem_create, None)
        fence_create = vk.VkFenceCreateInfo(
            flags=vk.VK_FENCE_CREATE_SIGNALED_BIT)
        self.in_flight = vk.vkCreateFence(self.device, fence_create, None)

    # -----------------------------------------------------------------
    # Render loop
    # -----------------------------------------------------------------

    def _draw_frame(self):
        vk.vkWaitForFences(self.device, 1, [self.in_flight], vk.VK_TRUE,
                            0xFFFFFFFFFFFFFFFF)
        vk.vkResetFences(self.device, 1, [self.in_flight])

        acquire = vk.vkGetDeviceProcAddr(
            self.device, 'vkAcquireNextImageKHR')
        image_index = acquire(self.device, self.swapchain,
                               0xFFFFFFFFFFFFFFFF,
                               self.image_available, vk.VK_NULL_HANDLE)

        submit = vk.VkSubmitInfo(
            waitSemaphoreCount=1, pWaitSemaphores=[self.image_available],
            pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffers[image_index]],
            signalSemaphoreCount=1, pSignalSemaphores=[self.render_finished],
        )
        vk.vkQueueSubmit(self.graphics_queue, 1, [submit], self.in_flight)

        present_info = vk.VkPresentInfoKHR(
            waitSemaphoreCount=1, pWaitSemaphores=[self.render_finished],
            swapchainCount=1, pSwapchains=[self.swapchain],
            pImageIndices=[image_index],
        )
        present = vk.vkGetDeviceProcAddr(
            self.device, 'vkQueuePresentKHR')
        present(self.present_queue, present_info)

    def run(self, max_frames: int | None = None):
        frame = 0
        t0 = time.monotonic()
        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                self._draw_frame()
                frame += 1
                if max_frames is not None and frame >= max_frames:
                    break
        finally:
            vk.vkDeviceWaitIdle(self.device)
            dt = time.monotonic() - t0
            if frame > 0:
                print(f'rendered {frame} frames in {dt:.2f}s '
                      f'({frame/dt:.1f} fps)')

    # -----------------------------------------------------------------
    # Cleanup — best effort; Wayland tears down the surface on exit.
    # -----------------------------------------------------------------

    def cleanup(self):
        if hasattr(self, 'device'):
            vk.vkDeviceWaitIdle(self.device)
            for fb in getattr(self, 'framebuffers', []):
                vk.vkDestroyFramebuffer(self.device, fb, None)
            for view in getattr(self, 'image_views', []):
                vk.vkDestroyImageView(self.device, view, None)
            for obj_attr in ('image_available', 'render_finished'):
                if hasattr(self, obj_attr):
                    vk.vkDestroySemaphore(self.device, getattr(self, obj_attr), None)
            if hasattr(self, 'in_flight'):
                vk.vkDestroyFence(self.device, self.in_flight, None)
            if hasattr(self, 'command_pool'):
                vk.vkDestroyCommandPool(self.device, self.command_pool, None)
            if hasattr(self, 'pipeline'):
                vk.vkDestroyPipeline(self.device, self.pipeline, None)
            if hasattr(self, 'pipeline_layout'):
                vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
            if hasattr(self, 'render_pass'):
                vk.vkDestroyRenderPass(self.device, self.render_pass, None)
            if hasattr(self, 'swapchain'):
                destroy_sc = vk.vkGetDeviceProcAddr(
                    self.device, 'vkDestroySwapchainKHR')
                destroy_sc(self.device, self.swapchain, None)
            vk.vkDestroyDevice(self.device, None)
        if hasattr(self, 'surface') and hasattr(self, 'instance'):
            destroy_surface = vk.vkGetInstanceProcAddr(
                self.instance, 'vkDestroySurfaceKHR')
            destroy_surface(self.instance, self.surface, None)
        if hasattr(self, 'instance'):
            vk.vkDestroyInstance(self.instance, None)
        if hasattr(self, 'window') and self.window is not None:
            glfw.destroy_window(self.window)
        glfw.terminate()


def main():
    demo = TriangleDemo()
    print(f'device: {demo.device_name}')
    print(f'swapchain: {demo.swapchain_extent.width}x'
          f'{demo.swapchain_extent.height}, '
          f'{len(demo.swapchain_images)} images, format={demo.swapchain_format}')
    print('rendering... close the window to exit')
    try:
        demo.run()
    finally:
        demo.cleanup()


if __name__ == '__main__':
    main()
