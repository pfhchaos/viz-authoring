"""Graphics pipeline + render pass builders.

The flame renderer needs many distinct pipelines (one per shader pair)
all targeting the same kind of render pass (single color attachment to
swapchain). These helpers cut the boilerplate at each pipeline site so
the caller writes "I want a pipeline using these shaders" rather than
restating 200 lines of VkPipeline* structs.

What's deliberately NOT generic yet:
  - Depth/stencil attachments (the flame path is 2D, no depth).
  - Multiple subpasses (single subpass covers everything we need).
  - Push constants (will add when the first pipeline needs them).
  - Descriptor sets (same — added when we have a real consumer).
"""
from __future__ import annotations

from pathlib import Path

import vulkan as vk

from .shader import compile_shader


def make_color_attachment_render_pass(ctx, color_format: int):
    """Single-subpass render pass: clear → draw → present-src layout.

    For rendering directly into a swapchain image. The image starts in
    UNDEFINED layout (whatever the swapchain last had it as), gets
    cleared, drawn into, and ends in PRESENT_SRC_KHR ready for
    presentation. Subpass dependency makes the implicit image-acquire
    happen-before the color-attachment-output stage.
    """
    color_attachment = vk.VkAttachmentDescription(
        format=color_format,
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
        colorAttachmentCount=1, pColorAttachments=[color_ref],
    )
    dependency = vk.VkSubpassDependency(
        srcSubpass=vk.VK_SUBPASS_EXTERNAL, dstSubpass=0,
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
    return vk.vkCreateRenderPass(ctx.device, rp_create, None)


class GraphicsPipeline:
    """One graphics pipeline + its layout. Hands callers `pipeline` and
    `layout` for command buffer recording; owns destruction."""

    def __init__(self, ctx, render_pass,
                 vertex_shader_path: Path | str,
                 fragment_shader_path: Path | str,
                 extent: 'vk.VkExtent2D',
                 topology: int = vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                 cull_mode: int = vk.VK_CULL_MODE_NONE,
                 # No vertex buffer by default — shaders that bake their
                 # vertices into gl_VertexIndex (like the triangle demo)
                 # need empty vertex input. Callers with vertex buffers
                 # will pass binding/attribute descriptions when the
                 # first real one shows up.
                 ):
        self.ctx = ctx
        vert_spv = compile_shader(vertex_shader_path)
        frag_spv = compile_shader(fragment_shader_path)

        vert_module = vk.vkCreateShaderModule(ctx.device,
            vk.VkShaderModuleCreateInfo(codeSize=len(vert_spv),
                                          pCode=vert_spv), None)
        frag_module = vk.vkCreateShaderModule(ctx.device,
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
        vertex_input = vk.VkPipelineVertexInputStateCreateInfo(
            vertexBindingDescriptionCount=0,
            vertexAttributeDescriptionCount=0,
        )
        input_assembly = vk.VkPipelineInputAssemblyStateCreateInfo(
            topology=topology, primitiveRestartEnable=vk.VK_FALSE,
        )
        viewport = vk.VkViewport(
            x=0.0, y=0.0,
            width=float(extent.width), height=float(extent.height),
            minDepth=0.0, maxDepth=1.0,
        )
        scissor = vk.VkRect2D(
            offset=vk.VkOffset2D(x=0, y=0), extent=extent)
        viewport_state = vk.VkPipelineViewportStateCreateInfo(
            viewportCount=1, pViewports=[viewport],
            scissorCount=1, pScissors=[scissor],
        )
        rasterizer = vk.VkPipelineRasterizationStateCreateInfo(
            depthClampEnable=vk.VK_FALSE,
            rasterizerDiscardEnable=vk.VK_FALSE,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            lineWidth=1.0,
            cullMode=cull_mode,
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
        self.layout = vk.vkCreatePipelineLayout(ctx.device, pl_create, None)

        gp_create = vk.VkGraphicsPipelineCreateInfo(
            stageCount=2, pStages=stages,
            pVertexInputState=vertex_input,
            pInputAssemblyState=input_assembly,
            pViewportState=viewport_state,
            pRasterizationState=rasterizer,
            pMultisampleState=multisampling,
            pColorBlendState=color_blending,
            layout=self.layout,
            renderPass=render_pass,
            subpass=0,
        )
        self.pipeline = vk.vkCreateGraphicsPipelines(
            ctx.device, vk.VK_NULL_HANDLE, 1, [gp_create], None)[0]

        # Shader modules can be destroyed once the pipeline holds the SPIR-V.
        vk.vkDestroyShaderModule(ctx.device, vert_module, None)
        vk.vkDestroyShaderModule(ctx.device, frag_module, None)

    def cleanup(self):
        if self.pipeline is not None:
            vk.vkDestroyPipeline(self.ctx.device, self.pipeline, None)
            vk.vkDestroyPipelineLayout(self.ctx.device, self.layout, None)
            self.pipeline = None
            self.layout = None


# ---------------------------------------------------------------------------
# Compute pipeline — for the chaos-game / histogram / tonemap-prep shaders
# ---------------------------------------------------------------------------

class ComputePipeline:
    """A compute pipeline bound against a fixed set of storage buffers.

    Created with a list of buffers in binding order — descriptor set
    layout is generated to match. Optional push constants for per-
    dispatch scalar params (frame index, decay value, etc.).

    Lifecycle: built once per (shader, buffer-set, push-size) combo;
    re-used across many dispatches. Caller is responsible for cleanup()
    when done.
    """

    def __init__(self, ctx,
                 shader_path: 'Path | str',
                 buffers: list,
                 push_constant_size: int = 0,
                 source_transform=None):
        from .shader import compile_shader
        self.ctx = ctx

        spv = compile_shader(shader_path, source_transform=source_transform)
        module = vk.vkCreateShaderModule(
            ctx.device,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(spv), pCode=spv),
            None)

        # Descriptor set layout: one storage buffer per provided buffer,
        # bindings 0, 1, 2, ... in the order passed. Matches the
        # `layout(std430, binding = N)` declarations in the shader.
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for i in range(len(buffers))
        ]
        dsl_create = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings), pBindings=bindings,
        )
        self.descriptor_set_layout = vk.vkCreateDescriptorSetLayout(
            ctx.device, dsl_create, None)

        # Pipeline layout: descriptor set + optional push constants.
        push_ranges = []
        if push_constant_size > 0:
            push_ranges.append(vk.VkPushConstantRange(
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                offset=0, size=push_constant_size,
            ))
        pl_create = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1, pSetLayouts=[self.descriptor_set_layout],
            pushConstantRangeCount=len(push_ranges),
            pPushConstantRanges=push_ranges,
        )
        self.layout = vk.vkCreatePipelineLayout(ctx.device, pl_create, None)

        # Descriptor pool sized for one set with len(buffers) storage
        # bindings. Per-pipeline pool keeps lifetimes obvious; if we hit
        # pool-fragmentation issues later, switch to a shared pool.
        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=max(1, len(buffers)),
        )
        dp_create = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=1, pPoolSizes=[pool_size],
            maxSets=1,
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(
            ctx.device, dp_create, None)

        ds_alloc = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.descriptor_set_layout],
        )
        self.descriptor_set = vk.vkAllocateDescriptorSets(
            ctx.device, ds_alloc)[0]

        # Bind the buffers into the descriptor set.
        writes = []
        for i, buf in enumerate(buffers):
            bi = vk.VkDescriptorBufferInfo(
                buffer=buf.buffer, offset=0, range=buf.size)
            writes.append(vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.descriptor_set,
                dstBinding=i, dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[bi],
            ))
        vk.vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

        # Build the pipeline.
        stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=module, pName='main',
        )
        cp_create = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage, layout=self.layout,
        )
        self.pipeline = vk.vkCreateComputePipelines(
            ctx.device, vk.VK_NULL_HANDLE, 1, [cp_create], None)[0]

        vk.vkDestroyShaderModule(ctx.device, module, None)

    def cleanup(self):
        if self.pipeline is not None:
            vk.vkDestroyPipeline(self.ctx.device, self.pipeline, None)
            vk.vkDestroyPipelineLayout(self.ctx.device, self.layout, None)
            vk.vkDestroyDescriptorPool(
                self.ctx.device, self.descriptor_pool, None)
            vk.vkDestroyDescriptorSetLayout(
                self.ctx.device, self.descriptor_set_layout, None)
            self.pipeline = None
