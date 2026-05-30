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
