use crate::renderer::{HZB_SAMPLED_IMAGE_CAPACITY, HZB_STORAGE_IMAGE_CAPACITY};
use ash::vk;

pub(crate) struct Pipelines {
    pub(crate) global_set_layout: vk::DescriptorSetLayout,
    pub(crate) hzb_set_layout: vk::DescriptorSetLayout,
    pub(crate) frame_set_layout: vk::DescriptorSetLayout,
    pub(crate) overdraw_set_layout: vk::DescriptorSetLayout,

    pub(crate) frustum_cull_pipeline_layout: vk::PipelineLayout,
    pub(crate) frustum_cull_pipeline: vk::Pipeline,
    pub(crate) render_pipeline_layout: vk::PipelineLayout,
    pub(crate) render_pipeline: vk::Pipeline,
    pub(crate) overdraw_render_pipeline_layout: vk::PipelineLayout,
    pub(crate) overdraw_render_pipeline: vk::Pipeline,
    pub(crate) overshade_render_pipeline: vk::Pipeline,
    pub(crate) overdraw_resolve_pipeline_layout: vk::PipelineLayout,
    pub(crate) overdraw_resolve_pipeline: vk::Pipeline,
    pub(crate) build_hzb_pipeline_layout: vk::PipelineLayout,
    pub(crate) build_hzb_pipeline: vk::Pipeline,
    pub(crate) occlusion_cull_pipeline_layout: vk::PipelineLayout,
    pub(crate) occlusion_cull_pipeline: vk::Pipeline,
}

impl Pipelines {
    pub(crate) unsafe fn new(device: &ash::Device, surface_format: vk::SurfaceFormatKHR) -> Self {
        let global_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .push_next(&mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&[
                        vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                        vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                    ]))
                    .bindings(&[
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1024)
                            .stage_flags(vk::ShaderStageFlags::ALL),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(1024)
                            .stage_flags(vk::ShaderStageFlags::ALL),
                    ])
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL),
                None,
            )
            .unwrap();

        let hzb_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .push_next(&mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&[
                        vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                        vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                    ]))
                    .bindings(&[
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(HZB_SAMPLED_IMAGE_CAPACITY)
                            .stage_flags(vk::ShaderStageFlags::ALL),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(HZB_STORAGE_IMAGE_CAPACITY)
                            .stage_flags(vk::ShaderStageFlags::ALL),
                    ])
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL),
                None,
            )
            .unwrap();

        let frame_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .push_next(&mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&[
                        vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                    ]))
                    .bindings(&[vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::ALL)])
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL),
                None,
            )
            .unwrap();

        let overdraw_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .push_next(&mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&[
                        vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                        vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                        vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                    ]))
                    .bindings(&[
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::ALL),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::ALL),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(2)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::ALL),
                    ])
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL),
                None,
            )
            .unwrap();

        let create_shader_module = |src: &[u8]| {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        p_code: src.as_ptr() as _,
                        code_size: src.len(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };

        let (frustum_cull_pipeline, frustum_cull_pipeline_layout) = {
            let comp_shader = create_shader_module(include_bytes!("frustum_cull.comp.spirv"));

            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&[frame_set_layout]).push_constant_ranges(&[
                        vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .offset(0)
                            .size(std::mem::size_of::<u32>() as u32),
                    ]),
                    None,
                )
                .unwrap();

            let pipeline = device
                .create_compute_pipelines(
                    vk::PipelineCache::default(),
                    &[
                        vk::ComputePipelineCreateInfo::default().layout(pipeline_layout).stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .name(c"main")
                                .module(comp_shader),
                        ),
                    ],
                    None,
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap();

            device.destroy_shader_module(comp_shader, None);

            (pipeline, pipeline_layout)
        };

        let (render_pipeline, render_pipeline_layout) = {
            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&[global_set_layout, frame_set_layout]),
                    None,
                )
                .unwrap();

            let vert_shader = create_shader_module(include_bytes!("render.vert.spirv"));
            let frag_shader = create_shader_module(include_bytes!("render.frag.spirv"));

            let pipeline = device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .push_next(
                            &mut vk::PipelineRenderingCreateInfo::default()
                                .color_attachment_formats(&[surface_format.format])
                                .depth_attachment_format(vk::Format::D32_SFLOAT),
                        )
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .module(vert_shader)
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .name(c"main"),
                            vk::PipelineShaderStageCreateInfo::default()
                                .module(frag_shader)
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .name(c"main"),
                        ])
                        .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                                .primitive_restart_enable(false),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default()
                                .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .depth_clamp_enable(false)
                                .rasterizer_discard_enable(false)
                                .polygon_mode(vk::PolygonMode::FILL)
                                .line_width(1.0)
                                .cull_mode(vk::CullModeFlags::BACK)
                                .front_face(vk::FrontFace::CLOCKWISE)
                                .depth_bias_enable(false),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .sample_shading_enable(false)
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default().logic_op_enable(false).attachments(&[
                                vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                                    .blend_enable(false),
                            ]),
                        )
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::default()
                                .depth_test_enable(true)
                                .depth_write_enable(true)
                                .depth_compare_op(vk::CompareOp::GREATER),
                        )
                        .layout(pipeline_layout)],
                    None,
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap();

            device.destroy_shader_module(vert_shader, None);
            device.destroy_shader_module(frag_shader, None);

            (pipeline, pipeline_layout)
        };

        let (overdraw_render_pipeline, overshade_render_pipeline, overdraw_render_pipeline_layout) = {
            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&[global_set_layout, overdraw_set_layout]),
                    None,
                )
                .unwrap();

            let vert_shader = create_shader_module(include_bytes!("render.vert.spirv"));
            let frag_shader = create_shader_module(include_bytes!("overdraw.frag.spirv"));
            let overshade_frag_shader = create_shader_module(include_bytes!("overshade.frag.spirv"));

            let pipeline = create_overdraw_graphics_pipeline(device, pipeline_layout, vert_shader, frag_shader);
            let overshade_pipeline =
                create_overdraw_graphics_pipeline(device, pipeline_layout, vert_shader, overshade_frag_shader);

            device.destroy_shader_module(vert_shader, None);
            device.destroy_shader_module(frag_shader, None);
            device.destroy_shader_module(overshade_frag_shader, None);

            (pipeline, overshade_pipeline, pipeline_layout)
        };

        let (overdraw_resolve_pipeline, overdraw_resolve_pipeline_layout) = {
            let comp_shader = create_shader_module(include_bytes!("overdraw_resolve.comp.spirv"));

            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&[global_set_layout, overdraw_set_layout]),
                    None,
                )
                .unwrap();

            let pipeline = device
                .create_compute_pipelines(
                    vk::PipelineCache::default(),
                    &[
                        vk::ComputePipelineCreateInfo::default().layout(pipeline_layout).stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .name(c"main")
                                .module(comp_shader),
                        ),
                    ],
                    None,
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap();

            device.destroy_shader_module(comp_shader, None);

            (pipeline, pipeline_layout)
        };

        let (build_hzb_pipeline, build_hzb_pipeline_layout) = {
            let comp_shader = create_shader_module(include_bytes!("build_hzb.comp.spirv"));

            let pipeline_layout = device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[hzb_set_layout]), None)
                .unwrap();

            let pipeline = device
                .create_compute_pipelines(
                    vk::PipelineCache::default(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .flags(vk::PipelineCreateFlags::DISPATCH_BASE)
                        .layout(pipeline_layout)
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .name(c"main")
                                .module(comp_shader),
                        )],
                    None,
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap();

            device.destroy_shader_module(comp_shader, None);

            (pipeline, pipeline_layout)
        };

        let (occlusion_cull_pipeline, occlusion_cull_pipeline_layout) = {
            let comp_shader = create_shader_module(include_bytes!("occlusion_cull.comp.spirv"));

            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&[hzb_set_layout, frame_set_layout]),
                    None,
                )
                .unwrap();

            let pipeline = device
                .create_compute_pipelines(
                    vk::PipelineCache::default(),
                    &[
                        vk::ComputePipelineCreateInfo::default().layout(pipeline_layout).stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .name(c"main")
                                .module(comp_shader),
                        ),
                    ],
                    None,
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap();

            device.destroy_shader_module(comp_shader, None);

            (pipeline, pipeline_layout)
        };

        Self {
            global_set_layout,
            hzb_set_layout,
            frame_set_layout,
            overdraw_set_layout,
            frustum_cull_pipeline_layout,
            frustum_cull_pipeline,
            render_pipeline_layout,
            render_pipeline,
            overdraw_render_pipeline_layout,
            overdraw_render_pipeline,
            overshade_render_pipeline,
            overdraw_resolve_pipeline_layout,
            overdraw_resolve_pipeline,
            build_hzb_pipeline_layout,
            build_hzb_pipeline,
            occlusion_cull_pipeline_layout,
            occlusion_cull_pipeline,
        }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn free(self, device: &ash::Device) {
        device.destroy_pipeline(self.frustum_cull_pipeline, None);
        device.destroy_pipeline(self.render_pipeline, None);
        device.destroy_pipeline(self.overdraw_render_pipeline, None);
        device.destroy_pipeline(self.overshade_render_pipeline, None);
        device.destroy_pipeline(self.overdraw_resolve_pipeline, None);
        device.destroy_pipeline(self.build_hzb_pipeline, None);
        device.destroy_pipeline(self.occlusion_cull_pipeline, None);

        device.destroy_pipeline_layout(self.frustum_cull_pipeline_layout, None);
        device.destroy_pipeline_layout(self.render_pipeline_layout, None);
        device.destroy_pipeline_layout(self.overdraw_render_pipeline_layout, None);
        device.destroy_pipeline_layout(self.overdraw_resolve_pipeline_layout, None);
        device.destroy_pipeline_layout(self.build_hzb_pipeline_layout, None);
        device.destroy_pipeline_layout(self.occlusion_cull_pipeline_layout, None);

        device.destroy_descriptor_set_layout(self.global_set_layout, None);
        device.destroy_descriptor_set_layout(self.hzb_set_layout, None);
        device.destroy_descriptor_set_layout(self.frame_set_layout, None);
        device.destroy_descriptor_set_layout(self.overdraw_set_layout, None);
    }
}

unsafe fn create_overdraw_graphics_pipeline(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
    vert_shader: vk::ShaderModule,
    frag_shader: vk::ShaderModule,
) -> vk::Pipeline {
    device
        .create_graphics_pipelines(
            vk::PipelineCache::null(),
            &[vk::GraphicsPipelineCreateInfo::default()
                .push_next(
                    &mut vk::PipelineRenderingCreateInfo::default()
                        .color_attachment_formats(&[])
                        .depth_attachment_format(vk::Format::D32_SFLOAT),
                )
                .stages(&[
                    vk::PipelineShaderStageCreateInfo::default()
                        .module(vert_shader)
                        .stage(vk::ShaderStageFlags::VERTEX)
                        .name(c"main"),
                    vk::PipelineShaderStageCreateInfo::default()
                        .module(frag_shader)
                        .stage(vk::ShaderStageFlags::FRAGMENT)
                        .name(c"main"),
                ])
                .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                .input_assembly_state(
                    &vk::PipelineInputAssemblyStateCreateInfo::default()
                        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                        .primitive_restart_enable(false),
                )
                .viewport_state(&vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1))
                .dynamic_state(
                    &vk::PipelineDynamicStateCreateInfo::default()
                        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]),
                )
                .rasterization_state(
                    &vk::PipelineRasterizationStateCreateInfo::default()
                        .depth_clamp_enable(false)
                        .rasterizer_discard_enable(false)
                        .polygon_mode(vk::PolygonMode::FILL)
                        .line_width(1.0)
                        .cull_mode(vk::CullModeFlags::BACK)
                        .front_face(vk::FrontFace::CLOCKWISE)
                        .depth_bias_enable(false),
                )
                .multisample_state(
                    &vk::PipelineMultisampleStateCreateInfo::default()
                        .sample_shading_enable(false)
                        .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                )
                .color_blend_state(
                    &vk::PipelineColorBlendStateCreateInfo::default().logic_op_enable(false).attachments(&[]),
                )
                .depth_stencil_state(
                    &vk::PipelineDepthStencilStateCreateInfo::default()
                        .depth_test_enable(true)
                        .depth_write_enable(true)
                        .depth_compare_op(vk::CompareOp::GREATER),
                )
                .layout(pipeline_layout)],
            None,
        )
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
}
