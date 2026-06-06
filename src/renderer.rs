use crate::buffer::Buffer;
use crate::core::Core;
use crate::glsl_types::*;
use crate::mesh::{Meshlet, load_mesh};
use crate::staging::StagingBuffer;
use crate::swapchain::Swapchain;
use ash::vk;
use glam::*;
use std::collections::HashMap;
use std::ffi::CStr;
use std::path::Path;
use std::path::PathBuf;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub struct MeshHandle(u32);
#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub struct ObjectHandle(u32);

#[derive(Copy, Clone, Default, Debug)]
struct FrameSyncPrimitives {
    image_available: vk::Semaphore,
    frame_in_flight: vk::Fence,
}

#[derive(Debug)]
pub(super) struct ObjectInstance {
    pub mesh: MeshHandle,
    pub position: Vec3,
    pub scale: f32,
    pub orientation: Quat,
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

/*
Plan:
0) Data upload
1) frustum_cull
2) render (only visible)
3) build_hzb
4) occlusion_cull
5) render (new vibile)
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum PipelineStage {
    DataUpload,
    FrustumCull,
    FirstDraw,
    BuildHzb,
    COUNT,
}

const MAX_PIPELINE_STAGES: usize = PipelineStage::COUNT as usize;

struct Scene {
    /* Images: */
    // Note: hzb mip 0 is half depth resolution.
    hzb_image: vk::Image, // ( TODO: make visiblity less serial )
    _hzb_alloc: vk_mem::Allocation,
    _hzb_test_view: vk::ImageView,
    hzb_build_src_views: Box<[vk::ImageView]>,
    hzb_build_dst_views: Box<[vk::ImageView]>,

    /* Various buffers. */
    visibility_buffer: Buffer<u32>, // per meshlet
    index_buffer: Buffer<u32>,
    object_buffer: Buffer<Object>,
    instance_buffer: Buffer<Instance>,
    meshlet_data_buffer: Buffer<MeshletData>,
    indirect_cmd_buffer: Buffer<vk::DrawIndexedIndirectCommand>,
    indirect_count_buffer: Buffer<u32>,
    scene_global_buffer: Buffer<SceneGlobal>,
    meshlet_cull_global_buffer: Buffer<MeshletCullGlobal>,
    meshlet_render_global_buffer: Buffer<MeshletRenderGlobal>,
}

pub struct Renderer {
    //
    core: Core,

    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    //transfer_queue: vk::Queue,

    /* Generic memory allocator: */
    allocator: vk_mem::Allocator,

    /* Swapchain data: */
    swapchain: Swapchain,

    /* Command pool: */
    query_pool: vk::QueryPool,
    _cmd_pool: vk::CommandPool,

    /* Desciptor set layouts: */
    _global_set_layout: vk::DescriptorSetLayout,
    _scene_set_layout: vk::DescriptorSetLayout,
    _render_set_layout: vk::DescriptorSetLayout,
    _cull_set_layout: vk::DescriptorSetLayout,

    /* Pipelines: */
    frustum_cull_pipeline_layout: vk::PipelineLayout,
    frustum_cull_pipeline: vk::Pipeline,
    render_pipeline_layout: vk::PipelineLayout,
    render_pipeline: vk::Pipeline,
    build_hzb_pipeline_layout: vk::PipelineLayout,
    build_hzb_pipeline: vk::Pipeline,

    /* Generic resource containers: */
    cwd: PathBuf,
    resource_counter: u32,
    meshes: HashMap<MeshHandle, (f32, Box<[Meshlet]>)>,
    objects: HashMap<ObjectHandle, ObjectInstance>,
    vertex_buffers: HashMap<MeshHandle, Buffer<Vertex>>,

    /* Staging: */
    staging_buffer: StagingBuffer,
    staging_cmd_buffer: vk::CommandBuffer,
    staging_fence: vk::Fence,

    /* Scene: */
    // Bindless set of all image views.
    _global_descriptor_pool: vk::DescriptorPool,
    _descriptor_pools: [vk::DescriptorPool; MAX_FRAMES_IN_FLIGHT],

    hzb_sampler: vk::Sampler,
    global_set: vk::DescriptorSet,
    scene_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    render_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    cull_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],

    // Used for sequencing stages, and other cross-frame syncing.
    pipeline_semaphore: vk::Semaphore,

    // Misc data that does not change across scene generations.
    sync_primitives: [FrameSyncPrimitives; MAX_FRAMES_IN_FLIGHT],
    cmd_buffers: [[vk::CommandBuffer; MAX_PIPELINE_STAGES]; MAX_FRAMES_IN_FLIGHT],
    depth_images: [(vk::Image, vk_mem::Allocation); MAX_FRAMES_IN_FLIGHT],
    depth_views: [vk::ImageView; MAX_FRAMES_IN_FLIGHT],
    render_finished: Box<[vk::Semaphore]>,

    // Scenes contain resources that may change across scene generations.
    current_scene: Option<Scene>,
    next_scene: Option<Scene>,

    // Various render state data.
    frame: usize,
    pub cam_pos: Vec3,
    pub cam_rot: Vec2, // YX
                       //last_timestamp: f32,
}

impl Drop for Renderer {
    fn drop(&mut self) {
        panic!(
            "{} dropped implicitly; call explicit renderer shutdown before drop",
            std::any::type_name::<Self>()
        );
    }
}

impl Renderer {
    pub fn new(
        cwd: impl AsRef<Path>,
        viewport_w: u32,
        viewport_h: u32,
        display: impl HasDisplayHandle + HasWindowHandle,
    ) -> Self {
        // Required Vulkan features (pass some of these in?).
        let device_extensions = [
            c"VK_KHR_dynamic_rendering",
            c"VK_EXT_descriptor_indexing",
            c"VK_KHR_swapchain",
        ];

        unsafe {
            let core = Core::new(viewport_w, viewport_h, display);
            let &Core {
                ref instance,
                physical_device,
                queue_family_index,
                surface_format,
                surface_extent,
                ..
            } = &core;

            // Create logical device and its associated queues.
            let (device, graphics_queue, present_queue) = {
                let features = vk::PhysicalDeviceFeatures::default()
                    .multi_draw_indirect(true)
                    .shader_int16(true);
                let extensions = device_extensions.map(|x: &CStr| x.as_ptr());

                let device = {
                    let mut vk11features = vk::PhysicalDeviceVulkan11Features::default()
                        .shader_draw_parameters(true)
                        .storage_buffer16_bit_access(true);

                    let mut vk12features = vk::PhysicalDeviceVulkan12Features::default()
                        .shader_int8(true)
                        .storage_buffer8_bit_access(true)
                        .draw_indirect_count(true)
                        .buffer_device_address(true)
                        .descriptor_binding_uniform_buffer_update_after_bind(true)
                        .descriptor_binding_storage_buffer_update_after_bind(true)
                        .descriptor_binding_storage_image_update_after_bind(true)
                        .descriptor_binding_partially_bound(true)
                        .descriptor_binding_sampled_image_update_after_bind(true)
                        .descriptor_indexing(true)
                        .runtime_descriptor_array(true)
                        .timeline_semaphore(true);

                    let mut vk13features = vk::PhysicalDeviceVulkan13Features::default()
                        .dynamic_rendering(true)
                        .synchronization2(true);

                    let priority = [1.0];

                    let queue_cinfo = [vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(queue_family_index)
                        .queue_priorities(&priority)];

                    let device_cinfo = vk::DeviceCreateInfo::default()
                        .push_next(&mut vk11features)
                        .push_next(&mut vk12features)
                        .push_next(&mut vk13features)
                        .queue_create_infos(&queue_cinfo)
                        .enabled_extension_names(&extensions)
                        .enabled_features(&features);

                    instance
                        .create_device(physical_device, &device_cinfo, None)
                        .unwrap()
                };

                // Extract queues.
                let graphics_queue = device.get_device_queue(queue_family_index, 0);
                let present_queue = device.get_device_queue(queue_family_index, 0);

                (device, graphics_queue, present_queue)
            };

            // AMD memory allocator.
            let mut allocator_cinfo =
                vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
            allocator_cinfo.flags |= vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
            let allocator = vk_mem::Allocator::new(allocator_cinfo).unwrap();

            // Build swapchain from core.
            let swapchain = Swapchain::new(&core, &device);

            // Query pool.
            let query_pool = device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(MAX_FRAMES_IN_FLIGHT as u32 * 2),
                    None,
                )
                .unwrap();

            // Descriptor set layout for all programs.
            let global_set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .push_next(
                            &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                                .binding_flags(&[
                                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                                ]),
                        )
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

            // Descriptor set layout for all programs.
            let scene_set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .push_next(
                            &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                                .binding_flags(&[vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND]),
                        )
                        .bindings(&[
                            // SceneGlobal
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(0)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .descriptor_count(1)
                                .stage_flags(vk::ShaderStageFlags::ALL),
                        ])
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL),
                    None,
                )
                .unwrap();

            // Desciptor set layout for the rendering program.
            let render_set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .push_next(
                            &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                                .binding_flags(&[vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND]),
                        )
                        .bindings(&[vk::DescriptorSetLayoutBinding::default()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::ALL)])
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL),
                    None,
                )
                .unwrap();

            // Descriptor set layout for cull compute program.
            let cull_set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .push_next(
                            &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                                .binding_flags(&[vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND]),
                        )
                        .bindings(&[vk::DescriptorSetLayoutBinding::default()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)])
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL),
                    None,
                )
                .unwrap();

            // Shader creation util function.
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

            // Create cull compute pipeline.
            let (frustum_cull_pipeline, frustum_cull_pipeline_layout) = {
                let comp_shader = create_shader_module(include_bytes!("frustum_cull.comp.spirv"));

                let pipeline_layout = device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default()
                            .set_layouts(&[scene_set_layout, cull_set_layout]),
                        None,
                    )
                    .unwrap();

                let pipeline = device
                    .create_compute_pipelines(
                        vk::PipelineCache::default(),
                        &[vk::ComputePipelineCreateInfo::default()
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

            // Create rendering pipeline.
            let (render_pipeline, render_pipeline_layout) = {
                let pipeline_layout = device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&[
                            global_set_layout,
                            scene_set_layout,
                            render_set_layout,
                        ]),
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
                                &vk::PipelineViewportStateCreateInfo::default()
                                    .viewports(&[vk::Viewport {
                                        x: 0.,
                                        y: 0.,
                                        width: viewport_w as f32,
                                        height: viewport_h as f32,
                                        min_depth: 0.0,
                                        max_depth: 1.0,
                                    }])
                                    .scissors(&[vk::Rect2D {
                                        offset: vk::Offset2D { x: 0, y: 0 },
                                        extent: vk::Extent2D {
                                            width: viewport_w,
                                            height: viewport_h,
                                        },
                                    }]),
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
                                &vk::PipelineColorBlendStateCreateInfo::default()
                                    .logic_op_enable(false)
                                    .attachments(&[
                                        vk::PipelineColorBlendAttachmentState::default()
                                            .color_write_mask(vk::ColorComponentFlags::RGBA)
                                            .blend_enable(false),
                                    ]),
                            )
                            .depth_stencil_state(
                                &vk::PipelineDepthStencilStateCreateInfo::default()
                                    .depth_test_enable(true)
                                    .depth_write_enable(true)
                                    .depth_compare_op(vk::CompareOp::LESS),
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

            // Create build hzb compute pipeline.
            let (build_hzb_pipeline, build_hzb_pipeline_layout) = {
                let comp_shader = create_shader_module(include_bytes!("build_hzb.comp.spirv"));

                let pipeline_layout = device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default()
                            .set_layouts(&[global_set_layout])
                            .push_constant_ranges(&[vk::PushConstantRange::default()
                                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                                .offset(0)
                                .size(size_of::<BuildHzbPushConstants>() as u32)]),
                        None,
                    )
                    .unwrap();

                let pipeline = device
                    .create_compute_pipelines(
                        vk::PipelineCache::default(),
                        &[vk::ComputePipelineCreateInfo::default()
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

            // Generic command pool.
            let cmd_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family_index)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();

            // Per-frame recorded render buffers.
            let cmd_buffers = std::array::from_fn(|_| {
                device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(cmd_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(MAX_PIPELINE_STAGES as _),
                    )
                    .unwrap()
                    .try_into()
                    .unwrap()
            });

            let render_finished = (0..swapchain.images.len())
                .into_iter()
                .map(|_| {
                    device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .unwrap()
                })
                .collect();

            // Staging data.
            let staging_buffer = StagingBuffer::new(10000000, &allocator);

            let staging_cmd_buffer = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()[0];

            let staging_fence = device
                .create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
                .unwrap();

            let sync_primitives = std::array::from_fn(|_| FrameSyncPrimitives {
                image_available: device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .unwrap(),
                frame_in_flight: device
                    .create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .unwrap(),
            });

            //
            let global_descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&[
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1024),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::STORAGE_IMAGE)
                                .descriptor_count(1024),
                        ])
                        .max_sets(1)
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                    None,
                )
                .unwrap();

            // Generic descriptor pool.
            let descriptor_pools = std::array::from_fn(|_| {
                device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .pool_sizes(&[vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                                .descriptor_count(3)])
                            .max_sets(3 * MAX_FRAMES_IN_FLIGHT as u32)
                            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                        None,
                    )
                    .unwrap()
            });

            // Create depth attachment for rendering.
            let depth_images = std::array::from_fn(|_| {
                use vk_mem::Alloc;
                allocator
                    .create_image(
                        &vk::ImageCreateInfo::default()
                            .image_type(vk::ImageType::TYPE_2D)
                            .extent(
                                vk::Extent3D::default()
                                    .width(surface_extent.width)
                                    .height(surface_extent.height)
                                    .depth(1),
                            )
                            .mip_levels(1)
                            .array_layers(1)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .format(vk::Format::D32_SFLOAT)
                            .usage(
                                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                                    | vk::ImageUsageFlags::SAMPLED,
                            ),
                        &vk_mem::AllocationCreateInfo {
                            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                            ..Default::default()
                        },
                    )
                    .unwrap()
            });

            let depth_views = std::array::from_fn(|i| {
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(depth_images[i].0)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(vk::Format::D32_SFLOAT)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::DEPTH,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            }),
                        None,
                    )
                    .unwrap()
            });

            // Semaphore for pipeline signalling.
            let pipeline_semaphore = device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(
                        &mut vk::SemaphoreTypeCreateInfo::default()
                            .semaphore_type(vk::SemaphoreType::TIMELINE)
                            .initial_value(0),
                    ),
                    None,
                )
                .unwrap();

            // Sampler.
            let hzb_sampler = device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::NEAREST)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .min_lod(0.0)
                        .max_lod(vk::LOD_CLAMP_NONE)
                        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                        .unnormalized_coordinates(false),
                    None,
                )
                .unwrap();

            // Descriptor sets.
            let global_set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(global_descriptor_pool)
                        .set_layouts(&[global_set_layout]),
                )
                .unwrap()[0];

            let scene_sets = std::array::from_fn(|fif| {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pools[fif])
                            .set_layouts(&[scene_set_layout]),
                    )
                    .unwrap()[0]
            });

            let render_sets = std::array::from_fn(|fif| {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pools[fif])
                            .set_layouts(&[render_set_layout]),
                    )
                    .unwrap()[0]
            });

            let cull_sets = std::array::from_fn(|fif| {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pools[fif])
                            .set_layouts(&[cull_set_layout]),
                    )
                    .unwrap()[0]
            });

            // Write the depth buffers to 0 & 1
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(global_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(2)
                    .image_info(&[
                        vk::DescriptorImageInfo::default()
                            .image_view(depth_views[0])
                            .sampler(hzb_sampler)
                            .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL),
                        vk::DescriptorImageInfo::default()
                            .image_view(depth_views[1])
                            .sampler(hzb_sampler)
                            .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL),
                    ])],
                &[],
            );

            //
            Self {
                core,

                device,
                graphics_queue,
                present_queue,

                allocator,

                swapchain,

                query_pool,
                _cmd_pool: cmd_pool,

                _global_set_layout: global_set_layout,
                _scene_set_layout: scene_set_layout,
                _render_set_layout: render_set_layout,
                _cull_set_layout: cull_set_layout,

                frustum_cull_pipeline_layout,
                frustum_cull_pipeline,
                render_pipeline_layout,
                render_pipeline,
                build_hzb_pipeline_layout,
                build_hzb_pipeline,

                cwd: cwd.as_ref().to_owned(),
                resource_counter: 0,
                meshes: HashMap::new(),
                objects: HashMap::new(),
                vertex_buffers: HashMap::new(),

                staging_buffer,
                staging_cmd_buffer,
                staging_fence,

                _global_descriptor_pool: global_descriptor_pool,
                _descriptor_pools: descriptor_pools,

                hzb_sampler,
                global_set,
                scene_sets,
                render_sets,
                cull_sets,

                pipeline_semaphore,

                sync_primitives,
                cmd_buffers,
                depth_images,
                depth_views,
                render_finished,

                current_scene: None,
                next_scene: None,

                frame: 0,
                cam_pos: Vec3::new(0., 0., 3.),
                cam_rot: <_>::default(),
            }
        }
    }

    pub fn render(&mut self, _timestamp: f32) {
        let frame_index = self.frame % MAX_FRAMES_IN_FLIGHT;
        let frame_count = self.frame;
        self.frame += 1;

        // A dirty hack, initialize first time scene.
        if self.current_scene.is_none() {
            unsafe {
                self.current_scene = Some(self.rebuild_scene(frame_index));
            }
        };

        // Attempt to clean up current_scene if there is a next.
        if self.next_scene.is_some() {
            // All frames are signalled.
            let signalled = self.sync_primitives.iter().fold(true, |acc, syncs| unsafe {
                acc & self.device.get_fence_status(syncs.frame_in_flight).unwrap()
            });

            if signalled {
                let next_scene = self.next_scene.take();
                let scene = std::mem::replace(&mut self.current_scene, next_scene).unwrap();

                unsafe {
                    self.free_scene(scene);
                }
                println!("Old scene cleared.");
            }
        }

        // Get current scene.
        let scene = match &self.next_scene {
            Some(scene) => scene,
            None => self.current_scene.as_ref().unwrap(),
        };

        // Sync primitives associated with this frame.
        let FrameSyncPrimitives {
            image_available,
            frame_in_flight,
        } = self.sync_primitives[frame_index];

        // Command buffer associated with this frame.
        let command_buffers = &self.cmd_buffers[frame_index];
        let data_upload = command_buffers[PipelineStage::DataUpload as usize];
        let frustum_cull = command_buffers[PipelineStage::FrustumCull as usize];
        let first_draw = command_buffers[PipelineStage::FirstDraw as usize];
        let build_hzb = command_buffers[PipelineStage::BuildHzb as usize];

        // Descriptor sets associated with this frame.
        let global_set = self.global_set;
        let scene_set = self.scene_sets[frame_index];
        let render_set = self.render_sets[frame_index];
        let cull_set = self.cull_sets[frame_index];

        unsafe {
            // Wait for next image to become available.
            self.device
                .wait_for_fences(&[frame_in_flight], true, u64::MAX)
                .unwrap();
            self.device.reset_fences(&[frame_in_flight]).unwrap();

            // TODO: keep this here? Its a per-fif variable.
            let depth_view = self.depth_views[frame_index];

            //
            let (image_index, _) = self
                .swapchain
                .swapchain_device
                .acquire_next_image(
                    self.swapchain.swapchain,
                    u64::MAX,
                    image_available,
                    vk::Fence::null(),
                )
                .unwrap();
            let render_finished = self.render_finished[image_index as usize];
            let swapchain_image = self.swapchain.images[image_index as usize];
            let swapchain_view = self.swapchain.views[image_index as usize];

            // Get the previous frame's timestamps.
            let mut data = [0u64; 2];
            if self.frame > MAX_FRAMES_IN_FLIGHT {
                self.device
                    .get_query_pool_results(
                        self.query_pool,
                        2 * frame_index as u32,
                        &mut data,
                        vk::QueryResultFlags::TYPE_64,
                    )
                    .unwrap();
                let multi =
                    self.core.physical_device_properties.limits.timestamp_period / 1_000_000.;
                println!("timestamp: {} ms", multi * (data[1] - data[0]) as f32);
            }

            // TODO: Make command buffers better.
            unsafe fn cmd_buffer_record(
                device: &ash::Device,
                cmd: vk::CommandBuffer,
                f: impl FnOnce(vk::CommandBuffer),
            ) {
                device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .unwrap();
                device
                    .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                    .unwrap();

                f(cmd);

                device.end_command_buffer(cmd).unwrap();
            }

            cmd_buffer_record(&self.device, data_upload, |cmd| {
                // Reset query pools and write a timestamp for frame keeping (keep this right at the top).
                self.device
                    .cmd_reset_query_pool(cmd, self.query_pool, 2 * frame_index as u32, 2);
                self.device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.query_pool,
                    (2 * frame_index + 0) as u32,
                );

                let object_data = self.objects.values().map(|obj| Object {
                    position: obj.position,
                    scale: obj.scale * self.meshes.get(&obj.mesh).unwrap().0,
                    orientation: obj.orientation,
                    vertex_buffer: self.device.get_buffer_device_address(
                        &vk::BufferDeviceAddressInfo::default()
                            .buffer(self.vertex_buffers.get(&obj.mesh).unwrap().vk_handle()),
                    ),
                    texture_id: 0,
                });

                // Upload global descriptor data & object data.
                let projection = Mat4::perspective_infinite_rh(
                    std::f32::consts::FRAC_PI_6,
                    self.core.surface_extent.width as f32 / self.core.surface_extent.height as f32,
                    0.1,
                );

                let p = Vec3::new(self.cam_rot[0].sin(), 0., -self.cam_rot[0].cos());
                let view = Mat4::look_to_rh(self.cam_pos, p, Vec3::new(0., 1., 0.));

                // Frustum plane data.
                let normalize_plane = |p: Vec4| p / p.xyz().length();
                let temp = projection.transpose();
                let frustum_x = normalize_plane(temp.w_axis + temp.x_axis);
                let frustum_y = normalize_plane(temp.w_axis + temp.y_axis);
                let frustum = Vec4::from([frustum_x.x, frustum_x.z, frustum_y.y, frustum_y.z]);

                // Upload scene data.
                self.staging_buffer.reset();

                self.staging_buffer.stage_buffer(
                    &self.device,
                    cmd,
                    &scene.object_buffer,
                    0,
                    object_data,
                );

                self.staging_buffer.stage_buffer(
                    &self.device,
                    cmd,
                    &scene.scene_global_buffer,
                    0,
                    [SceneGlobal {
                        pv: projection * view,
                        proj: projection,
                        view,
                        camera_position: self.cam_pos.extend(1.0),
                        camera_direction: p.extend(0.0),
                        light_position: Vec4::new(1.0, 0.0, 0.0, 1.0),
                        light_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
                    }],
                );

                self.staging_buffer.stage_buffer(
                    &self.device,
                    cmd,
                    &scene.meshlet_render_global_buffer,
                    0,
                    [MeshletRenderGlobal {
                        instance_buffer: self.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default()
                                .buffer(scene.instance_buffer.vk_handle()),
                        ),
                        object_buffer: self.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default()
                                .buffer(scene.object_buffer.vk_handle()),
                        ),
                    }],
                );
                self.staging_buffer.stage_buffer(
                    &self.device,
                    cmd,
                    &scene.indirect_count_buffer,
                    0,
                    [0u32],
                );
                self.staging_buffer.stage_buffer(
                    &self.device,
                    cmd,
                    &scene.meshlet_cull_global_buffer,
                    0,
                    [MeshletCullGlobal {
                        instances: scene.indirect_cmd_buffer.len,
                        frustum,
                        meshlet_visibility_buffer: self.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default()
                                .buffer(scene.visibility_buffer.vk_handle()),
                        ),
                        draw_count_buffer: self.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default()
                                .buffer(scene.indirect_count_buffer.vk_handle()),
                        ),
                        meshlet_buffer: self.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default()
                                .buffer(scene.meshlet_data_buffer.vk_handle()),
                        ),
                        draw_cmd_buffer: self.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default()
                                .buffer(scene.indirect_cmd_buffer.vk_handle()),
                        ),
                        instance_buffer: self.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default()
                                .buffer(scene.instance_buffer.vk_handle()),
                        ),
                        object_buffer: self.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default()
                                .buffer(scene.object_buffer.vk_handle()),
                        ),
                    }],
                );
            });

            cmd_buffer_record(&self.device, frustum_cull, |cmd| {
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.frustum_cull_pipeline_layout,
                    0,
                    &[scene_set, cull_set],
                    &[],
                );

                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.frustum_cull_pipeline,
                );

                self.device
                    .cmd_dispatch(cmd, scene.meshlet_data_buffer.len().div_ceil(64), 1, 1);
            });

            cmd_buffer_record(&self.device, first_draw, |cmd| {
                self.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        // Convert VK_IMAGE_LAYOUT_UNDEFINED -> VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL.
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .image(swapchain_image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                    ]),
                );

                // Use dynamic rendering.
                self.device.cmd_begin_rendering(
                    cmd,
                    &vk::RenderingInfo::default()
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: self.core.surface_extent.width,
                                height: self.core.surface_extent.height,
                            },
                        })
                        .layer_count(1)
                        .depth_attachment(
                            &vk::RenderingAttachmentInfo::default()
                                .image_view(depth_view)
                                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                                .load_op(vk::AttachmentLoadOp::CLEAR)
                                .store_op(vk::AttachmentStoreOp::STORE)
                                .clear_value(vk::ClearValue {
                                    depth_stencil: vk::ClearDepthStencilValue {
                                        depth: 1.0,
                                        stencil: 0,
                                    },
                                }),
                        )
                        .color_attachments(&[vk::RenderingAttachmentInfo::default()
                            .image_view(swapchain_view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .clear_value(vk::ClearValue {
                                color: vk::ClearColorValue {
                                    float32: [0.0, 0.0, 0.0, 1.0],
                                },
                            })]),
                );

                // Begin draw calls.
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.render_pipeline_layout,
                    0,
                    &[global_set, scene_set, render_set],
                    &[],
                );

                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.render_pipeline,
                );

                self.device.cmd_bind_index_buffer(
                    cmd,
                    scene.index_buffer.vk_handle(),
                    0,
                    vk::IndexType::UINT32,
                );

                self.device.cmd_draw_indexed_indirect_count(
                    cmd,
                    scene.indirect_cmd_buffer.vk_handle(),
                    0,
                    scene.indirect_count_buffer.vk_handle(),
                    0,
                    scene.meshlet_data_buffer.len(),
                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );

                self.device.cmd_end_rendering(cmd);

                // Transition swapchain image to present.
                self.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        // Convert VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL -> VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                            .image(swapchain_image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR),
                    ]),
                );

                // Run this at the end!
                self.device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.query_pool,
                    (2 * frame_index + 1) as u32,
                );
            });

            cmd_buffer_record(&self.device, build_hzb, |cmd| {
                self.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        // Prepare the HZB for writing.
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .image(scene.hzb_image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(vk::REMAINING_MIP_LEVELS)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .new_layout(vk::ImageLayout::GENERAL),
                        // Prepare the depth buffer for sampling @ first HZB reduction.
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(
                                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                            )
                            .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .image(self.depth_images[frame_index].0)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL),
                    ]),
                );

                let build_hzb = |src: u32, dst: u32| {
                    let pc = BuildHzbPushConstants { src, dst };

                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.build_hzb_pipeline_layout,
                        0,
                        &[global_set],
                        &[],
                    );

                    self.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.build_hzb_pipeline,
                    );

                    self.device.cmd_push_constants(
                        cmd,
                        self.build_hzb_pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        std::slice::from_raw_parts(
                            (&pc as *const BuildHzbPushConstants) as *const u8,
                            std::mem::size_of::<BuildHzbPushConstants>(),
                        ),
                    );

                    let w = self
                        .core
                        .surface_extent
                        .width
                        .div_ceil(2)
                        .checked_shr(dst)
                        .unwrap_or(0)
                        .max(1)
                        .div_ceil(8);
                    let h = self
                        .core
                        .surface_extent
                        .height
                        .div_ceil(2)
                        .checked_shr(dst)
                        .unwrap_or(0)
                        .max(1)
                        .div_ceil(8);
                    self.device.cmd_dispatch(cmd, w, h, 1);

                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[
                            vk::ImageMemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                                .image(scene.hzb_image)
                                .subresource_range(
                                    vk::ImageSubresourceRange::default()
                                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                                        .base_mip_level(dst)
                                        .level_count(1)
                                        .base_array_layer(0)
                                        .layer_count(1),
                                )
                                .old_layout(vk::ImageLayout::GENERAL)
                                .new_layout(vk::ImageLayout::GENERAL),
                        ]),
                    );
                };

                // For the first compute, the src view is the depth buffer, which
                // depends on the depth buffer.
                build_hzb(frame_index as u32, 0);

                // The rest are standard.
                let mips = scene.hzb_build_src_views.len();
                for i in 0..mips - 1 {
                    build_hzb(2 + i as u32, 1 + i as u32);
                }

                self.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        // Transition HZB back to read after all reductions.
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .image(scene.hzb_image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(vk::REMAINING_MIP_LEVELS)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                        // Transition the depth buffer back to depth attachment.
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .dst_stage_mask(
                                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                            )
                            .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                            .image(self.depth_images[frame_index].0)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .old_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
                            .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL),
                    ]),
                );
            });

            // TODO: Submit all queues.
            let subframe = |frame, subframe: PipelineStage| {
                frame as u64 * MAX_PIPELINE_STAGES as u64 + subframe as u64
            };
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[
                            vk::CommandBufferSubmitInfo::default().command_buffer(data_upload)
                        ])
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(subframe(frame_count, PipelineStage::DataUpload))
                            .stage_mask(vk::PipelineStageFlags2::TRANSFER)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(subframe(frame_count, PipelineStage::DataUpload) + 1)
                            .stage_mask(vk::PipelineStageFlags2::TRANSFER)])],
                    vk::Fence::null(),
                )
                .unwrap();
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[
                            vk::CommandBufferSubmitInfo::default().command_buffer(frustum_cull)
                        ])
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(subframe(frame_count, PipelineStage::FrustumCull))
                            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(subframe(frame_count, PipelineStage::FrustumCull) + 1)
                            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])],
                    vk::Fence::null(),
                )
                .unwrap();
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[
                            vk::CommandBufferSubmitInfo::default().command_buffer(first_draw)
                        ])
                        .wait_semaphore_infos(&[
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(image_available)
                                .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE),
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(self.pipeline_semaphore)
                                .value(subframe(frame_count, PipelineStage::FirstDraw))
                                .stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT),
                        ])
                        .signal_semaphore_infos(&[
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(self.pipeline_semaphore)
                                .value(subframe(frame_count, PipelineStage::FirstDraw) + 1)
                                .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(render_finished)
                                .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
                        ])],
                    vk::Fence::null(),
                )
                .unwrap();
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[
                            vk::CommandBufferSubmitInfo::default().command_buffer(build_hzb)
                        ])
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(subframe(frame_count, PipelineStage::BuildHzb))
                            .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(subframe(frame_count, PipelineStage::BuildHzb) + 1)
                            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])],
                    frame_in_flight,
                )
                .unwrap();

            // Swap backbuffer.
            self.swapchain
                .swapchain_device
                .queue_present(
                    self.present_queue,
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&[render_finished])
                        .swapchains(&[self.swapchain.swapchain])
                        .image_indices(&[image_index]),
                )
                .unwrap();
        }
    }

    unsafe fn free_scene(&mut self, mut scene: Scene) {
        scene.index_buffer.take().destroy(&self.allocator);
        scene.object_buffer.take().destroy(&self.allocator);
        scene.instance_buffer.take().destroy(&self.allocator);
        scene.meshlet_data_buffer.take().destroy(&self.allocator);
        scene.indirect_cmd_buffer.take().destroy(&self.allocator);
        scene.indirect_count_buffer.take().destroy(&self.allocator);
        scene
            .meshlet_cull_global_buffer
            .take()
            .destroy(&self.allocator);
        scene
            .meshlet_render_global_buffer
            .take()
            .destroy(&self.allocator);
    }

    unsafe fn rebuild_scene(&mut self, fif: usize) -> Scene {
        // Generate vertex data for newly added meshes.
        let new_meshes: HashMap<MeshHandle, Box<[Vertex]>> = self
            .meshes
            .iter()
            .filter(|(k, _)| !self.vertex_buffers.contains_key(k))
            .map(|(id, mesh)| {
                (
                    *id,
                    mesh.1
                        .iter()
                        .flat_map(|meshlet| {
                            (0..meshlet.positions.len()).map(|i| Vertex {
                                position: meshlet.positions[i],
                                normal: meshlet.normals[i],
                                uv: [0, 0],
                            })
                        })
                        .collect(),
                )
            })
            .collect();

        // Create new buffers for newly added meshes.
        for (id, vertices) in &new_meshes {
            self.vertex_buffers.insert(
                *id,
                Buffer::new(
                    &self.allocator,
                    vertices.len() as u32,
                    vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    vk_mem::MemoryUsage::AutoPreferDevice,
                ),
            );
        }

        // Generate index and meshlet data for object set.
        let mut indices = vec![];
        let mut meshlet_data = vec![];
        let mut instances = 0u32;
        let mut first_index = 0;
        for (i, object) in self.objects.values().enumerate() {
            // Get associated mesh and index offset.
            let mesh = &self.meshes.get(&object.mesh).unwrap().1;

            // Indices.
            let mut offset = 0u32;
            for meshlet in mesh {
                indices.extend(meshlet.indices.iter().map(|&index| index as u32 + offset));
                offset += meshlet.positions.len() as u32;
            }

            // Mesh data.
            for meshlet in mesh {
                meshlet_data.push(MeshletData {
                    center: Vec3::from(meshlet.center),
                    radius: meshlet.radius,
                    cone_apex: Vec3::from(meshlet.cone_apex),
                    pad0: 0.,
                    cone_axis: Vec3::from(meshlet.cone_axis),
                    cone_cutoff: meshlet.cone_cutoff,
                    object_id: i as u32,
                    index_count: meshlet.indices.len() as u32,
                    first_index,
                });
                first_index += meshlet.indices.len() as u32;
                instances += 1;
            }
        }

        use vk_mem::Alloc;
        let vk::Extent2D { width, height, .. } = self.core.surface_extent;
        let hzb_width = width.div_ceil(2);
        let hzb_height = height.div_ceil(2);
        let mipmaps = u32::max(hzb_width, hzb_height).ilog2() + 1;
        let (hzb_image, hzb_alloc) = self
            .allocator
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(vk::Extent3D {
                        width: hzb_width,
                        height: hzb_height,
                        depth: 1,
                    })
                    .mip_levels(mipmaps)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .format(vk::Format::R32_SFLOAT)
                    .usage(
                        vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::SAMPLED
                            | vk::ImageUsageFlags::STORAGE,
                    ),
                &vk_mem::AllocationCreateInfo {
                    required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    ..Default::default()
                },
            )
            .unwrap();

        let hzb_test_view = self
            .device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(hzb_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R32_SFLOAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
            .unwrap();

        let hzb_build_src_views = (0..mipmaps)
            .into_iter()
            .map(|level| {
                self.device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(hzb_image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(vk::Format::R32_SFLOAT)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: level,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            }),
                        None,
                    )
                    .unwrap()
            })
            .collect();

        let hzb_build_dst_views = (0..mipmaps)
            .into_iter()
            .map(|level| {
                self.device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(hzb_image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(vk::Format::R32_SFLOAT)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: level,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            }),
                        None,
                    )
                    .unwrap()
            })
            .collect();

        let frame = Scene {
            hzb_image,
            _hzb_alloc: hzb_alloc,
            _hzb_test_view: hzb_test_view,
            hzb_build_src_views,
            hzb_build_dst_views,

            visibility_buffer: Buffer::new(
                &self.allocator,
                instances,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            index_buffer: Buffer::new(
                &self.allocator,
                indices.len() as u32,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            object_buffer: Buffer::new(
                &self.allocator,
                self.objects.len() as u32,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            instance_buffer: Buffer::new(
                &self.allocator,
                instances,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            meshlet_data_buffer: Buffer::new(
                &self.allocator,
                instances,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            indirect_cmd_buffer: Buffer::new(
                &self.allocator,
                instances,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            indirect_count_buffer: Buffer::new(
                &self.allocator,
                1,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            scene_global_buffer: Buffer::new(
                &self.allocator,
                1,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            meshlet_cull_global_buffer: Buffer::new(
                &self.allocator,
                1,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            meshlet_render_global_buffer: Buffer::new(
                &self.allocator,
                1,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),
        };

        self.device
            .reset_command_buffer(
                self.staging_cmd_buffer,
                vk::CommandBufferResetFlags::empty(),
            )
            .unwrap();
        self.device
            .begin_command_buffer(
                self.staging_cmd_buffer,
                &vk::CommandBufferBeginInfo::default(),
            )
            .unwrap();

        // Upload.
        //
        self.staging_buffer.reset();
        self.staging_buffer.stage_buffer(
            &self.device,
            self.staging_cmd_buffer,
            &frame.index_buffer,
            0,
            indices,
        );
        for (id, vertices) in &new_meshes {
            self.staging_buffer.stage_buffer(
                &self.device,
                self.staging_cmd_buffer,
                self.vertex_buffers.get(id).unwrap(),
                0,
                vertices,
            );
        }
        self.staging_buffer.stage_buffer(
            &self.device,
            self.staging_cmd_buffer,
            &frame.meshlet_data_buffer,
            0,
            meshlet_data,
        );

        self.device.cmd_fill_buffer(
            self.staging_cmd_buffer,
            frame.visibility_buffer.vk_handle(),
            0,
            frame.visibility_buffer.size() as u64,
            1,
        );

        // Rerecord scene command buffers.
        {
            // Start with hzb transition.
            let mut tmp = vec![
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .image(hzb_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(vk::REMAINING_MIP_LEVELS)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            ];

            // Add depth transitions.
            tmp.extend((0..MAX_FRAMES_IN_FLIGHT).into_iter().map(|i| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    )
                    .image(self.depth_images[i].0)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            }));

            // Queue transitions.
            self.device.cmd_pipeline_barrier2(
                self.staging_cmd_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&tmp),
            )
        };

        // Submit (& wait at end of function).
        self.device
            .end_command_buffer(self.staging_cmd_buffer)
            .unwrap();
        self.device.reset_fences(&[self.staging_fence]).unwrap();
        self.device
            .queue_submit(
                self.graphics_queue,
                &[vk::SubmitInfo::default().command_buffers(&[self.staging_cmd_buffer])],
                self.staging_fence,
            )
            .unwrap();

        // TODO: this will NOT work with scene regeneration...
        // We can only regenerate these when the fif is not using them.
        {
            /* global_set part: */
            let global_set_0_info: Box<_> = (0..mipmaps)
                .into_iter()
                .map(|i| {
                    vk::DescriptorImageInfo::default()
                        .image_view(frame.hzb_build_src_views[i as usize])
                        .sampler(self.hzb_sampler)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                })
                .collect();

            let global_set_1_info: Box<_> = (0..mipmaps)
                .into_iter()
                .map(|i| {
                    vk::DescriptorImageInfo::default()
                        .image_view(frame.hzb_build_dst_views[i as usize])
                        .image_layout(vk::ImageLayout::GENERAL)
                })
                .collect();

            let mut tmp = vec![
                vk::WriteDescriptorSet::default()
                    .dst_set(self.global_set)
                    .dst_binding(0)
                    .dst_array_element(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(global_set_0_info.len() as u32)
                    .image_info(&global_set_0_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.global_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(global_set_1_info.len() as u32)
                    .image_info(&global_set_1_info),
            ];

            /* scene/cull/render part: */
            let scene_set_info = [vk::DescriptorBufferInfo::default()
                .buffer(frame.scene_global_buffer.vk_handle())
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let cull_set_info = [vk::DescriptorBufferInfo::default()
                .buffer(frame.meshlet_cull_global_buffer.vk_handle())
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let render_set_info = [vk::DescriptorBufferInfo::default()
                .buffer(frame.meshlet_render_global_buffer.vk_handle())
                .offset(0)
                .range(vk::WHOLE_SIZE)];

            tmp.extend((0..MAX_FRAMES_IN_FLIGHT).into_iter().flat_map(|i| {
                [
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.scene_sets[i])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(&scene_set_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.cull_sets[i])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(&cull_set_info),
                    //
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.render_sets[i])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(&render_set_info),
                ]
            }));

            /* What a mess */
            self.device.update_descriptor_sets(&tmp, &[]);
        }

        // Wait for transfer to complete before returning.
        self.device
            .wait_for_fences(&[self.staging_fence], true, u64::MAX)
            .unwrap();

        frame
    }

    pub fn create_object(
        &mut self,
        mesh: MeshHandle,
        position: Vec3,
        scale: f32,
        orientation: Quat,
    ) -> Option<ObjectHandle> {
        let handle = ObjectHandle(self.resource_counter);
        self.resource_counter += 1;
        self.objects.insert(
            handle,
            ObjectInstance {
                mesh,
                position,
                scale,
                orientation,
            },
        );
        Some(handle)
    }

    pub fn load_mesh(&mut self, filename: impl AsRef<Path>) -> Option<MeshHandle> {
        let mesh = load_mesh(self.cwd.join(filename))?;
        let handle = MeshHandle(self.resource_counter);
        self.resource_counter += 1;
        self.meshes.insert(handle, mesh);
        return Some(handle);
    }
}
