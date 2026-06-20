use crate::buffer::Buffer;
use crate::core::Core;
use crate::glsl_types::*;
use crate::mesh::{Meshlet, load_mesh};
use crate::profiling::PipelineProfiler;
use crate::staging::StagingBuffer;
use crate::swapchain::Swapchain;
use crate::util::{const_max, const_min, format_bytes, format_usize_commas};
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

#[derive(Debug)]
pub(super) struct Object {
    pub mesh: MeshHandle,
    pub position: Vec3,
    pub scale: f32,
    pub orientation: Quat,
}

pub(crate) const MAX_FRAMES_IN_FLIGHT: usize = 2;
const VISIBILITY_DEPTH: usize = 2;
const VISIBILITY_BUFFER_COUNT: usize = VISIBILITY_DEPTH + 1;
const STARTING_FRAME: usize = const_max([MAX_FRAMES_IN_FLIGHT, VISIBILITY_DEPTH]);

// Dedicated HZB/occlusion descriptor set.
const MAX_HZB_DIMENSION: u32 = 8192;
const MAX_HZB_MIPS: u32 = MAX_HZB_DIMENSION.div_ceil(2).ilog2() + 1;
const HZB_SAMPLED_IMAGE_CAPACITY: u32 = 1 + MAX_HZB_MIPS;
const HZB_STORAGE_IMAGE_CAPACITY: u32 = MAX_HZB_MIPS;

// The maximum number of FIF that can be in the BuildHzb of OcclusionCull phase of the pipeline.
const MAX_HZB_IN_FLIGHT: usize = const_min([MAX_FRAMES_IN_FLIGHT, VISIBILITY_DEPTH]);

/*
Plan:
0) Data upload
1) frustum_cull
2) render (only visible)
3) build_hzb
4) occlusion_cull
5) render (late visible)
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum PipelineStage {
    DataUpload,
    FrustumCull,
    EarlyDraw,
    BuildHzb,
    OcclusionCull,
    LateDraw,
    FrameEnd,
}

impl PipelineStage {
    const COUNT: usize = Self::FrameEnd as usize + 1;

    const fn start_value(self, frame: usize) -> u64 {
        frame as u64 * Self::COUNT as u64 + self as u64
    }

    const fn done_value(self, frame: usize) -> u64 {
        self.start_value(frame) + 1
    }
}

struct SwapchainResources {
    // HZB is per-frame scratch, but the ring only needs to be large enough to
    // cover the live visibility window.
    hzb_images: [vk::Image; MAX_HZB_IN_FLIGHT],
    hzb_allocs: [vk_mem::Allocation; MAX_HZB_IN_FLIGHT],
    hzb_build_depth_views: [vk::ImageView; MAX_HZB_IN_FLIGHT],
    hzb_build_src_views: [Box<[vk::ImageView]>; MAX_HZB_IN_FLIGHT],
    hzb_build_dst_views: [Box<[vk::ImageView]>; MAX_HZB_IN_FLIGHT],
    hzb_sets: [vk::DescriptorSet; MAX_HZB_IN_FLIGHT],
    overdraw_images: [vk::Image; MAX_FRAMES_IN_FLIGHT],
    overdraw_allocs: [vk_mem::Allocation; MAX_FRAMES_IN_FLIGHT],
    overdraw_views: [vk::ImageView; MAX_FRAMES_IN_FLIGHT],

    render_finished: Box<[vk::Semaphore]>,
    image_acquired_semaphores: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    cmd_buffers: [[vk::CommandBuffer; PipelineStage::COUNT]; MAX_FRAMES_IN_FLIGHT],
    depth_images: [(vk::Image, vk_mem::Allocation); MAX_FRAMES_IN_FLIGHT],
    depth_views: [vk::ImageView; MAX_FRAMES_IN_FLIGHT],
}

impl SwapchainResources {
    const HZB_SLOT_COUNT: usize = MAX_HZB_IN_FLIGHT;

    unsafe fn free(
        self,
        device: &ash::Device,
        allocator: &vk_mem::Allocator,
        cmd_pool: vk::CommandPool,
    ) -> [vk::DescriptorSet; MAX_HZB_IN_FLIGHT] {
        let Self {
            hzb_images,
            mut hzb_allocs,
            hzb_build_depth_views,
            hzb_build_src_views,
            hzb_build_dst_views,
            hzb_sets,
            overdraw_images,
            mut overdraw_allocs,
            overdraw_views,
            render_finished,
            image_acquired_semaphores,
            cmd_buffers,
            depth_images,
            depth_views,
        } = self;

        for slot in 0..Self::HZB_SLOT_COUNT {
            device.destroy_image_view(hzb_build_depth_views[slot], None);
            for view in hzb_build_src_views[slot].iter().copied() {
                device.destroy_image_view(view, None);
            }
            for view in hzb_build_dst_views[slot].iter().copied() {
                device.destroy_image_view(view, None);
            }
            allocator.destroy_image(hzb_images[slot], &mut hzb_allocs[slot]);
        }
        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            device.destroy_image_view(overdraw_views[slot], None);
            allocator.destroy_image(overdraw_images[slot], &mut overdraw_allocs[slot]);
        }

        for semaphore in render_finished {
            device.destroy_semaphore(semaphore, None);
        }
        for semaphore in image_acquired_semaphores {
            device.destroy_semaphore(semaphore, None);
        }

        for cmd_buffers in cmd_buffers {
            device.free_command_buffers(cmd_pool, &cmd_buffers);
        }

        for view in depth_views {
            device.destroy_image_view(view, None);
        }
        for (image, mut alloc) in depth_images {
            allocator.destroy_image(image, &mut alloc);
        }

        hzb_sets
    }
}

/* Resources that need regeneration when object set changes */
struct SceneResources {
    indirect_cmd_buffer: Buffer<GpuDrawCommandBuffer>,
    frustum_passing_meshlet_buffers: [Buffer<[u8]>; MAX_HZB_IN_FLIGHT],
    late_draw_cmd_buffers: [Buffer<GpuDrawCommandBuffer>; MAX_HZB_IN_FLIGHT],

    /* TODO: These are static after creation */
    index_buffer: Buffer<[u32]>,
    object_instance_buffer: Buffer<[GpuObjectInstance]>,
    meshlet_instance_buffer: Buffer<[GpuMeshletInstance]>,
}

impl SceneResources {
    unsafe fn free(self, allocator: &vk_mem::Allocator) {
        for buffer in self.frustum_passing_meshlet_buffers {
            buffer.destroy(&allocator);
        }
        for buffer in self.late_draw_cmd_buffers {
            buffer.destroy(&allocator);
        }
        self.index_buffer.destroy(&allocator);
        self.object_instance_buffer.destroy(&allocator);
        self.meshlet_instance_buffer.destroy(&allocator);
        self.indirect_cmd_buffer.destroy(&allocator);
    }
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
    profiler: PipelineProfiler,
    cmd_pool: vk::CommandPool,

    /* Desciptor set layouts: */
    _global_set_layout: vk::DescriptorSetLayout,
    hzb_set_layout: vk::DescriptorSetLayout,
    _frame_set_layout: vk::DescriptorSetLayout,
    overdraw_set_layout: vk::DescriptorSetLayout,

    /* Pipelines: */
    frustum_cull_pipeline_layout: vk::PipelineLayout,
    frustum_cull_pipeline: vk::Pipeline,
    render_pipeline_layout: vk::PipelineLayout,
    render_pipeline: vk::Pipeline,
    overdraw_render_pipeline_layout: vk::PipelineLayout,
    overdraw_render_pipeline: vk::Pipeline,
    overdraw_resolve_pipeline_layout: vk::PipelineLayout,
    overdraw_resolve_pipeline: vk::Pipeline,
    build_hzb_pipeline_layout: vk::PipelineLayout,
    build_hzb_pipeline: vk::Pipeline,
    occlusion_cull_pipeline_layout: vk::PipelineLayout,
    occlusion_cull_pipeline: vk::Pipeline,

    /* Generic resource containers: */
    cwd: PathBuf,
    resource_counter: u32,
    meshes: HashMap<MeshHandle, (f32, Box<[Meshlet]>)>,
    objects: Vec<(ObjectHandle, Object)>,
    vertex_buffers: HashMap<MeshHandle, Buffer<[GpuVertex]>>,
    // Canonical packed meshlet start index for each object.
    visibility_index_cache: HashMap<ObjectHandle, u32>,

    /* Staging: */
    staging_buffer: StagingBuffer,
    staging_cmd_buffer: vk::CommandBuffer,
    staging_fence: vk::Fence,

    /* Scene: */
    // Bindless set of all image views.
    global_descriptor_pool: vk::DescriptorPool,
    _descriptor_pools: [vk::DescriptorPool; MAX_FRAMES_IN_FLIGHT],
    overdraw_descriptor_pools: [vk::DescriptorPool; MAX_FRAMES_IN_FLIGHT],

    hzb_sampler: vk::Sampler,
    global_set: vk::DescriptorSet,
    frame_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    overdraw_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    frame_global_buffers: [Buffer<GpuFrameGlobal>; MAX_FRAMES_IN_FLIGHT],

    // Used for sequencing stages, and other cross-frame syncing.
    pipeline_semaphore: vk::Semaphore,

    // Visibility is shared across frames; replaced buffers retire once no
    // submitted frustum/occlusion pass can reference them.
    visibility_buffers: [Buffer<[u32]>; VISIBILITY_BUFFER_COUNT],
    visibility_buffer_retire_list: Vec<(u64, Buffer<[u32]>)>,

    // Dirty flags for resource regeneration.
    swapchain_resources_dirty: bool,
    scene_resources_dirty: bool,

    // When rendering a FIF
    swapchain_resources: Option<SwapchainResources>,
    scene_resources: Vec<SceneResources>,

    // Various render state data.
    frame: usize,
    pub cam_pos: Vec3,
    pub cam_rot: Vec2, // YX
    pub overdraw_enabled: bool,
}

impl Drop for Renderer {
    fn drop(&mut self) {
        panic!("{} dropped implicitly; call explicit renderer shutdown before drop", std::any::type_name::<Self>());
    }
}

impl Renderer {
    unsafe fn wait_for_pipeline_stage(&self, frame: usize, stage: PipelineStage) {
        self.device
            .wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&[self.pipeline_semaphore])
                    .values(&[stage.start_value(frame)]),
                u64::MAX,
            )
            .unwrap();
    }

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
                ..
            } = &core;

            // Create logical device and its associated queues.
            let (device, graphics_queue, present_queue) = {
                let features = vk::PhysicalDeviceFeatures::default()
                    .multi_draw_indirect(true)
                    .shader_int16(true)
                    .fragment_stores_and_atomics(true);
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
                        .shader_sampled_image_array_non_uniform_indexing(true)
                        .runtime_descriptor_array(true)
                        .timeline_semaphore(true);

                    let mut vk13features =
                        vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true).synchronization2(true);

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

                    instance.create_device(physical_device, &device_cinfo, None).unwrap()
                };

                // Extract queues.
                let graphics_queue = device.get_device_queue(queue_family_index, 0);
                let present_queue = device.get_device_queue(queue_family_index, 0);

                (device, graphics_queue, present_queue)
            };

            // AMD memory allocator.
            let mut allocator_cinfo = vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
            allocator_cinfo.flags |= vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
            let allocator = vk_mem::Allocator::new(allocator_cinfo).unwrap();

            // Build swapchain from core.
            let swapchain = Swapchain::new(&core, &device);

            let profiler = PipelineProfiler::new(&device);

            // Descriptor set layout for all programs.
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

            // Descriptor set layout for per-frame globals.
            let frame_set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .push_next(
                            &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                                .binding_flags(&[vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND]),
                        )
                        .bindings(&[
                            // FrameGlobal
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
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
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
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&[frame_set_layout]),
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

            // Create rendering pipeline.
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
                                        extent: vk::Extent2D { width: viewport_w, height: viewport_h },
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
                                &vk::PipelineColorBlendStateCreateInfo::default().logic_op_enable(false).attachments(
                                    &[vk::PipelineColorBlendAttachmentState::default()
                                        .color_write_mask(vk::ColorComponentFlags::RGBA)
                                        .blend_enable(false)],
                                ),
                            )
                            .depth_stencil_state(
                                &vk::PipelineDepthStencilStateCreateInfo::default()
                                    .depth_test_enable(true)
                                    .depth_write_enable(true)
                                    // Reverse-Z keeps the closest depth as the largest value.
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

            let (overdraw_render_pipeline, overdraw_render_pipeline_layout) = {
                let pipeline_layout = device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&[global_set_layout, overdraw_set_layout]),
                        None,
                    )
                    .unwrap();

                let vert_shader = create_shader_module(include_bytes!("render.vert.spirv"));
                let frag_shader = create_shader_module(include_bytes!("overdraw.frag.spirv"));

                let pipeline = device
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
                                        extent: vk::Extent2D { width: viewport_w, height: viewport_h },
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
                                    .attachments(&[]),
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

            // Create build hzb compute pipeline.
            let (build_hzb_pipeline, build_hzb_pipeline_layout) = {
                let comp_shader = create_shader_module(include_bytes!("build_hzb.comp.spirv"));

                let pipeline_layout = device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&[hzb_set_layout]),
                        None,
                    )
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

            // Create occlusion compute pipeline.
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

            // Generic command pool.
            let cmd_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family_index)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();

            // Staging data.
            let staging_buffer = StagingBuffer::new(1024 * 1024 * 1024, &allocator);

            let staging_cmd_buffer = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()[0];

            let staging_fence = device
                .create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)
                .unwrap();

            // The renderer starts at STARTING_FRAME so the steady-state render path
            // can read the previous timestamp slot without startup branches.
            device.begin_command_buffer(staging_cmd_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();
            profiler.bootstrap_queries(&device, staging_cmd_buffer);
            device.end_command_buffer(staging_cmd_buffer).unwrap();
            device
                .queue_submit(
                    graphics_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[staging_cmd_buffer])],
                    vk::Fence::null(),
                )
                .unwrap();
            device.queue_wait_idle(graphics_queue).unwrap();

            //
            let global_descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&[
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(
                                    1024 + (SwapchainResources::HZB_SLOT_COUNT as u32 * HZB_SAMPLED_IMAGE_CAPACITY),
                                ),
                            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(
                                1024 + (SwapchainResources::HZB_SLOT_COUNT as u32 * HZB_STORAGE_IMAGE_CAPACITY),
                            ),
                        ])
                        .max_sets(1 + SwapchainResources::HZB_SLOT_COUNT as u32)
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
                                .descriptor_count(1)])
                            .max_sets(1)
                            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                        None,
                    )
                    .unwrap()
            });

            let overdraw_descriptor_pools = std::array::from_fn(|_| {
                device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .pool_sizes(&[
                                vk::DescriptorPoolSize::default()
                                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                                    .descriptor_count(1),
                                vk::DescriptorPoolSize::default()
                                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                                    .descriptor_count(2),
                            ])
                            .max_sets(1)
                            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
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
                            .initial_value(PipelineStage::DataUpload.start_value(STARTING_FRAME)),
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

            let frame_sets = std::array::from_fn(|fif| {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pools[fif])
                            .set_layouts(&[frame_set_layout]),
                    )
                    .unwrap()[0]
            });

            let overdraw_sets = std::array::from_fn(|fif| {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(overdraw_descriptor_pools[fif])
                            .set_layouts(&[overdraw_set_layout]),
                    )
                    .unwrap()[0]
            });

            let frame_global_buffers = std::array::from_fn(|_| {
                Buffer::<GpuFrameGlobal>::new(
                    &allocator,
                    vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    vk_mem::MemoryUsage::AutoPreferDevice,
                )
            });

            for fif in 0..MAX_FRAMES_IN_FLIGHT {
                let descriptor_buffer_infos = [vk::DescriptorBufferInfo::default()
                    .buffer(frame_global_buffers[fif].vk_handle())
                    .offset(0)
                    .range(vk::WHOLE_SIZE)];

                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(frame_sets[fif])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(&descriptor_buffer_infos)],
                    &[],
                );

                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(overdraw_sets[fif])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(&descriptor_buffer_infos)],
                    &[],
                );
            }

            //
            Self {
                core,

                device,
                graphics_queue,
                present_queue,

                allocator,

                swapchain,

                profiler,
                cmd_pool,

                _global_set_layout: global_set_layout,
                hzb_set_layout,
                _frame_set_layout: frame_set_layout,
                overdraw_set_layout,

                frustum_cull_pipeline_layout,
                frustum_cull_pipeline,
                render_pipeline_layout,
                render_pipeline,
                overdraw_render_pipeline_layout,
                overdraw_render_pipeline,
                overdraw_resolve_pipeline_layout,
                overdraw_resolve_pipeline,
                build_hzb_pipeline_layout,
                build_hzb_pipeline,
                occlusion_cull_pipeline_layout,
                occlusion_cull_pipeline,

                cwd: cwd.as_ref().to_owned(),
                resource_counter: 0,
                meshes: HashMap::new(),
                objects: Vec::new(),
                vertex_buffers: HashMap::new(),
                visibility_index_cache: HashMap::new(),

                staging_buffer,
                staging_cmd_buffer,
                staging_fence,

                global_descriptor_pool,
                _descriptor_pools: descriptor_pools,
                overdraw_descriptor_pools,

                hzb_sampler,
                global_set,
                frame_sets,
                overdraw_sets,
                frame_global_buffers,

                pipeline_semaphore,
                visibility_buffers: std::array::from_fn(|_| Buffer::null()),
                visibility_buffer_retire_list: Vec::new(),

                swapchain_resources_dirty: true,
                scene_resources_dirty: true,

                swapchain_resources: None,
                scene_resources: vec![],

                frame: STARTING_FRAME,
                cam_pos: Vec3::new(0., 0., 3.),
                cam_rot: <_>::default(),
                overdraw_enabled: false,
            }
        }
    }

    pub fn render(&mut self, _timestamp: f32) {
        if self.swapchain_resources_dirty {
            self.swapchain_resources_dirty = false;
            unsafe {
                self.rebuild_swapchain();
            }
        }

        if self.scene_resources_dirty {
            self.scene_resources_dirty = false;
            unsafe {
                self.rebuild_scene();
            }
        }

        let frame_index = self.frame % MAX_FRAMES_IN_FLIGHT;
        let frame_count = self.frame;
        self.frame += 1;

        // Wait if we have too many frames in flight.
        unsafe {
            self.wait_for_pipeline_stage(frame_count - MAX_FRAMES_IN_FLIGHT + 1, PipelineStage::DataUpload);
        }

        // Try to clean up old visibility buffers.
        let completed_stage = unsafe { self.device.get_semaphore_counter_value(self.pipeline_semaphore).unwrap() };
        let allocator = &self.allocator;
        self.visibility_buffer_retire_list.retain_mut(|(retire_after, buffer)| {
            if completed_stage >= *retire_after {
                unsafe {
                    buffer.take().destroy(allocator);
                }
                false
            } else {
                true
            }
        });

        // Attempt to clean old scenes:
        let mut scene_resources = vec![self.scene_resources.pop().unwrap()];
        while let Some(scene) = self.scene_resources.pop() {
            let all_signalled = unsafe {
                self.device.get_semaphore_counter_value(self.pipeline_semaphore).unwrap()
                    >= PipelineStage::FrameEnd.done_value(frame_count - 1)
            };

            if all_signalled {
                unsafe {
                    scene.free(&self.allocator);
                }
                println!("Scene freed!");
                continue;
            }

            // This scene is still in use, push it back.
            scene_resources.push(scene);
        }
        scene_resources.reverse();
        self.scene_resources = scene_resources;

        let SceneResources {
            indirect_cmd_buffer,
            index_buffer,
            object_instance_buffer,
            meshlet_instance_buffer,
            frustum_passing_meshlet_buffers,
            late_draw_cmd_buffers,
            ..
        } = self.scene_resources.last_mut().unwrap();

        let SwapchainResources {
            hzb_images,
            depth_views,
            hzb_build_src_views,
            hzb_sets,
            overdraw_images,
            overdraw_views: _,
            render_finished,
            image_acquired_semaphores,
            cmd_buffers,
            depth_images,
            ..
        } = self.swapchain_resources.as_ref().unwrap();

        let image_acquired = image_acquired_semaphores[frame_index];
        let hzb_slot = frame_count % SwapchainResources::HZB_SLOT_COUNT;
        let hzb_image = hzb_images[hzb_slot];
        let hzb_set = hzb_sets[hzb_slot];
        let hzb_build_src_views = &hzb_build_src_views[hzb_slot];
        let frustum_passing_meshlet_buffer = &frustum_passing_meshlet_buffers[hzb_slot];
        let late_draw_cmd_buffer = &late_draw_cmd_buffers[hzb_slot];

        // Command buffer associated with this frame.
        let data_upload = cmd_buffers[frame_index][PipelineStage::DataUpload as usize];
        let frustum_cull = cmd_buffers[frame_index][PipelineStage::FrustumCull as usize];
        let early_draw = cmd_buffers[frame_index][PipelineStage::EarlyDraw as usize];
        let build_hzb = cmd_buffers[frame_index][PipelineStage::BuildHzb as usize];
        let occlusion_cull = cmd_buffers[frame_index][PipelineStage::OcclusionCull as usize];
        let late_draw = cmd_buffers[frame_index][PipelineStage::LateDraw as usize];
        let frame_end = cmd_buffers[frame_index][PipelineStage::FrameEnd as usize];

        // Descriptor sets associated with this frame.
        let global_set = self.global_set;
        let frame_set = self.frame_sets[frame_index];
        let overdraw_set = self.overdraw_sets[frame_index];
        let frame_global_buffer = &self.frame_global_buffers[frame_index];
        let overdraw_enabled = self.overdraw_enabled;

        // Visibility information.
        let visibility_buffer = &self.visibility_buffers[frame_count % VISIBILITY_BUFFER_COUNT];
        let last_visibility_buffer =
            &self.visibility_buffers[(frame_count - VISIBILITY_DEPTH) % VISIBILITY_BUFFER_COUNT];

        unsafe {
            // TODO: keep this here? It's a per-FIF variable.
            let depth_view = depth_views[frame_index];
            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(hzb_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .image_info(&[vk::DescriptorImageInfo::default()
                        .image_view(depth_view)
                        .sampler(self.hzb_sampler)
                        .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)])],
                &[],
            );

            //
            let (image_index, _) = self
                .swapchain
                .swapchain_device
                .acquire_next_image(self.swapchain.swapchain, u64::MAX, image_acquired, vk::Fence::null())
                .unwrap();
            let render_finished = render_finished[image_index as usize];
            let swapchain_image = self.swapchain.images[image_index as usize];
            let swapchain_view = self.swapchain.views[image_index as usize];
            let overdraw_image = overdraw_images[frame_index];

            if self.overdraw_enabled {
                let swapchain_info = [vk::DescriptorImageInfo::default()
                    .image_view(swapchain_view)
                    .image_layout(vk::ImageLayout::GENERAL)];
                self.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(overdraw_set)
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .image_info(&swapchain_info)],
                    &[],
                );
            }

            self.profiler.read_and_accumulate(
                &self.device,
                frame_index,
                self.core.physical_device_properties.limits.timestamp_period,
            );

            // TODO: Make command buffers better.
            unsafe fn cmd_buffer_record(
                device: &ash::Device,
                profiler: &PipelineProfiler,
                frame_index: usize,
                stage: PipelineStage,
                cmd: vk::CommandBuffer,
                f: impl FnOnce(vk::CommandBuffer),
            ) {
                device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
                device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();

                if stage as usize == 0 {
                    profiler.reset_frame(device, cmd, frame_index);
                    // Total frame timing is anchored to the stable first stage.
                    profiler.write_total_start(device, cmd, frame_index);
                }
                if stage != PipelineStage::FrameEnd {
                    profiler.write_stage_start(device, cmd, frame_index, stage);
                }

                f(cmd);

                if stage != PipelineStage::FrameEnd {
                    profiler.write_stage_end(device, cmd, frame_index, stage);
                }
                if stage as usize == PipelineStage::FrameEnd as usize - 1 {
                    // Total frame timing is anchored to the stable last real stage.
                    profiler.write_total_end(device, cmd, frame_index);
                }

                device.end_command_buffer(cmd).unwrap();
            }

            cmd_buffer_record(
                &self.device,
                &self.profiler,
                frame_index,
                PipelineStage::DataUpload,
                data_upload,
                |cmd| {
                    let object_data = self.objects.iter().map(|(_, obj)| GpuObjectInstance {
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
                    // Reverse-Z projection: near maps to 1.0, infinity tends toward 0.0.
                    let projection = Mat4::perspective_infinite_reverse_rh(
                        std::f32::consts::FRAC_PI_6,
                        self.core.surface_extent.width as f32 / self.core.surface_extent.height as f32,
                        0.1,
                    );

                    let p = Vec3::new(
                        self.cam_rot[0].sin() * self.cam_rot[1].cos(),
                        self.cam_rot[1].sin(),
                        -self.cam_rot[0].cos() * self.cam_rot[1].cos(),
                    );
                    let view = Mat4::look_to_rh(self.cam_pos, p, Vec3::new(0., 1., 0.));

                    // Frustum plane data.
                    let normalize_plane = |p: Vec4| p / p.xyz().length();
                    let temp = projection.transpose();
                    let frustum_x = normalize_plane(temp.w_axis + temp.x_axis);
                    let frustum_y = normalize_plane(temp.w_axis + temp.y_axis);
                    let frustum = Vec4::from([frustum_x.x, frustum_x.z, frustum_y.y, frustum_y.z]);

                    // Upload scene data.
                    self.staging_buffer.reset();

                    self.staging_buffer.stage(
                        &self.device,
                        cmd,
                        &object_instance_buffer,
                        0,
                        object_data.collect::<Vec<_>>(),
                    );

                    self.staging_buffer.stage(
                        &self.device,
                        cmd,
                        frame_global_buffer,
                        0,
                        [GpuFrameGlobal {
                            pv: projection * view,
                            proj: projection,
                            view,
                            camera_position: self.cam_pos.extend(1.0),
                            camera_direction: p.extend(0.0),
                            light_position: Vec4::new(1.0, 0.0, 0.0, 1.0),
                            light_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
                            frustum,
                            screen_info: Vec4::new(
                                self.core.surface_extent.width as f32,
                                self.core.surface_extent.height as f32,
                                0.0,
                                0.0,
                            ),
                            meshlet_visibility_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default().buffer(visibility_buffer.vk_handle()),
                            ),
                            meshlet_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default().buffer(meshlet_instance_buffer.vk_handle()),
                            ),
                            draw_cmd_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default().buffer(indirect_cmd_buffer.vk_handle()),
                            ),
                            late_draw_cmd_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default().buffer(late_draw_cmd_buffer.vk_handle()),
                            ),
                            object_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default().buffer(object_instance_buffer.vk_handle()),
                            ),
                            frustum_passing_meshlet_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(frustum_passing_meshlet_buffer.vk_handle()),
                            ),
                            instances: meshlet_instance_buffer.len(),
                        }],
                    );
                    self.device.cmd_fill_buffer(
                        cmd,
                        indirect_cmd_buffer.vk_handle(),
                        GpuDrawCommandBuffer::LEN_OFFSET,
                        std::mem::size_of::<u32>() as u64,
                        0,
                    );
                    if !frustum_passing_meshlet_buffer.is_null() {
                        self.device.cmd_fill_buffer(
                            cmd,
                            frustum_passing_meshlet_buffer.vk_handle(),
                            GpuFrustumPassingMeshletBuffer::LEN_OFFSET,
                            std::mem::size_of::<u32>() as u64,
                            0,
                        );
                    }
                    if !late_draw_cmd_buffer.is_null() {
                        self.device.cmd_fill_buffer(
                            cmd,
                            late_draw_cmd_buffer.vk_handle(),
                            GpuDrawCommandBuffer::LEN_OFFSET,
                            std::mem::size_of::<u32>() as u64,
                            0,
                        );
                    }

                    let mut buffer_barriers = vec![
                        vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(
                                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                            )
                            .buffer(indirect_cmd_buffer.vk_handle())
                            .offset(0)
                            .size(indirect_cmd_buffer.size() as u64),
                    ];
                    if !frustum_passing_meshlet_buffer.is_null() {
                        buffer_barriers.push(
                            vk::BufferMemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                .dst_access_mask(
                                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                                )
                                .buffer(frustum_passing_meshlet_buffer.vk_handle())
                                .offset(0)
                                .size(frustum_passing_meshlet_buffer.size() as u64),
                        );
                    }
                    if !late_draw_cmd_buffer.is_null() {
                        buffer_barriers.push(
                            vk::BufferMemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                .dst_access_mask(
                                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                                )
                                .buffer(late_draw_cmd_buffer.vk_handle())
                                .offset(0)
                                .size(late_draw_cmd_buffer.size() as u64),
                        );
                    }
                },
            );

            cmd_buffer_record(
                &self.device,
                &self.profiler,
                frame_index,
                PipelineStage::FrustumCull,
                frustum_cull,
                |cmd| {
                    // Copy visibility history into the current frame's buffer.
                    let visibility_copy_size = visibility_buffer.size().min(last_visibility_buffer.size());
                    if visibility_copy_size > 0 {
                        self.device.cmd_copy_buffer(
                            cmd,
                            last_visibility_buffer.vk_handle(),
                            visibility_buffer.vk_handle(),
                            &[vk::BufferCopy::default().src_offset(0).dst_offset(0).size(visibility_copy_size as u64)],
                        );
                    }

                    // TMP: start newly appended visibility entries as visible for testing.
                    if visibility_copy_size < visibility_buffer.size() {
                        self.device.cmd_fill_buffer(
                            cmd,
                            visibility_buffer.vk_handle(),
                            visibility_copy_size as u64,
                            (visibility_buffer.size() - visibility_copy_size) as u64,
                            1,
                        );
                    }

                    // Visibility history was filled/copy-updated on TRANSFER, so make it visible to the frustum cull
                    // compute pass.
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().buffer_memory_barriers(&[vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(
                                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                            )
                            .buffer(visibility_buffer.vk_handle())
                            .offset(0)
                            .size(visibility_buffer.size() as u64)]),
                    );

                    // The indirect buffer is also rewritten during upload and needs to be visible before compute reads
                    // it.
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().buffer_memory_barriers(&[vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(
                                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                            )
                            .buffer(indirect_cmd_buffer.vk_handle())
                            .offset(0)
                            .size(indirect_cmd_buffer.size() as u64)]),
                    );

                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.frustum_cull_pipeline_layout,
                        0,
                        &[frame_set],
                        &[],
                    );

                    self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.frustum_cull_pipeline);

                    self.device.cmd_dispatch(cmd, meshlet_instance_buffer.len().div_ceil(64), 1, 1);
                },
            );

            cmd_buffer_record(&self.device, &self.profiler, frame_index, PipelineStage::EarlyDraw, early_draw, |cmd| {
                self.device.cmd_bind_index_buffer(cmd, index_buffer.vk_handle(), 0, vk::IndexType::UINT32);

                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        // Reverse-Z clears to the farthest depth value.
                        depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                    });

                let render_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: self.core.surface_extent.width,
                            height: self.core.surface_extent.height,
                        },
                    })
                    .layer_count(1)
                    .depth_attachment(&depth_attachment);

                if overdraw_enabled {
                    self.device.cmd_clear_color_image(
                        cmd,
                        overdraw_image,
                        vk::ImageLayout::GENERAL,
                        &vk::ClearColorValue { uint32: [0, 0, 0, 0] },
                        &[vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)],
                    );

                    // The overdraw count image was cleared by transfer; make it available to fragment shader atomics.
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .image(overdraw_image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::GENERAL)]),
                    );

                    self.device.cmd_begin_rendering(cmd, &render_info.color_attachments(&[]));
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.overdraw_render_pipeline_layout,
                        0,
                        &[global_set, overdraw_set],
                        &[],
                    );
                    self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.overdraw_render_pipeline);
                } else {
                    // Swapchain image must move from presentable usage to color attachment usage for the normal render
                    // path.
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
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
                            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)]),
                    );

                    self.device.cmd_begin_rendering(
                        cmd,
                        &render_info.color_attachments(&[vk::RenderingAttachmentInfo::default()
                            .image_view(swapchain_view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .clear_value(vk::ClearValue {
                                color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
                            })]),
                    );

                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.render_pipeline_layout,
                        0,
                        &[global_set, frame_set],
                        &[],
                    );
                    self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.render_pipeline);
                }

                self.device.cmd_draw_indexed_indirect_count(
                    cmd,
                    indirect_cmd_buffer.vk_handle(),
                    GpuDrawCommandBuffer::DATA_OFFSET,
                    indirect_cmd_buffer.vk_handle(),
                    GpuDrawCommandBuffer::LEN_OFFSET,
                    meshlet_instance_buffer.len(),
                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );

                self.device.cmd_end_rendering(cmd);
            });

            cmd_buffer_record(&self.device, &self.profiler, frame_index, PipelineStage::BuildHzb, build_hzb, |cmd| {
                // HZB is sampled from the previous frame and rewritten by this frame's reduction passes.
                self.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        // Prepare the HZB for writing.
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .image(hzb_image)
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
                            .image(depth_images[frame_index].0)
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

                let build_hzb = |level: u32| {
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.build_hzb_pipeline_layout,
                        0,
                        &[hzb_set],
                        &[],
                    );

                    self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.build_hzb_pipeline);

                    let w =
                        self.core.surface_extent.width.div_ceil(2).checked_shr(level).unwrap_or(0).max(1).div_ceil(8);
                    let h =
                        self.core.surface_extent.height.div_ceil(2).checked_shr(level).unwrap_or(0).max(1).div_ceil(8);
                    self.device.cmd_dispatch_base(cmd, 0, 0, level, w, h, 1);

                    // Keep each mip level coherent as the reduction chain walks down the pyramid.
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .image(hzb_image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(level)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::GENERAL)]),
                    );
                };

                // For the first compute, the src view is the depth buffer, which
                // depends on the depth buffer.
                let mips = hzb_build_src_views.len() as u32;
                for level in 0..mips {
                    build_hzb(level);
                }

                // Return the HZB to sampled-read and the depth buffer to attachment-write for the next render frame.
                self.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        // Transition HZB back to read after all reductions.
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
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
                            .image(depth_images[frame_index].0)
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

            cmd_buffer_record(
                &self.device,
                &self.profiler,
                frame_index,
                PipelineStage::OcclusionCull,
                occlusion_cull,
                |cmd| {
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.occlusion_cull_pipeline_layout,
                        0,
                        &[hzb_set, frame_set],
                        &[],
                    );

                    self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.occlusion_cull_pipeline);

                    self.device.cmd_dispatch(cmd, meshlet_instance_buffer.len().div_ceil(64), 1, 1);
                },
            );

            cmd_buffer_record(&self.device, &self.profiler, frame_index, PipelineStage::LateDraw, late_draw, |cmd| {
                self.device.cmd_bind_index_buffer(cmd, index_buffer.vk_handle(), 0, vk::IndexType::UINT32);

                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE);

                let render_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: self.core.surface_extent.width,
                            height: self.core.surface_extent.height,
                        },
                    })
                    .layer_count(1)
                    .depth_attachment(&depth_attachment);

                if overdraw_enabled {
                    self.device.cmd_begin_rendering(cmd, &render_info.color_attachments(&[]));
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.overdraw_render_pipeline_layout,
                        0,
                        &[global_set, overdraw_set],
                        &[],
                    );
                    self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.overdraw_render_pipeline);
                } else {
                    self.device.cmd_begin_rendering(
                        cmd,
                        &render_info.color_attachments(&[vk::RenderingAttachmentInfo::default()
                            .image_view(swapchain_view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE)]),
                    );

                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.render_pipeline_layout,
                        0,
                        &[global_set, frame_set],
                        &[],
                    );

                    self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.render_pipeline);
                }

                self.device.cmd_draw_indexed_indirect_count(
                    cmd,
                    late_draw_cmd_buffer.vk_handle(),
                    GpuDrawCommandBuffer::DATA_OFFSET,
                    late_draw_cmd_buffer.vk_handle(),
                    GpuDrawCommandBuffer::LEN_OFFSET,
                    meshlet_instance_buffer.len(),
                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );

                self.device.cmd_end_rendering(cmd);

                if overdraw_enabled {
                    // In overdraw mode, the swapchain image becomes a storage image for the resolve compute pass.
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .image(swapchain_image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                            .new_layout(vk::ImageLayout::GENERAL)]),
                    );

                    // The resolve pass reads the accumulated counts after the fragment shader atomics finish.
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                            .image(overdraw_image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::GENERAL)]),
                    );

                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.overdraw_resolve_pipeline_layout,
                        0,
                        &[global_set, overdraw_set],
                        &[],
                    );
                    self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.overdraw_resolve_pipeline);
                    self.device.cmd_dispatch(
                        cmd,
                        self.core.surface_extent.width.div_ceil(8),
                        self.core.surface_extent.height.div_ceil(8),
                        1,
                    );

                    // The compute resolve writes the final swapchain image, so transition it back to presentable usage.
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
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
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)]),
                    );
                }

                if !overdraw_enabled {
                    // Hand the swapchain image from color attachment output to presentation.
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
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
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)]),
                    );
                }
            });

            cmd_buffer_record(&self.device, &self.profiler, frame_index, PipelineStage::FrameEnd, frame_end, |_cmd| {
                // FrameEnd is intentionally empty; it only preserves the stage accounting / timeline structure.
            });

            let early_draw_waits = if overdraw_enabled {
                vec![
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(self.pipeline_semaphore)
                        .value(PipelineStage::EarlyDraw.start_value(frame_count))
                        .stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT),
                ]
            } else {
                vec![
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(image_acquired)
                        .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE),
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(self.pipeline_semaphore)
                        .value(PipelineStage::EarlyDraw.start_value(frame_count))
                        .stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT),
                ]
            };

            let frame_end_waits = if overdraw_enabled {
                vec![
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(self.pipeline_semaphore)
                        .value(PipelineStage::LateDraw.done_value(frame_count))
                        .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE),
                ]
            } else {
                vec![
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(self.pipeline_semaphore)
                        .value(PipelineStage::LateDraw.done_value(frame_count))
                        .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER),
                ]
            };

            // TODO: Submit all queues.
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(data_upload)])
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(PipelineStage::DataUpload.start_value(frame_count))
                            .stage_mask(vk::PipelineStageFlags2::TRANSFER)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(PipelineStage::DataUpload.done_value(frame_count))
                            .stage_mask(vk::PipelineStageFlags2::TRANSFER)])],
                    vk::Fence::null(),
                )
                .unwrap();
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(frustum_cull)])
                        .wait_semaphore_infos(&[
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(self.pipeline_semaphore)
                                .value(PipelineStage::FrustumCull.start_value(frame_count))
                                .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER),
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(self.pipeline_semaphore)
                                .value(PipelineStage::FrameEnd.done_value(frame_count - VISIBILITY_DEPTH))
                                .stage_mask(vk::PipelineStageFlags2::TRANSFER),
                        ])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(PipelineStage::FrustumCull.done_value(frame_count))
                            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])],
                    vk::Fence::null(),
                )
                .unwrap();
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(early_draw)])
                        .wait_semaphore_infos(&early_draw_waits)
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(PipelineStage::EarlyDraw.done_value(frame_count))
                            .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)])],
                    vk::Fence::null(),
                )
                .unwrap();
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(build_hzb)])
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(PipelineStage::BuildHzb.start_value(frame_count))
                            .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(PipelineStage::BuildHzb.done_value(frame_count))
                            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])],
                    vk::Fence::null(),
                )
                .unwrap();
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(occlusion_cull)])
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(PipelineStage::BuildHzb.done_value(frame_count))
                            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(PipelineStage::OcclusionCull.done_value(frame_count))
                            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])],
                    vk::Fence::null(),
                )
                .unwrap();
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(late_draw)])
                        .wait_semaphore_infos(&if overdraw_enabled {
                            vec![
                                vk::SemaphoreSubmitInfo::default()
                                    .semaphore(image_acquired)
                                    .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE),
                                vk::SemaphoreSubmitInfo::default()
                                    .semaphore(self.pipeline_semaphore)
                                    .value(PipelineStage::OcclusionCull.done_value(frame_count))
                                    .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER),
                            ]
                        } else {
                            vec![
                                vk::SemaphoreSubmitInfo::default()
                                    .semaphore(self.pipeline_semaphore)
                                    .value(PipelineStage::OcclusionCull.done_value(frame_count))
                                    .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER),
                            ]
                        })
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(PipelineStage::LateDraw.done_value(frame_count))
                            .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)])],
                    vk::Fence::null(),
                )
                .unwrap();

            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(frame_end)])
                        .wait_semaphore_infos(&frame_end_waits)
                        .signal_semaphore_infos(&[
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(self.pipeline_semaphore)
                                .value(PipelineStage::FrameEnd.done_value(frame_count))
                                .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(render_finished)
                                .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
                        ])],
                    vk::Fence::null(),
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

    unsafe fn rebuild_scene(&mut self) {
        let old_visibility_index_cache = self.visibility_index_cache.clone();

        // Generate vertex data for newly added meshes.
        let new_meshes: HashMap<MeshHandle, Box<[GpuVertex]>> = self
            .meshes
            .iter()
            .filter(|(k, _)| !self.vertex_buffers.contains_key(k))
            .map(|(id, mesh)| {
                (
                    *id,
                    mesh.1
                        .iter()
                        .flat_map(|meshlet| {
                            (0..meshlet.positions.len()).map(|i| GpuVertex {
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
                Buffer::<[GpuVertex]>::new(
                    &self.allocator,
                    vertices.len() as u32,
                    vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    vk_mem::MemoryUsage::AutoPreferDevice,
                ),
            );
        }

        // Generate index data once per unique mesh, then reference those packed ranges from objects.
        let mut sorted_meshes: Vec<_> = self.meshes.iter().collect();
        sorted_meshes.sort_by_key(|(handle, _)| handle.0);

        let mut indices = vec![];
        let mut mesh_index_offset_cache = HashMap::with_capacity(sorted_meshes.len());
        let mut meshlet_first_index_cache = HashMap::with_capacity(sorted_meshes.len());
        for (handle, mesh) in sorted_meshes {
            let mesh_base_index = indices.len() as u32;
            mesh_index_offset_cache.insert(*handle, mesh_base_index);

            let meshlets = &mesh.1;
            let mut first_index = 0u32;
            let mut meshlet_first_indices = Vec::with_capacity(meshlets.len());
            let mut offset = 0u32;
            for meshlet in meshlets {
                meshlet_first_indices.push(first_index);
                indices.extend(meshlet.indices.iter().map(|&index| index as u32 + offset));
                offset += meshlet.positions.len() as u32;
                first_index += meshlet.indices.len() as u32;
            }
            meshlet_first_index_cache.insert(*handle, meshlet_first_indices.into_boxed_slice());
        }

        let mut meshlet_data = vec![];
        let mut new_visibility_index_cache = HashMap::with_capacity(self.objects.len());
        let mut object_meshlet_counts = HashMap::with_capacity(self.objects.len());
        let mut instances = 0u32;
        let mut triangle_count = 0usize;
        for (i, (handle, object)) in self.objects.iter().enumerate() {
            // Get associated mesh and its packed index range.
            let mesh = &self.meshes.get(&object.mesh).unwrap().1;
            let meshlet_start = instances;
            new_visibility_index_cache.insert(*handle, meshlet_start);
            object_meshlet_counts.insert(*handle, mesh.len() as u32);

            // Mesh data.
            let mesh_base_index = mesh_index_offset_cache.get(&object.mesh).copied().unwrap();
            let meshlet_first_indices = meshlet_first_index_cache.get(&object.mesh).unwrap();
            for (meshlet_idx, meshlet) in mesh.iter().enumerate() {
                triangle_count += meshlet.indices.len() / 3;
                meshlet_data.push(GpuMeshletInstance {
                    center: Vec3::from(meshlet.center),
                    radius: meshlet.radius,
                    cone_apex: Vec3::from(meshlet.cone_apex),
                    pad0: 0.,
                    cone_axis: Vec3::from(meshlet.cone_axis),
                    cone_cutoff: meshlet.cone_cutoff,
                    object_id: i as u32,
                    index_count: meshlet.indices.len() as u32,
                    first_index: mesh_base_index + meshlet_first_indices[meshlet_idx],
                });
                instances += 1;
            }
        }

        // Print some information about the scene.
        // TODO: Move this to the end?
        let object_count = self.objects.len();
        let meshlet_count = instances as usize;
        let new_mesh_count = new_meshes.len();
        let index_upload_bytes = indices.len() * std::mem::size_of::<u32>();
        let vertex_upload_bytes =
            new_meshes.iter().map(|(_, vertices)| vertices.len() * std::mem::size_of::<GpuVertex>()).sum::<usize>();
        let meshlet_upload_bytes = meshlet_data.len() * std::mem::size_of::<GpuMeshletInstance>();
        let total_upload_bytes = index_upload_bytes + vertex_upload_bytes + meshlet_upload_bytes;
        println!(
            "New scene data:\n  objects = {}\n  meshlets = {}\n  triangles = {}\n  new_meshes = {}\n  upload = {}\n  indices = {}\n  vertices = {}\n  meshlets = {}",
            format_usize_commas(object_count),
            format_usize_commas(meshlet_count),
            format_usize_commas(triangle_count),
            format_usize_commas(new_mesh_count),
            format_bytes(total_upload_bytes),
            format_bytes(index_upload_bytes),
            format_bytes(vertex_upload_bytes),
            format_bytes(meshlet_upload_bytes),
        );

        let index_buffer = Buffer::<[u32]>::new(
            &self.allocator,
            indices.len() as u32,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let object_instance_buffer = Buffer::<[GpuObjectInstance]>::new(
            &self.allocator,
            self.objects.len() as u32,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let meshlet_instance_buffer = Buffer::<[GpuMeshletInstance]>::new(
            &self.allocator,
            instances,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let indirect_cmd_buffer = Buffer::<GpuDrawCommandBuffer>::new_sized(
            &self.allocator,
            GpuDrawCommandBuffer::byte_size(instances),
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let frustum_passing_meshlet_buffers = std::array::from_fn(|_| {
            if instances > 0 {
                Buffer::<[u8]>::new(
                    &self.allocator,
                    GpuFrustumPassingMeshletBuffer::byte_size(instances) as u32,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    vk_mem::MemoryUsage::AutoPreferDevice,
                )
            } else {
                Buffer::null()
            }
        });

        let late_draw_cmd_buffers = std::array::from_fn(|_| {
            if instances > 0 {
                Buffer::<GpuDrawCommandBuffer>::new_sized(
                    &self.allocator,
                    GpuDrawCommandBuffer::byte_size(instances),
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::INDIRECT_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    vk_mem::MemoryUsage::AutoPreferDevice,
                )
            } else {
                Buffer::null()
            }
        });

        let visibility_stride = std::mem::size_of::<u32>() as u64;
        let old_visibility_buffers =
            std::mem::replace(&mut self.visibility_buffers, std::array::from_fn(|_| Buffer::null()));
        let mut new_visibility_buffers = std::array::from_fn(|_| Buffer::null());
        for slot in 0..VISIBILITY_BUFFER_COUNT {
            new_visibility_buffers[slot] = if instances > 0 {
                Buffer::<[u32]>::new(
                    &self.allocator,
                    instances,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    vk_mem::MemoryUsage::AutoPreferDevice,
                )
            } else {
                Buffer::null()
            };
        }

        self.device.wait_for_fences(&[self.staging_fence], true, u64::MAX).unwrap();
        self.device.reset_fences(&[self.staging_fence]).unwrap();

        self.device.reset_command_buffer(self.staging_cmd_buffer, vk::CommandBufferResetFlags::empty()).unwrap();
        self.device.begin_command_buffer(self.staging_cmd_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

        // Upload.
        self.staging_buffer.reset();
        self.staging_buffer.stage(&self.device, self.staging_cmd_buffer, &index_buffer, 0, indices);
        for (id, vertices) in &new_meshes {
            self.staging_buffer.stage(
                &self.device,
                self.staging_cmd_buffer,
                self.vertex_buffers.get(id).unwrap(),
                0,
                vertices.as_ref(),
            );
        }
        self.staging_buffer.stage(&self.device, self.staging_cmd_buffer, &meshlet_instance_buffer, 0, meshlet_data);
        for slot in 0..VISIBILITY_BUFFER_COUNT {
            let new_buffer = &new_visibility_buffers[slot];
            if new_buffer.is_null() {
                continue;
            }

            self.device.cmd_fill_buffer(
                self.staging_cmd_buffer,
                new_buffer.vk_handle(),
                0,
                new_buffer.size() as u64,
                1,
            );

            let old_buffer = &old_visibility_buffers[slot];
            if old_buffer.is_null() {
                continue;
            }

            for (handle, old_start) in &old_visibility_index_cache {
                let Some(new_start) = new_visibility_index_cache.get(handle).copied() else {
                    continue;
                };
                let Some(meshlet_count) = object_meshlet_counts.get(handle).copied() else {
                    continue;
                };
                let copy_size = meshlet_count as u64 * visibility_stride;
                if copy_size == 0 {
                    continue;
                }

                self.device.cmd_copy_buffer(
                    self.staging_cmd_buffer,
                    old_buffer.vk_handle(),
                    new_buffer.vk_handle(),
                    &[vk::BufferCopy::default()
                        .src_offset(*old_start as u64 * visibility_stride)
                        .dst_offset(new_start as u64 * visibility_stride)
                        .size(copy_size)],
                );
            }
        }

        // Submit (& wait at end of function).
        self.device.end_command_buffer(self.staging_cmd_buffer).unwrap();
        self.device
            .queue_submit(
                self.graphics_queue,
                &[vk::SubmitInfo::default().command_buffers(&[self.staging_cmd_buffer])],
                self.staging_fence,
            )
            .unwrap();

        // Wait for transfer to complete before returning.
        self.device.wait_for_fences(&[self.staging_fence], true, u64::MAX).unwrap();

        self.visibility_buffers = new_visibility_buffers;
        self.visibility_index_cache = new_visibility_index_cache;
        let retire_after = PipelineStage::OcclusionCull.done_value(self.frame.saturating_sub(1));
        for old_buffer in old_visibility_buffers {
            if !old_buffer.is_null() {
                self.visibility_buffer_retire_list.push((retire_after, old_buffer));
            }
        }

        // Scene is ready, push.
        self.scene_resources.push(SceneResources {
            index_buffer,
            object_instance_buffer,
            meshlet_instance_buffer,
            indirect_cmd_buffer,
            frustum_passing_meshlet_buffers,
            late_draw_cmd_buffers,
        });
    }

    unsafe fn rebuild_swapchain(&mut self) {
        use vk_mem::Alloc;

        // Swapchain rebuilds are fairly rare, and fundamentally experience intrusive, so for the sake of simplicity,
        // we're going to wait until the pipeline is caught up to now (IE, there are no FIF) before rebuilding.
        if self.swapchain_resources.is_some() && self.frame > 0 {
            self.wait_for_pipeline_stage(self.frame, PipelineStage::DataUpload);
        }

        // Attempt to take and free the current swapchain resources.
        let hzb_sets = self
            .swapchain_resources
            .take()
            .map(|r| r.free(&self.device, &self.allocator, self.cmd_pool))
            .unwrap_or_else(|| {
                self.device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(self.global_descriptor_pool)
                            .set_layouts(&[self.hzb_set_layout; SwapchainResources::HZB_SLOT_COUNT]),
                    )
                    .unwrap()
                    .try_into()
                    .unwrap()
            });

        // TODO: Eventually, we will want variable resolutions.
        let vk::Extent2D { width, height, .. } = self.core.surface_extent;
        if width > MAX_HZB_DIMENSION || height > MAX_HZB_DIMENSION {
            panic!("HZB/occlusion descriptor set only supports up to {MAX_HZB_DIMENSION}; got {width}x{height}");
        }

        // Calculate hzb dimensions for later.
        let hzb_width = width.div_ceil(2); // div2 for half resolution
        let hzb_height = height.div_ceil(2); // ..
        let mipmaps = u32::max(hzb_width, hzb_height).ilog2() + 1;

        if mipmaps > MAX_HZB_MIPS {
            panic!("HZB mip chain exceeds reserved descriptor range: {mipmaps} mips > {MAX_HZB_MIPS}");
        }

        /* Build the HZB images and image views: */
        let mut hzb_allocs: [Option<vk_mem::Allocation>; SwapchainResources::HZB_SLOT_COUNT] =
            std::array::from_fn(|_| None);
        let hzb_images = std::array::from_fn(|slot| {
            let (hzb_image, hzb_alloc) = self
                .allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .extent(vk::Extent3D { width: hzb_width, height: hzb_height, depth: 1 })
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
            hzb_allocs[slot] = Some(hzb_alloc);
            hzb_image
        });
        let hzb_allocs = hzb_allocs.map(|alloc| alloc.unwrap());

        let hzb_build_src_views = std::array::from_fn(|slot| {
            (0..mipmaps)
                .into_iter()
                .map(|level| {
                    self.device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(hzb_images[slot])
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
                .collect::<Vec<_>>()
                .into_boxed_slice()
        });

        let hzb_build_dst_views = std::array::from_fn(|slot| {
            (0..mipmaps)
                .into_iter()
                .map(|level| {
                    self.device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(hzb_images[slot])
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
                .collect::<Vec<_>>()
                .into_boxed_slice()
        });

        let render_finished = (0..self.swapchain.images.len())
            .into_iter()
            .map(|_| self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap())
            .collect();

        let image_acquired_semaphores =
            std::array::from_fn(|_| self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap());

        // Per-frame recorded render buffers.
        let cmd_buffers = std::array::from_fn(|_| {
            self.device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(PipelineStage::COUNT as _),
                )
                .unwrap()
                .try_into()
                .unwrap()
        });

        // Create depth attachment for rendering.
        let depth_images = std::array::from_fn(|_| {
            use vk_mem::Alloc;
            self.allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .extent(
                            vk::Extent3D::default()
                                .width(self.core.surface_extent.width)
                                .height(self.core.surface_extent.height)
                                .depth(1),
                        )
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .format(vk::Format::D32_SFLOAT)
                        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED),
                    &vk_mem::AllocationCreateInfo {
                        required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        ..Default::default()
                    },
                )
                .unwrap()
        });

        let depth_views = std::array::from_fn(|i| {
            self.device
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

        let mut overdraw_allocs: [Option<vk_mem::Allocation>; MAX_FRAMES_IN_FLIGHT] = std::array::from_fn(|_| None);
        let overdraw_images = std::array::from_fn(|i| {
            let (image, alloc) = self
                .allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .extent(
                            vk::Extent3D::default()
                                .width(self.core.surface_extent.width)
                                .height(self.core.surface_extent.height)
                                .depth(1),
                        )
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .format(vk::Format::R32_UINT)
                        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST),
                    &vk_mem::AllocationCreateInfo {
                        required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        ..Default::default()
                    },
                )
                .unwrap();
            overdraw_allocs[i] = Some(alloc);
            image
        });
        let overdraw_allocs = overdraw_allocs.map(|alloc| alloc.unwrap());

        let overdraw_views = std::array::from_fn(|i| {
            self.device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(overdraw_images[i])
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R32_UINT)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
                .unwrap()
        });

        debug_assert!(SwapchainResources::HZB_SLOT_COUNT <= MAX_FRAMES_IN_FLIGHT);
        let hzb_build_depth_views = std::array::from_fn(|slot| {
            debug_assert!(slot < depth_views.len());
            depth_views[slot]
        });

        // Write descriptors for each HZB scratch slot.
        for slot in 0..SwapchainResources::HZB_SLOT_COUNT {
            let hzb_src_infos: Box<_> = hzb_build_src_views[slot]
                .iter()
                .map(|&image_view| {
                    vk::DescriptorImageInfo::default()
                        .image_view(image_view)
                        .sampler(self.hzb_sampler)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                })
                .collect();

            let hzb_dst_infos: Box<_> = hzb_build_dst_views[slot]
                .iter()
                .map(|&image_view| {
                    vk::DescriptorImageInfo::default().image_view(image_view).image_layout(vk::ImageLayout::GENERAL)
                })
                .collect();

            let depth_info = [vk::DescriptorImageInfo::default()
                .image_view(hzb_build_depth_views[slot])
                .sampler(self.hzb_sampler)
                .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)];

            self.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(hzb_sets[slot])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(depth_info.len() as u32)
                        .image_info(&depth_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(hzb_sets[slot])
                        .dst_binding(0)
                        .dst_array_element(1)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(hzb_src_infos.len() as u32)
                        .image_info(&hzb_src_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(hzb_sets[slot])
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(hzb_dst_infos.len() as u32)
                        .image_info(&hzb_dst_infos),
                ],
                &[],
            );
        }

        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            let overdraw_info = [vk::DescriptorImageInfo::default()
                .image_view(overdraw_views[slot])
                .image_layout(vk::ImageLayout::GENERAL)];

            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(self.overdraw_sets[slot])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .image_info(&overdraw_info)],
                &[],
            );
        }

        // Transition some of the resouces we just made.
        {
            // Start with hzb transition.
            let mut tmp = Vec::with_capacity(SwapchainResources::HZB_SLOT_COUNT + MAX_FRAMES_IN_FLIGHT * 2);
            for slot in 0..SwapchainResources::HZB_SLOT_COUNT {
                tmp.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .image(hzb_images[slot])
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
                );
            }

            tmp.extend((0..MAX_FRAMES_IN_FLIGHT).into_iter().map(|i| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .image(overdraw_images[i])
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
            }));

            // Add depth transitions.
            tmp.extend((0..MAX_FRAMES_IN_FLIGHT).into_iter().map(|i| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    )
                    .image(depth_images[i].0)
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

            /* Write command buffer and queue. */
            self.device.wait_for_fences(&[self.staging_fence], true, u64::MAX).unwrap();
            self.device.reset_fences(&[self.staging_fence]).unwrap();

            self.device.reset_command_buffer(self.staging_cmd_buffer, vk::CommandBufferResetFlags::empty()).unwrap();
            self.device.begin_command_buffer(self.staging_cmd_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();
            // Initialize the HZB images and depth images before the first frame uses them.
            self.device.cmd_pipeline_barrier2(
                self.staging_cmd_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&tmp),
            );
            self.device.end_command_buffer(self.staging_cmd_buffer).unwrap();

            self.device
                .queue_submit(
                    self.graphics_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[self.staging_cmd_buffer])],
                    self.staging_fence,
                )
                .unwrap();
        };

        // Wait for transfer to complete before returning.
        self.device.wait_for_fences(&[self.staging_fence], true, u64::MAX).unwrap();

        self.swapchain_resources = Some(SwapchainResources {
            hzb_images,
            hzb_allocs,
            hzb_build_depth_views,
            hzb_build_src_views,
            hzb_build_dst_views,
            hzb_sets,
            overdraw_images,
            overdraw_allocs,
            overdraw_views,

            render_finished,
            image_acquired_semaphores,
            cmd_buffers,
            depth_images,
            depth_views,
        });

        println!("SwapchainResource rebuilt!");
    }

    pub fn create_object(
        &mut self,
        mesh: MeshHandle,
        position: Vec3,
        scale: f32,
        orientation: Quat,
    ) -> Option<ObjectHandle> {
        self.scene_resources_dirty = true;
        let handle = ObjectHandle(self.resource_counter);
        self.resource_counter += 1;
        self.objects.push((handle, Object { mesh, position, scale, orientation }));
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
