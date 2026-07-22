use crate::buffer::Buffer;
use crate::core::Core;
use crate::generation_queue::{Generation, GenerationQueue, Pending};
use crate::glsl_types::*;
use crate::image::{Image, ImageView};
use crate::mesh::{Mesh, load_mesh};
use crate::profiling::PipelineProfiler;
use crate::rw_queue::{ResourceQueue, WaitStrategy};
use crate::staging::{StagingBlock, StagingBuffer, Whole};
use crate::swapchain::Swapchain;
use crate::util::{format_bytes, format_usize_commas, wait_semaphores_any_fallback};
use crate::vk_helpers::*;
use ash::vk;
use glam::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::CStr;
use std::mem::offset_of;
use std::ops::Range;
use std::path::Path;
use std::path::PathBuf;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[derive(Debug)]
pub struct HandleCounter(u32);

#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Handle(u32);

impl Iterator for HandleCounter {
    type Item = Handle;
    fn next(&mut self) -> Option<Self::Item> {
        self.0 += 1;
        return Some(Handle(self.0));
    }
}

type MeshHandle = Handle;
type ObjectHandle = Handle;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Object {
    pub mesh: MeshHandle,
    pub position: Vec3,
    pub scale: f32,
    pub orientation: Quat,
}

#[derive(Debug, Clone, Default)]
struct WorldKeys {
    objects: BTreeSet<ObjectHandle>,
    meshes: BTreeSet<MeshHandle>,
}

#[allow(unused)]
const B: u64 = 1;
#[allow(non_upper_case_globals, unused)]
const KiB: u64 = 1024 * B;
#[allow(non_upper_case_globals, unused)]
const MiB: u64 = 1024 * KiB;
#[allow(non_upper_case_globals, unused)]
const GiB: u64 = 1024 * MiB;

pub(crate) const STAGING_ARENA_SIZE: u64 = 512 * MiB;
pub(crate) const STAGING_FIF_BLOCK_SIZE: u64 = 32 * MiB;
pub(crate) const STAGING_PENDING_BLOCK_SIZE: u64 = 64 * MiB;

pub(crate) const MAX_FRAMES_IN_FLIGHT: usize = 2;
pub(crate) const VISIBILITY_DEPTH: usize = 2;
pub(crate) const VISIBILITY_RESOURCE_QUEUE_LEN: usize = VISIBILITY_DEPTH + MAX_FRAMES_IN_FLIGHT;

// Dedicated HZB/occlusion descriptor set.
pub(crate) const MAX_HZB_DIMENSION: u32 = 8192;
pub(crate) const MAX_HZB_MIPS: u32 = MAX_HZB_DIMENSION.div_ceil(2).ilog2() + 1;
pub(crate) const HZB_SAMPLED_IMAGE_CAPACITY: u32 = 1 + MAX_HZB_MIPS;
pub(crate) const HZB_STORAGE_IMAGE_CAPACITY: u32 = MAX_HZB_MIPS;

/*Generate index
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

    const fn wait_value(self, base: u64) -> u64 {
        base as u64 as u64 + self as u64
    }

    const fn signal_value(self, base: u64) -> u64 {
        self.wait_value(base) + 1
    }
}

struct SwapchainResources {
    // HZB is per-frame scratch, but the ring only needs to be large enough to
    // cover the live visibility window.
    hzb_descriptor_pool: vk::DescriptorPool,
    hzb_images: [Image; MAX_FRAMES_IN_FLIGHT],
    hzb_build_src_views: [Box<[ImageView]>; MAX_FRAMES_IN_FLIGHT],
    hzb_build_dst_views: [Box<[ImageView]>; MAX_FRAMES_IN_FLIGHT],
    hzb_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    overdraw_images: [Image; MAX_FRAMES_IN_FLIGHT],
    overdraw_views: [ImageView; MAX_FRAMES_IN_FLIGHT],

    render_finished: Box<[vk::Semaphore]>,
    image_acquired_semaphores: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    depth_images: [Image; MAX_FRAMES_IN_FLIGHT],
    depth_views: [ImageView; MAX_FRAMES_IN_FLIGHT],
}

impl SwapchainResources {
    unsafe fn free(self, device: &ash::Device, allocator: &vk_mem::Allocator) {
        let Self {
            hzb_descriptor_pool,
            hzb_images,
            hzb_build_src_views,
            hzb_build_dst_views,
            hzb_sets: _,
            overdraw_images,
            overdraw_views,
            render_finished,
            image_acquired_semaphores,
            depth_images,
            depth_views,
        } = self;

        for views in hzb_build_src_views {
            for view in views {
                device.destroy_image_view(view.view, None);
            }
        }
        for views in hzb_build_dst_views {
            for view in views {
                device.destroy_image_view(view.view, None);
            }
        }
        for mut image in hzb_images {
            unsafe { allocator.destroy_image(image.image, &mut image.alloc) }
        }
        for view in overdraw_views {
            device.destroy_image_view(view.view, None);
        }
        for mut image in overdraw_images {
            unsafe { allocator.destroy_image(image.image, &mut image.alloc) }
        }

        for semaphore in render_finished {
            device.destroy_semaphore(semaphore, None);
        }
        for semaphore in image_acquired_semaphores {
            device.destroy_semaphore(semaphore, None);
        }

        for view in depth_views {
            device.destroy_image_view(view.view, None);
        }
        for mut image in depth_images {
            unsafe { allocator.destroy_image(image.image, &mut image.alloc) }
        }

        device.destroy_descriptor_pool(hzb_descriptor_pool, None);
    }
}

/* Resources that need regeneration when object set changes */
struct SceneResources {
    indirect_cmd_buffer: Buffer<GpuDrawCommandBuffer>,
    scene_index_buffer: Buffer<[GpuIndex]>,
    frustum_passing_meshlet_buffers: [Buffer<GpuFrustumPassingMeshletBuffer>; MAX_FRAMES_IN_FLIGHT],
    visibility_buffers: HashMap<ObjectHandle, ResourceQueue<Buffer<[u32]>>>,
    maximum_meshlets: u32,

    /* TODO: These are static after creation */
    object_instance_buffer: Buffer<[GpuObjectInstance]>,

    // Bookkeeping
    world_keys: WorldKeys,
    scene_index_offsets: HashMap<MeshHandle, u32>,
}

impl SceneResources {
    unsafe fn free(self, device: &ash::Device, allocator: &vk_mem::Allocator) {
        for buffer in self.frustum_passing_meshlet_buffers {
            buffer.destroy(&allocator);
        }
        self.scene_index_buffer.destroy(&allocator);
        for (_, queue) in self.visibility_buffers {
            for buffer in queue.free(device) {
                buffer.destroy(&allocator);
            }
        }
        self.object_instance_buffer.destroy(&allocator);
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
    _cmd_pool: vk::CommandPool,

    /* Desciptor set layouts: */
    _global_set_layout: vk::DescriptorSetLayout,
    hzb_set_layout: vk::DescriptorSetLayout,
    _frame_set_layout: vk::DescriptorSetLayout,
    _overdraw_set_layout: vk::DescriptorSetLayout,

    /* Pipelines: */
    frustum_cull_pipeline_layout: vk::PipelineLayout,
    frustum_cull_pipeline: vk::Pipeline,
    render_pipeline_layout: vk::PipelineLayout,
    render_pipeline: vk::Pipeline,
    overdraw_render_pipeline_layout: vk::PipelineLayout,
    overdraw_render_pipeline: vk::Pipeline,
    overshade_render_pipeline: vk::Pipeline,
    overdraw_resolve_pipeline_layout: vk::PipelineLayout,
    overdraw_resolve_pipeline: vk::Pipeline,
    build_hzb_pipeline_layout: vk::PipelineLayout,
    build_hzb_pipeline: vk::Pipeline,
    occlusion_cull_pipeline_layout: vk::PipelineLayout,
    occlusion_cull_pipeline: vk::Pipeline,

    /* Generic resource containers: */
    cwd: PathBuf,
    resource_counter: HandleCounter,
    objects: BTreeMap<ObjectHandle, Object>,
    meshes: BTreeMap<MeshHandle, Mesh>,

    // Some cpu -> gpu resources.
    vertex_buffers: HashMap<MeshHandle, Buffer<[GpuVertex]>>,
    index_buffers: HashMap<MeshHandle, Buffer<[GpuIndex]>>,
    meshlet_buffers: HashMap<MeshHandle, (Buffer<[GpuMeshlet]>, HashMap<u8, Range<u16>>)>,

    /* Staging: */
    staging: StagingBlock,

    /* Lone sampler: */
    hzb_sampler: vk::Sampler,

    /* Scene: */
    _descriptor_pools: [vk::DescriptorPool; MAX_FRAMES_IN_FLIGHT],
    _overdraw_descriptor_pools: [vk::DescriptorPool; MAX_FRAMES_IN_FLIGHT],

    // Submit command buffers.
    cmd_buffers: [[vk::CommandBuffer; PipelineStage::COUNT]; MAX_FRAMES_IN_FLIGHT],

    // Reusable FIF staging buffers.
    staging_buffers: [StagingBuffer; MAX_FRAMES_IN_FLIGHT],

    // Global & per FIF desciptor sets.
    global_set: vk::DescriptorSet,
    // (frame_sets and overdraw_sets are arrays of samplers)
    frame_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    overdraw_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],

    // Global buffer.
    frame_global_buffers: [Buffer<GpuFrameGlobal>; MAX_FRAMES_IN_FLIGHT],

    // Used for sequencing stages, and other cross-frame syncing.
    pipeline_semaphores: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],

    // Scene generation currently associated with each FIF slot.
    fif_scene_generations: [Generation; MAX_FRAMES_IN_FLIGHT],

    // Next frame timeline wait value for each slot.
    fif_timeline_waits: [u64; MAX_FRAMES_IN_FLIGHT],

    // Dirty flags for resource regeneration.
    swapchain_resources_dirty: bool,
    scene_resources_dirty: bool,

    // When rendering a FIF.
    scene_generation_manager: GenerationQueue,
    swapchain_generation_manager: GenerationQueue,
    scene_resources: BTreeMap<Generation, SceneResources>,
    swapchain_resources: BTreeMap<Generation, SwapchainResources>,

    // Various render state data.
    frame: usize,
    pub cam_pos: Vec3,
    pub cam_rot: Vec2, // YX
    pub overdraw_enabled: bool,
    pub overshade_enabled: bool,
}

const LOD_DISTANCE_BIAS: f32 = 2.0;
const LOD_DISTANCE_OFFSET: f32 = 0.25;

impl Drop for Renderer {
    fn drop(&mut self) {
        panic!("{} dropped implicitly; call explicit renderer shutdown before drop", std::any::type_name::<Self>());
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
                ..
            } = &core;

            // Create logical device and its associated queues.
            let (device, graphics_queue, present_queue) = {
                let features = vk::PhysicalDeviceFeatures::default()
                    .multi_draw_indirect(true)
                    .shader_int16(true)
                    .fragment_stores_and_atomics(true)
                    .vertex_pipeline_stores_and_atomics(true);
                let extensions = device_extensions.map(|x: &CStr| x.as_ptr());

                let device = {
                    let mut vk11features = vk::PhysicalDeviceVulkan11Features::default()
                        .shader_draw_parameters(true)
                        .storage_buffer16_bit_access(true)
                        .storage_push_constant16(true);

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

            let profiler = PipelineProfiler::new(&device, queue_family_index, graphics_queue);

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
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
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
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&[frame_set_layout]).push_constant_ranges(
                            &[vk::PushConstantRange::default()
                                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                                .offset(0)
                                .size(std::mem::size_of::<u32>() as u32)],
                        ),
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

                let overshade_pipeline = device
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
                                    .module(overshade_frag_shader)
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

            // Per-frame recorded render buffers.
            let cmd_buffers = std::array::from_fn(|_| {
                device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(cmd_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(PipelineStage::COUNT as _),
                    )
                    .unwrap()
                    .try_into()
                    .unwrap()
            });

            // Staging data.
            let mut staging = StagingBlock::new(&allocator, STAGING_ARENA_SIZE);
            let staging_buffers = std::array::from_fn(|_| StagingBuffer::new(&mut staging, STAGING_FIF_BLOCK_SIZE));

            //
            let global_descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&[
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1024 + (MAX_FRAMES_IN_FLIGHT as u32 * HZB_SAMPLED_IMAGE_CAPACITY)),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::STORAGE_IMAGE)
                                .descriptor_count(1024 + (MAX_FRAMES_IN_FLIGHT as u32 * HZB_STORAGE_IMAGE_CAPACITY)),
                        ])
                        .max_sets(1 + MAX_FRAMES_IN_FLIGHT as u32)
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
                                .ty(vk::DescriptorType::STORAGE_BUFFER)
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
                                    .ty(vk::DescriptorType::STORAGE_BUFFER)
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

            // Timeline semaphores for per-FIF stage sequencing.
            let pipeline_semaphores = std::array::from_fn(|_| {
                device
                    .create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(
                            &mut vk::SemaphoreTypeCreateInfo::default()
                                .semaphore_type(vk::SemaphoreType::TIMELINE)
                                .initial_value(0),
                        ),
                        None,
                    )
                    .unwrap()
            });

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
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    vk_mem::MemoryUsage::AutoPreferDevice,
                )
            });
            let scene_generation_manager = GenerationQueue::new(&device, queue_family_index);
            let swapchain_generation_manager = GenerationQueue::new(&device, queue_family_index);

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
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(&descriptor_buffer_infos)],
                    &[],
                );

                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(overdraw_sets[fif])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
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
                _cmd_pool: cmd_pool,

                _global_set_layout: global_set_layout,
                hzb_set_layout,
                _frame_set_layout: frame_set_layout,
                _overdraw_set_layout: overdraw_set_layout,

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

                cwd: cwd.as_ref().to_owned(),
                resource_counter: HandleCounter(0),
                objects: BTreeMap::new(),
                meshes: BTreeMap::new(),

                vertex_buffers: HashMap::new(),
                index_buffers: HashMap::new(),
                meshlet_buffers: HashMap::new(),

                staging,

                hzb_sampler,

                cmd_buffers,

                staging_buffers,

                _descriptor_pools: descriptor_pools,
                _overdraw_descriptor_pools: overdraw_descriptor_pools,

                global_set,
                frame_sets,
                overdraw_sets,
                frame_global_buffers,

                pipeline_semaphores,
                fif_scene_generations: [0; MAX_FRAMES_IN_FLIGHT],
                fif_timeline_waits: [0; MAX_FRAMES_IN_FLIGHT],
                swapchain_resources_dirty: true,
                scene_resources_dirty: true,

                scene_generation_manager,
                swapchain_generation_manager,
                scene_resources: BTreeMap::new(),
                swapchain_resources: BTreeMap::new(),

                frame: 0,
                cam_pos: Vec3::new(0., 0., 3.),
                cam_rot: <_>::default(),
                overdraw_enabled: false,
                overshade_enabled: false,
            }
        }
    }

    pub fn render(&mut self, _timestamp: f32) {
        self.frame += 1;

        if self.swapchain_resources_dirty {
            self.swapchain_resources_dirty = false;
            unsafe {
                self.build_swapchain();
            }
        }

        if self.scene_resources_dirty {
            self.scene_resources_dirty = false;
            unsafe {
                self.build_scene();
            }
        }

        // Wait for an availiable FIF slot.
        let (frame_index, frame_timeline_base) = unsafe {
            wait_semaphores_any_fallback(&self.device, &self.pipeline_semaphores, &self.fif_timeline_waits).unwrap();

            let index = self
                .pipeline_semaphores
                .iter()
                .zip(self.fif_timeline_waits.iter())
                .enumerate()
                .find(|(_, (semaphore, wait))| self.device.get_semaphore_counter_value(**semaphore).unwrap() == **wait)
                .unwrap()
                .0;

            let timeline = self.fif_timeline_waits[index];
            self.fif_timeline_waits[index] += PipelineStage::COUNT as u64;
            (index, timeline)
        };

        let pipeline_semaphore = self.pipeline_semaphores[frame_index];

        // Advance the scene and swapchain generation managers, then free retired generations.
        let scene_generation =
            unsafe { self.scene_generation_manager.next(&self.device, &mut self.staging, frame_index) };
        let swapchain_generation =
            unsafe { self.swapchain_generation_manager.next(&self.device, &mut self.staging, frame_index) };
        self.fif_scene_generations[frame_index] = scene_generation;

        // Clean any old generations.
        unsafe {
            // Scene:
            self.scene_generation_manager.retired_scenes(&self.device, &mut self.staging).for_each(|generation| {
                self.scene_resources
                    .remove(&generation)
                    .expect("Scene {generation} should exist!")
                    .free(&self.device, &self.allocator);
            });

            // Swapchain:
            self.swapchain_generation_manager.retired_scenes(&self.device, &mut self.staging).for_each(|generation| {
                self.swapchain_resources
                    .remove(&generation)
                    .expect("Swapchain {generation} should exist!")
                    .free(&self.device, &self.allocator)
            });
        }

        /* Post generation reserve resource extraction: */

        let SceneResources {
            indirect_cmd_buffer,
            scene_index_buffer,
            visibility_buffers,
            scene_index_offsets,
            maximum_meshlets,
            object_instance_buffer,
            frustum_passing_meshlet_buffers,
            world_keys,
            ..
        } = self.scene_resources.get_mut(&scene_generation).unwrap();
        let SwapchainResources {
            hzb_images,
            depth_views,
            hzb_build_src_views,
            hzb_sets,
            overdraw_images,
            overdraw_views: _,
            render_finished,
            image_acquired_semaphores,
            depth_images,
            ..
        } = self.swapchain_resources.get(&swapchain_generation).unwrap();

        let image_acquired = image_acquired_semaphores[frame_index];
        let hzb_image = &hzb_images[frame_index];
        let hzb_set = hzb_sets[frame_index];
        let hzb_build_src_views = &hzb_build_src_views[frame_index];
        let frustum_passing_meshlet_buffer = &frustum_passing_meshlet_buffers[frame_index];
        let visibility_wait_strategy = WaitStrategy {
            semaphore: pipeline_semaphore,
            value: PipelineStage::OcclusionCull.signal_value(frame_timeline_base),
        };

        // Command buffer associated with this frame.
        let data_upload = self.cmd_buffers[frame_index][PipelineStage::DataUpload as usize];
        let frustum_cull = self.cmd_buffers[frame_index][PipelineStage::FrustumCull as usize];
        let early_draw = self.cmd_buffers[frame_index][PipelineStage::EarlyDraw as usize];
        let build_hzb = self.cmd_buffers[frame_index][PipelineStage::BuildHzb as usize];
        let occlusion_cull = self.cmd_buffers[frame_index][PipelineStage::OcclusionCull as usize];
        let late_draw = self.cmd_buffers[frame_index][PipelineStage::LateDraw as usize];
        let frame_end = self.cmd_buffers[frame_index][PipelineStage::FrameEnd as usize];

        // Descriptor sets associated with this frame.
        let global_set = self.global_set;
        let frame_set = self.frame_sets[frame_index];
        let overdraw_set = self.overdraw_sets[frame_index];
        let frame_global_buffer = &self.frame_global_buffers[frame_index];
        let overdraw_enabled = self.overdraw_enabled;
        let overshade_enabled = self.overshade_enabled;
        let debug_draw_enabled = overdraw_enabled || overshade_enabled;
        let hzb_base_width = self.core.surface_extent.width.div_ceil(2).max(1);
        let hzb_base_height = self.core.surface_extent.height.div_ceil(2).max(1);

        unsafe {
            // TODO: keep this here? It's a per-FIF variable.
            let depth_view = &depth_views[frame_index];
            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(hzb_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .image_info(&[vk::DescriptorImageInfo::default()
                        .image_view(depth_view.view)
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
            let overdraw_image = &overdraw_images[frame_index];

            if debug_draw_enabled {
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
            let mut object_dispatch = Vec::with_capacity(world_keys.objects.len());
            let visibility_resource_waits = record_cmd_buffer(
                &self.device,
                &self.profiler,
                frame_index,
                PipelineStage::DataUpload,
                data_upload,
                |cmd| {
                    let mut visibility_resource_waits = Vec::new();
                    let camera_forward = Vec3::new(
                        self.cam_rot[0].sin() * self.cam_rot[1].cos(),
                        self.cam_rot[1].sin(),
                        -self.cam_rot[0].cos() * self.cam_rot[1].cos(),
                    );
                    let mut object_data = Vec::with_capacity(world_keys.objects.len());
                    for handle in world_keys.objects.iter() {
                        let obj = self.objects.get(handle).unwrap();

                        let mesh = self.meshes.get(&obj.mesh).unwrap();
                        let vertex_buffer = self.vertex_buffers.get(&obj.mesh).unwrap();
                        let (meshlet_buffer, lod_map) = self.meshlet_buffers.get(&obj.mesh).unwrap();
                        let scene_index_offset = *scene_index_offsets.get(&obj.mesh).unwrap();

                        let distance = (self.cam_pos - obj.position).length();
                        let object_radius = obj.scale * mesh.scale * mesh.radius;
                        let lod_ratio = ((distance.max(1e-5) / object_radius.max(1e-5)) - LOD_DISTANCE_OFFSET)
                            .max(1e-5)
                            * LOD_DISTANCE_BIAS;
                        let lod_id =
                            lod_ratio.log2().floor().clamp(0.0, (mesh.lod_count.saturating_sub(1)) as f32) as u8;
                        let meshlet_subrange = lod_map[&lod_id.min(mesh.lod_count)].clone();

                        // Push dispatch.
                        object_dispatch.push((object_data.len() as u16, meshlet_subrange.end - meshlet_subrange.start));

                        let (visibility_buffer, previous_visibility_buffer, waits) = {
                            let visibility_queue = visibility_buffers.get_mut(handle).unwrap();
                            let previous_visibility_buffer =
                                visibility_queue.read(&self.device, visibility_wait_strategy).vk_handle();
                            let (visibility_buffer, waits) =
                                visibility_queue.write(&self.device, frame_index, visibility_wait_strategy).unwrap();
                            (visibility_buffer.vk_handle(), previous_visibility_buffer, waits)
                        };
                        visibility_resource_waits.extend(waits);

                        object_data.push(GpuObjectInstance {
                            position: obj.position,
                            scale: obj.scale * mesh.scale,
                            orientation: obj.orientation,
                            vertex_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.vk_handle()),
                            ),
                            // This BDA is corrected for LOD subrange.
                            meshlet_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default().buffer(meshlet_buffer.vk_handle()),
                            ) + meshlet_subrange.start as u64
                                * std::mem::size_of::<GpuMeshlet>() as u64,
                            visibility_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default().buffer(visibility_buffer),
                            ),
                            previous_visibility_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default().buffer(previous_visibility_buffer),
                            ),
                            texture_id: 0,
                            scene_index_offset,
                        });
                    }

                    // Sort objects by camera distance.
                    // TODO: consider gpu sorting?
                    object_dispatch.sort_unstable_by(|a, b| {
                        let a_pos = object_data[usize::from(a.0)].position;
                        let b_pos = object_data[usize::from(b.0)].position;
                        let a_distance = (self.cam_pos - a_pos).length_squared();
                        let b_distance = (self.cam_pos - b_pos).length_squared();
                        a_distance.total_cmp(&b_distance)
                    });

                    // Upload global descriptor data & object data.
                    // Reverse-Z projection: near maps to 1.0, infinity tends toward 0.0.
                    let projection = Mat4::perspective_infinite_reverse_rh(
                        std::f32::consts::FRAC_PI_6,
                        self.core.surface_extent.width as f32 / self.core.surface_extent.height as f32,
                        0.1,
                    );

                    let p = camera_forward;
                    let view = Mat4::look_to_rh(self.cam_pos, p, Vec3::new(0., 1., 0.));

                    // Frustum plane data.
                    let normalize_plane = |p: Vec4| p / p.xyz().length();
                    let temp = projection.transpose();
                    let frustum_x = normalize_plane(temp.w_axis + temp.x_axis);
                    let frustum_y = normalize_plane(temp.w_axis + temp.y_axis);
                    let frustum = Vec4::from([frustum_x.x, frustum_x.z, frustum_y.y, frustum_y.z]);

                    // Upload scene data.
                    let staging = &mut self.staging_buffers[frame_index];
                    staging.reset();
                    staging.stage(&self.device, cmd, &object_instance_buffer, Whole(object_data));

                    staging.stage(
                        &self.device,
                        cmd,
                        frame_global_buffer,
                        Whole(GpuFrameGlobal {
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
                            draw_cmd_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default().buffer(indirect_cmd_buffer.vk_handle()),
                            ),
                            object_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default().buffer(object_instance_buffer.vk_handle()),
                            ),
                            frustum_passing_meshlet_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(frustum_passing_meshlet_buffer.vk_handle()),
                            ),
                            occlusion_dispatch: vk::DispatchIndirectCommand { x: 0, y: 1, z: 1 },
                        }),
                    );

                    // Set indirect & frustum_passing lens to 0.
                    self.device.cmd_fill_buffer(
                        cmd,
                        indirect_cmd_buffer.vk_handle(),
                        0,
                        std::mem::size_of::<u32>() as u64,
                        0,
                    );
                    self.device.cmd_fill_buffer(
                        cmd,
                        frustum_passing_meshlet_buffer.vk_handle(),
                        0,
                        std::mem::size_of::<u32>() as u64,
                        0,
                    );

                    visibility_resource_waits
                },
            );
            record_cmd_buffer(
                &self.device,
                &self.profiler,
                frame_index,
                PipelineStage::FrustumCull,
                frustum_cull,
                |cmd| {
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.frustum_cull_pipeline_layout,
                        0,
                        &[frame_set],
                        &[],
                    );

                    self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.frustum_cull_pipeline);

                    for (object_index, meshlet_count) in object_dispatch {
                        let push_constants = u32::from(object_index) | (u32::from(meshlet_count) << 16);
                        self.device.cmd_push_constants(
                            cmd,
                            self.frustum_cull_pipeline_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            &push_constants.to_ne_bytes(),
                        );

                        self.device.cmd_dispatch(cmd, (meshlet_count as u32).div_ceil(64), 1, 1);
                    }
                },
            );

            record_cmd_buffer(&self.device, &self.profiler, frame_index, PipelineStage::EarlyDraw, early_draw, |cmd| {
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view.view)
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

                if debug_draw_enabled {
                    self.device.cmd_clear_color_image(
                        cmd,
                        overdraw_image.image,
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
                            .image(overdraw_image.image)
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
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
                    self.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        if overshade_enabled { self.overshade_render_pipeline } else { self.overdraw_render_pipeline },
                    );
                } else {
                    // Swapchain image must move from presentable usage to color attachment usage for the normal render
                    // path.
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .image(swapchain_image)
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
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

                self.device.cmd_bind_index_buffer(cmd, scene_index_buffer.vk_handle(), 0, vk::IndexType::UINT32);
                self.device.cmd_draw_indexed_indirect_count(
                    cmd,
                    indirect_cmd_buffer.vk_handle(),
                    std::mem::size_of::<GpuIndex>() as u64,
                    indirect_cmd_buffer.vk_handle(),
                    0,
                    *maximum_meshlets,
                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );

                self.device.cmd_end_rendering(cmd);
            });

            record_cmd_buffer(&self.device, &self.profiler, frame_index, PipelineStage::BuildHzb, build_hzb, |cmd| {
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
                            .image(hzb_image.image)
                            .subresource_range(vk::ImageSubresourceRange {
                                level_count: vk::REMAINING_MIP_LEVELS,
                                ..COLOR_2D_SUBRESOURCE_RANGE
                            })
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
                            .image(depth_images[frame_index].image)
                            .subresource_range(DEPTH_2D_SUBRESOURCE_RANGE)
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

                    let w = hzb_base_width.checked_shr(level).unwrap_or(0).max(1).div_ceil(8);
                    let h = hzb_base_height.checked_shr(level).unwrap_or(0).max(1).div_ceil(8);
                    self.device.cmd_dispatch_base(cmd, 0, 0, level, w, h, 1);

                    // Keep each mip level coherent as the reduction chain walks down the pyramid.
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .image(hzb_image.image)
                            .subresource_range(vk::ImageSubresourceRange {
                                base_mip_level: level,
                                level_count: 1,
                                ..COLOR_2D_SUBRESOURCE_RANGE
                            })
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
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .image(hzb_image.image)
                            .subresource_range(vk::ImageSubresourceRange {
                                level_count: vk::REMAINING_MIP_LEVELS,
                                ..COLOR_2D_SUBRESOURCE_RANGE
                            })
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .dst_stage_mask(
                                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                            )
                            .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                            .image(depth_images[frame_index].image)
                            .subresource_range(DEPTH_2D_SUBRESOURCE_RANGE)
                            .old_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
                            .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL),
                    ]),
                );
            });

            record_cmd_buffer(
                &self.device,
                &self.profiler,
                frame_index,
                PipelineStage::OcclusionCull,
                occlusion_cull,
                |cmd| {
                    // Reuse the indirect buffer for the late list only after early draw has consumed it.
                    self.device.cmd_fill_buffer(
                        cmd,
                        indirect_cmd_buffer.vk_handle(),
                        0,
                        std::mem::size_of::<u32>() as u64,
                        0,
                    );

                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.occlusion_cull_pipeline_layout,
                        0,
                        &[hzb_set, frame_set],
                        &[],
                    );

                    self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.occlusion_cull_pipeline);

                    self.device.cmd_dispatch_indirect(
                        cmd,
                        frame_global_buffer.vk_handle(),
                        offset_of!(GpuFrameGlobal, occlusion_dispatch) as u64,
                    );
                },
            );

            record_cmd_buffer(&self.device, &self.profiler, frame_index, PipelineStage::LateDraw, late_draw, |cmd| {
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view.view)
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

                if debug_draw_enabled {
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

                self.device.cmd_bind_index_buffer(cmd, scene_index_buffer.vk_handle(), 0, vk::IndexType::UINT32);
                self.device.cmd_draw_indexed_indirect_count(
                    cmd,
                    indirect_cmd_buffer.vk_handle(),
                    std::mem::size_of::<GpuIndex>() as u64,
                    indirect_cmd_buffer.vk_handle(),
                    0,
                    *maximum_meshlets,
                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );

                self.device.cmd_end_rendering(cmd);

                if debug_draw_enabled {
                    // In overdraw mode, the swapchain image becomes a storage image for the resolve compute pass.
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .image(swapchain_image)
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
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
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)]),
                    );
                }

                if !debug_draw_enabled {
                    // Hand the swapchain image from color attachment output to presentation.
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                            .image(swapchain_image)
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)]),
                    );
                }
            });

            record_cmd_buffer(&self.device, &self.profiler, frame_index, PipelineStage::FrameEnd, frame_end, |_cmd| {
                // FrameEnd is intentionally empty; it only preserves the stage accounting / timeline structure.
            });

            // TODO: Submit all queues.
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(data_upload)])
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::DataUpload.wait_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::TRANSFER)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::DataUpload.signal_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::TRANSFER)])],
                    vk::Fence::null(),
                )
                .unwrap();
            let frustum_cmd_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(frustum_cull)];
            let frustum_signal_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(pipeline_semaphore)
                .value(PipelineStage::FrustumCull.signal_value(frame_timeline_base))
                .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)];
            let mut frustum_waits = Vec::with_capacity(2 + visibility_resource_waits.len());
            frustum_waits.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(pipeline_semaphore)
                    .value(PipelineStage::FrustumCull.wait_value(frame_timeline_base))
                    .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER),
            );
            frustum_waits.extend(visibility_resource_waits);
            let frustum_submit_infos = [vk::SubmitInfo2::default()
                .command_buffer_infos(&frustum_cmd_infos)
                .wait_semaphore_infos(&frustum_waits)
                .signal_semaphore_infos(&frustum_signal_infos)];
            self.device.queue_submit2(self.graphics_queue, &frustum_submit_infos, vk::Fence::null()).unwrap();
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(early_draw)])
                        .wait_semaphore_infos(&[
                            // We can delay this wait until late_draw when overdraw mode is on, but its probably not a
                            // useful optimization.
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(pipeline_semaphore)
                                .value(PipelineStage::EarlyDraw.wait_value(frame_timeline_base))
                                .stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT),
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(image_acquired)
                                .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE),
                        ])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::EarlyDraw.signal_value(frame_timeline_base))
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
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::BuildHzb.wait_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::BuildHzb.signal_value(frame_timeline_base))
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
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::BuildHzb.signal_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::OcclusionCull.signal_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])],
                    vk::Fence::null(),
                )
                .unwrap();
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(late_draw)])
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::OcclusionCull.signal_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::LateDraw.signal_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)])],
                    vk::Fence::null(),
                )
                .unwrap();
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(frame_end)])
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::LateDraw.signal_value(frame_timeline_base))
                            .stage_mask(match debug_draw_enabled {
                                true => vk::PipelineStageFlags2::TOP_OF_PIPE,
                                false => vk::PipelineStageFlags2::COMPUTE_SHADER,
                            })])
                        .signal_semaphore_infos(&[
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(pipeline_semaphore)
                                .value(PipelineStage::FrameEnd.signal_value(frame_timeline_base))
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

    unsafe fn build_scene(&mut self) {
        // Register generation.
        let (generation, pending) =
            self.scene_generation_manager.register(&self.device, &mut self.staging, STAGING_PENDING_BLOCK_SIZE);
        let Pending { cmd, staging, timeline, signal_value } = pending;
        let timeline_wait_value = generation;
        let timeline_signal_value = *signal_value;

        staging.reset();
        self.device.reset_command_buffer(*cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        self.device.begin_command_buffer(*cmd, &vk::CommandBufferBeginInfo::default()).unwrap();

        // Clone world keys.
        let new_world_keys = WorldKeys {
            objects: self.objects.keys().cloned().collect(),
            meshes: self.meshes.keys().cloned().collect(),
        };

        let previous_generation = generation.wrapping_sub(1);

        //  Cycle the keys back.
        let old_world_keys =
            self.scene_resources.get(&previous_generation).map(|s| s.world_keys.clone()).unwrap_or_default();
        let old_world_keys = &old_world_keys;
        let added_meshes: Vec<_> = new_world_keys.meshes.difference(&old_world_keys.meshes).copied().collect();
        let removed_meshes: Vec<_> = old_world_keys.meshes.difference(&new_world_keys.meshes).copied().collect();

        // TODO: Write this comment.
        for mesh_id in &added_meshes {
            let mesh = self.meshes.get(mesh_id).unwrap();

            // Extract vertex and index information for the entire LOD set.
            let mut vertices = vec![];
            let mut indices = vec![];
            let mut meshlets = vec![];
            let mut meshlet_lod_to_offset = HashMap::new();
            for lod in 0..mesh.lod_count {
                // Record lod to offset.
                let meshlet_offset = meshlets.len() as u16;

                // Push vertex/index model data.
                for meshlet in &mesh.lods[lod as usize] {
                    // Push this meshlet's metadata.
                    meshlets.push(GpuMeshlet {
                        center: Vec3::from(meshlet.center),
                        radius: meshlet.radius,
                        aabb_min: Vec3::from(meshlet.aabb_min),
                        cone_cutoff: meshlet.cone_cutoff,
                        aabb_max: Vec3::from(meshlet.aabb_max),
                        index_count: meshlet.indices.len() as u32,
                        cone_apex: Vec3::from(meshlet.cone_apex),
                        first_index: indices.len() as u32,
                        cone_axis: Vec3::from(meshlet.cone_axis),
                    });

                    // Push this meshlet's indices.
                    indices.extend(meshlet.indices.iter().map(|i| *i as u32 + vertices.len() as u32));

                    // Push this meshlet's vertices.
                    vertices.extend((0..meshlet.positions.len()).map(|i| GpuVertex {
                        position: meshlet.positions[i],
                        normal: meshlet.normals[i],
                        uv: [0, 0],
                    }));
                }

                // Log lod offset.
                meshlet_lod_to_offset.insert(lod, meshlet_offset..meshlets.len() as u16);
            }

            // Allocate index buffer.
            let meshlet_buffer = Buffer::<[GpuMeshlet]>::new(
                &self.allocator,
                meshlets.len() as u32,
                // TODO: confirm these.
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );

            // Allocate index buffer.
            let index_buffer = Buffer::<[GpuIndex]>::new(
                &self.allocator,
                indices.len() as u32,
                // TODO: confirm these.
                vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );

            // Allocate vertex buffer.
            let vertex_buffer = Buffer::<[GpuVertex]>::new(
                &self.allocator,
                vertices.len() as u32,
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );

            // Insert.
            self.vertex_buffers.insert(*mesh_id, vertex_buffer);
            self.index_buffers.insert(*mesh_id, index_buffer);
            self.meshlet_buffers.insert(*mesh_id, (meshlet_buffer, meshlet_lod_to_offset));

            // Stage the prebaked mesh payload while the source data is still here.
            staging.stage(
                &self.device,
                *cmd,
                &self.meshlet_buffers.get(mesh_id).unwrap().0,
                Whole(meshlets.as_slice()),
            );
            staging.stage(&self.device, *cmd, self.index_buffers.get(mesh_id).unwrap(), Whole(indices.as_slice()));
            staging.stage(&self.device, *cmd, self.vertex_buffers.get(mesh_id).unwrap(), Whole(vertices.as_slice()));
        }

        let mut scene_index_offsets = HashMap::with_capacity(new_world_keys.meshes.len());
        let mut scene_index_count = 0u32;
        for mesh_id in &new_world_keys.meshes {
            let index_buffer = self.index_buffers.get(mesh_id).unwrap();
            scene_index_offsets.insert(*mesh_id, scene_index_count);
            scene_index_count += index_buffer.len();
        }

        let scene_index_buffer = Buffer::<[GpuIndex]>::new(
            &self.allocator,
            scene_index_count.max(1),
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let scene_index_buffer_barriers: Vec<_> = new_world_keys
            .meshes
            .iter()
            .map(|mesh_id| {
                let index_buffer = self.index_buffers.get(mesh_id).unwrap();
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .buffer(index_buffer.vk_handle())
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
            })
            .collect();

        self.device.cmd_pipeline_barrier2(
            *cmd,
            &vk::DependencyInfo::default().buffer_memory_barriers(&scene_index_buffer_barriers),
        );

        for mesh_id in &new_world_keys.meshes {
            let index_buffer = self.index_buffers.get(mesh_id).unwrap();
            let dst_offset = scene_index_offsets[mesh_id] as u64 * std::mem::size_of::<GpuIndex>() as u64;
            self.device.cmd_copy_buffer(
                *cmd,
                index_buffer.vk_handle(),
                scene_index_buffer.vk_handle(),
                &[vk::BufferCopy::default().src_offset(0).dst_offset(dst_offset).size(index_buffer.size() as u64)],
            );
        }

        // The maximum meshlets that can be visible on screen.
        let maximum_scene_meshlets = new_world_keys
            .objects
            .iter()
            .map(|object_id| {
                let object = self.objects.get(object_id).unwrap();
                let mesh = self.meshes.get(&object.mesh).unwrap();
                mesh.lods.iter().map(|lod| lod.len() as u32).max().unwrap_or(0)
            })
            .sum::<u32>();

        let mut new_visibility_buffers = HashMap::with_capacity(new_world_keys.objects.len());
        let mut visibility_buffer_bytes = 0usize;
        for object_id in &new_world_keys.objects {
            let object = self.objects.get(object_id).unwrap();
            let mesh = self.meshes.get(&object.mesh).unwrap();
            let maximum_mesh_meshlets = mesh.lods.iter().map(|lod| lod.len()).max().unwrap_or(0);

            let visibility_buffers = (0..VISIBILITY_RESOURCE_QUEUE_LEN)
                .map(|_| {
                    Buffer::<[u32]>::new(
                        &self.allocator,
                        maximum_mesh_meshlets as u32,
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST,
                        vk_mem::MemoryUsage::AutoPreferDevice,
                    )
                })
                .collect::<Vec<_>>();
            visibility_buffer_bytes += visibility_buffers.iter().map(|buffer| buffer.size()).sum::<usize>();

            let previous_visibility = self
                .scene_resources
                .get_mut(&previous_generation)
                .and_then(|s| s.visibility_buffers.get_mut(object_id))
                .map(|queue| {
                    let buffer =
                        queue.read(&self.device, WaitStrategy { semaphore: *timeline, value: timeline_signal_value });
                    (buffer.vk_handle(), buffer.size())
                });

            match previous_visibility {
                // Buffer copy case.
                Some((previous_visibility, previous_visibility_size)) => {
                    for visibility_buffer in &visibility_buffers {
                        let copy_size = previous_visibility_size.min(visibility_buffer.size()) as u64;
                        if copy_size < visibility_buffer.size() as u64 {
                            self.device.cmd_fill_buffer(
                                *cmd,
                                visibility_buffer.vk_handle(),
                                copy_size,
                                visibility_buffer.size() as u64 - copy_size,
                                0,
                            );
                        }
                        if copy_size > 0 {
                            self.device.cmd_copy_buffer(
                                *cmd,
                                previous_visibility,
                                visibility_buffer.vk_handle(),
                                &[vk::BufferCopy::default().size(copy_size)],
                            );
                        }
                    }
                }
                // Buffer fill case.
                None => {
                    for visibility_buffer in &visibility_buffers {
                        self.device.cmd_fill_buffer(
                            *cmd,
                            visibility_buffer.vk_handle(),
                            0,
                            visibility_buffer.size() as u64,
                            0,
                        );
                    }
                }
            }

            new_visibility_buffers.insert(*object_id, ResourceQueue::new(MAX_FRAMES_IN_FLIGHT, visibility_buffers));
        }

        let object_instance_buffer = Buffer::<[GpuObjectInstance]>::new(
            &self.allocator,
            new_world_keys.objects.len() as u32,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let indirect_cmd_buffer = Buffer::<GpuDrawCommandBuffer>::new_trailing(
            &self.allocator,
            maximum_scene_meshlets,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let frustum_passing_meshlet_buffers = std::array::from_fn(|_| {
            Buffer::<GpuFrustumPassingMeshletBuffer>::new_trailing(
                &self.allocator,
                maximum_scene_meshlets,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            )
        });

        let total_meshlet_buffer_count = new_world_keys
            .meshes
            .iter()
            .map(|mesh_id| self.meshlet_buffers.get(mesh_id).unwrap().0.len() as usize)
            .sum::<usize>();
        let total_index_count = scene_index_count as usize;
        let total_triangle_count = total_index_count / 3;
        let scene_index_copy_bytes = total_index_count * std::mem::size_of::<GpuIndex>();
        let visibility_buffer_count = new_visibility_buffers.values().map(|queue| queue.len()).sum::<usize>();
        let object_instance_bytes = object_instance_buffer.size();
        let indirect_cmd_bytes = indirect_cmd_buffer.size();
        let frustum_passing_bytes = frustum_passing_meshlet_buffers.iter().map(|buffer| buffer.size()).sum::<usize>();
        let staging_bytes_used = staging.size() as usize;

        let mesh_upload_bytes = |mesh_id: &MeshHandle| -> usize {
            self.meshlet_buffers.get(mesh_id).unwrap().0.size()
                + self.index_buffers.get(mesh_id).unwrap().size()
                + self.vertex_buffers.get(mesh_id).unwrap().size()
        };
        let added_mesh_upload_bytes = added_meshes.iter().map(mesh_upload_bytes).sum::<usize>();

        println!(
            "Scene {} created ({} staged):\n  objects = {}\n  meshes = {} (+{}, -{}, {} new mesh payload)",
            generation,
            format_bytes(staging_bytes_used),
            format_usize_commas(new_world_keys.objects.len()),
            format_usize_commas(new_world_keys.meshes.len()),
            format_usize_commas(added_meshes.len()),
            format_usize_commas(removed_meshes.len()),
            format_bytes(added_mesh_upload_bytes),
        );

        for mesh_id in &added_meshes {
            let mesh = self.meshes.get(mesh_id).unwrap();
            let meshlet_count = mesh.lods.iter().map(|lod| lod.len()).sum::<usize>();
            let max_lod_meshlets = mesh.lods.iter().map(|lod| lod.len()).max().unwrap_or(0);
            let meshlet_buffer = &self.meshlet_buffers.get(mesh_id).unwrap().0;
            let index_buffer = self.index_buffers.get(mesh_id).unwrap();
            let vertex_buffer = self.vertex_buffers.get(mesh_id).unwrap();
            println!(
                "    + {:?}: lods = {}, meshlets = {} ({}) max_lod_meshlets = {}, indices = {} ({}), vertices = {} ({}), upload = {}",
                mesh_id,
                mesh.lod_count,
                format_usize_commas(meshlet_count),
                format_bytes(meshlet_buffer.size()),
                format_usize_commas(max_lod_meshlets),
                format_usize_commas(index_buffer.len() as usize),
                format_bytes(index_buffer.size()),
                format_usize_commas(vertex_buffer.len() as usize),
                format_bytes(vertex_buffer.size()),
                format_bytes(mesh_upload_bytes(mesh_id)),
            );
        }
        for mesh_id in &removed_meshes {
            println!("    - {:?}", mesh_id);
        }
        println!(
            "  meshlet buffers = {} meshlets\n  max visible meshlets = {}\n  scene indices = {} ({} triangles, {} GPU copy)\n  scene buffers = {} objects ({}), {} indirect commands ({}), {} frustum candidates ({})\n  visibility buffers = {} buffers ({})",
            format_usize_commas(total_meshlet_buffer_count),
            format_usize_commas(maximum_scene_meshlets as usize),
            format_usize_commas(total_index_count),
            format_usize_commas(total_triangle_count),
            format_bytes(scene_index_copy_bytes),
            format_usize_commas(new_world_keys.objects.len()),
            format_bytes(object_instance_bytes),
            format_usize_commas(maximum_scene_meshlets as usize),
            format_bytes(indirect_cmd_bytes),
            format_usize_commas(maximum_scene_meshlets as usize),
            format_bytes(frustum_passing_bytes),
            format_usize_commas(visibility_buffer_count),
            format_bytes(visibility_buffer_bytes),
        );

        // Submit and let the lifetime manager decide when the upload is safe to retire.
        self.device.end_command_buffer(*cmd).unwrap();
        self.device
            .queue_submit2(
                self.graphics_queue,
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(*cmd)])
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(*timeline)
                        .value(timeline_wait_value)
                        .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(*timeline)
                        .value(timeline_signal_value)
                        .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)])],
                vk::Fence::null(),
            )
            .unwrap();

        // Scene is ready, push.
        self.scene_resources.insert(
            generation,
            SceneResources {
                object_instance_buffer,
                indirect_cmd_buffer,
                scene_index_buffer,
                frustum_passing_meshlet_buffers,
                visibility_buffers: new_visibility_buffers,
                scene_index_offsets,
                maximum_meshlets: maximum_scene_meshlets,
                world_keys: new_world_keys,
            },
        );
    }

    unsafe fn build_swapchain(&mut self) {
        // Register generation.
        let (generation, pending) =
            self.swapchain_generation_manager.register(&self.device, &mut self.staging, STAGING_PENDING_BLOCK_SIZE);
        let Pending {
            cmd: staging_cmd_buffer,
            timeline: timeline_semaphore,
            signal_value: timeline_signal_value,
            ..
        } = pending;
        let timeline_wait_value = generation;
        let timeline_signal_value = *timeline_signal_value;

        /* Write command buffer and queue. */
        self.device.reset_command_buffer(*staging_cmd_buffer, vk::CommandBufferResetFlags::empty()).unwrap();
        self.device.begin_command_buffer(*staging_cmd_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

        let hzb_descriptor_pool = self
            .device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&[
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32 * HZB_SAMPLED_IMAGE_CAPACITY),
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32 * HZB_STORAGE_IMAGE_CAPACITY),
                    ])
                    .max_sets(MAX_FRAMES_IN_FLIGHT as u32)
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                None,
            )
            .unwrap();
        let hzb_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT] = self
            .device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(hzb_descriptor_pool)
                    .set_layouts(&[self.hzb_set_layout; MAX_FRAMES_IN_FLIGHT]),
            )
            .unwrap()
            .try_into()
            .unwrap();

        let vk::Extent2D { width, height, .. } = self.core.surface_extent;
        if width > MAX_HZB_DIMENSION || height > MAX_HZB_DIMENSION {
            panic!("HZB/occlusion descriptor set only supports up to {MAX_HZB_DIMENSION}; got {width}x{height}");
        }

        // Round the half-res base up to the next power of two so the mip chain is regular.
        let hzb_width = width.div_ceil(2).max(1).next_power_of_two();
        let hzb_height = height.div_ceil(2).max(1).next_power_of_two();
        let mipmaps = u32::max(hzb_width, hzb_height).ilog2() + 1;

        if mipmaps > MAX_HZB_MIPS {
            panic!("HZB mip chain exceeds reserved descriptor range: {mipmaps} mips > {MAX_HZB_MIPS}");
        }

        let create_image =
            |extent: vk::Extent2D, format: vk::Format, usage: vk::ImageUsageFlags, mip_levels: u32| -> Image {
                let create_info = image2d_create_info()
                    .extent(extent3d_from_extent2d(extent))
                    .format(format)
                    .usage(usage)
                    .mip_levels(mip_levels);
                let (image, alloc) = unsafe {
                    vk_mem::Alloc::create_image(&self.allocator, &create_info, &device_local_alloc()).unwrap()
                };
                Image { image, alloc }
            };

        let create_view = |image: vk::Image,
                           format: vk::Format,
                           aspect: vk::ImageAspectFlags,
                           base_mip_level: u32,
                           level_count: u32|
         -> ImageView {
            unsafe {
                let view = self
                    .device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: aspect,
                                base_mip_level,
                                level_count,
                                base_array_layer: 0,
                                layer_count: 1,
                            }),
                        None,
                    )
                    .unwrap();
                ImageView { view }
            }
        };

        /* Build the HZB images and image views: */
        let hzb_images = std::array::from_fn(|_| {
            create_image(
                vk::Extent2D { width: hzb_width, height: hzb_height },
                vk::Format::R32_SFLOAT,
                vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
                mipmaps,
            )
        });

        let hzb_build_src_views = std::array::from_fn(|slot| {
            (0..mipmaps)
                .into_iter()
                .map(|level| {
                    create_view(hzb_images[slot].image, vk::Format::R32_SFLOAT, vk::ImageAspectFlags::COLOR, level, 1)
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        });

        let hzb_build_dst_views = std::array::from_fn(|slot| {
            (0..mipmaps)
                .into_iter()
                .map(|level| {
                    create_view(hzb_images[slot].image, vk::Format::R32_SFLOAT, vk::ImageAspectFlags::COLOR, level, 1)
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

        // Create depth attachment for rendering.
        let depth_images = std::array::from_fn(|_| {
            create_image(
                self.core.surface_extent,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                1,
            )
        });

        let depth_views = std::array::from_fn(|i| {
            create_view(depth_images[i].image, vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH, 0, 1)
        });

        let overdraw_images = std::array::from_fn(|_| {
            create_image(
                self.core.surface_extent,
                vk::Format::R32_UINT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
                1,
            )
        });

        let overdraw_views = std::array::from_fn(|i| {
            create_view(overdraw_images[i].image, vk::Format::R32_UINT, vk::ImageAspectFlags::COLOR, 0, 1)
        });

        // Write descriptors for each HZB scratch slot.
        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            let hzb_src_infos: Box<_> = hzb_build_src_views[slot]
                .iter()
                .map(|image_view| {
                    vk::DescriptorImageInfo::default()
                        .image_view(image_view.view)
                        .sampler(self.hzb_sampler)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                })
                .collect();

            let hzb_dst_infos: Box<_> = hzb_build_dst_views[slot]
                .iter()
                .map(|image_view| {
                    vk::DescriptorImageInfo::default()
                        .image_view(image_view.view)
                        .image_layout(vk::ImageLayout::GENERAL)
                })
                .collect();

            let depth_info = [vk::DescriptorImageInfo::default()
                .image_view(depth_views[slot].view)
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
                .image_view(overdraw_views[slot].view)
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
            let mut tmp = Vec::new();
            for slot in 0..MAX_FRAMES_IN_FLIGHT {
                tmp.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .image(hzb_images[slot].image)
                        .subresource_range(COLOR_2D_SUBRESOURCE_RANGE.level_count(vk::REMAINING_MIP_LEVELS))
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL),
                );
            }

            tmp.extend((0..MAX_FRAMES_IN_FLIGHT).into_iter().map(|i| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .image(overdraw_images[i].image)
                    .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
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
                    .image(depth_images[i].image)
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

            // Initialize the HZB images and depth images before the first frame uses them.
            self.device
                .cmd_pipeline_barrier2(*staging_cmd_buffer, &vk::DependencyInfo::default().image_memory_barriers(&tmp));

            for slot in 0..MAX_FRAMES_IN_FLIGHT {
                self.device.cmd_clear_color_image(
                    *staging_cmd_buffer,
                    hzb_images[slot].image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] },
                    &[vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(mipmaps)
                        .base_array_layer(0)
                        .layer_count(1)],
                );
            }

            let tmp = Vec::from_iter((0..MAX_FRAMES_IN_FLIGHT).map(|slot| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .image(hzb_images[slot].image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(vk::REMAINING_MIP_LEVELS)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            }));

            self.device
                .cmd_pipeline_barrier2(*staging_cmd_buffer, &vk::DependencyInfo::default().image_memory_barriers(&tmp));

            self.device.end_command_buffer(*staging_cmd_buffer).unwrap();
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[
                            vk::CommandBufferSubmitInfo::default().command_buffer(*staging_cmd_buffer)
                        ])
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(*timeline_semaphore)
                            .value(timeline_wait_value)
                            .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(*timeline_semaphore)
                            .value(timeline_signal_value)
                            .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)])],
                    vk::Fence::null(),
                )
                .unwrap();
        };

        self.swapchain_resources.insert(
            generation,
            SwapchainResources {
                hzb_descriptor_pool,
                hzb_images,
                hzb_build_src_views,
                hzb_build_dst_views,
                hzb_sets,
                overdraw_images,
                overdraw_views,

                render_finished,
                image_acquired_semaphores,
                depth_images,
                depth_views,
            },
        );

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
        let handle = self.resource_counter.next().unwrap();
        self.objects.insert(handle, Object { mesh, position, scale, orientation });
        Some(handle)
    }

    pub fn load_mesh(&mut self, filename: impl AsRef<Path>) -> Option<MeshHandle> {
        let mesh = load_mesh(self.cwd.join(filename))?;
        let handle = self.resource_counter.next().unwrap();
        self.meshes.insert(handle, mesh);
        return Some(handle);
    }
}
