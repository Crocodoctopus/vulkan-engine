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

#[derive(Debug)]
pub(super) struct Object {
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

/* Resources that need regeneration when Swapchain changes */
struct SwapchainResources {
    /* TODO: Currently, only 1 FIF can use these, so we only need 1 */
    hzb_image: vk::Image,
    _hzb_alloc: vk_mem::Allocation,
    _hzb_test_view: vk::ImageView,
    hzb_build_src_views: Box<[vk::ImageView]>,
    _hzb_build_dst_views: Box<[vk::ImageView]>,

    render_finished: Box<[vk::Semaphore]>,
    image_acquired_semaphores: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    cmd_buffers: [[vk::CommandBuffer; PipelineStage::COUNT]; MAX_FRAMES_IN_FLIGHT],
    depth_images: [(vk::Image, vk_mem::Allocation); MAX_FRAMES_IN_FLIGHT],
    depth_views: [vk::ImageView; MAX_FRAMES_IN_FLIGHT],
}

impl SwapchainResources {
    unsafe fn free(
        mut self,
        device: &ash::Device,
        allocator: &vk_mem::Allocator,
        cmd_pool: vk::CommandPool,
    ) {
        device.destroy_image_view(self._hzb_test_view, None);
        for view in self.hzb_build_src_views {
            device.destroy_image_view(view, None);
        }
        for view in self._hzb_build_dst_views {
            device.destroy_image_view(view, None);
        }
        allocator.destroy_image(self.hzb_image, &mut self._hzb_alloc);

        for semaphore in self.render_finished {
            device.destroy_semaphore(semaphore, None);
        }
        for semaphore in self.image_acquired_semaphores {
            device.destroy_semaphore(semaphore, None);
        }

        for cmd_buffers in self.cmd_buffers {
            device.free_command_buffers(cmd_pool, &cmd_buffers);
        }

        for view in self.depth_views {
            device.destroy_image_view(view, None);
        }
        for (image, mut alloc) in self.depth_images {
            allocator.destroy_image(image, &mut alloc);
        }
    }
}

/* Resources that need regeneration when object set changes */
struct SceneResources {
    visibility_buffer: Buffer<u32>,
    indirect_cmd_buffer: Buffer<vk::DrawIndexedIndirectCommand>,
    indirect_count_buffer: Buffer<u32>,

    /* TODO: These are static after creation */
    index_buffer: Buffer<u32>,
    object_instance_buffer: Buffer<GpuObjectInstance>,
    meshlet_instance_buffer: Buffer<GpuMeshletInstance>,
}

impl SceneResources {
    unsafe fn free(self, allocator: &vk_mem::Allocator) {
        // Almost certainly this is wrong
        self.visibility_buffer.destroy(&allocator);

        self.index_buffer.destroy(&allocator);
        self.object_instance_buffer.destroy(&allocator);
        self.meshlet_instance_buffer.destroy(&allocator);
        self.indirect_cmd_buffer.destroy(&allocator);
        self.indirect_count_buffer.destroy(&allocator);
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
    query_pool: vk::QueryPool,
    cmd_pool: vk::CommandPool,

    /* Desciptor set layouts: */
    _global_set_layout: vk::DescriptorSetLayout,
    _frame_set_layout: vk::DescriptorSetLayout,

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
    objects: Vec<(ObjectHandle, Object)>,
    vertex_buffers: HashMap<MeshHandle, Buffer<GpuVertex>>,

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
    frame_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    frame_global_buffers: [Buffer<FrameGlobal>; MAX_FRAMES_IN_FLIGHT],

    // Used for sequencing stages, and other cross-frame syncing.
    pipeline_semaphore: vk::Semaphore,

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
                        &vk::PipelineLayoutCreateInfo::default()
                            .set_layouts(&[global_set_layout, frame_set_layout]),
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
                                .descriptor_count(1)])
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

            let frame_sets = std::array::from_fn(|fif| {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pools[fif])
                            .set_layouts(&[frame_set_layout]),
                    )
                    .unwrap()[0]
            });

            let frame_global_buffers = std::array::from_fn(|_| {
                Buffer::new(
                    &allocator,
                    1,
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
            }

            //
            Self {
                core,

                device,
                graphics_queue,
                present_queue,

                allocator,

                swapchain,

                query_pool,
                cmd_pool,

                _global_set_layout: global_set_layout,
                _frame_set_layout: frame_set_layout,

                frustum_cull_pipeline_layout,
                frustum_cull_pipeline,
                render_pipeline_layout,
                render_pipeline,
                build_hzb_pipeline_layout,
                build_hzb_pipeline,

                cwd: cwd.as_ref().to_owned(),
                resource_counter: 0,
                meshes: HashMap::new(),
                objects: Vec::new(),
                vertex_buffers: HashMap::new(),

                staging_buffer,
                staging_cmd_buffer,
                staging_fence,

                _global_descriptor_pool: global_descriptor_pool,
                _descriptor_pools: descriptor_pools,

                hzb_sampler,
                global_set,
                frame_sets,
                frame_global_buffers,

                pipeline_semaphore,

                swapchain_resources_dirty: true,
                scene_resources_dirty: true,

                swapchain_resources: None,
                scene_resources: vec![],

                frame: 0,
                cam_pos: Vec3::new(0., 0., 3.),
                cam_rot: <_>::default(),
            }
        }
    }

    pub fn render(&mut self, _timestamp: f32) {
        if self.swapchain_resources_dirty {
            self.swapchain_resources_dirty = false;
            unsafe {
                self.rebuild_swapchain();
            }
            println!("SwapchainResource regenerated!");
        }

        if self.scene_resources_dirty {
            self.scene_resources_dirty = false;
            unsafe {
                self.rebuild_scene();
            }
            println!("SceneResource regenerated!");
        }

        let frame_index = self.frame % MAX_FRAMES_IN_FLIGHT;
        let frame_count = self.frame;
        self.frame += 1;

        // Wait if we have too many frames in flight.
        if frame_count >= MAX_FRAMES_IN_FLIGHT {
            unsafe {
                self.wait_for_pipeline_stage(
                    frame_count - MAX_FRAMES_IN_FLIGHT + 1,
                    PipelineStage::DataUpload,
                );
            }
        }

        // Attempt to clean old scenes:
        let mut scene_resources = vec![self.scene_resources.pop().unwrap()];
        while let Some(scene) = self.scene_resources.pop() {
            let all_signalled = if frame_count == 0 {
                false
            } else {
                unsafe {
                    self.device
                        .get_semaphore_counter_value(self.pipeline_semaphore)
                        .unwrap()
                        >= PipelineStage::FrameEnd.done_value(frame_count - 1)
                }
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
            visibility_buffer,
            indirect_cmd_buffer,
            indirect_count_buffer,
            index_buffer,
            object_instance_buffer,
            meshlet_instance_buffer,
            ..
        } = self.scene_resources.last_mut().unwrap();

        let SwapchainResources {
            hzb_image,
            hzb_build_src_views,
            render_finished,
            image_acquired_semaphores,
            cmd_buffers,
            depth_images,
            depth_views,
            ..
        } = self.swapchain_resources.as_ref().unwrap();

        let image_acquired = image_acquired_semaphores[frame_index];

        // Command buffer associated with this frame.
        let data_upload = cmd_buffers[frame_index][PipelineStage::DataUpload as usize];
        let frustum_cull = cmd_buffers[frame_index][PipelineStage::FrustumCull as usize];
        let first_draw = cmd_buffers[frame_index][PipelineStage::FirstDraw as usize];
        let build_hzb = cmd_buffers[frame_index][PipelineStage::BuildHzb as usize];

        // Descriptor sets associated with this frame.
        let global_set = self.global_set;
        let frame_set = self.frame_sets[frame_index];
        let frame_global_buffer = &self.frame_global_buffers[frame_index];

        unsafe {
            // TODO: keep this here? Its a per-fif variable.
            let depth_view = depth_views[frame_index];

            //
            let (image_index, _) = self
                .swapchain
                .swapchain_device
                .acquire_next_image(
                    self.swapchain.swapchain,
                    u64::MAX,
                    image_acquired,
                    vk::Fence::null(),
                )
                .unwrap();
            let render_finished = render_finished[image_index as usize];
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
                    &object_instance_buffer,
                    0,
                    object_data,
                );

                self.staging_buffer.stage_buffer(
                    &self.device,
                    cmd,
                    frame_global_buffer,
                    0,
                    [FrameGlobal {
                        pv: projection * view,
                        proj: projection,
                        view,
                        camera_position: self.cam_pos.extend(1.0),
                        camera_direction: p.extend(0.0),
                        light_position: Vec4::new(1.0, 0.0, 0.0, 1.0),
                        light_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
                        frustum,
                        meshlet_visibility_buffer: self.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default()
                                .buffer(visibility_buffer.vk_handle()),
                        ),
                        draw_count_buffer: self.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default()
                                .buffer(indirect_count_buffer.vk_handle()),
                        ),
                        meshlet_buffer: self.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default()
                                .buffer(meshlet_instance_buffer.vk_handle()),
                        ),
                        draw_cmd_buffer: self.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default()
                                .buffer(indirect_cmd_buffer.vk_handle()),
                        ),
                        object_buffer: self.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default()
                                .buffer(object_instance_buffer.vk_handle()),
                        ),
                        instances: indirect_cmd_buffer.len,
                    }],
                );
                self.staging_buffer.stage_buffer(
                    &self.device,
                    cmd,
                    indirect_count_buffer,
                    0,
                    [0u32],
                );
            });

            cmd_buffer_record(&self.device, frustum_cull, |cmd| {
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.frustum_cull_pipeline_layout,
                    0,
                    &[frame_set],
                    &[],
                );

                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.frustum_cull_pipeline,
                );

                self.device
                    .cmd_dispatch(cmd, meshlet_instance_buffer.len().div_ceil(64), 1, 1);
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
                    &[global_set, frame_set],
                    &[],
                );

                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.render_pipeline,
                );

                self.device.cmd_bind_index_buffer(
                    cmd,
                    index_buffer.vk_handle(),
                    0,
                    vk::IndexType::UINT32,
                );

                self.device.cmd_draw_indexed_indirect_count(
                    cmd,
                    indirect_cmd_buffer.vk_handle(),
                    0,
                    indirect_count_buffer.vk_handle(),
                    0,
                    meshlet_instance_buffer.len(),
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
                            .image(*hzb_image)
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
                                .image(*hzb_image)
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
                let mips = hzb_build_src_views.len();
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
                            .image(*hzb_image)
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

            // TODO: Submit all queues.
            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[
                            vk::CommandBufferSubmitInfo::default().command_buffer(data_upload)
                        ])
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
                        .command_buffer_infos(&[
                            vk::CommandBufferSubmitInfo::default().command_buffer(frustum_cull)
                        ])
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(PipelineStage::FrustumCull.start_value(frame_count))
                            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])
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
                        .command_buffer_infos(&[
                            vk::CommandBufferSubmitInfo::default().command_buffer(first_draw)
                        ])
                        .wait_semaphore_infos(&[
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(image_acquired)
                                .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE),
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(self.pipeline_semaphore)
                                .value(PipelineStage::FirstDraw.start_value(frame_count))
                                .stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT),
                        ])
                        .signal_semaphore_infos(&[
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(self.pipeline_semaphore)
                                .value(PipelineStage::FirstDraw.done_value(frame_count))
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
                            .value(PipelineStage::BuildHzb.start_value(frame_count))
                            .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.pipeline_semaphore)
                            .value(PipelineStage::FrameEnd.done_value(frame_count))
                            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])],
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
        for (i, (_, object)) in self.objects.iter().enumerate() {
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
                meshlet_data.push(GpuMeshletInstance {
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

        let visibility_buffer = Buffer::new(
            &self.allocator,
            instances,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let index_buffer = Buffer::new(
            &self.allocator,
            indices.len() as u32,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let object_instance_buffer = Buffer::new(
            &self.allocator,
            self.objects.len() as u32,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let meshlet_instance_buffer = Buffer::new(
            &self.allocator,
            instances,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let indirect_cmd_buffer = Buffer::new(
            &self.allocator,
            instances,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let indirect_count_buffer = Buffer::new(
            &self.allocator,
            1,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        self.device
            .wait_for_fences(&[self.staging_fence], true, u64::MAX)
            .unwrap();
        self.device.reset_fences(&[self.staging_fence]).unwrap();

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
        self.staging_buffer.reset();
        self.staging_buffer.stage_buffer(
            &self.device,
            self.staging_cmd_buffer,
            &index_buffer,
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
            &meshlet_instance_buffer,
            0,
            meshlet_data,
        );

        self.device.cmd_fill_buffer(
            self.staging_cmd_buffer,
            visibility_buffer.vk_handle(),
            0,
            visibility_buffer.size() as u64,
            1,
        );

        // Submit (& wait at end of function).
        self.device
            .end_command_buffer(self.staging_cmd_buffer)
            .unwrap();
        self.device
            .queue_submit(
                self.graphics_queue,
                &[vk::SubmitInfo::default().command_buffers(&[self.staging_cmd_buffer])],
                self.staging_fence,
            )
            .unwrap();

        // Wait for transfer to complete before returning.
        self.device
            .wait_for_fences(&[self.staging_fence], true, u64::MAX)
            .unwrap();

        // Scene is ready, push.
        self.scene_resources.push(SceneResources {
            visibility_buffer,
            index_buffer,
            object_instance_buffer,
            meshlet_instance_buffer,
            indirect_cmd_buffer,
            indirect_count_buffer,
        });
    }

    unsafe fn rebuild_swapchain(&mut self) {
        // If swapchain resources already exist, in-flight frames may still be
        // using their depth/HZB views through descriptors. Swapchain rebuilds are
        // rare, so wait for every FIF before replacing this resource group.
        if self.swapchain_resources.is_some() && self.frame > 0 {
            self.wait_for_pipeline_stage(self.frame, PipelineStage::DataUpload);
        }

        // Free the previous SwapchainResources.
        if let Some(swapchain_resources) = self.swapchain_resources.take() {
            swapchain_resources.free(&self.device, &self.allocator, self.cmd_pool);
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

        let hzb_build_src_views: Box<[vk::ImageView]> = (0..mipmaps)
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

        let hzb_build_dst_views: Box<[vk::ImageView]> = (0..mipmaps)
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

        let render_finished = (0..self.swapchain.images.len())
            .into_iter()
            .map(|_| {
                self.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .unwrap()
            })
            .collect();

        let image_acquired_semaphores = std::array::from_fn(|_| {
            self.device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap()
        });

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

        let hzb_src_infos: Box<_> = hzb_build_src_views
            .iter()
            .map(|&image_view| {
                vk::DescriptorImageInfo::default()
                    .image_view(image_view)
                    .sampler(self.hzb_sampler)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();

        let hzb_dst_infos: Box<_> = hzb_build_dst_views
            .iter()
            .map(|&image_view| {
                vk::DescriptorImageInfo::default()
                    .image_view(image_view)
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect();

        let depth_infos = [
            vk::DescriptorImageInfo::default()
                .image_view(depth_views[0])
                .sampler(self.hzb_sampler)
                .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL),
            vk::DescriptorImageInfo::default()
                .image_view(depth_views[1])
                .sampler(self.hzb_sampler)
                .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL),
        ];

        // Write the depth buffers to index 0 & 1, followed by HZB mips.
        // This is safe because all FIF are done.
        self.device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::default()
                    .dst_set(self.global_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(depth_infos.len() as u32)
                    .image_info(&depth_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.global_set)
                    .dst_binding(0)
                    .dst_array_element(depth_infos.len() as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(hzb_src_infos.len() as u32)
                    .image_info(&hzb_src_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.global_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(hzb_dst_infos.len() as u32)
                    .image_info(&hzb_dst_infos),
            ],
            &[],
        );

        // Transition some of the resouces we just made.
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
            self.device
                .wait_for_fences(&[self.staging_fence], true, u64::MAX)
                .unwrap();
            self.device.reset_fences(&[self.staging_fence]).unwrap();

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
            self.device.cmd_pipeline_barrier2(
                self.staging_cmd_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&tmp),
            );
            self.device
                .end_command_buffer(self.staging_cmd_buffer)
                .unwrap();

            self.device
                .queue_submit(
                    self.graphics_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[self.staging_cmd_buffer])],
                    self.staging_fence,
                )
                .unwrap();
        };

        // Wait for transfer to complete before returning.
        self.device
            .wait_for_fences(&[self.staging_fence], true, u64::MAX)
            .unwrap();

        self.swapchain_resources = Some(SwapchainResources {
            hzb_image,
            _hzb_alloc: hzb_alloc,
            _hzb_test_view: hzb_test_view,
            hzb_build_src_views,
            _hzb_build_dst_views: hzb_build_dst_views,

            render_finished,
            image_acquired_semaphores,
            cmd_buffers,
            depth_images,
            depth_views,
        });
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
        self.objects.push((
            handle,
            Object {
                mesh,
                position,
                scale,
                orientation,
            },
        ));
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
