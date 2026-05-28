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

struct Frame {
    //
    scene_cmd_buffers: Box<[vk::CommandBuffer]>,

    // Descriptor sets.
    scene_set: vk::DescriptorSet,
    render_set: vk::DescriptorSet,
    cull_set: vk::DescriptorSet,

    // Various buffers.
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

impl Frame {
    fn null() -> Self {
        Self {
            scene_cmd_buffers: <_>::default(),
            scene_set: vk::DescriptorSet::null(),
            render_set: vk::DescriptorSet::null(),
            cull_set: vk::DescriptorSet::null(),
            index_buffer: Buffer::null(),
            object_buffer: Buffer::null(),
            instance_buffer: Buffer::null(),
            meshlet_data_buffer: Buffer::null(),
            indirect_cmd_buffer: Buffer::null(),
            indirect_count_buffer: Buffer::null(),
            scene_global_buffer: Buffer::null(),
            meshlet_cull_global_buffer: Buffer::null(),
            meshlet_render_global_buffer: Buffer::null(),
        }
    }
}

pub struct Renderer {
    //
    core: Core,

    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    //transfer_queue: vk::Queue,

    // Generic memory allocator.
    allocator: vk_mem::Allocator,

    // Swapchain data.
    swapchain: Swapchain,

    // Command pool.
    query_pool: vk::QueryPool,
    cmd_pool: vk::CommandPool,

    // Desciptor set layouts.
    scene_set_layout: vk::DescriptorSetLayout,
    render_set_layout: vk::DescriptorSetLayout,
    cull_set_layout: vk::DescriptorSetLayout,

    // Pipelines.
    render_pipeline_layout: vk::PipelineLayout,
    render_pipeline: vk::Pipeline,
    cull_pipeline_layout: vk::PipelineLayout,
    cull_pipeline: vk::Pipeline,

    // Generic resource containers.
    cwd: PathBuf,
    resource_counter: u32,
    meshes: HashMap<MeshHandle, (f32, Box<[Meshlet]>)>,
    objects: HashMap<ObjectHandle, ObjectInstance>,
    vertex_buffers: HashMap<MeshHandle, Buffer<Vertex>>,

    // Staging.
    staging_buffer: StagingBuffer,
    staging_cmd_buffer: vk::CommandBuffer,
    staging_fence: vk::Fence,

    // Frame.
    sync_primitives: [FrameSyncPrimitives; MAX_FRAMES_IN_FLIGHT],
    render_cmd_buffers: [vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT],
    descriptor_pools: [vk::DescriptorPool; MAX_FRAMES_IN_FLIGHT],
    render_finished: Box<[vk::Semaphore]>,
    current_frame: Frame,
    next_frame: Option<Frame>,

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
                        .descriptor_binding_partially_bound(true)
                        .descriptor_binding_sampled_image_update_after_bind(true)
                        .descriptor_indexing(true)
                        .runtime_descriptor_array(true);

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
            let swapchain = Swapchain::new(&core, &device, &allocator);

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
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .descriptor_count(1)
                                .stage_flags(vk::ShaderStageFlags::ALL),
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(1)
                                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1024)
                                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        ])
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

            // Create rendering pipeline.
            let (render_pipeline, render_pipeline_layout) = {
                let pipeline_layout = device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default()
                            .set_layouts(&[scene_set_layout, render_set_layout]),
                        None,
                    )
                    .unwrap();

                let vert_shader = create_shader_module(include_bytes!("shader.vert.spirv"));
                let frag_shader = create_shader_module(include_bytes!("shader.frag.spirv"));

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

            // Create cull compute pipeline.
            let (cull_pipeline, cull_pipeline_layout) = {
                let comp_shader = create_shader_module(include_bytes!("shader.comp.spirv"));

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
            let render_cmd_buffers = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(MAX_FRAMES_IN_FLIGHT as _),
                )
                .unwrap()
                .try_into()
                .unwrap();

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

            // Generic descriptor pool.
            let descriptor_pools = std::array::from_fn(|_| {
                device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .pool_sizes(&[
                                vk::DescriptorPoolSize::default()
                                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                                    .descriptor_count(3),
                                vk::DescriptorPoolSize::default()
                                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                    .descriptor_count(1024),
                            ])
                            .max_sets(3)
                            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                        None,
                    )
                    .unwrap()
            });

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

                scene_set_layout,
                render_set_layout,
                cull_set_layout,

                render_pipeline_layout,
                render_pipeline,
                cull_pipeline_layout,
                cull_pipeline,

                cwd: cwd.as_ref().to_owned(),
                resource_counter: 0,
                meshes: HashMap::new(),
                objects: HashMap::new(),
                vertex_buffers: HashMap::new(),

                staging_buffer,
                staging_cmd_buffer,
                staging_fence,

                sync_primitives,
                render_cmd_buffers,
                descriptor_pools,
                render_finished,
                current_frame: Frame::null(),
                next_frame: None,

                frame: 0,
                cam_pos: <_>::default(),
                cam_rot: <_>::default(),
            }
        }
    }

    pub fn render(&mut self, _timestamp: f32) {
        let frame_index = self.frame % MAX_FRAMES_IN_FLIGHT;
        self.frame += 1;

        unsafe {
            // A dirty hack. When a rebuild occurs, wait for transfer to fully complete.
            if self.frame == 1 {
                // Rebuild scene elements.
                self.current_frame = self.rebuild_scene(frame_index);
            }
        }

        // Attempt to clean up current_frame if there is a next.
        if self.next_frame.is_some() {
            // All frames are signalled.
            let signalled = self.sync_primitives.iter().fold(true, |acc, syncs| unsafe {
                acc & self.device.get_fence_status(syncs.frame_in_flight).unwrap()
            });

            if signalled {
                let frame =
                    std::mem::replace(&mut self.current_frame, self.next_frame.take().unwrap());

                // TODO: free self.current_frame.
                unsafe {
                    self.free_frame(frame);
                }
                println!("Old frame cleared.");
            }
        }

        // Get current working frame.
        let frame = match &self.next_frame {
            Some(frame) => frame,
            None => &self.current_frame,
        };

        // Sync primitives associated with this frame.
        let FrameSyncPrimitives {
            image_available,
            frame_in_flight,
        } = self.sync_primitives[frame_index];

        // Command buffer associated with this frame.
        let command_buffer = self.render_cmd_buffers[frame_index];

        unsafe {
            // Wait for next image to become available.
            self.device
                .wait_for_fences(&[frame_in_flight], true, u64::MAX)
                .unwrap();
            self.device.reset_fences(&[frame_in_flight]).unwrap();

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
            let scene_command_buffer = frame.scene_cmd_buffers[image_index as usize];

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

            // Reset and record.
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            self.device.cmd_reset_query_pool(
                command_buffer,
                self.query_pool,
                2 * frame_index as u32,
                2,
            );
            self.device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                (2 * frame_index + 0) as u32,
            );

            // Transfer some global state data.
            {
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

                // Upload data.
                {
                    self.staging_buffer.reset();

                    self.staging_buffer.stage_buffer(
                        &self.device,
                        command_buffer,
                        &frame.object_buffer,
                        0,
                        object_data,
                    );

                    self.staging_buffer.stage_buffer(
                        &self.device,
                        command_buffer,
                        &frame.scene_global_buffer,
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
                        command_buffer,
                        &frame.meshlet_render_global_buffer,
                        0,
                        [MeshletRenderGlobal {
                            instance_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(frame.instance_buffer.vk_handle()),
                            ),
                            object_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(frame.object_buffer.vk_handle()),
                            ),
                        }],
                    );
                    self.staging_buffer.stage_buffer(
                        &self.device,
                        command_buffer,
                        &frame.indirect_count_buffer,
                        0,
                        [0u32],
                    );
                    self.staging_buffer.stage_buffer(
                        &self.device,
                        command_buffer,
                        &frame.meshlet_cull_global_buffer,
                        0,
                        [MeshletCullGlobal {
                            instances: frame.indirect_cmd_buffer.len,
                            frustum,
                            draw_count_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(frame.indirect_count_buffer.vk_handle()),
                            ),
                            meshlet_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(frame.meshlet_data_buffer.vk_handle()),
                            ),
                            draw_cmd_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(frame.indirect_cmd_buffer.vk_handle()),
                            ),
                            instance_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(frame.instance_buffer.vk_handle()),
                            ),
                            object_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(frame.object_buffer.vk_handle()),
                            ),
                        }],
                    );
                }
            }

            self.device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                (2 * frame_index + 1) as u32,
            );
            self.device
                .cmd_execute_commands(command_buffer, &[scene_command_buffer]);

            self.device.end_command_buffer(command_buffer).unwrap();

            // Execute command buffer.
            self.device
                .queue_submit(
                    self.graphics_queue,
                    &[vk::SubmitInfo::default()
                        .wait_semaphores(&[image_available])
                        .signal_semaphores(&[render_finished])
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .command_buffers(&[command_buffer])],
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

    unsafe fn free_frame(&mut self, mut frame: Frame) {
        if frame.scene_cmd_buffers.len() > 0 {
            self.device
                .free_command_buffers(self.cmd_pool, &frame.scene_cmd_buffers);
        }
        frame.index_buffer.take().destroy(&self.allocator);
        frame.object_buffer.take().destroy(&self.allocator);
        frame.instance_buffer.take().destroy(&self.allocator);
        frame.meshlet_data_buffer.take().destroy(&self.allocator);
        frame.indirect_cmd_buffer.take().destroy(&self.allocator);
        frame.indirect_count_buffer.take().destroy(&self.allocator);
        frame
            .meshlet_cull_global_buffer
            .take()
            .destroy(&self.allocator);
        frame
            .meshlet_render_global_buffer
            .take()
            .destroy(&self.allocator);
    }

    unsafe fn rebuild_scene(&mut self, fif: usize) -> Frame {
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

        // Descriptor sets.
        let [scene_set, render_set, cull_set] = self
            .device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pools[fif])
                    .set_layouts(&[
                        self.scene_set_layout,
                        self.render_set_layout,
                        self.cull_set_layout,
                    ]),
            )
            .unwrap()
            .try_into()
            .unwrap();

        // TODO: split this up.
        let frame = Frame {
            scene_cmd_buffers: self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.cmd_pool)
                        .level(vk::CommandBufferLevel::SECONDARY)
                        .command_buffer_count(self.swapchain.images.len() as u32),
                )
                .unwrap()
                .into_boxed_slice(),

            scene_set,
            render_set,
            cull_set,

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

        // Rerecord scene command buffers.
        for i in 0..self.swapchain.images.len() {
            let command_buffer = frame.scene_cmd_buffers[i];
            let image = self.swapchain.images[i];
            let color_view = self.swapchain.color_views[i];
            let depth_view = self.swapchain.depth_views[i];

            unsafe {
                // Begin recording.
                self.device
                    .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                    .unwrap();
                self.device
                    .begin_command_buffer(
                        command_buffer,
                        &vk::CommandBufferBeginInfo::default()
                            .inheritance_info(&vk::CommandBufferInheritanceInfo::default()),
                    )
                    .unwrap();

                // Compute prepass.
                {
                    self.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        self.cull_pipeline_layout,
                        0,
                        &[frame.scene_set, frame.cull_set],
                        &[],
                    );

                    self.device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        self.cull_pipeline,
                    );

                    self.device
                        .cmd_dispatch(command_buffer, instances.div_ceil(64), 1, 1);
                }

                self.device.cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfo::default()
                        .memory_barriers(&[
                            //
                            vk::MemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                                .dst_stage_mask(vk::PipelineStageFlags2::INDEX_INPUT)
                                .dst_access_mask(vk::AccessFlags2::MEMORY_READ),
                        ])
                        .image_memory_barriers(&[
                            // Convert VK_IMAGE_LAYOUT_UNDEFINED -> VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL.
                            vk::ImageMemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                                .image(image)
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
                            //
                            vk::ImageMemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                                .image(self.swapchain.depth_image.0)
                                .subresource_range(
                                    vk::ImageSubresourceRange::default()
                                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                        .base_mip_level(0)
                                        .level_count(1)
                                        .base_array_layer(0)
                                        .layer_count(1),
                                )
                                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                                .old_layout(vk::ImageLayout::UNDEFINED)
                                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL),
                        ]),
                );

                // Begin rendering.
                {
                    self.device.cmd_begin_rendering(
                        command_buffer,
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
                                .image_view(color_view)
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
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.render_pipeline_layout,
                        0,
                        &[frame.scene_set, frame.render_set],
                        &[],
                    );

                    self.device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.render_pipeline,
                    );

                    self.device.cmd_bind_index_buffer(
                        command_buffer,
                        frame.index_buffer.vk_handle(),
                        0,
                        vk::IndexType::UINT32,
                    );

                    self.device.cmd_draw_indexed_indirect_count(
                        command_buffer,
                        frame.indirect_cmd_buffer.vk_handle(),
                        0,
                        frame.indirect_count_buffer.vk_handle(),
                        0,
                        instances,
                        size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                    );

                    self.device.cmd_end_rendering(command_buffer);
                }

                self.device.cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        // Convert VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL -> VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                            .image(image)
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
                        //
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                            .image(self.swapchain.depth_image.0)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR),
                    ]),
                );

                // End recording.
                self.device.end_command_buffer(command_buffer).unwrap();

                //
                self.device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(frame.scene_set)
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .buffer_info(&[vk::DescriptorBufferInfo::default()
                                .buffer(frame.scene_global_buffer.vk_handle())
                                .offset(0)
                                .range(vk::WHOLE_SIZE)]),
                        vk::WriteDescriptorSet::default()
                            .dst_set(frame.cull_set)
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .buffer_info(&[vk::DescriptorBufferInfo::default()
                                .buffer(frame.meshlet_cull_global_buffer.vk_handle())
                                .offset(0)
                                .range(vk::WHOLE_SIZE)]),
                        //
                        vk::WriteDescriptorSet::default()
                            .dst_set(frame.render_set)
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .buffer_info(&[vk::DescriptorBufferInfo::default()
                                .buffer(frame.meshlet_render_global_buffer.vk_handle())
                                .offset(0)
                                .range(vk::WHOLE_SIZE)]),
                    ],
                    &[],
                );
            }
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
