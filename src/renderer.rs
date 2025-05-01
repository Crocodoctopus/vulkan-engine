//use crate::util::Map2;
use ash::vk::{Extent2D, ImageUsageFlags};
use ash::{khr, vk};
use glam::*;
use itertools::Itertools;
use std::collections::HashMap;
use std::ffi::CStr;
use std::path::Path;
use std::path::PathBuf;
use vk_mem::Alloc;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::staging::StagingBuffer;

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
struct MeshletRenderGlobal {
    pv: Mat4,
    vertex_buffer: vk::DeviceAddress,
    instance_buffer: vk::DeviceAddress,
    object_buffer: vk::DeviceAddress,
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
struct MeshletCullGlobal {
    view: [f32; 4 * 4],
    camera_position: [f32; 3],
    instances: u32,

    frustum: [f32; 4],

    draw_count_buffer: vk::DeviceAddress,
    meshlet_buffer: vk::DeviceAddress,
    draw_cmd_buffer: vk::DeviceAddress,
    instance_buffer: vk::DeviceAddress,
    object_buffer: vk::DeviceAddress,
}

#[derive(Clone, Debug)]
#[repr(C, align(16))]
struct Object {
    model: Mat4,
    texture_id: u32,
}

#[derive(Clone, Debug)]
#[repr(C, align(16))]
struct Instance {
    object_id: u32,
}

#[derive(Clone, Debug)]
#[repr(C, align(16))]
struct MeshletData {
    // Culling.
    center: [f32; 3],
    radius: f32,
    cone_apex: [f32; 3],
    pad0: f32,
    cone_axis: [f32; 3],
    cone_cutoff: f32,

    // Draw cmd.
    object_id: u32,
    index_count: u32,
    first_index: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Debug, Default)]
struct Vertex {
    position: Vec3,
    u: f32,
    normal: Vec3,
    v: f32,
    color: Vec4,
}

#[derive(Debug)]
pub struct Buffer<T> {
    phantom: std::marker::PhantomData<T>,
    buffer: vk::Buffer,
    alloc: Option<vk_mem::Allocation>,
    len: u32,
}

fn create_buffer<T>(
    allocator: &vk_mem::Allocator,
    len: u32,
    vk_usage: vk::BufferUsageFlags,
    vma_usage: vk_mem::MemoryUsage,
) -> Buffer<T> {
    unsafe {
        let (buffer, alloc) = allocator
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(len as u64 * size_of::<T>() as u64)
                    .usage(vk_usage),
                &vk_mem::AllocationCreateInfo {
                    usage: vma_usage,
                    ..Default::default()
                },
            )
            .unwrap();

        Buffer {
            phantom: std::marker::PhantomData,
            buffer,
            alloc: Some(alloc),
            len,
        }
    }
}

unsafe fn destroy_buffer<T>(allocator: &vk_mem::Allocator, mut buffer: Buffer<T>) {
    if let Some(mut alloc) = buffer.alloc {
        allocator.destroy_buffer(buffer.buffer, &mut alloc);
    }
}

impl<T> Buffer<T> {
    pub fn null() -> Self {
        Self {
            phantom: std::marker::PhantomData,
            buffer: vk::Buffer::null(),
            alloc: None,
            len: 0,
        }
    }
    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn size(&self) -> usize {
        self.len as usize * size_of::<T>()
    }

    pub fn stride(&self) -> u32 {
        size_of::<T>() as u32
    }

    pub fn vk_handle(&self) -> vk::Buffer {
        self.buffer
    }
}

pub struct Meshlet {
    pub center: [f32; 3],
    pub radius: f32,
    pub cone_apex: [f32; 3],
    pub cone_axis: [f32; 3],
    pub cone_cutoff: f32,
    pub indices: Box<[u8]>,
    pub positions: Box<[f32]>,
    pub texcoords: Box<[f32]>,
}

struct ObjectInstance {
    mesh: MeshHandle,
    position: Vec3,
    orientation: Quat,
    scale: Vec3,
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct MeshHandle(u32);
#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct ObjectHandle(u32);

#[derive(Copy, Clone)]
struct FrameSyncPrimitives {
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    frame_in_flight: vk::Fence,
}

struct Frame {
    //
    sync_primitives: Box<[FrameSyncPrimitives]>,

    //
    scene_cmd_buffers: Box<[vk::CommandBuffer]>,

    // Various buffers.
    index_buffer: Buffer<u32>,
    vertex_buffer: Buffer<Vertex>,
    object_buffer: Buffer<Object>,
    instance_buffer: Buffer<Instance>,
    meshlet_data_buffer: Buffer<MeshletData>,
    indirect_cmd_buffer: Buffer<vk::DrawIndexedIndirectCommand>,
    indirect_count_buffer: Buffer<u32>,
    meshlet_cull_global_buffer: Buffer<MeshletCullGlobal>,
    meshlet_render_global_buffer: Buffer<MeshletRenderGlobal>,
}

impl Frame {
    fn null() -> Self {
        Self {
            sync_primitives: Box::new([]),
            scene_cmd_buffers: Box::new([]),
            index_buffer: Buffer::null(),
            vertex_buffer: Buffer::null(),
            object_buffer: Buffer::null(),
            instance_buffer: Buffer::null(),
            meshlet_data_buffer: Buffer::null(),
            indirect_cmd_buffer: Buffer::null(),
            indirect_count_buffer: Buffer::null(),
            meshlet_cull_global_buffer: Buffer::null(),
            meshlet_render_global_buffer: Buffer::null(),
        }
    }
}

pub struct Renderer {
    // Various Vulkan state data.
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,

    pub surface_instance: khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,
    pub surface_extent: vk::Extent2D,

    pub device: ash::Device,
    pub queue_family_index: u32,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    //pub transfer_queue: vk::Queue,

    // Generic memory allocator.
    pub allocator: vk_mem::Allocator,

    // Swapchain data.
    pub swapchain_device: khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub depth_image: (vk::Image, vk_mem::Allocation),
    pub swapchain_images: Box<[vk::Image]>,
    pub swapchain_color_views: Box<[vk::ImageView]>,
    pub swapchain_depth_views: Box<[vk::ImageView]>,

    // Command pool.
    cmd_pool: vk::CommandPool,

    // Desciptor sets.
    render_set_layout: vk::DescriptorSetLayout,
    render_set: vk::DescriptorSet,
    cull_set_layout: vk::DescriptorSetLayout,
    cull_set: vk::DescriptorSet,

    // Pipelines.
    render_pipeline_layout: vk::PipelineLayout,
    render_pipeline: vk::Pipeline,
    cull_pipeline_layout: vk::PipelineLayout,
    cull_pipeline: vk::Pipeline,

    // Generic resource containers.
    cwd: PathBuf,
    resource_counter: u32,
    meshes: HashMap<MeshHandle, Box<[Meshlet]>>,
    objects: HashMap<ObjectHandle, ObjectInstance>,

    // Staging.
    staging_buffer: StagingBuffer,
    staging_cmd_buffer: vk::CommandBuffer,
    staging_fence: vk::Fence,

    // Frame.
    render_cmd_buffers: Box<[vk::CommandBuffer]>,
    current_frame: Frame,
    next_frame: Option<Frame>,

    // Various render state data.
    frame: usize,
    pub cam_pos: Vec3,
    pub cam_rot: Vec2, // YX
                       //last_timestamp: f32,
}

impl Renderer {
    pub fn render(&mut self, timestamp: f32) {
        unsafe {
            // A dirty hack. When a rebuild occurs, wait for transfer to fully complete.
            if self.frame == 0 {
                // Rebuild scene elements.
                self.next_frame = Some(self.rebuild_scene());
            }
        }

        // Attempt to clean up current_frame if there is a next.
        if self.next_frame.is_some() {
            // All frames are signalled.
            let signalled =
                self.current_frame
                    .sync_primitives
                    .iter()
                    .fold(true, |acc, syncs| unsafe {
                        acc & self.device.get_fence_status(syncs.frame_in_flight).unwrap()
                    });

            if signalled {
                let frame =
                    std::mem::replace(&mut self.current_frame, self.next_frame.take().unwrap());

                // TODO: free self.current_frame.
                unsafe {
                    for sync in &frame.sync_primitives {
                        self.device.destroy_semaphore(sync.image_available, None);
                        self.device.destroy_semaphore(sync.render_finished, None);
                        self.device.destroy_fence(sync.frame_in_flight, None);
                    }
                    if frame.scene_cmd_buffers.len() > 0 {
                        self.device
                            .free_command_buffers(self.cmd_pool, &frame.scene_cmd_buffers);
                    }
                    destroy_buffer(&self.allocator, frame.index_buffer);
                    destroy_buffer(&self.allocator, frame.vertex_buffer);
                    destroy_buffer(&self.allocator, frame.object_buffer);
                    destroy_buffer(&self.allocator, frame.instance_buffer);
                    destroy_buffer(&self.allocator, frame.meshlet_data_buffer);
                    destroy_buffer(&self.allocator, frame.indirect_cmd_buffer);
                    destroy_buffer(&self.allocator, frame.indirect_count_buffer);
                    destroy_buffer(&self.allocator, frame.meshlet_cull_global_buffer);
                    destroy_buffer(&self.allocator, frame.meshlet_render_global_buffer);
                    println!("Old frame cleared.");
                }

                //
            }
        }

        // Get current working frame.
        let frame = match &self.next_frame {
            Some(frame) => frame,
            None => &self.current_frame,
        };

        println!("Starting on {}.", self.frame);
        let frame_index = self.frame % frame.sync_primitives.len();
        self.frame += 1;

        // Sync primitives associated with this frame.
        let FrameSyncPrimitives {
            image_available,
            render_finished,
            frame_in_flight,
        } = frame.sync_primitives[frame_index];

        // Command buffer associated with this frame.
        let command_buffer = self.render_cmd_buffers[frame_index];

        unsafe {
            // Wait for next image to become available.
            self.device
                .wait_for_fences(&[frame_in_flight], true, u64::MAX)
                .unwrap();
            self.device.reset_fences(&[frame_in_flight]).unwrap();

            let (image_index, _) = self
                .swapchain_device
                .acquire_next_image(self.swapchain, u64::MAX, image_available, vk::Fence::null())
                .unwrap();
            let scene_command_buffer = frame.scene_cmd_buffers[image_index as usize];

            // Reset and record.
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();

            // Transfer some global state data.
            {
                let object_data = self.objects.values().map(|obj| Object {
                    model: Mat4::from_translation(obj.position)
                        * Mat4::from_scale(obj.scale)
                        * Mat4::from_quat(obj.orientation),
                    texture_id: 0,
                });

                // Upload global descriptor data & object data.
                let projection = Mat4::perspective_infinite_rh(
                    std::f32::consts::FRAC_PI_4,
                    self.surface_extent.width as f32 / self.surface_extent.height as f32,
                    0.1,
                    //2.0,
                );
                let view = Mat4::from_rotation_translation(
                    Quat::from_euler(
                        EulerRot::XYZ,
                        -std::f32::consts::FRAC_PI_8,
                        self.cam_rot[0],
                        0.,
                    ),
                    Vec3::new(0., 0., 0.),
                ) * Mat4::from_translation(Vec3::new(
                    self.cam_pos[0],
                    self.cam_pos[1],
                    self.cam_pos[2],
                ));

                // Frustum plane data.
                let normalize_plane = |p: Vec4| p / p.xyz().length();
                let temp = projection.transpose();
                let frustum_x = normalize_plane(temp.w_axis + temp.x_axis);
                let frustum_y = normalize_plane(temp.w_axis + temp.y_axis);
                let frustum = [frustum_x.x, frustum_x.z, frustum_y.y, frustum_y.z];

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
                        &frame.meshlet_render_global_buffer,
                        0,
                        [MeshletRenderGlobal {
                            pv: projection * view,
                            vertex_buffer: self.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(frame.vertex_buffer.vk_handle()),
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
                            view: view.to_cols_array(),
                            camera_position: self.cam_pos.to_array(),
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
            self.swapchain_device
                .queue_present(
                    self.present_queue,
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&[render_finished])
                        .swapchains(&[self.swapchain])
                        .image_indices(&[image_index]),
                )
                .unwrap();
        }
    }

    unsafe fn rebuild_scene(&mut self) -> Frame {
        // Build scene buffer data.
        let (indices, vertices, meshlet_data) = {
            // Create a single gigabuffer.
            let mut vertices = vec![];
            let mut indices = vec![];
            let mut offset_map = HashMap::<MeshHandle, u32>::new();
            for (id, mesh) in self.meshes.iter() {
                // Push the base index offset for this mesh.
                offset_map.insert(*id, indices.len() as u32);

                // Extend index and vertex buffers.
                for meshlet in mesh {
                    let offset = vertices.len();
                    indices.extend(
                        meshlet
                            .indices
                            .iter()
                            .map(|&index| index as u32 + offset as u32),
                    );

                    vertices.extend((0..meshlet.positions.len() / 3).map(|i| Vertex {
                        position: Vec3::new(
                            meshlet.positions[3 * i],
                            meshlet.positions[3 * i + 1],
                            meshlet.positions[3 * i + 2],
                        ),
                        u: 0., //meshlet.texcoords[2 * i],
                        normal: Vec3::splat(0.0),
                        /*normals: Vec3::new(
                            meshlet.normals[3 * i],
                            meshlet.normals[3 * i + 1],
                            meshlet.normals[3 * i + 2],
                        )*/
                        v: 0., //meshlet.texcoords[2 * i + 1],
                        color: Vec4::splat(1.0),
                    }));
                }
            }

            let mut meshlet_data = vec![];
            for (i, object) in self.objects.values().enumerate() {
                // Get associated mesh and index offset.
                let mesh = self.meshes.get(&object.mesh).unwrap();
                let mut first_index = *offset_map.get(&object.mesh).unwrap();

                // Generate meshlet data.
                for meshlet in mesh {
                    meshlet_data.push(MeshletData {
                        center: meshlet.center,
                        radius: meshlet.radius,
                        cone_apex: meshlet.cone_apex,
                        pad0: 0.,
                        cone_axis: meshlet.cone_axis,
                        cone_cutoff: meshlet.cone_cutoff,
                        object_id: i as u32,
                        index_count: meshlet.indices.len() as u32,
                        first_index,
                    });
                    first_index += meshlet.indices.len() as u32;
                }
            }

            (indices, vertices, meshlet_data)
        };
        let instances = meshlet_data.len() as u32;

        let max_frames_in_flight = 2;
        let frame = Frame {
            sync_primitives: (0..max_frames_in_flight)
                .map(|_| FrameSyncPrimitives {
                    image_available: self
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .unwrap(),
                    render_finished: self
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .unwrap(),
                    frame_in_flight: self
                        .device
                        .create_fence(
                            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                            None,
                        )
                        .unwrap(),
                })
                .collect(),

            scene_cmd_buffers: self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.cmd_pool)
                        .level(vk::CommandBufferLevel::SECONDARY)
                        .command_buffer_count(self.swapchain_images.len() as u32),
                )
                .unwrap()
                .into_boxed_slice(),

            index_buffer: create_buffer(
                &self.allocator,
                indices.len() as u32,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),
            vertex_buffer: create_buffer(
                &self.allocator,
                vertices.len() as u32,
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),
            object_buffer: create_buffer(
                &self.allocator,
                self.objects.len() as u32,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            instance_buffer: create_buffer(
                &self.allocator,
                instances,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            meshlet_data_buffer: create_buffer(
                &self.allocator,
                instances,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            indirect_cmd_buffer: create_buffer(
                &self.allocator,
                instances,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            indirect_count_buffer: create_buffer(
                &self.allocator,
                1,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            meshlet_cull_global_buffer: create_buffer(
                &self.allocator,
                1,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),

            meshlet_render_global_buffer: create_buffer(
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
        self.staging_buffer.stage_buffer(
            &self.device,
            self.staging_cmd_buffer,
            &frame.vertex_buffer,
            0,
            vertices,
        );
        self.staging_buffer.stage_buffer(
            &self.device,
            self.staging_cmd_buffer,
            &frame.meshlet_data_buffer,
            0,
            meshlet_data,
        );

        // Submit & wait.
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
        for i in 0..self.swapchain_images.len() {
            let command_buffer = frame.scene_cmd_buffers[i];
            let image = self.swapchain_images[i];
            let color_view = self.swapchain_color_views[i];
            let depth_view = self.swapchain_depth_views[i];

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
                        &[self.cull_set],
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

                // Convert VK_IMAGE_LAYOUT_UNDEFINED -> VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL.
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[
                        vk::ImageMemoryBarrier::default()
                            .image(image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                        vk::ImageMemoryBarrier::default()
                            .image(self.depth_image.0)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL),
                    ],
                );

                // Begin rendering.
                self.device.cmd_begin_rendering(
                    command_buffer,
                    &vk::RenderingInfo::default()
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: self.surface_extent.width,
                                height: self.surface_extent.height,
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
                    &[self.render_set],
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

                // Convert VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL -> VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[
                        vk::ImageMemoryBarrier::default()
                            .image(image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR),
                        vk::ImageMemoryBarrier::default()
                            .image(self.depth_image.0)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR),
                    ],
                );

                // End recording.
                self.device.end_command_buffer(command_buffer).unwrap();

                //
                self.device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.cull_set)
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
                            .dst_set(self.render_set)
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .buffer_info(&[vk::DescriptorBufferInfo::default()
                                .buffer(frame.meshlet_render_global_buffer.vk_handle())
                                .offset(0)
                                .range(vk::WHOLE_SIZE)]),
                        /*vk::WriteDescriptorSet::default()
                        .dst_set(global_set)
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .image_info(&[vk::DescriptorImageInfo::default()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(viking_room_view)
                            .sampler(viking_room_sampler)]),*/
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.render_set)
                            .dst_binding(2)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .buffer_info(&[vk::DescriptorBufferInfo::default()
                                .buffer(frame.object_buffer.vk_handle())
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
        orientation: Quat,
        scale: Vec3,
    ) -> Option<ObjectHandle> {
        let handle = ObjectHandle(self.resource_counter);
        self.resource_counter += 1;
        self.objects.insert(
            handle,
            ObjectInstance {
                mesh,
                position,
                orientation,
                scale,
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

    pub fn new(
        cwd: impl AsRef<Path>,
        viewport_w: u32,
        viewport_h: u32,
        display: impl HasDisplayHandle + HasWindowHandle,
    ) -> Self {
        let raw_display_handle = display.display_handle().unwrap().as_raw();
        let raw_window_handle = display.window_handle().unwrap().as_raw();

        // Required Vulkan features (pass some of these in?).
        let instance_extensions = [];
        let validation_layers = [c"VK_LAYER_KHRONOS_validation"];
        let device_extensions = [
            c"VK_KHR_dynamic_rendering",
            c"VK_EXT_descriptor_indexing",
            c"VK_KHR_swapchain",
        ];

        unsafe {
            let entry = ash::Entry::load().expect("Failed to load vulkan functions.");

            let instance = {
                let required_extensions =
                    ash_window::enumerate_required_extensions(raw_display_handle).unwrap();
                let extensions = [
                    required_extensions,
                    &instance_extensions.map(|x: &CStr| x.as_ptr()),
                ]
                .concat();

                let app_info = vk::ApplicationInfo::default()
                    .application_name(c"Raytrace")
                    .api_version(vk::make_api_version(0, 1, 3, 0));
                let layers = validation_layers.map(|x: &CStr| x.as_ptr());
                let instance_cinfo = vk::InstanceCreateInfo::default()
                    .application_info(&app_info)
                    .enabled_layer_names(&layers)
                    .enabled_extension_names(&extensions);
                entry
                    .create_instance(&instance_cinfo, None)
                    .expect("Failed to create vulkan instance.")
            };

            // Find first descrete GPU.
            let physical_device = instance
                .enumerate_physical_devices()
                .expect("Could not find any Vulkan compatible devices.")
                .into_iter()
                .find(|&physical_device| {
                    instance
                        .get_physical_device_properties(physical_device)
                        .device_type
                        == vk::PhysicalDeviceType::DISCRETE_GPU
                })
                .unwrap();

            let surface_instance = khr::surface::Instance::new(&entry, &instance);
            let surface = ash_window::create_surface(
                &entry,
                &instance,
                raw_display_handle,
                raw_window_handle,
                None,
            )
            .unwrap();

            let surface_format = surface_instance
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()
                .into_iter()
                .next()
                .unwrap();
            let surface_capabilities = surface_instance
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap();

            // Find a queue family that is capable of both present and graphics commands.
            let queue_family_index = instance
                .get_physical_device_queue_family_properties(physical_device)
                .into_iter()
                .enumerate()
                .find_map(|(index, properties)| {
                    let graphics = properties.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                    let present = surface_instance
                        .get_physical_device_surface_support(physical_device, index as u32, surface)
                        .unwrap();
                    (graphics && present).then_some(index as u32)
                })
                .expect("Could not find a suitable graphics queue.");

            // Find a queue family that is capable of just transfer commands.
            /*let queue_family_index = instance
            .get_physical_device_queue_family_properties(physical_device)
            .into_iter()
            .enumerate()
            .find_map(|(index, properties)| {
                println!("{:?}", properties.queue_flags);
                properties
                    .queue_flags
                    .eq(&vk::QueueFlags::TRANSFER)
                    .then_some(index as u32)
            })
            .expect("Could not find a suitable graphics queue.");*/

            // Create logical device and its associated queues.
            let (device, graphics_queue, present_queue) = {
                let features = vk::PhysicalDeviceFeatures::default().multi_draw_indirect(true);
                let extensions = device_extensions.map(|x: &CStr| x.as_ptr());

                let device = {
                    let mut vk11features =
                        vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);

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

                    let mut vk13features =
                        vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);

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
            use vk_mem::Alloc;
            let mut allocator_cinfo =
                vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
            allocator_cinfo.flags |= vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
            let allocator = vk_mem::Allocator::new(allocator_cinfo).unwrap();

            // Create depth attachment for rendering.
            let (depth_image, depth_alloc) = allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .extent(
                            vk::Extent3D::default()
                                .width(viewport_w)
                                .height(viewport_h)
                                .depth(1),
                        )
                        .mip_levels(1)
                        .array_layers(1)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .format(vk::Format::D32_SFLOAT)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
                    &vk_mem::AllocationCreateInfo {
                        required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        ..Default::default()
                    },
                )
                .unwrap();

            // Swapchain.
            let swapchain_device = khr::swapchain::Device::new(&instance, &device);
            let swapchain = swapchain_device
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(surface)
                        .min_image_count(3)
                        .image_format(surface_format.format)
                        .image_color_space(surface_format.color_space)
                        .image_extent(Extent2D {
                            width: viewport_w,
                            height: viewport_h,
                        })
                        .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .pre_transform(surface_capabilities.current_transform)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(vk::PresentModeKHR::FIFO)
                        .clipped(true)
                        .image_array_layers(1),
                    None,
                )
                .unwrap();

            // Create image views.
            let swapchain_images = swapchain_device
                .get_swapchain_images(swapchain)
                .unwrap()
                .into_boxed_slice();
            let (swapchain_color_views, swapchain_depth_views) = {
                let n = swapchain_images.len();
                let mut color_views = vec![vk::ImageView::null(); n].into_boxed_slice();
                let mut depth_views = vec![vk::ImageView::null(); n].into_boxed_slice();
                for i in 0..n {
                    let swapchain_view = device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(swapchain_images[i])
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .format(surface_format.format)
                                .subresource_range(vk::ImageSubresourceRange {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                }),
                            None,
                        )
                        .unwrap();

                    let depth_view = device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(depth_image)
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
                        .unwrap();

                    color_views[i] = swapchain_view;
                    depth_views[i] = depth_view;
                }
                (color_views, depth_views)
            };

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
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(2)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                .descriptor_count(1)
                                .stage_flags(vk::ShaderStageFlags::VERTEX),
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

            // Generic descriptor pool
            let descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&[vk::DescriptorPoolSize::default().descriptor_count(3)])
                        .max_sets(3)
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                    None,
                )
                .unwrap();

            // Create a render and cull set from previous layouts.
            let (render_set, cull_set) = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[render_set_layout, cull_set_layout]),
                )
                .ok()
                .and_then(|v| v.into_iter().tuples().next())
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
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&[render_set_layout]),
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
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&[cull_set_layout]),
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
                        .command_buffer_count(2),
                )
                .unwrap()
                .into_boxed_slice();

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
            Self {
                entry,
                instance,
                physical_device,

                surface_instance,
                surface,
                surface_format,
                surface_capabilities,
                surface_extent: vk::Extent2D {
                    width: viewport_w,
                    height: viewport_h,
                },

                device,
                queue_family_index,
                graphics_queue,
                present_queue,

                allocator,

                swapchain_device,
                swapchain,
                depth_image: (depth_image, depth_alloc),
                swapchain_images,
                swapchain_color_views,
                swapchain_depth_views,

                cmd_pool,
                render_cmd_buffers,

                render_set,
                render_set_layout,
                cull_set,
                cull_set_layout,

                render_pipeline_layout,
                render_pipeline,
                cull_pipeline_layout,
                cull_pipeline,

                cwd: cwd.as_ref().to_owned(),
                resource_counter: 0,
                meshes: HashMap::new(),
                objects: HashMap::new(),

                staging_buffer,
                staging_cmd_buffer,
                staging_fence,

                current_frame: Frame::null(),
                next_frame: None,

                frame: 0,
                cam_pos: <_>::default(),
                cam_rot: <_>::default(),
            }
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        /*
        unsafe {
            // Free buffers.
            for (buffer, alloc) in std::mem::take(&mut self.buffer_allocs).into_iter() {
                self.vk_delete_buffer(buffer, alloc);
            }

            // Free images.
            self.allocator
                .destroy_image(self.depth_image.0, &mut self.depth_image.1);

            // Free images.
            for i in 0..3 {
                self.device
                    .destroy_image_view(self.swapchain_depth_views[i], None);
                self.device
                    .destroy_image_view(self.swapchain_color_views[i], None);
            }

            // Free the rest.
            self.swapchain_device
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_instance.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
        */
    }
}

pub fn load_mesh(filename: impl AsRef<Path>) -> Option<Box<[Meshlet]>> {
    let model = {
        use std::io::BufReader;
        let data = std::fs::read(filename).ok()?;
        let (models, _) =
            tobj::load_obj_buf(&mut BufReader::new(&data[..]), |_| unreachable!()).unwrap();
        models.into_iter().next()?.mesh
    };

    let mut indices: Box<[u32]> = model.indices.into_boxed_slice();
    let mut positions: Box<[[f32; 3]]> = model
        .positions
        .chunks_exact(3)
        .map(|v| [v[0], v[1], v[2]])
        .collect();

    // Optimize index count.
    meshopt::optimize_vertex_cache_in_place(&mut indices, positions.len());

    // Optimize overdraw.
    meshopt::optimize_overdraw_in_place_decoder(&mut indices, &positions, 1.05);

    // Optimize vertex fetch.
    meshopt::optimize_vertex_fetch_in_place(&mut indices, &mut positions);

    let adapter = meshopt::VertexDataAdapter {
        reader: std::io::Cursor::new(unsafe {
            std::slice::from_raw_parts(
                positions.as_ptr() as *const u8,
                3 * positions.len() * size_of::<f32>(),
            )
        }),
        vertex_count: positions.len(),
        vertex_stride: 3 * size_of::<f32>(),
        position_offset: 0,
    };

    let meshlets = meshopt::build_meshlets(&indices, &adapter, 64, 124, 0.5)
        .iter()
        .map(|meshlet| {
            let bounds = meshopt::compute_meshlet_bounds_decoder(meshlet, &positions);
            Meshlet {
                center: bounds.cone_apex,
                radius: bounds.radius,
                cone_apex: bounds.cone_apex,
                cone_axis: bounds.cone_axis,
                cone_cutoff: bounds.cone_cutoff,
                indices: meshlet.triangles.to_owned().into_boxed_slice(),
                positions: meshlet
                    .vertices
                    .iter()
                    .flat_map(|&i: &u32| &positions[i as usize])
                    .copied()
                    .collect(),
                texcoords: Box::new([]),
            }
        })
        .collect();

    Some(meshlets)
}
