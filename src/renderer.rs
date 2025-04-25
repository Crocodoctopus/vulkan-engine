//use crate::util::Map2;
use ash::vk::{Extent2D, ImageUsageFlags};
use ash::{khr, vk};
use glam::*;
use std::collections::HashMap;
use std::ffi::CStr;
use std::path::Path;
use vk_mem::Alloc;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

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

#[derive(Clone)]
#[repr(C, align(16))]
struct Object {
    model: Mat4,
    texture_id: u32,
}

#[derive(Clone)]
#[repr(C, align(16))]
struct Instance {
    object_id: u32,
}

#[derive(Clone)]
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
pub struct Shader(vk::ShaderModule);

impl Drop for Shader {
    fn drop(&mut self) {
        panic!("This type must be dropped via Renderer::delete_buffer!");
    }
}

#[derive(Debug)]
pub struct Buffer<T> {
    phantom: std::marker::PhantomData<T>,
    buffer: vk::Buffer,
    len: u32,
}

impl<T> Buffer<T> {
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

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        panic!("This type must be dropped via Renderer::delete_buffer!");
    }
}

struct Meshlet {
    center: [f32; 3],
    radius: f32,
    cone_apex: [f32; 3],
    cone_axis: [f32; 3],
    cone_cutoff: f32,
    indices: Box<[u8]>,
    positions: Box<[f32]>,
    texcoords: Box<[f32]>,
}

struct ObjectInstance {
    mesh: MeshHandle,
    position: Vec3,
    orientation: Quat,
    scale: Vec3,
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
struct MeshHandle(u32);
#[derive(Copy, Clone, Hash, Eq, PartialEq)]
struct ObjectHandle(u32);

pub struct Renderer {
    // Generic resource containers.
    //pub resource_counter: u32,
    //pub meshes: HashMap<MeshHandle, Box<[Meshlet]>>,
    //pub objects: HashMap<ObjectHandle, ObjectInstance>,

    // Various Vulkan state data.
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,

    pub surface_instance: khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,

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

    //pub pipeline_layouts: Map2<u16, vk::PipelineLayout>,
    //pub pipelines: Map2<u16, vk::Pipeline>,
    //pub buffers: Map2<u16, vk::Buffer>,
    buffer_allocs: HashMap<vk::Buffer, vk_mem::Allocation>,
}

impl Renderer {
    /*pub fn create_object(
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
                println!("{:?} vs {:?}", bounds.center, bounds.cone_apex);
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

        let handle = MeshHandle(self.resource_counter);
        self.resource_counter += 1;
        self.meshes.insert(handle, meshlets);
        return Some(handle);
    }*/

    pub fn new(
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

            /*
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
            let command_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family_index)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();

            //
            let [staging] = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()[..];

            // Create 3 sub command buffers for each swapchain image.
            let scene_cmd_buffers = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::SECONDARY)
                        .command_buffer_count(3),
                )
                .unwrap();

            // Synchronization primitives for each frame.
            let max_frames_in_flight = 2;
            let image_available: Box<[vk::Semaphore]> = (0..max_frames_in_flight)
                .map(|_| device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None))
                .collect::<Result<_, _>>()
                .unwrap();
            let render_finished: Box<[vk::Semaphore]> = (0..max_frames_in_flight)
                .map(|_| device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None))
                .collect::<Result<_, _>>()
                .unwrap();
            let frame_in_flight: Box<[vk::Fence]> = (0..max_frames_in_flight)
                .map(|_| {
                    device.create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                })
                .collect::<Result<_, _>>()
                .unwrap();

            // Various buffers.
            let mut staging_buffer = StagingBuffer::new(10000000, &renderer.allocator);

            /*let index_count = bunny_meshlets
                .iter()
                .fold(0, |acc, meshlet| acc + meshlet.indices.len());
            let index_buffer: Buffer<u32> = renderer.create_buffer(
                index_count as u32,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );*/

            let object_buffer: Buffer<Object> = renderer.create_buffer(
                3,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );

            let instance_buffer: Buffer<Instance> = renderer.create_buffer(
                3 * bunny_meshlets.len() as u32,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );

            let meshlet_data_buffer: Buffer<MeshletData> = renderer.create_buffer(
                3 * bunny_meshlets.len() as u32,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );

            let meshlet_cull_global_buffer: Buffer<MeshletCullGlobal> = renderer.create_buffer(
                1,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );

            let indirect_cmd_buffer: Buffer<vk::DrawIndexedIndirectCommand> = renderer
                .create_buffer(
                    3 * bunny_meshlets.len() as u32,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::INDIRECT_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    vk_mem::MemoryUsage::AutoPreferDevice,
                );

            let comp_buffer: Buffer<u32> = renderer.create_buffer(
                1,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );

            let vertex_count = bunny_meshlets
                .iter()
                .fold(0, |acc, meshlet| acc + meshlet.positions.len() / 3);
            let vertex_buffer: Buffer<Vertex> = renderer.create_buffer(
                vertex_count as u32,
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );

            let global_buffer: Buffer<MeshletRenderGlobal> = renderer.create_buffer(
                1,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );*/

            Self {
                entry,
                instance,
                physical_device,

                surface_instance,
                surface,
                surface_format,
                surface_capabilities,

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

                buffer_allocs: HashMap::new(),
            }
        }
    }

    pub fn create_shader_from_bytes(&self, src: &[u8]) -> Shader {
        Shader(unsafe {
            self.device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        p_code: src.as_ptr() as _,
                        code_size: src.len(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        })
    }

    pub fn delete_shader(&self, shader: Shader) {
        unsafe { self.device.destroy_shader_module(shader.0, None) };
        std::mem::forget(shader);
    }

    pub fn create_buffer<T>(
        &mut self,
        len: u32,
        vk_usage: vk::BufferUsageFlags,
        vma_usage: vk_mem::MemoryUsage,
    ) -> Buffer<T> {
        unsafe {
            let (buffer, alloc) = self
                .allocator
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

            self.buffer_allocs.insert(buffer, alloc);

            Buffer {
                phantom: std::marker::PhantomData,
                buffer,
                len,
            }
        }
    }

    pub fn delete_buffer<T>(&mut self, buffer: Buffer<T>) {
        let alloc = self.buffer_allocs.remove(&buffer.buffer).unwrap();
        self.vk_delete_buffer(buffer.buffer, alloc);
        std::mem::forget(buffer);
    }

    fn vk_delete_buffer(&mut self, buffer: vk::Buffer, mut alloc: vk_mem::Allocation) {
        unsafe {
            self.allocator.destroy_buffer(buffer, &mut alloc);
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
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
    }
}
