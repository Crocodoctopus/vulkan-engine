extern crate ash;
extern crate ash_window;
extern crate glam;
extern crate itertools;
extern crate png;
extern crate tobj;
extern crate vk_mem;
extern crate winit;

mod staging;

use ash::vk::{Extent2D, ImageUsageFlags};
use ash::{khr, vk, Entry};
use glam::*;
use itertools::Itertools;
use std::f32::consts::FRAC_PI_2;
use std::ffi::CStr;
use std::io::BufReader;
use std::mem::size_of;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::staging::StagingBuffer;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct GlobalDescriptorSet {
    proj: Mat4,
    view: Mat4,
}

fn main() {
    // File IO.
    let (viking_room_tex, viking_room_tex_w, viking_room_tex_h) = {
        let viking_room_png = include_bytes!("../resources/textures/viking_room.png");
        let decoder = png::Decoder::new(&viking_room_png[..]);
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!(info.buffer_size(), buf.len());
        (buf, info.width, info.height)
    };

    let viking_room_model = {
        let viking_room_obj = include_bytes!("../resources/models/viking_room.obj");
        let (viking_room_models, _) = tobj::load_obj_buf(
            &mut BufReader::new(&viking_room_obj[..]),
            |_| unreachable!(),
        )
        .unwrap();
        viking_room_models.into_iter().next().unwrap().mesh
    };

    // Required Vulkan features.
    let instance_extensions = [];
    let validation_layers = [c"VK_LAYER_KHRONOS_validation"];
    let device_extensions = [
        c"VK_KHR_dynamic_rendering",
        c"VK_EXT_descriptor_indexing",
        c"VK_KHR_swapchain",
    ];
    let (viewport_w, viewport_h) = (1080_u32, 720_u32);

    // Create window.
    let mut event_loop = EventLoop::new().expect("Could not create window event loop.");
    #[allow(deprecated)]
    let window = event_loop
        .create_window(
            Window::default_attributes()
                .with_resizable(false)
                .with_inner_size(PhysicalSize::new(viewport_w, viewport_h)),
        )
        .expect("Could not create window.");
    let raw_display_handle = window.display_handle().unwrap().as_raw();
    let raw_window_handle = window.window_handle().unwrap().as_raw();

    unsafe {
        let entry = Entry::load().expect("Failed to load vulkan functions.");

        let instance = {
            //let supported_extensions = entry.enumerate_instance_extension_properties(None).unwrap();
            //println!("{supported_extensions:?}");
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
        let pdevice = instance
            .enumerate_physical_devices()
            .expect("Could not find any Vulkan compatible devices.")
            .into_iter()
            .find(|&pdevice| {
                instance.get_physical_device_properties(pdevice).device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .unwrap();

        let surface = ash_window::create_surface(
            &entry,
            &instance,
            raw_display_handle,
            raw_window_handle,
            None,
        )
        .unwrap();

        let surface_instance = khr::surface::Instance::new(&entry, &instance);
        let surface_format = surface_instance
            .get_physical_device_surface_formats(pdevice, surface)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let surface_capabilities = surface_instance
            .get_physical_device_surface_capabilities(pdevice, surface)
            .unwrap();

        // Find a queue family that is capable of both present and graphics commands.
        let queue_family_index = instance
            .get_physical_device_queue_family_properties(pdevice)
            .into_iter()
            .enumerate()
            .find_map(|(index, properties)| {
                let graphics = properties.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                let present = surface_instance
                    .get_physical_device_surface_support(pdevice, index as u32, surface)
                    .unwrap();
                (graphics && present).then_some(index as u32)
            })
            .expect("Could not find a suitable graphics queue.");

        /*
            let test = instance
                .get_physical_device_queue_family_properties(pdevice)
                .into_iter()
                .enumerate()
                .find_map(|(index, properties)| {
                    let graphics = properties.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                    let transfer = properties.queue_flags.contains(vk::QueueFlags::TRANSFER);
                    (!graphics && transfer).then_some(index as u32)
                });
            println!("{test:?}");
        */

        let (device, graphics_queue, present_queue) = {
            let features = vk::PhysicalDeviceFeatures::default();
            let extensions = device_extensions.map(|x: &CStr| x.as_ptr());

            let device = {
                let mut descriptor_indexing =
                    vk::PhysicalDeviceDescriptorIndexingFeatures::default()
                        .descriptor_binding_uniform_buffer_update_after_bind(true)
                        .descriptor_binding_partially_bound(true)
                        .descriptor_binding_sampled_image_update_after_bind(true);

                let mut dynamic_rendering =
                    vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

                let priority = [1.0];

                let queue_cinfo = [vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&priority)];

                let device_cinfo = vk::DeviceCreateInfo::default()
                    .push_next(&mut descriptor_indexing)
                    .push_next(&mut dynamic_rendering)
                    .queue_create_infos(&queue_cinfo)
                    .enabled_extension_names(&extensions)
                    .enabled_features(&features);

                instance
                    .create_device(pdevice, &device_cinfo, None)
                    .unwrap()
            };

            // Extract queues.
            let graphics_queue = device.get_device_queue(queue_family_index, 0);
            let present_queue = device.get_device_queue(queue_family_index, 0);

            (device, graphics_queue, present_queue)
        };

        // AMD memory allocator.
        use vk_mem::Alloc;
        let allocator = vk_mem::Allocator::new(vk_mem::AllocatorCreateInfo::new(
            &instance, &device, pdevice,
        ))
        .unwrap();

        // Depth
        let (depth_image, mut depth_alloc) = allocator
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
        let swapchain_images = swapchain_device.get_swapchain_images(swapchain).unwrap();
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

        let create_shader_module = |src: &[u8]| {
            let shader_module_cinfo = vk::ShaderModuleCreateInfo {
                p_code: src.as_ptr() as _,
                code_size: src.len(),
                ..Default::default()
            };
            device
                .create_shader_module(&shader_module_cinfo, None)
                .unwrap()
        };

        // Global descriptor set.
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

        let vert_shader = create_shader_module(include_bytes!("shader.vert.spirv"));
        let frag_shader = create_shader_module(include_bytes!("shader.frag.spirv"));

        let (pipeline, pipeline_layout) = {
            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&[global_set_layout])
                        .push_constant_ranges(&[vk::PushConstantRange::default()
                            .offset(0)
                            .size(size_of::<Mat4>() as u32)
                            .stage_flags(vk::ShaderStageFlags::VERTEX)]),
                    None,
                )
                .unwrap();

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
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::default()
                                .vertex_binding_descriptions(&[
                                    vk::VertexInputBindingDescription::default()
                                        .binding(0)
                                        .stride(size_of::<Vec3>() as u32)
                                        .input_rate(vk::VertexInputRate::VERTEX),
                                    vk::VertexInputBindingDescription::default()
                                        .binding(1)
                                        .stride(size_of::<Vec2>() as u32)
                                        .input_rate(vk::VertexInputRate::VERTEX),
                                ])
                                .vertex_attribute_descriptions(&[
                                    vk::VertexInputAttributeDescription::default()
                                        .binding(0)
                                        .location(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .offset(0),
                                    vk::VertexInputAttributeDescription::default()
                                        .binding(1)
                                        .location(1)
                                        .format(vk::Format::R32G32_SFLOAT)
                                        .offset(0),
                                ]),
                        )
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
                                .attachments(&[vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                                    .blend_enable(false)]),
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

            (pipeline, pipeline_layout)
        };

        let descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&[vk::DescriptorPoolSize::default().descriptor_count(3)])
                    .max_sets(3)
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                None,
            )
            .unwrap();

        let global_sets = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[global_set_layout, global_set_layout, global_set_layout]),
            )
            .unwrap()
            .into_boxed_slice();

        let command_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .unwrap();

        let command_buffers = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(4),
            )
            .unwrap()
            .into_boxed_slice();

        let graphics_command_buffers = [command_buffers[0], command_buffers[1], command_buffers[2]];
        let staging_command_buffer = command_buffers[1];
        drop(command_buffers);

        // Synchronization primitives for each frame.
        let image_available: Box<[vk::Semaphore]> = (0..3)
            .map(|_| device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None))
            .collect::<Result<_, _>>()
            .unwrap();
        let render_finished: Box<[vk::Semaphore]> = (0..3)
            .map(|_| device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None))
            .collect::<Result<_, _>>()
            .unwrap();
        let frame_in_flight: Box<[vk::Fence]> = (0..3)
            .map(|_| {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            })
            .collect::<Result<_, _>>()
            .unwrap();

        let (viking_room_image, mut viking_room_alloc) = allocator
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(vk::Extent3D {
                        width: viking_room_tex_w,
                        height: viking_room_tex_h,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .samples(vk::SampleCountFlags::TYPE_1),
                &vk_mem::AllocationCreateInfo {
                    ..Default::default()
                },
            )
            .unwrap();

        let viking_room_view = device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(viking_room_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    ),
                None,
            )
            .unwrap();

        let viking_room_sampler = device
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .unnormalized_coordinates(false)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .mip_lod_bias(0.0)
                    .min_lod(0.0)
                    .max_lod(0.0),
                None,
            )
            .unwrap();

        // TODO: delete
        let mut staging_buffer = StagingBuffer::new(10000000, &allocator);

        let (index_buffer, mut index_alloc) = allocator
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size((viking_room_model.indices.len() * size_of::<u32>()) as u64)
                    .usage(vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo::default(),
            )
            .unwrap();

        let (position_buffer, mut position_alloc) = allocator
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size((viking_room_model.positions.len() * size_of::<f32>()) as u64)
                    .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo::default(),
            )
            .unwrap();

        let (uv_buffer, mut uv_alloc) = allocator
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size((viking_room_model.texcoords.len() * size_of::<f32>()) as u64)
                    .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo::default(),
            )
            .unwrap();

        let (matrix_buffer, mut matrix_alloc) = allocator
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(2 * size_of::<Mat4>() as u64)
                    .usage(
                        vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo::default(),
            )
            .unwrap();

        // Upload vertex buffer data.
        {
            device
                .reset_command_buffer(staging_command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
            device
                .begin_command_buffer(
                    staging_command_buffer,
                    &vk::CommandBufferBeginInfo::default(),
                )
                .unwrap();

            staging_buffer
                .begin_transfer(&device, staging_command_buffer)
                .stage_buffer::<u32>(index_buffer, 0, &viking_room_model.indices)
                .stage_buffer(position_buffer, 0, viking_room_model.positions)
                .stage_buffer(uv_buffer, 0, viking_room_model.texcoords)
                .stage_image(
                    viking_room_image,
                    viking_room_tex_w,
                    viking_room_tex_h,
                    viking_room_tex
                        .iter()
                        .tuples()
                        .flat_map(|(&x, &y, &z)| [x, y, z, 255]),
                )
                .finish();

            device.end_command_buffer(staging_command_buffer).unwrap();

            // Create wait fence.
            let wait = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();

            // Submit.
            device
                .queue_submit(
                    graphics_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[staging_command_buffer])],
                    wait,
                )
                .unwrap();

            // Wait.
            device.wait_for_fences(&[wait], true, u64::MAX).unwrap();
            device.destroy_fence(wait, None);
        }

        // "Gameloop"
        //let mut timestamp = 0_u64;
        let mut time = 0_f32;
        let dt = 0.016666_f32;
        // Misc.
        let mut cam_x = 0_f32;
        let cam_y = 0_f32;
        let mut cam_z = 0_f32;
        let mut cam_hr = 0_f32;
        let mut cam_vr = 0_f32;
        let mut w_down = false;
        let mut a_down = false;
        let mut s_down = false;
        let mut d_down = false;
        let mut q_down = false;
        let mut e_down = false;
        for frame in (0..3).cycle() {
            // Input.
            let mut exit = false;
            use winit::platform::pump_events::EventLoopExtPumpEvents;
            #[allow(deprecated)]
            let _status = event_loop.pump_events(Some(std::time::Duration::ZERO), |event, _| {
                match event {
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => exit = true,

                    Event::WindowEvent {
                        event:
                            WindowEvent::KeyboardInput {
                                event:
                                    KeyEvent {
                                        physical_key: PhysicalKey::Code(key),
                                        state,
                                        repeat: false,
                                        ..
                                    },
                                ..
                            },
                        ..
                    } => {
                        // Skip repeats.
                        let var = match key {
                            KeyCode::KeyW => &mut w_down,
                            KeyCode::KeyA => &mut a_down,
                            KeyCode::KeyS => &mut s_down,
                            KeyCode::KeyD => &mut d_down,
                            KeyCode::KeyQ => &mut q_down,
                            KeyCode::KeyE => &mut e_down,
                            _ => return,
                        };

                        match state {
                            ElementState::Pressed => *var = true,
                            ElementState::Released => *var = false,
                        }
                    }

                    // Unhandled.
                    _ => {}
                }
            });

            if exit {
                break;
            }

            // Update.

            // Forward.
            if w_down && !s_down {
                cam_z += dt * cam_hr.cos();
                cam_x -= dt * cam_hr.sin();
            }
            // Backward.
            if !w_down && s_down {
                cam_z -= dt * cam_hr.cos();
                cam_x += dt * cam_hr.sin();
            }

            // Strafe left.
            if a_down && !d_down {
                cam_hr -= dt;
            }
            // Strafe right.
            if !a_down && d_down {
                cam_hr += dt;
            }

            // Turn left.
            if q_down && !e_down {
                cam_x += dt * cam_hr.cos();
                cam_z += dt * cam_hr.sin();
            }
            // Turn right.
            if !q_down && e_down {
                cam_x -= dt * cam_hr.cos();
                cam_z -= dt * cam_hr.sin();
            }

            cam_vr = cam_vr.clamp(-FRAC_PI_2, FRAC_PI_2);

            // Draw.
            let command_buffer = graphics_command_buffers[frame];
            let frame_in_flight = frame_in_flight[frame];
            let render_finished = render_finished[frame];
            let image_available = image_available[frame];

            // Wait for next image to become available.
            device
                .wait_for_fences(&[frame_in_flight], true, u64::MAX)
                .unwrap();
            device.reset_fences(&[frame_in_flight]).unwrap();

            let (image_index, _) = swapchain_device
                .acquire_next_image(swapchain, u64::MAX, image_available, vk::Fence::null())
                .unwrap();
            let image = swapchain_images[image_index as usize];
            let color_view = swapchain_color_views[image_index as usize];
            let depth_view = swapchain_depth_views[image_index as usize];

            // Reset and record.
            device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
            device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();

            // Upload global descriptor data.
            staging_buffer
                .begin_transfer(&device, command_buffer)
                .stage_buffer(
                    matrix_buffer,
                    0,
                    std::iter::once_with(|| GlobalDescriptorSet {
                        proj: Mat4::perspective_rh_gl(
                            std::f32::consts::FRAC_PI_4,
                            viewport_w as f32 / viewport_h as f32,
                            0.01,
                            10.0,
                        ),
                        view: Mat4::from_rotation_translation(
                            Quat::from_euler(
                                EulerRot::XYZ,
                                -std::f32::consts::FRAC_PI_8,
                                cam_hr,
                                0.,
                            ),
                            Vec3::new(0., 0., 0.),
                        ) * Mat4::from_translation(Vec3::new(cam_x, cam_y, cam_z)),
                    }),
                )
                .finish();

            // Convert VK_IMAGE_LAYOUT_UNDEFINED -> VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL.
            device.cmd_pipeline_barrier(
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
                        .image(depth_image)
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
            device.cmd_begin_rendering(
                command_buffer,
                &vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: viewport_w,
                            height: viewport_h,
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
            {
                device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(global_sets[frame])
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .buffer_info(&[vk::DescriptorBufferInfo::default()
                                .buffer(matrix_buffer)
                                .offset(0)
                                .range(vk::WHOLE_SIZE)]),
                        vk::WriteDescriptorSet::default()
                            .dst_set(global_sets[frame])
                            .dst_binding(1)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1)
                            .image_info(&[vk::DescriptorImageInfo::default()
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(viking_room_view)
                                .sampler(viking_room_sampler)]),
                    ],
                    &[],
                );

                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &[global_sets[frame]],
                    &[],
                );

                let model = Mat4::from_translation(Vec3::new(0., 1., 0.));
                let model = model * Mat4::from_rotation_y(time * std::f32::consts::FRAC_PI_2);
                let model = model * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);
                device.cmd_push_constants(
                    command_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    std::slice::from_raw_parts(
                        model.to_cols_array().as_ptr() as _,
                        size_of::<Mat4>(),
                    ),
                );

                device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
                device.cmd_bind_index_buffer(
                    command_buffer,
                    index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    &[position_buffer, uv_buffer],
                    &[0, 0],
                );
                device.cmd_draw_indexed(
                    command_buffer,
                    viking_room_model.indices.len() as u32,
                    1,
                    0,
                    0,
                    0,
                );
            }

            device.cmd_end_rendering(command_buffer);

            // Convert VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL -> VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.
            device.cmd_pipeline_barrier(
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
                        .image(depth_image)
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

            device.end_command_buffer(command_buffer).unwrap();

            // Execute command buffer.
            let waits = [image_available];
            let signals = [render_finished];
            let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&waits)
                .signal_semaphores(&signals)
                .wait_dst_stage_mask(&stages)
                .command_buffers(&command_buffers);
            device
                .queue_submit(graphics_queue, &[submit_info], frame_in_flight)
                .unwrap();

            //
            let waits = [render_finished];
            let swapchains = [swapchain];
            let images = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&waits)
                .swapchains(&swapchains)
                .image_indices(&images);
            swapchain_device
                .queue_present(present_queue, &present_info)
                .unwrap();

            //timestamp += 16666;
            time += 0.016666 * 0.1;
            //panic!();
        }

        // Block until the gpu is finished before proceeding to clean up.
        device
            .wait_for_fences(&frame_in_flight, true, u64::MAX)
            .unwrap();

        // Clean up.
        device.destroy_sampler(viking_room_sampler, None);
        device.destroy_image_view(viking_room_view, None);
        allocator.destroy_image(viking_room_image, &mut viking_room_alloc);
        allocator.destroy_buffer(matrix_buffer, &mut matrix_alloc);
        allocator.destroy_buffer(position_buffer, &mut position_alloc);
        allocator.destroy_buffer(uv_buffer, &mut uv_alloc);
        allocator.destroy_buffer(index_buffer, &mut index_alloc);
        staging_buffer.destroy(&allocator);
        allocator.destroy_image(depth_image, &mut depth_alloc);
        drop(allocator);
        for i in 0..3 {
            device.destroy_fence(frame_in_flight[i], None);
            device.destroy_semaphore(render_finished[i], None);
            device.destroy_semaphore(image_available[i], None); // bleh
            device.destroy_image_view(swapchain_depth_views[i], None);
            device.destroy_image_view(swapchain_color_views[i], None);
        }
        device.destroy_command_pool(command_pool, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_descriptor_set_layout(global_set_layout, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_shader_module(vert_shader, None);
        device.destroy_shader_module(frag_shader, None);
        swapchain_device.destroy_swapchain(swapchain, None);
        device.destroy_device(None);
        surface_instance.destroy_surface(surface, None);
        instance.destroy_instance(None);
    }
}
