extern crate ash;
extern crate ash_window;
extern crate winit;

use ash::vk::{Extent2D, ImageUsageFlags};
use ash::{khr, vk, Entry};
use std::ffi::CStr;
use std::mem::size_of;
use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoop;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

#[repr(C)]
struct GlVec2(f32, f32);
#[repr(C)]
struct GlVec3(f32, f32, f32);

fn main() {
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

        // Physical device.
        let pdevice = instance
            .enumerate_physical_devices()
            .expect("Could not find any Vulkan compatible devices.")
            .into_iter()
            .next()
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
            .expect("Could not find suitable queue.");

        let (device, graphics_queue, present_queue) = {
            let features = vk::PhysicalDeviceFeatures::default();
            let extensions = device_extensions.map(|x: &CStr| x.as_ptr());

            let device = {
                let mut dynamic_rendering =
                    vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

                let priority = [1.0];

                let queue_cinfo = [vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&priority)];

                let device_cinfo = vk::DeviceCreateInfo::default()
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

        // Swapchain.
        let swapchain_device = khr::swapchain::Device::new(&instance, &device);
        let swapchain = {
            let swapchain_cinfo = vk::SwapchainCreateInfoKHR::default()
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
                .image_array_layers(1);
            swapchain_device
                .create_swapchain(&swapchain_cinfo, None)
                .unwrap()
        };

        // Extract swapchain images and create image views for them.
        let swapchain_images = swapchain_device.get_swapchain_images(swapchain).unwrap();
        let swapchain_image_views = swapchain_images
            .iter()
            .map(|img| {
                let image_view_cinfo = vk::ImageViewCreateInfo::default()
                    .image(*img)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                device.create_image_view(&image_view_cinfo, None).unwrap()
            })
            .collect::<Vec<vk::ImageView>>();

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

        let vert_shader = create_shader_module(include_bytes!("shader.vert.spirv"));
        let frag_shader = create_shader_module(include_bytes!("shader.frag.spirv"));

        let (pipeline, pipeline_layout) = {
            let binding_desc0 = vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(size_of::<GlVec2>() as u32) // [float, float]
                .input_rate(vk::VertexInputRate::VERTEX);
            let attribute_desc0 = vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0);

            let binding_desc1 = vk::VertexInputBindingDescription::default()
                .binding(1)
                .stride(size_of::<GlVec3>() as u32) // [float, float, float]
                .input_rate(vk::VertexInputRate::VERTEX);
            let attribute_desc1 = vk::VertexInputAttributeDescription::default()
                .binding(1)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0);

            let binding_descs = [binding_desc0, binding_desc1];
            let attribute_descs = [attribute_desc0, attribute_desc1];

            let vertex_input_cinfo = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&binding_descs)
                .vertex_attribute_descriptions(&attribute_descs);

            let input_assembly_cinfo = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);

            let viewports = [vk::Viewport {
                x: 0.,
                y: 0.,
                width: viewport_w as f32,
                height: viewport_h as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: viewport_w,
                    height: viewport_h,
                },
            }];

            let viewport_cinfo = vk::PipelineViewportStateCreateInfo::default()
                .viewports(&viewports)
                .scissors(&scissors);

            let rasterization_cinfo = vk::PipelineRasterizationStateCreateInfo::default()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false);

            let multisample_cinfo = vk::PipelineMultisampleStateCreateInfo::default()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            let color_blend_states = [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false)];
            let color_blend_cinfo = vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op_enable(false)
                .attachments(&color_blend_states);

            let pipeline_layout_cinfo = vk::PipelineLayoutCreateInfo::default();
            let pipeline_layout = device
                .create_pipeline_layout(&pipeline_layout_cinfo, None)
                .unwrap();

            let shader_stage_cinfos = [
                vk::PipelineShaderStageCreateInfo::default()
                    .module(vert_shader)
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .name(c"main"),
                vk::PipelineShaderStageCreateInfo::default()
                    .module(frag_shader)
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .name(c"main"),
            ];

            let tmp = [surface_format.format];
            let mut pipeline_rendering_cinfo =
                vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&tmp);

            let pipeline_cinfo = vk::GraphicsPipelineCreateInfo::default()
                .push_next(&mut pipeline_rendering_cinfo)
                .stages(&shader_stage_cinfos)
                .vertex_input_state(&vertex_input_cinfo)
                .viewport_state(&viewport_cinfo)
                .input_assembly_state(&input_assembly_cinfo)
                .rasterization_state(&rasterization_cinfo)
                .multisample_state(&multisample_cinfo)
                .color_blend_state(&color_blend_cinfo)
                .layout(pipeline_layout);
            let pipeline = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_cinfo], None)
                .unwrap()
                .into_iter()
                .next()
                .unwrap();

            (pipeline, pipeline_layout)
        };

        let command_pool = {
            let command_pool_cinfo = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            device
                .create_command_pool(&command_pool_cinfo, None)
                .unwrap()
        };

        let command_buffers = {
            let command_buffer_alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(3);
            device
                .allocate_command_buffers(&command_buffer_alloc)
                .unwrap()
                .into_boxed_slice()
        };

        // Synchronization primitives for each frame.
        let image_available: Box<[vk::Semaphore]> = (0..3)
            .into_iter()
            .map(|_| device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None))
            .collect::<Result<_, _>>()
            .unwrap();
        let render_finished: Box<[vk::Semaphore]> = (0..3)
            .into_iter()
            .map(|_| device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None))
            .collect::<Result<_, _>>()
            .unwrap();
        let frame_in_flight: Box<[vk::Fence]> = (0..3)
            .into_iter()
            .map(|_| {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            })
            .collect::<Result<_, _>>()
            .unwrap();

        let memory_properties = instance.get_physical_device_memory_properties(pdevice);

        let find_memory_type = |type_support: u32, flags| {
            for i in 0..memory_properties.memory_type_count {
                // Check if this resource supports this memory type.
                if type_support & (i << 1) == 0 {
                    continue;
                }

                // Check if this memory type has the property flags needed.
                if memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
                {
                    return i;
                }
            }
            panic!();
        };

        let (position_buffer, position_buffer_mem) = {
            let buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(3 * size_of::<GlVec2>() as u64) // [f32, f32]
                        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap();
            let req = device.get_buffer_memory_requirements(buffer);
            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(find_memory_type(
                            req.memory_type_bits,
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )),
                    None,
                )
                .unwrap();
            device.bind_buffer_memory(buffer, memory, 0).unwrap();

            (buffer, memory)
        };

        let (color_buffer, color_buffer_mem) = {
            let buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(3 * size_of::<GlVec3>() as u64) // [f32, f32, f32]
                        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap();
            let req = device.get_buffer_memory_requirements(buffer);
            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(find_memory_type(
                            req.memory_type_bits,
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )),
                    None,
                )
                .unwrap();
            device.bind_buffer_memory(buffer, memory, 0).unwrap();

            (buffer, memory)
        };

        {
            let ptr = device
                .map_memory(
                    position_buffer_mem,
                    0,
                    3 * size_of::<GlVec2>() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            let position_buffer_map: &mut [GlVec2] = std::slice::from_raw_parts_mut(ptr as _, 3);

            let ptr = device
                .map_memory(
                    color_buffer_mem,
                    0,
                    3 * size_of::<GlVec3>() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            let color_buffer_map: &mut [GlVec3] = std::slice::from_raw_parts_mut(ptr as _, 3);

            //device.flush_mapped_memory_ranges(&[position_buffer_mem, color_buffer_mem]);
            position_buffer_map[0] = GlVec2(0.0, -0.5);
            position_buffer_map[1] = GlVec2(0.5, 0.5);
            position_buffer_map[2] = GlVec2(-0.5, 0.5);
            color_buffer_map[0] = GlVec3(1.0, 0.0, 0.0);
            color_buffer_map[1] = GlVec3(0.0, 1.0, 0.0);
            color_buffer_map[2] = GlVec3(0.0, 0.0, 1.0);

            device.unmap_memory(position_buffer_mem);
            device.unmap_memory(color_buffer_mem);
        }

        // "Gameloop"
        let mut n = 0;
        for frame in (0..3).into_iter().cycle() {
            // Input.
            let mut exit = false;
            use winit::platform::pump_events::EventLoopExtPumpEvents;
            let _status = event_loop.pump_events(Some(std::time::Duration::ZERO), |event, _| {
                match event {
                    winit::event::Event::WindowEvent {
                        event: winit::event::WindowEvent::CloseRequested,
                        ..
                    } => exit = true,

                    // Unhandled.
                    _ => {}
                }
            });

            if exit {
                break;
            }

            // Update.
            n += 1;
            if n > 60 {
                println!("1s");
                n -= 60;
            }

            // Draw.
            let command_buffer = command_buffers[frame];
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
            let image_view = swapchain_image_views[image_index as usize];

            // Reset and record.
            let begin_info = vk::CommandBufferBeginInfo::default();
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
            device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
            device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();

            // Used to transmute the layout of the next swapchain image.
            let color_image_memory_barrier = vk::ImageMemoryBarrier::default()
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            // Convert VK_IMAGE_LAYOUT_UNDEFINED -> VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL.
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[color_image_memory_barrier
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)],
            );

            // Begin rendering.
            let color_attachment_infos = [vk::RenderingAttachmentInfo::default()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                })];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: viewport_w,
                        height: viewport_h,
                    },
                })
                .layer_count(1)
                .color_attachments(&color_attachment_infos);
            device.cmd_begin_rendering(command_buffer, &rendering_info);

            // Begin draw calls.
            {
                device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
                device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    &[position_buffer, color_buffer],
                    &[0, 0],
                );
                device.cmd_draw(command_buffer, 3, 1, 0, 0);
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
                &[color_image_memory_barrier
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)],
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
        }

        // Block until the gpu is finished before proceeding to clean up.
        device
            .wait_for_fences(&frame_in_flight, true, u64::MAX)
            .unwrap();

        // Clean up.
        device.destroy_buffer(position_buffer, None);
        device.destroy_buffer(color_buffer, None);
        device.free_memory(position_buffer_mem, None);
        device.free_memory(color_buffer_mem, None);
        for i in 0..3 {
            device.destroy_fence(frame_in_flight[i], None);
            device.destroy_semaphore(render_finished[i], None);
            device.destroy_semaphore(image_available[i], None); // bleh
            device.destroy_image_view(swapchain_image_views[i], None);
        }
        device.destroy_command_pool(command_pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_shader_module(vert_shader, None);
        device.destroy_shader_module(frag_shader, None);
        swapchain_device.destroy_swapchain(swapchain, None);
        device.destroy_device(None);
        surface_instance.destroy_surface(surface, None);
        instance.destroy_instance(None);
    }
}
