extern crate ash;
extern crate ash_window;
extern crate winit;

use ash::vk::{Extent2D, ImageUsageFlags, ShaderModuleCreateInfo};
use ash::{khr, vk, Device, Entry, Instance};
use std::ffi::CStr;
use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoop;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

fn main() {
    // Required Vulkan features.
    let instance_extensions = [];
    let validation_layers = [c"VK_LAYER_KHRONOS_validation"];
    let device_extensions = [
        c"VK_KHR_dynamic_rendering",
        c"VK_EXT_descriptor_indexing",
        c"VK_KHR_swapchain",
    ];
    let (viewport_w, viewport_h) = (720_u32, 1080_u32);

    // Create window.
    let event_loop = EventLoop::new().expect("Could not create window event loop.");
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
                .api_version(vk::make_api_version(0, 1, 2, 0));
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
            let vertex_input_cinfo = vk::PipelineVertexInputStateCreateInfo::default();
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

        // Cleanup.
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        swapchain_image_views
            .into_iter()
            .for_each(|v| device.destroy_image_view(v, None));
        device.destroy_shader_module(vert_shader, None);
        device.destroy_shader_module(frag_shader, None);
        swapchain_device.destroy_swapchain(swapchain, None);
        device.destroy_device(None);
        surface_instance.destroy_surface(surface, None);
        instance.destroy_instance(None);
    }
}
