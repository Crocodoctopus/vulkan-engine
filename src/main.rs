extern crate ash;
extern crate ash_window;
extern crate winit;

use ash::google::surfaceless_query;
use ash::vk::ShaderModuleCreateInfo;
use ash::{khr, vk, Device, Entry, Instance};
use std::ffi::c_char;
use std::sync::Arc;
use vulkano::device::QueueCreateFlags;
use winit::event_loop::EventLoop;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

fn main() {
    // Create window.
    let event_loop = EventLoop::new().expect("Could not create window event loop.");
    let window = event_loop
        .create_window(Window::default_attributes())
        .expect("Could not create window.");
    let raw_display_handle = window.display_handle().unwrap().as_raw();
    let raw_window_handle = window.window_handle().unwrap().as_raw();

    unsafe {
        let entry = Entry::load().expect("Failed to load vulkan functions.");

        // Create instance.
        let instance = {
            // Validation layers.
            let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];

            // Instance extensions.
            //let supported_extensions = entry.enumerate_instance_extension_properties(None).unwrap();
            //println!("{supported_extensions:?}");
            let mut extensions: Vec<*const c_char> =
                ash_window::enumerate_required_extensions(raw_display_handle)
                    .unwrap()
                    .to_vec();

            /*
            for e in &extensions {
                let s = std::ffi::CStr::from_ptr(*e);
                println!("{s:?}");
            }
            */

            let app_info = vk::ApplicationInfo::default()
                .application_name(c"Raytrace")
                .api_version(vk::make_api_version(0, 1, 2, 0));
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

        // Surface.
        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        let surface = ash_window::create_surface(
            &entry,
            &instance,
            raw_display_handle,
            raw_window_handle,
            None,
        )
        .unwrap();

        let surface_format = surface_loader
            .get_physical_device_surface_formats(pdevice, surface)
            .unwrap();
        let surface_capabilities = surface_loader
            .get_physical_device_surface_capabilities(pdevice, surface)
            .unwrap();

        // Device.
        let (device, queue) = {
            // Get queue family capable of graphics.
            let queue_family_index = instance
                .get_physical_device_queue_family_properties(pdevice)
                .iter()
                .enumerate()
                .find(|(index, desc)| desc.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .expect("Could not find a graphics queue in {pdevice:?}.")
                .0 as u32;

            // Extensions.
            #[rustfmt::skip]
            let extensions = [
                c"VK_KHR_dynamic_rendering"
            ].map(|t| t.as_ptr());

            // Features.
            let features = vk::PhysicalDeviceFeatures {
                ..Default::default()
            };

            // Create device.
            let prio = [1.0];
            let queue_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&prio);

            let queue_infos = [queue_info];
            let device_cinfo = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&extensions)
                .enabled_features(&features);

            let device = instance
                .create_device(pdevice, &device_cinfo, None)
                .expect("Could not create device.");

            // Queue.
            let queue = device.get_device_queue(queue_family_index, 0);

            (device, queue)
        };

        let vert_shader = {
            let shader_src = include_bytes!("shader.vert.spirv");

            let (a, tmp_src, b) = shader_src.align_to();
            assert_eq!(a.len(), 0);
            assert_eq!(b.len(), 0);
            let shader_module_cinfo = vk::ShaderModuleCreateInfo::default().code(tmp_src);

            device
                .create_shader_module(&shader_module_cinfo, None)
                .unwrap()
        };

        let frag_shader = {
            let shader_src = include_bytes!("shader.vert.spirv");

            let (a, tmp_src, b) = shader_src.align_to();
            assert_eq!(a.len(), 0);
            assert_eq!(b.len(), 0);
            let shader_module_cinfo = vk::ShaderModuleCreateInfo::default().code(tmp_src);

            device
                .create_shader_module(&shader_module_cinfo, None)
                .unwrap()
        };

        let vert_pipeline_cinfo = vk::PipelineShaderStageCreateInfo::default()
            .module(vert_shader)
            .name(c"main");

        // Cleanup.
        device.destroy_shader_module(vert_shader, None);
        device.destroy_shader_module(frag_shader, None);
        device.destroy_device(None);
        //device.destroy_swapchain(swapchain, None);
        //instance.destroy_surface(surface, None);
        instance.destroy_instance(None);
    }
}
/*
    // Create base Vulkan library.
    let vulkano = VulkanLibrary::new().unwrap();

    // List of required Vulkan extensions.
    let extensions = InstanceExtensions {
        khr_surface: true,
        khr_xlib_surface: true,
        ..InstanceExtensions::empty()
    };

    // Create Vulkan instance.
    let instance = Instance::new(
        vulkano,
        InstanceCreateInfo {
            enabled_extensions: extensions,
            ..Default::default()
        },
    )
    .expect("Unable to create Vulkan instance.");

    // Create surface.
    let surface = Surface::from_window(instance.clone(), window_handle).expect("Could not create Vulkan surface.");

    // List of required Device features.
    let features = Features {
        dynamic_rendering: true,
        ..Features::empty()
    };

    // Enumerate physical devices, select one.
    let pdevice = instance
        .enumerate_physical_devices()
        .unwrap()
        .next()
        .expect("Unable to find a physical device.");
    println!(
        "Physical device found: {:?}",
        pdevice.properties().device_name
    );

    //
    let surface_formats = pdevice.surface_formats(&surface, SurfaceInfo::default()).unwrap();
    println!("{:?}", surface_formats);

    // Get queue.
    let graphics_queue_index = pdevice
        .queue_family_properties()
        .iter()
        .enumerate()
        .filter_map(|(index, property)| property.queue_flags
            .contains(QueueFlags::GRAPHICS)
            .then(|| index as u32))
        .next()
        .expect("No graphics queue found.");
    /*
    let mut graphics_queue = None;
    //let mut present_queue = None;
    for (index, queue_property) in pdevice.queue_family_properties().iter().enumerate() {
        if queue_property.queue_flags.contains(QueueFlags::GRAPHICS) {
            graphics_queue = Some(index);
        }
    }
    let graphics_queue = graphics_queue.expect("No graphics queue found.");
    //assert!(present_queue.is_some());
    */

    // List of required Device extensions.
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    // Create logical device from physical device.
    let (device, mut queues) = Device::new(
        pdevice,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index: graphics_queue_index,
                ..Default::default()
            }],
            enabled_features: features,
            enabled_extensions: device_extensions,
            ..Default::default()
        },
    )
    .expect("Unable to create a logical device.");

    // Extract queues.
    let graphics_queue = queues.next();
    // let present_queue = queues.next();

    // Create swapchain and image views.
    let (swapchain, images) = Swapchain::new(device.clone(), surface, SwapchainCreateInfo {
        present_mode: PresentMode::Fifo,
        min_image_count: 3,
        image_format: Format::B8G8R8A8_UNORM,
        image_usage: ImageUsage::COLOR_ATTACHMENT,
        image_color_space: ColorSpace::SrgbNonLinear,
        image_extent: [640, 480],
        ..Default::default()
    }).expect("Could not create Vulkan swapchain.");
    let image_views: Vec<Arc<ImageView>> = images
        .iter()
        .cloned()
        .map(|image| ImageView::new(image, ImageViewCreateInfo {
            view_type: vulkano::image::view::ImageViewType::Dim2d,
            format: Format::B8G8R8A8_UNORM,
            usage: ImageUsage::COLOR_ATTACHMENT,
            ..Default::default()
        }))
        .collect::<Result<_, _>>()
        .unwrap();
    //

    // Default command allocator.
    let command_allocator = StandardCommandBufferAllocator::new(device.clone(), StandardCommandBufferAllocatorCreateInfo {
        primary_buffer_count: 1,
        secondary_buffer_count: 0,
        ..Default::default()
    });

    // Create 3 command bufers.
    let command_buffers: Vec<_> = (0..3).into_iter().map(|i| {
        // Create rendering info.
        let rendering_info = RenderingInfo {
            render_area_offset: [0, 0],
            render_area_extent: [0, 0],
            layer_count: 1,
            view_mask: 0,
            color_attachments: vec![Some(RenderingAttachmentInfo{
                image_view: image_views[i].clone(),
                image_layout: ImageLayout::ColorAttachmentOptimal,
                resolve_info: None,
                load_op: AttachmentLoadOp::DontCare,
                store_op: AttachmentStoreOp::DontCare,
                clear_value: None,
                _ne: NonExhaustive::default(),
            })],
            ..Default::default()

        };

        // Record command buffer.
        let command_buffer = AutoCommandBufferBuilder::primary(&command_pool, graphics_queue_index, CommandBufferUsage::MultipleSubmit)
            .unwrap()
            .begin_rendering()
        }).collect();
*/
// Create pipeline.
