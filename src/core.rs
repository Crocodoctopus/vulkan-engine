use ash::{khr, vk};
use std::ffi::CStr;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub(crate) struct VulkanCore {
    // Various Vulkan state data.
    pub _entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub queue_family_index: u32,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub cmd_pool: vk::CommandPool,
    pub allocator: vk_mem::Allocator,

    pub _surface_instance: khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,
}

impl VulkanCore {
    pub(crate) unsafe fn new(display: impl HasDisplayHandle + HasWindowHandle) -> Self {
        let device_extensions = [
            c"VK_KHR_dynamic_rendering",
            c"VK_EXT_descriptor_indexing",
            c"VK_KHR_swapchain",
        ];
        let raw_display_handle = display.display_handle().unwrap().as_raw();
        let raw_window_handle = display.window_handle().unwrap().as_raw();

        // TODO: Pass this in?
        let instance_extensions = [];
        let validation_layers = [c"VK_LAYER_KHRONOS_validation"];

        let entry = ash::Entry::load().expect("Failed to load vulkan functions.");

        let instance = {
            let required_extensions = ash_window::enumerate_required_extensions(raw_display_handle).unwrap();
            let extensions = [
                required_extensions,
                &instance_extensions.map(|x: &CStr| x.as_ptr()),
            ]
            .concat();

            let driver_api_version =
                entry.try_enumerate_instance_version().unwrap_or(None).unwrap_or(vk::API_VERSION_1_0);
            let app_info = vk::ApplicationInfo::default()
                .application_name(c"Raytrace")
                .api_version(driver_api_version.min(vk::API_VERSION_1_3));
            let layers = validation_layers.map(|x: &CStr| x.as_ptr());
            let instance_cinfo = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&layers)
                .enabled_extension_names(&extensions);
            entry.create_instance(&instance_cinfo, None).expect("Failed to create vulkan instance.")
        };

        // Prefer an integrated GPU, but fall back to the first available device if needed.
        let physical_device = instance
            .enumerate_physical_devices()
            .expect("Could not find any Vulkan compatible devices.")
            .into_iter()
            .find(|&physical_device| {
                instance.get_physical_device_properties(physical_device).device_type
                    == vk::PhysicalDeviceType::INTEGRATED_GPU
            })
            .or_else(|| {
                instance
                    .enumerate_physical_devices()
                    .expect("Could not find any Vulkan compatible devices.")
                    .into_iter()
                    .next()
            })
            .expect("Could not find any Vulkan compatible devices.");

        // For later.
        let physical_device_properties = instance.get_physical_device_properties(physical_device);

        let surface_instance = khr::surface::Instance::new(&entry, &instance);
        let surface =
            ash_window::create_surface(&entry, &instance, raw_display_handle, raw_window_handle, None).unwrap();

        let surface_formats = surface_instance.get_physical_device_surface_formats(physical_device, surface).unwrap();
        let surface_format = surface_formats
            .iter()
            .copied()
            .find(|format| format.format == vk::Format::R8G8B8A8_UNORM)
            .or_else(|| surface_formats.iter().copied().find(|format| format.format == vk::Format::B8G8R8A8_UNORM))
            .or_else(|| surface_formats.first().copied())
            .unwrap();
        let surface_capabilities =
            surface_instance.get_physical_device_surface_capabilities(physical_device, surface).unwrap();

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
                .sampler_filter_minmax(true)
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

        let graphics_queue = device.get_device_queue(queue_family_index, 0);
        let present_queue = device.get_device_queue(queue_family_index, 0);
        let cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .unwrap();

        let mut allocator_cinfo = vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
        allocator_cinfo.flags |= vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        let allocator = vk_mem::Allocator::new(allocator_cinfo).unwrap();

        Self {
            _entry: entry,
            instance,
            physical_device,
            queue_family_index,
            physical_device_properties,
            device,
            graphics_queue,
            present_queue,
            cmd_pool,
            allocator,

            _surface_instance: surface_instance,
            surface,
            surface_format,
            surface_capabilities,
        }
    }
}
