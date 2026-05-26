use ash::{khr, vk};
use std::ffi::CStr;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub(crate) struct Core {
    // Various Vulkan state data.
    pub _entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_index: u32,

    pub _surface_instance: khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,
    pub surface_extent: vk::Extent2D,
}

impl Core {
    pub(crate) unsafe fn new(
        viewport_w: u32,
        viewport_h: u32,
        display: impl HasDisplayHandle + HasWindowHandle,
    ) -> Self {
        let raw_display_handle = display.display_handle().unwrap().as_raw();
        let raw_window_handle = display.window_handle().unwrap().as_raw();

        // TODO: Pass this in?
        let instance_extensions = [];
        let validation_layers = [c"VK_LAYER_KHRONOS_validation"];

        let entry = ash::Entry::load().expect("Failed to load vulkan functions.");

        let instance = {
            let required_extensions =
                ash_window::enumerate_required_extensions(raw_display_handle).unwrap();
            let extensions = [
                required_extensions,
                &instance_extensions.map(|x: &CStr| x.as_ptr()),
            ]
            .concat();

            let driver_api_version = entry
                .try_enumerate_instance_version()
                .unwrap_or(None)
                .unwrap_or(vk::API_VERSION_1_0);
            let app_info = vk::ApplicationInfo::default()
                .application_name(c"Raytrace")
                .api_version(driver_api_version.min(vk::API_VERSION_1_3));
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
            /*.find(|&physical_device| {
                println!(
                    "{:?}",
                    instance
                        .get_physical_device_properties(physical_device)
                        .device_type
                );
                instance
                    .get_physical_device_properties(physical_device)
                    .device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
            })*/
            .next()
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

        Self {
            _entry: entry,
            instance,
            physical_device,
            queue_family_index,

            _surface_instance: surface_instance,
            surface,
            surface_format,
            surface_capabilities,
            surface_extent: vk::Extent2D {
                width: viewport_w,
                height: viewport_h,
            },
        }
    }
}
