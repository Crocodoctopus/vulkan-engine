use ash::{khr, vk};
use std::ffi::CStr;
use std::sync::{Arc, Mutex};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

fn format_version(version: u32) -> String {
    format!("{}.{}.{}", vk::api_version_major(version), vk::api_version_minor(version), vk::api_version_patch(version))
}

fn format_memory_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 { format!("{bytes} B") } else { format!("{value:.1} {}", UNITS[unit]) }
}

pub(crate) struct VulkanCore {
    // Various Vulkan state data.
    pub _entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub queue_family_index: u32,
    pub compute_queue_family_index: u32,
    pub transfer_queue_family_index: u32,
    pub device: ash::Device,
    // These may alias the same underlying queue and therefore share the same mutex.
    pub graphics_queue: Arc<Mutex<vk::Queue>>,
    pub present_queue: Arc<Mutex<vk::Queue>>,
    pub compute_queue: Arc<Mutex<vk::Queue>>,
    pub transfer_queue: Arc<Mutex<vk::Queue>>,
    // Render-thread command pools. Move these out of VulkanCore once scene and
    // mesh jobs use worker-owned command pools.
    pub cmd_pool: vk::CommandPool,
    pub compute_cmd_pool: vk::CommandPool,
    pub transfer_cmd_pool: vk::CommandPool,
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

        let queue_family_properties = instance.get_physical_device_queue_family_properties(physical_device);

        // Find a queue family that is capable of both present and graphics commands.
        let queue_family_index = queue_family_properties
            .iter()
            .enumerate()
            .find_map(|(index, properties)| {
                let graphics = properties.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                let present = surface_instance
                    .get_physical_device_surface_support(physical_device, index as u32, surface)
                    .unwrap();
                (graphics && present).then_some(index as u32)
            })
            .expect("Could not find a suitable graphics queue.");

        // Prefer compute without graphics, falling back to the universal
        // graphics family.
        let compute_queue_family_index = queue_family_properties
            .iter()
            .enumerate()
            .find_map(|(index, properties)| {
                let compute = properties.queue_flags.contains(vk::QueueFlags::COMPUTE);
                let graphics = properties.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                (compute && !graphics).then_some(index as u32)
            })
            .unwrap_or(queue_family_index);

        // Prefer transfer without graphics or compute, then the selected
        // non-graphics compute family, and finally the universal graphics family.
        let transfer_queue_family_index = queue_family_properties
            .iter()
            .enumerate()
            .find_map(|(index, properties)| {
                let transfer = properties.queue_flags.contains(vk::QueueFlags::TRANSFER);
                let graphics_or_compute =
                    properties.queue_flags.intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE);
                (transfer && !graphics_or_compute).then_some(index as u32)
            })
            // Compute-capable queues implicitly support transfer operations.
            .or_else(|| (compute_queue_family_index != queue_family_index).then_some(compute_queue_family_index))
            .unwrap_or(queue_family_index);

        let graphics_family_queue_count = queue_family_properties[queue_family_index as usize].queue_count;
        let present_queue_index = u32::from(graphics_family_queue_count > 1);
        let compute_queue_index = if compute_queue_family_index == queue_family_index {
            if graphics_family_queue_count > 2 { 2 } else { 0 }
        } else {
            0
        };
        let transfer_queue_index =
            if transfer_queue_family_index == queue_family_index && compute_queue_family_index == queue_family_index {
                let next_index = present_queue_index.max(compute_queue_index) + 1;
                if graphics_family_queue_count > next_index { next_index } else { compute_queue_index }
            } else if transfer_queue_family_index == compute_queue_family_index {
                let queue_count = queue_family_properties[transfer_queue_family_index as usize].queue_count;
                if queue_count > compute_queue_index + 1 { compute_queue_index + 1 } else { compute_queue_index }
            } else if transfer_queue_family_index == queue_family_index {
                let next_index = present_queue_index.max(compute_queue_index) + 1;
                if graphics_family_queue_count > next_index { next_index } else { 0 }
            } else {
                0
            };

        let mut requested_queue_counts = vec![(queue_family_index, present_queue_index + 1)];
        for (family_index, queue_index) in [
            (compute_queue_family_index, compute_queue_index),
            (transfer_queue_family_index, transfer_queue_index),
        ] {
            if let Some((_, count)) = requested_queue_counts.iter_mut().find(|(family, _)| *family == family_index) {
                *count = (*count).max(queue_index + 1);
            } else {
                requested_queue_counts.push((family_index, queue_index + 1));
            }
        }

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

            let priorities = [1.0; 4];
            let queue_cinfo = requested_queue_counts
                .iter()
                .map(|&(family_index, queue_count)| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(family_index)
                        .queue_priorities(&priorities[..queue_count as usize])
                })
                .collect::<Vec<_>>();

            let device_cinfo = vk::DeviceCreateInfo::default()
                .push_next(&mut vk11features)
                .push_next(&mut vk12features)
                .push_next(&mut vk13features)
                .queue_create_infos(&queue_cinfo)
                .enabled_extension_names(&extensions)
                .enabled_features(&features);

            instance.create_device(physical_device, &device_cinfo, None).unwrap()
        };

        let graphics_queue = Arc::new(Mutex::new(device.get_device_queue(queue_family_index, 0)));
        let present_queue = if present_queue_index > 0 {
            Arc::new(Mutex::new(device.get_device_queue(queue_family_index, present_queue_index)))
        } else {
            Arc::clone(&graphics_queue)
        };
        let compute_queue = if compute_queue_family_index == queue_family_index && compute_queue_index == 0 {
            Arc::clone(&graphics_queue)
        } else {
            Arc::new(Mutex::new(device.get_device_queue(compute_queue_family_index, compute_queue_index)))
        };
        let transfer_queue = if transfer_queue_family_index == queue_family_index && transfer_queue_index == 0 {
            Arc::clone(&graphics_queue)
        } else if transfer_queue_family_index == compute_queue_family_index
            && transfer_queue_index == compute_queue_index
        {
            Arc::clone(&compute_queue)
        } else {
            Arc::new(Mutex::new(device.get_device_queue(transfer_queue_family_index, transfer_queue_index)))
        };
        let cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .unwrap();
        let compute_cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(compute_queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .unwrap();
        let transfer_cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(transfer_queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .unwrap();

        let mut allocator_cinfo = vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
        allocator_cinfo.flags |= vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        let allocator = vk_mem::Allocator::new(allocator_cinfo).unwrap();

        let memory_properties = instance.get_physical_device_memory_properties(physical_device);
        let device_name = CStr::from_ptr(physical_device_properties.device_name.as_ptr()).to_string_lossy();
        let memory_heaps = &memory_properties.memory_heaps[..memory_properties.memory_heap_count as usize];
        let memory_types = &memory_properties.memory_types[..memory_properties.memory_type_count as usize];
        let device_local_memory = memory_heaps
            .iter()
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .sum::<u64>();

        println!("Vulkan device:");
        println!("  name = {device_name}");
        println!(
            "  type = {:?}, vendor = {:#06x}, device = {:#06x}",
            physical_device_properties.device_type,
            physical_device_properties.vendor_id,
            physical_device_properties.device_id,
        );
        println!(
            "  Vulkan API = {}, driver version = {:#010x}",
            format_version(physical_device_properties.api_version),
            physical_device_properties.driver_version,
        );
        println!(
            "  advertised device-local memory = {} across {} heap(s)",
            format_memory_size(device_local_memory),
            memory_heaps.iter().filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)).count(),
        );
        for (index, heap) in memory_heaps.iter().enumerate() {
            println!("    heap {index}: {} {:?}", format_memory_size(heap.size), heap.flags);
        }
        for (index, memory_type) in memory_types.iter().enumerate() {
            println!("    memory type {index}: heap {} {:?}", memory_type.heap_index, memory_type.property_flags,);
        }

        println!("  selected queues:");
        println!("    graphics = family {queue_family_index}, queue 0");
        println!(
            "    present = family {queue_family_index}, queue {present_queue_index}{}",
            if Arc::ptr_eq(&present_queue, &graphics_queue) { " (shares graphics queue)" } else { "" },
        );
        println!(
            "    compute = family {compute_queue_family_index}, queue {compute_queue_index}{}",
            if Arc::ptr_eq(&compute_queue, &graphics_queue) {
                " (shares graphics queue)"
            } else if compute_queue_family_index == queue_family_index {
                " (separate queue in graphics family)"
            } else {
                " (dedicated compute family)"
            },
        );
        println!(
            "    transfer = family {transfer_queue_family_index}, queue {transfer_queue_index}{}",
            if Arc::ptr_eq(&transfer_queue, &graphics_queue) {
                " (shares graphics queue)"
            } else if Arc::ptr_eq(&transfer_queue, &compute_queue) {
                " (shares compute queue)"
            } else if transfer_queue_family_index == compute_queue_family_index {
                " (separate queue in compute family)"
            } else {
                " (dedicated transfer family)"
            },
        );
        println!("  queue families:");
        for (index, properties) in queue_family_properties.iter().enumerate() {
            let present =
                surface_instance.get_physical_device_surface_support(physical_device, index as u32, surface).unwrap();
            println!(
                "    family {index}: queues = {}, flags = {:?}, present = {present}, timestamp bits = {}",
                properties.queue_count, properties.queue_flags, properties.timestamp_valid_bits,
            );
        }

        Self {
            _entry: entry,
            instance,
            physical_device,
            queue_family_index,
            compute_queue_family_index,
            transfer_queue_family_index,
            physical_device_properties,
            device,
            graphics_queue,
            present_queue,
            compute_queue,
            transfer_queue,
            cmd_pool,
            compute_cmd_pool,
            transfer_cmd_pool,
            allocator,

            _surface_instance: surface_instance,
            surface,
            surface_format,
            surface_capabilities,
        }
    }
}
