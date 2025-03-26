use ash::vk::{Extent2D, ImageUsageFlags};
use ash::{khr, vk};
use glam::*;
use std::ffi::CStr;
use vk_mem::Alloc;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[derive(Debug)]
pub struct Shader(vk::ShaderModule);

impl Drop for Shader {
    fn drop(&mut self) {
        panic!("This type must be dropped via Renderer::delete_buffer!");
    }
}

#[derive(Debug)]
pub struct Buffer {
    buffer: vk::Buffer,
    alloc: vk_mem::Allocation,
    size: u64,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        panic!("This type must be dropped via Renderer::delete_buffer!");
    }
}

pub struct Renderer {
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

    pub allocator: vk_mem::Allocator,

    pub swapchain_device: khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub depth_image: (vk::Image, vk_mem::Allocation),
    pub swapchain_images: Box<[vk::Image]>,
    pub swapchain_color_views: Box<[vk::ImageView]>,
    pub swapchain_depth_views: Box<[vk::ImageView]>,
}

impl Renderer {
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

            // Create logical device and its associated queues.
            let (device, graphics_queue, present_queue) = {
                let features = vk::PhysicalDeviceFeatures::default();
                let extensions = device_extensions.map(|x: &CStr| x.as_ptr());

                let device = {
                    let mut device_address =
                        vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
                            .buffer_device_address(true);

                    let mut descriptor_indexing =
                        vk::PhysicalDeviceDescriptorIndexingFeatures::default()
                            .descriptor_binding_uniform_buffer_update_after_bind(true)
                            .descriptor_binding_partially_bound(true)
                            .descriptor_binding_sampled_image_update_after_bind(true);

                    let mut dynamic_rendering =
                        vk::PhysicalDeviceDynamicRenderingFeatures::default()
                            .dynamic_rendering(true);

                    let priority = [1.0];

                    let queue_cinfo = [vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(queue_family_index)
                        .queue_priorities(&priority)];

                    let device_cinfo = vk::DeviceCreateInfo::default()
                        .push_next(&mut descriptor_indexing)
                        .push_next(&mut dynamic_rendering)
                        .push_next(&mut device_address)
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

    pub fn create_buffer(
        &self,
        size: u64,
        vk_usage: vk::BufferUsageFlags,
        vma_usage: vk_mem::MemoryUsage,
    ) -> Buffer {
        unsafe {
            let (buffer, alloc) = self
                .allocator
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(size).usage(vk_usage),
                    &vk_mem::AllocationCreateInfo {
                        usage: vma_usage,
                        ..Default::default()
                    },
                )
                .unwrap();

            Buffer {
                buffer,
                alloc,
                size,
            }
        }
    }

    pub fn delete_buffer(&self, mut buffer: Buffer) {
        unsafe {
            self.allocator
                .destroy_buffer(buffer.buffer, &mut buffer.alloc);
        }
        std::mem::forget(buffer);
    }
}
