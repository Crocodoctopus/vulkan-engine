use crate::core::Core;
use crate::image::ImageView2D;
use ash::{khr, vk};

pub(crate) struct Swapchain {
    pub swapchain_device: khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub images: Box<[vk::Image]>,
    pub views: Box<[vk::ImageView]>,
}

impl Swapchain {
    pub(crate) unsafe fn new(core: &Core, device: &ash::Device) -> Self {
        let &Core {
            ref instance,
            surface,
            surface_format,
            surface_capabilities,
            surface_extent,
            ..
        } = core;

        // Swapchain.
        let swapchain_device = khr::swapchain::Device::new(&instance, device);
        let swapchain_image_count = 3.max(surface_capabilities.min_image_count);
        let swapchain = swapchain_device
            .create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(surface)
                    .min_image_count(swapchain_image_count)
                    .image_format(surface_format.format)
                    .image_color_space(surface_format.color_space)
                    .image_extent(vk::Extent2D { width: surface_extent.width, height: surface_extent.height })
                    .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE)
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
        let images = swapchain_device.get_swapchain_images(swapchain).unwrap().into_boxed_slice();
        let color_views = (0..images.len())
            .into_iter()
            .map(|i| unsafe {
                ImageView2D::new()
                    .format(surface_format.format)
                    .aspect(vk::ImageAspectFlags::COLOR)
                    .build_raw(device, images[i], surface_format.format)
                    .unwrap()
                    .vk_handle()
            })
            .collect();

        Swapchain { swapchain_device, swapchain, images, views: color_views }
    }
}
