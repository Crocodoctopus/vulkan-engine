use crate::core::Core;
use ash::{khr, vk};

pub(crate) struct Swapchain {
    pub swapchain_device: khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub depth_image: (vk::Image, vk_mem::Allocation),
    pub images: Box<[vk::Image]>,
    pub color_views: Box<[vk::ImageView]>,
    pub depth_views: Box<[vk::ImageView]>,
}

impl Swapchain {
    pub(crate) unsafe fn new(
        core: &Core,
        device: &ash::Device,
        allocator: &vk_mem::Allocator,
    ) -> Self {
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
                    .image_extent(vk::Extent2D {
                        width: surface_extent.width,
                        height: surface_extent.height,
                    })
                    .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(surface_capabilities.current_transform)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(vk::PresentModeKHR::FIFO)
                    .clipped(true)
                    .image_array_layers(1),
                None,
            )
            .unwrap();

        // Create depth attachment for rendering.
        use vk_mem::Alloc;
        let (depth_image, depth_alloc) = allocator
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(
                        vk::Extent3D::default()
                            .width(surface_extent.width)
                            .height(surface_extent.height)
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

        // Create image views.
        let images = swapchain_device
            .get_swapchain_images(swapchain)
            .unwrap()
            .into_boxed_slice();
        let n = images.len();
        let mut color_views = vec![vk::ImageView::null(); n].into_boxed_slice();
        let mut depth_views = vec![vk::ImageView::null(); n].into_boxed_slice();
        for i in 0..n {
            let swapchain_view = device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(images[i])
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

        Swapchain {
            swapchain_device,
            swapchain,
            depth_image: (depth_image, depth_alloc),
            images,
            color_views,
            depth_views,
        }
    }
}
