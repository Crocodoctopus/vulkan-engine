use ash::vk;

pub(crate) fn extent3d_from_extent2d(extent: vk::Extent2D) -> vk::Extent3D {
    vk::Extent3D { width: extent.width, height: extent.height, depth: 1 }
}

#[allow(dead_code)]
pub(crate) fn extent2d_from_extent3d(extent: vk::Extent3D) -> vk::Extent2D {
    vk::Extent2D { width: extent.width, height: extent.height }
}

pub(crate) fn image2d_create_info() -> vk::ImageCreateInfo<'static> {
    vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D { width: 1, height: 1, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .format(vk::Format::UNDEFINED)
        .usage(vk::ImageUsageFlags::empty())
}

pub(crate) fn device_local_alloc() -> vk_mem::AllocationCreateInfo {
    vk_mem::AllocationCreateInfo {
        required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ..Default::default()
    }
}

pub(crate) const COLOR_2D_SUBRESOURCE_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

pub(crate) const DEPTH_2D_SUBRESOURCE_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::DEPTH,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

pub(crate) unsafe fn record_cmd_buffer<T>(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    f: impl FnOnce(vk::CommandBuffer) -> T,
) -> T {
    device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
    device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();

    let result = f(cmd);

    device.end_command_buffer(cmd).unwrap();
    result
}
