use ash::vk;

use crate::vk_helpers::{device_local_alloc, extent3d_from_extent2d, image2d_create_info};

#[derive(Debug)]
pub(crate) struct Image2D {
    pub image: vk::Image,
    pub alloc: Option<vk_mem::Allocation>,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
}

impl Image2D {
    pub fn new() -> Image2DBuilder {
        Image2DBuilder {
            create_info: image2d_create_info(),
            alloc_info: device_local_alloc(),
        }
    }

    pub fn vk_handle(&self) -> vk::Image {
        self.image
    }

    #[allow(dead_code)]
    pub fn format(&self) -> vk::Format {
        self.format
    }

    #[allow(dead_code)]
    pub fn extent(&self) -> vk::Extent3D {
        self.extent
    }

    #[allow(dead_code)]
    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    #[allow(dead_code)]
    pub fn array_layers(&self) -> u32 {
        self.array_layers
    }

    pub fn destroy(mut self, allocator: &vk_mem::Allocator) {
        if let Some(alloc) = self.alloc.as_mut() {
            unsafe {
                allocator.destroy_image(self.image, alloc);
            }
        }
        std::mem::forget(self);
    }
}

impl Drop for Image2D {
    fn drop(&mut self) {
        if self.alloc.is_some() {
            panic!("Active {} dropped implicitly", std::any::type_name::<Self>());
        }
    }
}

#[derive(Clone)]
pub(crate) struct Image2DBuilder {
    pub create_info: vk::ImageCreateInfo<'static>,
    pub alloc_info: vk_mem::AllocationCreateInfo,
}

impl Image2DBuilder {
    pub fn extent(mut self, extent: vk::Extent2D) -> Self {
        self.create_info = self.create_info.extent(extent3d_from_extent2d(extent));
        self
    }

    #[allow(dead_code)]
    pub fn extent3d(mut self, extent: vk::Extent3D) -> Self {
        self.create_info = self.create_info.extent(extent);
        self
    }

    pub fn format(mut self, format: vk::Format) -> Self {
        self.create_info = self.create_info.format(format);
        self
    }

    pub fn usage(mut self, usage: vk::ImageUsageFlags) -> Self {
        self.create_info = self.create_info.usage(usage);
        self
    }

    pub fn mip_levels(mut self, mip_levels: u32) -> Self {
        self.create_info = self.create_info.mip_levels(mip_levels);
        self
    }

    #[allow(dead_code)]
    pub fn array_layers(mut self, array_layers: u32) -> Self {
        self.create_info = self.create_info.array_layers(array_layers);
        self
    }

    #[allow(dead_code)]
    pub fn samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.create_info = self.create_info.samples(samples);
        self
    }

    #[allow(dead_code)]
    pub fn alloc_info(mut self, alloc_info: vk_mem::AllocationCreateInfo) -> Self {
        self.alloc_info = alloc_info;
        self
    }

    pub fn build(self, allocator: &vk_mem::Allocator) -> ash::prelude::VkResult<Image2D> {
        let (image, alloc) = unsafe { vk_mem::Alloc::create_image(allocator, &self.create_info, &self.alloc_info)? };
        Ok(Image2D {
            image,
            alloc: Some(alloc),
            format: self.create_info.format,
            extent: self.create_info.extent,
            mip_levels: self.create_info.mip_levels,
            array_layers: self.create_info.array_layers,
        })
    }
}

#[derive(Debug)]
pub(crate) struct ImageView2D {
    pub view: vk::ImageView,
}

impl ImageView2D {
    pub fn new() -> ImageView2DBuilder {
        ImageView2DBuilder {
            create_info: vk::ImageViewCreateInfo::default()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::UNDEFINED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        }
    }

    pub fn vk_handle(&self) -> vk::ImageView {
        self.view
    }

    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ImageView2DBuilder {
    pub create_info: vk::ImageViewCreateInfo<'static>,
}

impl ImageView2DBuilder {
    pub fn format(mut self, format: vk::Format) -> Self {
        self.create_info = self.create_info.format(format);
        self
    }

    pub fn aspect(mut self, aspect: vk::ImageAspectFlags) -> Self {
        self.create_info = self
            .create_info
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: aspect, ..self.create_info.subresource_range });
        self
    }

    pub fn base_mip_level(mut self, base_mip_level: u32) -> Self {
        self.create_info = self
            .create_info
            .subresource_range(vk::ImageSubresourceRange { base_mip_level, ..self.create_info.subresource_range });
        self
    }

    #[allow(dead_code)]
    pub fn level_count(mut self, level_count: u32) -> Self {
        self.create_info = self
            .create_info
            .subresource_range(vk::ImageSubresourceRange { level_count, ..self.create_info.subresource_range });
        self
    }

    #[allow(dead_code)]
    pub fn base_array_layer(mut self, base_array_layer: u32) -> Self {
        self.create_info = self
            .create_info
            .subresource_range(vk::ImageSubresourceRange { base_array_layer, ..self.create_info.subresource_range });
        self
    }

    #[allow(dead_code)]
    pub fn layer_count(mut self, layer_count: u32) -> Self {
        self.create_info = self
            .create_info
            .subresource_range(vk::ImageSubresourceRange { layer_count, ..self.create_info.subresource_range });
        self
    }

    pub fn build(self, device: &ash::Device, image: &Image2D) -> ash::prelude::VkResult<ImageView2D> {
        unsafe { self.build_raw(device, image.vk_handle(), image.format) }
    }

    pub unsafe fn build_raw(
        self,
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
    ) -> ash::prelude::VkResult<ImageView2D> {
        let view = device.create_image_view(
            &self.create_info.image(image).format(if self.create_info.format == vk::Format::UNDEFINED {
                format
            } else {
                self.create_info.format
            }),
            None,
        )?;
        Ok(ImageView2D { view })
    }
}
