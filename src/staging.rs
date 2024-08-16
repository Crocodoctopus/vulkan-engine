use ash::vk;
use std::borrow::Borrow;

pub struct StagingBuffer {
    buffer: vk::Buffer,
    alloc: vk_mem::Allocation,
    map: *mut u8,
}

impl StagingBuffer {
    pub unsafe fn new(len: u64, allocator: &vk_mem::Allocator) -> Self {
        use vk_mem::Alloc;
        let (buffer, mut alloc) = allocator
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(len as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo {
                    flags: vk_mem::AllocationCreateFlags::MAPPED
                        | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                    usage: vk_mem::MemoryUsage::AutoPreferHost,
                    required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                    ..Default::default()
                },
            )
            .unwrap();

        let map = allocator.map_memory(&mut alloc).unwrap();

        Self { buffer, alloc, map }
    }

    pub unsafe fn begin_transfer<'a>(
        &'a mut self,
        device: &'a ash::Device,
        command_buffer: vk::CommandBuffer,
    ) -> Staging {
        Staging {
            device,
            ptr: self.map,
            command_buffer,
            buffer: self,
        }
    }

    pub unsafe fn destroy(mut self, alloc: &vk_mem::Allocator) {
        alloc.unmap_memory(&mut self.alloc);
        alloc.destroy_buffer(self.buffer, &mut self.alloc);
        std::mem::forget(self)
    }
}

impl Drop for StagingBuffer {
    fn drop(&mut self) {
        println!(
            "Warning: {} must be dropped with {}::destroy!",
            std::any::type_name::<Self>(),
            std::any::type_name::<Self>()
        );
    }
}

pub struct Staging<'a> {
    device: &'a ash::Device,
    command_buffer: vk::CommandBuffer,
    ptr: *mut u8,
    buffer: &'a mut StagingBuffer,
}

impl<'a> Staging<'a> {
    pub unsafe fn stage_buffer<T: Copy>(
        mut self,
        dst: vk::Buffer,
        offset: u64,
        data: impl IntoIterator<Item = impl Borrow<T>>,
    ) -> Self {
        let start = self.ptr;
        for t in data {
            *(self.ptr as *mut T) = *t.borrow();
            self.ptr = self.ptr.wrapping_add(std::mem::size_of::<T>());
        }

        self.device.cmd_copy_buffer(
            self.command_buffer,
            self.buffer.buffer,
            dst,
            &[vk::BufferCopy::default()
                .src_offset(start.byte_offset_from(self.buffer.map) as u64)
                .dst_offset(offset)
                .size((self.ptr).byte_offset_from(start) as u64)],
        );

        self
    }

    pub unsafe fn stage_image(
        mut self,
        image: vk::Image,
        width: u32,
        height: u32,
        data: impl IntoIterator<Item = u8>,
    ) -> Self {
        let start = self.ptr;
        for t in data {
            *self.ptr = t;
            self.ptr = self.ptr.add(1);
        }

        self.device.cmd_pipeline_barrier(
            self.command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
        );

        self.device.cmd_copy_buffer_to_image(
            self.command_buffer,
            self.buffer.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy::default()
                .buffer_offset(start.offset_from(self.buffer.map) as u64)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })],
        );

        self.device.cmd_pipeline_barrier(
            self.command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)],
        );

        self
    }

    pub unsafe fn finish(self) {}
}
