use ash::vk;
use std::borrow::Borrow;

use crate::Buffer;

pub struct StagingBuffer {
    buffer: vk::Buffer,
    alloc: vk_mem::Allocation,
    base: *mut u8, // Base.
    head: *mut u8,
}

impl StagingBuffer {
    pub unsafe fn new(len: u64, allocator: &vk_mem::Allocator) -> Self {
        use vk_mem::Alloc;
        let (buffer, mut alloc) = allocator
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(len)
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

        Self {
            buffer,
            alloc,
            base: map,
            head: map,
        }
    }

    pub unsafe fn reset(&mut self) {
        self.head = self.base;
    }

    pub unsafe fn stage_buffer<T: Clone>(
        &mut self,
        device: &ash::Device,
        cmd_buffer: vk::CommandBuffer,
        dst: &Buffer<T>,
        offset: u64,
        data: impl IntoIterator<Item = impl Borrow<T>>,
    ) {
        // Correct alignment.
        let alignment = std::mem::align_of::<T>();
        self.head = (self.head as usize / alignment * alignment + alignment) as *mut u8;

        // Push data to staging buffer.
        let start = self.head;
        for t in data {
            *(self.head as *mut T) = t.borrow().clone();
            self.head = self.head.wrapping_add(std::mem::size_of::<T>());
        }

        // Record transfer from staging to dst.
        device.cmd_copy_buffer(
            cmd_buffer,
            self.buffer,
            dst.vk_handle(),
            &[vk::BufferCopy::default()
                .src_offset(start.byte_offset_from(self.base) as u64)
                .dst_offset(offset)
                .size((self.head).byte_offset_from(start) as u64)],
        );
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

/*
impl Staging<'_> {
    pub unsafe fn stage_buffer<T: Clone>(
        &mut self,
        dst: vk::Buffer,
        offset: u64,
        data: impl IntoIterator<Item = impl Borrow<T>>,
    ) {
        // Correct alignment.
        let alignment = std::mem::align_of::<T>();
        self.ptr = (self.ptr as usize / alignment * alignment + alignment) as *mut u8;

        // Push data to staging buffer.
        let start = self.ptr;
        for t in data {
            *(self.ptr as *mut T) = t.borrow().clone();
            self.ptr = self.ptr.wrapping_add(std::mem::size_of::<T>());
        }

        // Record transfer from staging to dst.
        self.device.cmd_copy_buffer(
            self.command_buffer,
            self.buffer.buffer,
            dst,
            &[vk::BufferCopy::default()
                .src_offset(start.byte_offset_from(self.buffer.map) as u64)
                .dst_offset(offset)
                .size((self.ptr).byte_offset_from(start) as u64)],
        );
    }

    pub unsafe fn stage_image(
        &mut self,
        image: vk::Image,
        width: u32,
        height: u32,
        data: impl IntoIterator<Item = u8>,
    ) {
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
    }
}
*/
