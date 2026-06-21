use ash::vk;

use crate::buffer::Buffer;

pub struct StagingBuffer {
    buffer: vk::Buffer,
    alloc: vk_mem::Allocation,
    base: *mut u8,
    head: *mut u8,
    end: *mut u8,
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
                    required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
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
            end: map.wrapping_add(len as usize),
        }
    }

    pub unsafe fn reset(&mut self) {
        self.head = self.base;
    }

    pub unsafe fn stage<B: ?Sized, S, T>(
        &mut self,
        device: &ash::Device,
        cmd_buffer: vk::CommandBuffer,
        dst: &Buffer<B>,
        offset: u64,
        data: S,
    ) where
        S: AsRef<[T]>,
        T: Clone,
    {
        let data = data.as_ref();
        if data.is_empty() {
            return;
        }

        let alignment = std::mem::align_of::<T>();
        self.head = (((self.head as usize + alignment - 1) / alignment) * alignment) as *mut u8;

        let start = self.head;
        let size = data.len() * std::mem::size_of::<T>();
        assert!((start as usize + size) <= self.end as usize, "staging buffer exhausted");
        assert!(offset + size as u64 <= dst.size() as u64, "destination buffer too small");

        for t in data {
            *(self.head as *mut T) = t.clone();
            self.head = self.head.wrapping_add(std::mem::size_of::<T>());
        }

        device.cmd_copy_buffer(
            cmd_buffer,
            self.buffer,
            dst.vk_handle(),
            &[vk::BufferCopy::default()
                .src_offset(start.byte_offset_from(self.base) as u64)
                .dst_offset(offset)
                .size(size as u64)],
        );
    }

    pub unsafe fn stage_bytes<B: ?Sized>(
        &mut self,
        device: &ash::Device,
        cmd_buffer: vk::CommandBuffer,
        dst: &Buffer<B>,
        offset: u64,
        bytes: &[u8],
    ) {
        if bytes.is_empty() {
            return;
        }

        let alignment = 4usize;
        self.head = (((self.head as usize + alignment - 1) / alignment) * alignment) as *mut u8;

        let start = self.head;
        let size = bytes.len();
        assert!((start as usize + size) <= self.end as usize, "staging buffer exhausted");
        assert!(offset + size as u64 <= dst.size() as u64, "destination buffer too small");

        let staging_slice = std::slice::from_raw_parts_mut(start, size);
        staging_slice.copy_from_slice(bytes);
        self.head = self.head.wrapping_add(size);

        device.cmd_copy_buffer(
            cmd_buffer,
            self.buffer,
            dst.vk_handle(),
            &[vk::BufferCopy::default()
                .src_offset(start.byte_offset_from(self.base) as u64)
                .dst_offset(offset)
                .size(size as u64)],
        );
    }
}

impl Drop for StagingBuffer {
    fn drop(&mut self) {
        panic!(
            "{} dropped implicitly; destroy/unmap staging Vulkan resources explicitly first",
            std::any::type_name::<Self>(),
        );
    }
}
