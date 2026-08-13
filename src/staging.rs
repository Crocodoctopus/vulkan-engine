use ash::vk;
use std::sync::Mutex;

use crate::buffer::{Buffer, Trailing};

pub struct Whole<T>(pub T);

pub struct Partial<T>(pub usize, pub T);

#[allow(dead_code)]
pub struct Raw<'a>(pub usize, pub &'a [u8]);

pub(crate) trait UploadOp<D: ?Sized> {
    unsafe fn apply(
        self,
        staging: &StagingPool,
        span: &mut StagingSpan,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        dst: &Buffer<D>,
    );
}

pub struct StagingSpan {
    allocation: vk_mem::VirtualAllocation,
    offset: u32,
    len: u32,
    capacity: u32,
}

impl StagingSpan {
    pub unsafe fn reset(&mut self) {
        self.len = 0;
    }

    pub fn size(&self) -> u32 {
        self.len
    }
}

#[allow(dead_code)]
pub(crate) struct StagingPool {
    block: Mutex<vk_mem::VirtualBlock>,
    buffer_allocation: vk_mem::Allocation,
    buffer: vk::Buffer,
    base: *mut u8,
}

// SAFETY: TODO: Review this.
unsafe impl Send for StagingPool {}
unsafe impl Sync for StagingPool {}

impl StagingPool {
    pub unsafe fn new(allocator: &vk_mem::Allocator, size: u64) -> Self {
        use vk_mem::Alloc;

        let (buffer, mut buffer_allocation) = allocator
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
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
        let base = allocator.map_memory(&mut buffer_allocation).unwrap();
        let block = vk_mem::VirtualBlock::new(vk_mem::VirtualBlockCreateInfo {
            size,
            flags: Default::default(),
            allocation_callbacks: None,
        })
        .unwrap();

        Self { block: Mutex::new(block), buffer_allocation, buffer, base }
    }

    pub unsafe fn alloc(&self, len: u64) -> StagingSpan {
        self.try_alloc(len).unwrap()
    }

    pub unsafe fn try_alloc(&self, len: u64) -> ash::prelude::VkResult<StagingSpan> {
        let (allocation, info) = {
            let mut block = self.block.lock().unwrap();
            let (allocation, offset) = block.allocate(vk_mem::VirtualAllocationCreateInfo {
                size: len,
                alignment: 1,
                user_data: 0,
                flags: vk_mem::VirtualAllocationCreateFlags::empty(),
            })?;
            let info = block.get_allocation_info(&allocation).unwrap();
            debug_assert_eq!(info.offset, offset);
            (allocation, info)
        };

        Ok(StagingSpan {
            allocation,
            offset: info.offset as u32,
            len: 0,
            capacity: info.size as u32,
        })
    }

    pub unsafe fn free_span(&self, span: StagingSpan) {
        let mut allocation = std::ptr::read(&span.allocation);
        self.block.lock().unwrap().free(&mut allocation);
        std::mem::forget(allocation);
        std::mem::forget(span);
    }

    unsafe fn stage_bytes<D: ?Sized>(
        &self,
        span: &mut StagingSpan,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        dst: &Buffer<D>,
        dst_offset: usize,
        bytes: &[u8],
    ) {
        assert!(dst_offset + bytes.len() <= dst.size(), "destination buffer too small");
        assert!(span.len + bytes.len() as u32 <= span.capacity, "staging span exhausted");

        // Write to gpu staging memory.
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            self.base.wrapping_add(span.offset as usize + span.len as usize),
            bytes.len(),
        );

        // Record transfer from staging -> buffer.
        device.cmd_copy_buffer(
            cmd,
            self.buffer,
            dst.vk_handle(),
            &[vk::BufferCopy::default()
                .src_offset(span.offset as u64 + span.len as u64)
                .dst_offset(dst_offset as u64)
                .size(bytes.len() as u64)],
        );

        span.len += bytes.len() as u32;
    }

    pub unsafe fn stage<D: ?Sized, O: UploadOp<D>>(
        &self,
        span: &mut StagingSpan,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        dst: &Buffer<D>,
        op: O,
    ) {
        op.apply(self, span, device, cmd, dst);
    }

    #[allow(dead_code)]
    pub unsafe fn free(self, allocator: &vk_mem::Allocator) {
        let mut buffer_allocation = std::ptr::read(&self.buffer_allocation);
        allocator.destroy_buffer(self.buffer, &mut buffer_allocation);
        std::mem::forget(buffer_allocation);
        std::mem::forget(self);
    }
}

impl<T: Copy> UploadOp<T> for Whole<T> {
    unsafe fn apply(
        self,
        staging: &StagingPool,
        span: &mut StagingSpan,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        dst: &Buffer<T>,
    ) {
        assert!(dst.size() == std::mem::size_of::<T>(), "destination buffer size mismatch");
        let bytes = std::slice::from_raw_parts((&self.0 as *const T).cast::<u8>(), std::mem::size_of::<T>());
        staging.stage_bytes(span, device, cmd, dst, 0, bytes);
    }
}

impl<T: Copy> UploadOp<[T]> for Whole<Vec<T>> {
    unsafe fn apply(
        self,
        staging: &StagingPool,
        span: &mut StagingSpan,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        dst: &Buffer<[T]>,
    ) {
        assert!(dst.len() as usize == self.0.len(), "destination buffer length mismatch");
        let bytes = std::slice::from_raw_parts(self.0.as_ptr().cast::<u8>(), self.0.len() * std::mem::size_of::<T>());
        staging.stage_bytes(span, device, cmd, dst, 0, bytes);
    }
}

impl<'a, T: Copy> UploadOp<[T]> for Whole<&'a [T]> {
    unsafe fn apply(
        self,
        staging: &StagingPool,
        span: &mut StagingSpan,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        dst: &Buffer<[T]>,
    ) {
        assert!(dst.len() as usize == self.0.len(), "destination buffer length mismatch");
        let bytes = std::slice::from_raw_parts(self.0.as_ptr().cast::<u8>(), self.0.len() * std::mem::size_of::<T>());
        staging.stage_bytes(span, device, cmd, dst, 0, bytes);
    }
}

impl<T: Copy, const N: usize> UploadOp<[T]> for Whole<[T; N]> {
    unsafe fn apply(
        self,
        staging: &StagingPool,
        span: &mut StagingSpan,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        dst: &Buffer<[T]>,
    ) {
        assert!(dst.len() as usize == N, "destination buffer length mismatch");
        let bytes = std::slice::from_raw_parts(self.0.as_ptr().cast::<u8>(), N * std::mem::size_of::<T>());
        staging.stage_bytes(span, device, cmd, dst, 0, bytes);
    }
}

impl<T: Copy, S: AsRef<[T]>> UploadOp<[T]> for Partial<S> {
    unsafe fn apply(
        self,
        staging: &StagingPool,
        span: &mut StagingSpan,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        dst: &Buffer<[T]>,
    ) {
        let data = self.1.as_ref();
        assert!(self.0 + data.len() <= dst.len() as usize, "destination buffer range out of bounds");
        let dst_offset = self.0 * std::mem::size_of::<T>();
        let bytes = std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * std::mem::size_of::<T>());
        staging.stage_bytes(span, device, cmd, dst, dst_offset, bytes);
    }
}

impl<H: Trailing, S: AsRef<[H::Tail]>> UploadOp<H> for Whole<(H, S)>
where
    H: Copy,
    H::Tail: Copy,
{
    unsafe fn apply(
        self,
        staging: &StagingPool,
        span: &mut StagingSpan,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        dst: &Buffer<H>,
    ) {
        let tail = self.0.1.as_ref();
        assert!(dst.len() as usize == tail.len(), "destination trailing tail length mismatch");

        let header_bytes = std::slice::from_raw_parts((&self.0.0 as *const H).cast::<u8>(), std::mem::size_of::<H>());
        staging.stage_bytes(span, device, cmd, dst, 0, header_bytes);

        let tail_bytes =
            std::slice::from_raw_parts(tail.as_ptr().cast::<u8>(), tail.len() * std::mem::size_of::<H::Tail>());
        staging.stage_bytes(span, device, cmd, dst, H::tail_offset() as usize, tail_bytes);
    }
}

impl<H: Trailing, S: AsRef<[H::Tail]>> UploadOp<H> for Partial<S>
where
    H::Tail: Copy,
{
    unsafe fn apply(
        self,
        staging: &StagingPool,
        span: &mut StagingSpan,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        dst: &Buffer<H>,
    ) {
        let tail = self.1.as_ref();
        assert!(self.0 + tail.len() <= dst.len() as usize, "destination trailing tail range out of bounds");
        let tail_offset = H::tail_offset() as usize + self.0 * std::mem::size_of::<H::Tail>();
        let bytes = std::slice::from_raw_parts(tail.as_ptr().cast::<u8>(), tail.len() * std::mem::size_of::<H::Tail>());
        staging.stage_bytes(span, device, cmd, dst, tail_offset, bytes);
    }
}

impl<D: ?Sized> UploadOp<D> for Raw<'_> {
    unsafe fn apply(
        self,
        staging: &StagingPool,
        span: &mut StagingSpan,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        dst: &Buffer<D>,
    ) {
        staging.stage_bytes(span, device, cmd, dst, self.0, self.1);
    }
}

impl Drop for StagingSpan {
    fn drop(&mut self) {
        println!("WARNING: {} dropped implicitly!", std::any::type_name::<Self>());
    }
}
