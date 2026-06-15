use ash::vk;
use vk_mem::Alloc;

#[derive(Debug)]
pub(crate) struct Buffer<T: ?Sized> {
    pub phantom: std::marker::PhantomData<T>,
    pub buffer: vk::Buffer,
    pub alloc: Option<vk_mem::Allocation>,
    pub len: u32,
    pub size_bytes: u64,
}

impl<T: ?Sized> Drop for Buffer<T> {
    fn drop(&mut self) {
        if self.alloc.is_some() {
            panic!("Active {} dropped implicitly", std::any::type_name::<Self>());
        }
    }
}

#[allow(unused)]
impl<T: ?Sized> Buffer<T> {
    pub(crate) fn null() -> Self {
        Self {
            phantom: std::marker::PhantomData,
            buffer: vk::Buffer::null(),
            alloc: None,
            len: 0,
            size_bytes: 0,
        }
    }

    pub(crate) fn is_null(&self) -> bool {
        self.alloc.is_none()
    }

    pub(crate) fn vk_handle(&self) -> vk::Buffer {
        self.buffer
    }

    pub(crate) fn take(&mut self) -> Self {
        Self {
            phantom: self.phantom,
            buffer: self.buffer,
            len: self.len,
            size_bytes: self.size_bytes,
            alloc: self.alloc.take(),
        }
    }

    pub(crate) unsafe fn destroy(mut self, allocator: &vk_mem::Allocator) {
        if let Some(alloc) = self.alloc.as_mut() {
            unsafe {
                allocator.destroy_buffer(self.buffer, alloc);
            }
        }
        std::mem::forget(self);
    }

    pub(crate) fn size(&self) -> usize {
        self.size_bytes as usize
    }
}

#[allow(unused)]
impl<T> Buffer<T> {
    pub(crate) fn new(
        allocator: &vk_mem::Allocator,
        vk_usage: vk::BufferUsageFlags,
        vma_usage: vk_mem::MemoryUsage,
    ) -> Self {
        unsafe {
            let (buffer, alloc) = allocator
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(size_of::<T>() as u64).usage(vk_usage),
                    &vk_mem::AllocationCreateInfo { usage: vma_usage, ..Default::default() },
                )
                .unwrap();

            Buffer {
                phantom: std::marker::PhantomData,
                buffer,
                alloc: Some(alloc),
                len: 1,
                size_bytes: size_of::<T>() as u64,
            }
        }
    }
}

#[allow(unused)]
impl<T> Buffer<[T]> {
    pub(crate) fn new(
        allocator: &vk_mem::Allocator,
        len: u32,
        vk_usage: vk::BufferUsageFlags,
        vma_usage: vk_mem::MemoryUsage,
    ) -> Self {
        unsafe {
            let (buffer, alloc) = allocator
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(len as u64 * size_of::<T>() as u64).usage(vk_usage),
                    &vk_mem::AllocationCreateInfo { usage: vma_usage, ..Default::default() },
                )
                .unwrap();

            Buffer {
                phantom: std::marker::PhantomData,
                buffer,
                alloc: Some(alloc),
                len,
                size_bytes: len as u64 * size_of::<T>() as u64,
            }
        }
    }

    pub(crate) fn len(&self) -> u32 {
        self.len
    }

    pub(crate) fn stride(&self) -> u32 {
        size_of::<T>() as u32
    }
}
