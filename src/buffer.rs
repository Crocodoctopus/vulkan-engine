use ash::vk;

struct Buffer {
    buffer: vk::Buffer,
    alloc: vk_mem::Allocation,
    alloc_info: vk_mem::AllocationInfo,
}

impl Buffer {
    pub fn new(size: usize, usage: vk::BufferUsageFlags, vma_usage: vk_mem::MemoryUsage) -> Self {
        let buffer_cinfo = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(usage);
        let buffer_acinfo = vk_mem::AllocationCreateInfo {
            usage: vma_usage,
            ..Default::default()
        };
    }
}

/*
use alloc::alloc;
use ash::vk;
use ash::{Device, Instance};

struct VertexBuffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
}

unsafe fn find_memory_type(
    instance: ash::Instance,
    pdevice: vk::PhysicalDevice,
    device: ash::Device,
    buffer: vk::Buffer,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let memory_properties = instance.get_physical_device_memory_properties(pdevice);
    let memory_requirements = device.get_buffer_memory_requirements(buffer);

    for i in 0..memory_properties.memory_type_count {
        // Check if this resource supports this memory type.
        if memory_requirements.memory_type_bits & (i << 1) == 0 {
            continue;
        }

        // Check if this memory type has the property flags needed.
        if memory_properties.memory_types[i as usize]
            .property_flags
            .contains(flags)
        {
            return Some(i);
        }
    }
    return None;
}

impl VertexBuffer {
    unsafe fn new(
        instance: ash::Instance,
        pdevice: vk::PhysicalDevice,
        device: ash::Device,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = device.create_buffer(&buffer_info, None).unwrap();

        let index = find_memory_type(
            instance,
            pdevice,
            device,
            buffer,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();

        let memory_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size as u64)
            .memory_type_index(index);

        let memory = device.allocate_memory(memory_info, None).unwrap();

        device.bind_buffer_memory(buffer, memory, 0);
    }
}
*/
