use ash::vk;
use encase::ArrayLength;
use glam::*;

#[derive(Clone, Copy, Debug, Default, encase::ShaderType)]
pub(super) struct GpuDrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

#[derive(Clone, Debug, Default, encase::ShaderType)]
pub(super) struct GpuDrawCommandBuffer {
    pub len: ArrayLength,
    #[shader(align(4), size(runtime))]
    pub data: Vec<GpuDrawIndexedIndirectCommand>,
}

impl GpuDrawCommandBuffer {
    pub const LEN_OFFSET: u64 = <Self as encase::ShaderType>::METADATA.offset(0);
    pub const DATA_OFFSET: u64 = <Self as encase::ShaderType>::METADATA.offset(1);

    pub fn new(data: impl IntoIterator<Item = GpuDrawIndexedIndirectCommand>) -> Self {
        Self { len: ArrayLength, data: data.into_iter().collect() }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut buffer = encase::StorageBuffer::new(&mut bytes);
            buffer.write(&self).unwrap();
        }
        bytes
    }
}

impl From<Vec<GpuDrawIndexedIndirectCommand>> for GpuDrawCommandBuffer {
    fn from(data: Vec<GpuDrawIndexedIndirectCommand>) -> Self {
        Self::new(data)
    }
}

impl<const N: usize> From<[GpuDrawIndexedIndirectCommand; N]> for GpuDrawCommandBuffer {
    fn from(data: [GpuDrawIndexedIndirectCommand; N]) -> Self {
        Self::new(data)
    }
}

impl From<&[GpuDrawIndexedIndirectCommand]> for GpuDrawCommandBuffer {
    fn from(data: &[GpuDrawIndexedIndirectCommand]) -> Self {
        Self::new(data.iter().copied())
    }
}

impl FromIterator<GpuDrawIndexedIndirectCommand> for GpuDrawCommandBuffer {
    fn from_iter<T: IntoIterator<Item = GpuDrawIndexedIndirectCommand>>(iter: T) -> Self {
        Self::new(iter)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub(super) struct FrameGlobal {
    // Matrices.
    pub pv: Mat4,
    pub proj: Mat4,
    pub view: Mat4,

    // Misc.
    pub camera_position: Vec4,
    pub camera_direction: Vec4, // XYZ
    pub light_position: Vec4,
    pub light_color: Vec4,

    pub frustum: Vec4,

    pub meshlet_visibility_buffer: vk::DeviceAddress,
    pub meshlet_buffer: vk::DeviceAddress,
    pub draw_cmd_buffer: vk::DeviceAddress,
    pub object_buffer: vk::DeviceAddress,

    pub instances: u32,
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(super) struct GpuObjectInstance {
    pub position: Vec3,
    pub scale: f32,
    pub orientation: Quat,
    pub vertex_buffer: vk::DeviceAddress,
    pub texture_id: u32,
}

#[derive(Clone, Debug)]
#[repr(C, align(16))]
pub(super) struct GpuMeshletInstance {
    // Culling.
    pub center: Vec3,
    pub radius: f32,
    pub cone_apex: Vec3,
    pub pad0: f32,
    pub cone_axis: Vec3,
    pub cone_cutoff: f32,

    // Draw cmd.
    pub object_id: u32,
    pub index_count: u32,
    pub first_index: u32,
}

#[derive(Clone, Debug, Default)]
#[repr(C)]
pub(super) struct GpuVertex {
    pub position: [i16; 3],
    pub uv: [i16; 2],
    pub normal: [i8; 3],
}
