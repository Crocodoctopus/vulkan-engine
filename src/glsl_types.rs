use ash::vk;
use glam::*;
use std::mem::offset_of;

use crate::buffer::Trailing;

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub(super) struct GpuDrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub(super) struct GpuDrawCommandBuffer {
    pub len: u32,
    pub data: [GpuDrawIndexedIndirectCommand; 0],
}

impl Trailing for GpuDrawCommandBuffer {
    type Tail = GpuDrawIndexedIndirectCommand;

    fn tail_offset() -> u64 {
        offset_of!(Self, data) as u64
    }

    fn byte_size(len: u32) -> u64 {
        Self::tail_offset() + len as u64 * std::mem::size_of::<Self::Tail>() as u64
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub(super) struct GpuFrustumPassingMeshletBuffer {
    pub len: u32,
    pub data: [u32; 0],
}

impl Trailing for GpuFrustumPassingMeshletBuffer {
    type Tail = u32;

    fn tail_offset() -> u64 {
        offset_of!(Self, data) as u64
    }

    fn byte_size(len: u32) -> u64 {
        Self::tail_offset() + len as u64 * std::mem::size_of::<Self::Tail>() as u64
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub(super) struct GpuActiveMeshletBuffer {
    pub len: u32,
    pub data: [u32; 0],
}

impl Trailing for GpuActiveMeshletBuffer {
    type Tail = u32;

    fn tail_offset() -> u64 {
        offset_of!(Self, data) as u64
    }

    fn byte_size(len: u32) -> u64 {
        Self::tail_offset() + len as u64 * std::mem::size_of::<Self::Tail>() as u64
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub(super) struct GpuFrameGlobal {
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
    pub screen_info: Vec4,

    pub active_meshlet_buffer: vk::DeviceAddress,
    pub meshlet_visibility_buffer: vk::DeviceAddress,
    pub meshlet_buffer: vk::DeviceAddress,
    pub draw_cmd_buffer: vk::DeviceAddress,
    pub object_buffer: vk::DeviceAddress,
    pub frustum_passing_meshlet_buffer: vk::DeviceAddress,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub(super) struct GpuObjectInstance {
    pub position: Vec3,
    pub scale: f32,
    pub orientation: Quat,
    pub vertex_buffer: [vk::DeviceAddress; 8],
    pub texture_id: u32,
}

#[derive(Clone, Copy, Debug, Default)]
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

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub(super) struct GpuVertex {
    pub position: [i16; 3],
    pub uv: [i16; 2],
    pub normal: [i8; 3],
}
