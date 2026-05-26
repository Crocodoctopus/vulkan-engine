use ash::vk;
use glam::*;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub(super) struct SceneGlobal {
    // Matrices.
    pub pv: Mat4,
    pub proj: Mat4,
    pub view: Mat4,

    // Misc.
    pub camera_position: Vec4,
    pub camera_direction: Vec4, // XYZ
    pub light_position: Vec4,
    pub light_color: Vec4,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub(super) struct MeshletRenderGlobal {
    pub instance_buffer: vk::DeviceAddress,
    pub object_buffer: vk::DeviceAddress,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(super) struct MeshletCullGlobal {
    pub frustum: Vec4,

    pub draw_count_buffer: vk::DeviceAddress,
    pub meshlet_buffer: vk::DeviceAddress,
    pub draw_cmd_buffer: vk::DeviceAddress,
    pub instance_buffer: vk::DeviceAddress,
    pub object_buffer: vk::DeviceAddress,

    pub instances: u32,
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(super) struct Object {
    pub position: Vec3,
    pub scale: f32,
    pub orientation: Quat,
    pub vertex_buffer: vk::DeviceAddress,
    pub texture_id: u32,
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(super) struct Instance {
    pub object_id: u32,
}

#[derive(Clone, Debug)]
#[repr(C, align(16))]
pub(super) struct MeshletData {
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
pub(super) struct Vertex {
    pub position: [i16; 3],
    pub uv: [i16; 2],
    pub normal: [i8; 3],
}
