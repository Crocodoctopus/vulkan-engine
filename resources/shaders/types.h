#ifndef TYPES_H_INCLUDED
#define TYPES_H_INCLUDED

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

#define CONCAT_INNER(a, b) a##b
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define BUFFER_NAME(line) CONCAT(TMP_, line)
#define UNIFORM uniform BUFFER_NAME(__LINE__)

struct Vertex {
    int16_t x, y, z;
    int16_t u, v;
    int8_t nx, ny, nz;
};

layout (buffer_reference, scalar) readonly buffer VertexBuffer {
    Vertex data[];
};

struct ObjectInstance {
    vec3 position;
    float scale;
    vec4 orientation;
    VertexBuffer vertex_buffer;
    uint tex_id;
};

layout (buffer_reference, std430) readonly buffer ObjectInstanceBuffer {
    ObjectInstance data[];
};

struct MeshletInstance {
    // Culling.
    vec3 center;
    float radius;
    vec3 cone_apex;
    float pad0;
    vec3 cone_axis;
    float cone_cutoff;

    // Draw cmd.
    uint object_id;
    uint index_count;
    uint first_index;
};

layout (buffer_reference, std430) readonly buffer MeshletInstanceBuffer {
    MeshletInstance data[];
};

layout (buffer_reference, scalar) buffer MeshletVisibilityBuffer {
    bool data[];
};

// Meshlets that pass frustum culling and still need occlusion testing.
layout (buffer_reference, scalar) buffer FrustumPassingMeshletBuffer {
    uint len;
    uint meshlet_ids[];
};

struct VkDrawIndexedIndirectCommand {
    uint index_count;
    uint instance_count;
    uint first_index;
    int vertex_offset;
    uint first_instance;
};

layout (buffer_reference, std430) writeonly buffer DrawCmdBuffer {
    // Counter prefix for vkCmdDrawIndexedIndirectCount; commands start immediately after it.
    uint len;
    VkDrawIndexedIndirectCommand data[];
};

struct FrameGlobal {
    mat4 pv;
    mat4 proj;
    mat4 view;

    vec3 camera_position;
    vec3 camera_direction;
    vec3 light_position;
    vec4 light_color;

    vec4 frustum;

    MeshletVisibilityBuffer meshlet_visibility_buffer;
    MeshletInstanceBuffer meshlet_buffer;
    DrawCmdBuffer draw_cmd_buffer;
    DrawCmdBuffer late_draw_cmd_buffer;
    ObjectInstanceBuffer object_buffer;

    FrustumPassingMeshletBuffer frustum_passing_meshlet_buffer;

    uint instances;
};

#endif
