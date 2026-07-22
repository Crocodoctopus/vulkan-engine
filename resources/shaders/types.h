#ifndef TYPES_H_INCLUDED
#define TYPES_H_INCLUDED

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

#define CONCAT_INNER(a, b) a##b
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define BUFFER_NAME(line) CONCAT(TMP_, line)
#define UNIFORM uniform BUFFER_NAME(__LINE__)
#define BUFFER buffer BUFFER_NAME(__LINE__)

struct Vertex {
    int16_t x, y, z;
    int16_t u, v;
    int8_t nx, ny, nz;
};

struct Meshlet {
    // Culling.
    vec3 center;
    float radius;
    vec3 aabb_min;
    float cone_cutoff;
    vec3 aabb_max;
    uint index_count;
    vec3 cone_apex;
    uint first_index;
    vec3 cone_axis;
};

layout (buffer_reference, scalar) readonly buffer VertexBuffer {
    Vertex data[];
};

layout (buffer_reference, std430) readonly buffer MeshletBuffer {
    Meshlet data[];
};

layout (buffer_reference, scalar) buffer VisibilityBuffer {
    uint data[];
};

struct ObjectInstance {
    vec3 position;
    float scale;
    vec4 orientation;
    VertexBuffer vertex_buffer;
    MeshletBuffer meshlet_buffer;
    VisibilityBuffer visibility_buffer;
    VisibilityBuffer previous_visibility_buffer;
    uint tex_id;
    uint scene_index_offset;
};

layout (buffer_reference, std430) readonly buffer ObjectInstanceBuffer {
    ObjectInstance data[];
};

struct MeshletLookup {
    uint16_t object_index;
    uint16_t meshlet_index;
};

// Meshlets that pass frustum culling and still need occlusion testing.
layout (buffer_reference, scalar) buffer FrustumPassingMeshletBuffer {
    uint len;
    MeshletLookup data[];
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

struct VkDispatchIndirectCommand {
    uint x;
    uint y;
    uint z;
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
    vec4 screen_info;

    DrawCmdBuffer draw_cmd_buffer;
    ObjectInstanceBuffer object_buffer;
    FrustumPassingMeshletBuffer frustum_passing_meshlet_buffer;

    VkDispatchIndirectCommand occlusion_dispatch;
};

#endif
