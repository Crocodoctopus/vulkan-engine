#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

struct Vertex {
    int16_t x, y, z;
    int16_t u, v;
    int8_t nx, ny, nz;
};

layout (buffer_reference, scalar) readonly buffer VertexBuffer {
    Vertex data[];
};

struct Object {
    vec3 position;
    float scale;
    vec4 orientation;
    VertexBuffer vertex_buffer;
    uint tex_id;
};

struct Instance {
    uint object_id;   
};

layout (buffer_reference, std430) readonly buffer InstanceBuffer {
    Instance data[];
};

layout (buffer_reference, std430) readonly buffer ObjectBuffer {
    Object data[];
};

layout (buffer_reference, std430) buffer DrawCountBuffer {
    uint counter;
};

struct VkDrawIndexedIndirectCommand {
    uint index_count;
    uint instance_count;
    uint first_index;
    int vertex_offset;
    uint first_instance;
};

struct MeshletData {
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

layout (buffer_reference, std430) readonly buffer MeshletBuffer {
    MeshletData data[];
};

layout (buffer_reference, std430) writeonly buffer DrawCmdBuffer {
    VkDrawIndexedIndirectCommand data[];  
};
