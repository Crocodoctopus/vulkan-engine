struct Vertex {
    vec3 position;
    float u;
    vec3 normal;
    float v;
    vec4 color;
};

layout (buffer_reference, std430) readonly buffer VertexBuffer {
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
