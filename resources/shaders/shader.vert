#version 450
#extension GL_EXT_buffer_reference : require

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

struct Instance {
    mat4 model;
    VertexBuffer vertex_buffer;
    uint tex_id;
};

layout (set = 0, binding = 0) uniform Global {
    mat4 proj;
    mat4 view;
};

layout (std140, set = 0, binding = 2) readonly buffer InstanceBuffer {
    Instance instances[];
};

layout (location = 0) out vec2 frag_uv;
layout (location = 1) out uint frag_tex_id;

void main() {
    Instance instance = instances[gl_InstanceIndex];
    Vertex vert = instance.vertex_buffer.data[gl_VertexIndex];
    gl_Position = proj * view * instance.model * vec4(vert.position, 1.0);
    frag_uv = vec2(vert.u, vert.v);
    frag_tex_id = instance.tex_id;
}
