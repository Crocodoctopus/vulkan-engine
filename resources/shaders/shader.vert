#version 450
#extension GL_EXT_buffer_reference : require

struct Vertex {
    vec3 position;
    float u;
    vec3 normal;
    float v;
    vec4 color;
};

layout (set = 0, binding = 0) uniform Global {
    mat4 proj;
    mat4 view;
};

layout (buffer_reference, std430) readonly buffer VertexBuffer {
    Vertex data[];
};

layout (push_constant) uniform Constants {
    mat4 model;
    VertexBuffer vertex_buffer;
    uint tex_id;
};

layout (location = 0) out vec2 frag_uv;
layout (location = 1) out uint frag_tex_id;

void main() {
    Vertex vert = vertex_buffer.data[gl_VertexIndex];
    gl_Position = proj * view * model * vec4(vert.position, 1.0);
    frag_uv = vec2(vert.u, vert.v);
    frag_tex_id = tex_id;
}
