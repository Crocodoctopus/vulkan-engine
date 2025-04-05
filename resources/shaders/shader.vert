#version 460
#extension GL_EXT_buffer_reference : require

struct Vertex {
    vec3 position;
    float u;
    vec3 normal;
    float v;
    vec4 color;
};

struct Instance {
    uint object_id;   
};

struct Object {
    mat4 model;
    uint tex_id;
};

layout (buffer_reference, std430) readonly buffer VertexBuffer {
    Vertex data[];
};

layout (buffer_reference, std430) readonly buffer InstanceBuffer {
    Instance data[];
};

layout (buffer_reference, std430) readonly buffer ObjectBuffer {
    Object data[];
};


layout (set = 0, binding = 0) uniform Global {
    mat4 proj;
    mat4 view;
    VertexBuffer vertex_buffer;
    InstanceBuffer instance_buffer;
    ObjectBuffer object_buffer;
};


layout (location = 0) out vec2 frag_uv;
layout (location = 1) out uint frag_tex_id;
layout (location = 2) out vec4 frag_color;

vec4 colors[7] = {
    vec4(0.0, 0.0, 1.0, 1.0),
    vec4(0.0, 1.0, 0.0, 1.0),
    vec4(0.0, 1.0, 1.0, 1.0),
    vec4(1.0, 0.0, 0.0, 1.0),
    vec4(1.0, 0.0, 1.0, 1.0),
    vec4(1.0, 1.0, 0.0, 1.0),
    vec4(1.0, 1.0, 1.0, 1.0),
};

void main() {
    // Currently only one object.
    Instance instance = instance_buffer.data[gl_DrawID];
    Object object = object_buffer.data[instance.object_id];
    Vertex vert = vertex_buffer.data[gl_VertexIndex];

    //
    gl_Position = proj * view * object.model * vec4(vert.position, 1.0);
    frag_uv = vec2(vert.u, vert.v);
    frag_tex_id = object.tex_id;
    frag_color = colors[gl_DrawID % 7];
}
