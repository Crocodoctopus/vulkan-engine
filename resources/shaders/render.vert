#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

#include "types.h"

layout(std430, set = 1, binding = 0) UNIFORM {
    FrameGlobal frame_global;
};

layout (location = 0) out vec2 frag_uv;
layout (location = 1) out uint frag_tex_id;
layout (location = 2) out vec4 frag_color;
layout (location = 3) out vec3 frag_normal;
layout (location = 4) out vec3 frag_position;

vec4 colors[14] = {
    vec4(0.0, 0.0, 1.0, 1.0),
    vec4(0.0, 1.0, 0.0, 1.0),
    vec4(0.0, 1.0, 1.0, 1.0),
    vec4(1.0, 0.0, 0.0, 1.0),
    vec4(1.0, 0.0, 1.0, 1.0),
    vec4(1.0, 1.0, 0.0, 1.0),
    vec4(1.0, 1.0, 1.0, 1.0),

    vec4(0.25, 0.25, 1.0, 1.0),
    vec4(0.25, 1.0, 0.25, 1.0),
    vec4(0.25, 1.0, 1.0, 1.0),
    vec4(1.0, 0.25, 0.25, 1.0),
    vec4(1.0, 0.25, 1.0, 1.0),
    vec4(1.0, 1.0, 0.25, 1.0),
    vec4(1.0, 1.0, 1.0, 1.0),
};

vec3 rotate_quat(vec3 v, vec4 q) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main() {
    ObjectInstance object = frame_global.object_buffer.data[gl_BaseVertex];
    Vertex vert = object.vertex_buffer.data[gl_VertexIndex - gl_BaseVertex];

    // Decompress.
    vec3 position = vec3(vert.x, vert.y, vert.z) / 32767.0f;
    vec3 normal = vec3(vert.nx, vert.ny, vert.nz) / 127.0f;
    vec2 uv = vec2(vert.u, vert.v) / 32767.0f;

    //
    vec3 wpos = object.position + object.scale * rotate_quat(position, object.orientation);
    gl_Position = frame_global.pv * vec4(wpos, 1.0);
    frag_position = wpos;
    frag_normal = rotate_quat(normal, object.orientation);
    frag_uv = uv;
    frag_tex_id = object.tex_id;
    frag_color = colors[gl_BaseInstance % 14];
}
