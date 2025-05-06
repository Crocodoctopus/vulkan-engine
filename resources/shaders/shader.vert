#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#include "types.h"

layout (std430, set = 0, binding = 0) uniform SceneGlobal {
    mat4 pv;
    mat4 proj;
    mat4 view;
    
    vec3 camera_position;
    vec3 camera_direction;
    vec3 light_position;
    vec4 light_color;
};

layout (set = 1, binding = 0) uniform MeshletRenderGlobal {
    InstanceBuffer instance_buffer;
    ObjectBuffer object_buffer;
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
    Instance instance = instance_buffer.data[gl_DrawID];
    Object object = object_buffer.data[instance.object_id];
    Vertex vert = object.vertex_buffer.data[gl_VertexIndex];

    //
    //gl_Position = pv * object.model * vec4(vert.position, 1.0);
    vec3 wpos = object.position + object.scale * rotate_quat(vert.position, object.orientation);
    gl_Position = pv * vec4(wpos, 1.0);
    frag_position = wpos;
    frag_normal = rotate_quat(vert.normal, object.orientation);
    frag_uv = vec2(vert.u, vert.v);
    frag_tex_id = object.tex_id;
    frag_color = colors[gl_DrawID % 14];
}
