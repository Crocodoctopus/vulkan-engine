#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_KHR_shader_subgroup_quad : require

#include "types.h"

layout(std430, set = 1, binding = 0) BUFFER {
    FrameGlobal frame_global;
};

layout(r32ui, set = 1, binding = 1) uniform uimage2D overdraw_counts;

layout(location = 0) in vec2 frag_uv;
layout(location = 1) flat in uint frag_tex_id;
layout(location = 2) in vec4 frag_color;
layout(location = 3) in vec3 frag_normal;
layout(location = 4) in vec3 frag_position;

layout(early_fragment_tests) in;

void main() {
    uvec2 screen = uvec2(frame_global.screen_info.xy);
    vec2 p0 = subgroupQuadBroadcast(gl_FragCoord.xy, 0);
    vec2 p1 = subgroupQuadBroadcast(gl_FragCoord.xy, 1);
    vec2 p2 = subgroupQuadBroadcast(gl_FragCoord.xy, 2);
    vec2 p3 = subgroupQuadBroadcast(gl_FragCoord.xy, 3);
    uint helper0 = subgroupQuadBroadcast(gl_HelperInvocation ? 1u : 0u, 0);
    uint helper1 = subgroupQuadBroadcast(gl_HelperInvocation ? 1u : 0u, 1);
    uint helper2 = subgroupQuadBroadcast(gl_HelperInvocation ? 1u : 0u, 2);
    uint helper3 = subgroupQuadBroadcast(gl_HelperInvocation ? 1u : 0u, 3);

    bool is_writer = !gl_HelperInvocation;
    vec2 writer_p = gl_FragCoord.xy;

    if (helper0 == 0u) {
        is_writer = is_writer && (writer_p.y <= p0.y && (writer_p.y < p0.y || writer_p.x <= p0.x));
    }
    if (helper1 == 0u) {
        is_writer = is_writer && (writer_p.y <= p1.y && (writer_p.y < p1.y || writer_p.x <= p1.x));
    }
    if (helper2 == 0u) {
        is_writer = is_writer && (writer_p.y <= p2.y && (writer_p.y < p2.y || writer_p.x <= p2.x));
    }
    if (helper3 == 0u) {
        is_writer = is_writer && (writer_p.y <= p3.y && (writer_p.y < p3.y || writer_p.x <= p3.x));
    }

    if (!is_writer) {
        return;
    }

    if (helper0 != 0u && p0.x < screen.x && p0.y < screen.y) {
        imageAtomicAdd(overdraw_counts, ivec2(p0), 1u);
    }
    if (helper1 != 0u && p1.x < screen.x && p1.y < screen.y) {
        imageAtomicAdd(overdraw_counts, ivec2(p1), 1u);
    }
    if (helper2 != 0u && p2.x < screen.x && p2.y < screen.y) {
        imageAtomicAdd(overdraw_counts, ivec2(p2), 1u);
    }
    if (helper3 != 0u && p3.x < screen.x && p3.y < screen.y) {
        imageAtomicAdd(overdraw_counts, ivec2(p3), 1u);
    }
}
