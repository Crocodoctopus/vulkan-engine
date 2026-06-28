#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#include "types.h"

layout(std430, set = 1, binding = 0) BUFFER {
    FrameGlobal frame_global;
};

layout(r32ui, set = 1, binding = 1) uniform uimage2D overdraw_counts;

layout (location = 0) in vec2 frag_uv;
layout (location = 1) flat in uint frag_tex_id;
layout (location = 2) in vec4 frag_color;
layout (location = 3) in vec3 frag_normal;
layout (location = 4) in vec3 frag_position;

layout(early_fragment_tests) in;

void main() {
    uvec2 screen = uvec2(frame_global.screen_info.xy);
    uvec2 p = uvec2(gl_FragCoord.xy);
    if (p.x < screen.x && p.y < screen.y) {
        imageAtomicAdd(overdraw_counts, ivec2(p), 1u);
    }
}
