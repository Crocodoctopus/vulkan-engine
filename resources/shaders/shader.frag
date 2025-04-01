#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout (set = 0, binding = 1) uniform sampler2D samplers[];

layout (location = 0) in vec2 frag_uv;
layout (location = 1) flat in uint frag_tex_id;
layout (location = 2) in vec4 frag_color;

layout (location = 0) out vec4 out_color;

void main() {
    vec2 uv = frag_uv;
    uv.y = 1.0f - frag_uv[1];
    out_color = frag_color;//texture(samplers[frag_tex_id], uv); 
}
