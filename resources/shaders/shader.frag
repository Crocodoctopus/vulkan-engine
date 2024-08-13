#version 450

layout(set = 0, binding = 1) uniform sampler2D samplers[];

layout(location = 0) in vec2 frag_texcoord;

layout(location = 0) out vec4 out_color;

void main() {
    vec2 uv = frag_texcoord;
    uv.y = 1.0f - frag_texcoord.y;
    out_color = texture(samplers[0], uv); 
}
