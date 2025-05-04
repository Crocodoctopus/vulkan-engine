#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout (set = 0, binding = 1) uniform sampler2D samplers[];

layout (location = 0) in vec2 frag_uv;
layout (location = 1) flat in uint frag_tex_id;
layout (location = 2) in vec4 frag_color;
layout (location = 3) in vec3 frag_normal;

layout (location = 0) out vec4 out_color;

void main() {
    vec3 lrgb = vec3(1.0, 1.0, 1.0);
    vec3 lsrc = vec3(1.0, 0.0, 0.0);
    vec3 normal = normalize(frag_normal);
    
    vec3 ambient = vec3(.5, .5, .5);
    vec3 diffuse = max(0.0, dot(lsrc, normal)) * lrgb;
    vec3 specular = normalize(reflect(-lsrc, frag_normal));
    
    vec2 uv = frag_uv;
    uv.y = 1.0f - frag_uv[1];
    out_color = vec4(frag_color.xyz * (ambient + diffuse + 0*specular), 1.0);//texture(samplers[frag_tex_id], uv); 
}
