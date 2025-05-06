#version 450
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

layout (std430, set = 0, binding = 0) uniform SceneGlobal {
    mat4 pv;
    mat4 proj;
    mat4 view;
    
    vec3 camera_position;
    vec3 camera_direction;
    vec3 light_position;
    vec4 light_color;
};

layout (set = 1, binding = 1) uniform sampler2D samplers[];

layout (location = 0) in vec2 frag_uv;
layout (location = 1) flat in uint frag_tex_id;
layout (location = 2) in vec4 frag_color;
layout (location = 3) flat in vec3 frag_normal;
layout (location = 4) in vec3 frag_position;

layout (location = 0) out vec4 out_color;

void main() {
    vec3 normal = normalize(frag_normal);
    vec3 incident_ray = normalize(light_position - frag_position);
    vec3 reflected_ray = normalize(reflect(-incident_ray, normal));
    vec3 to_camera_ray = normalize(camera_position - frag_position);

    float ambient = .75;
    float diffuse = max(0.0, dot(incident_ray, normal));
    float specular = pow(max(0.0, dot(to_camera_ray, reflected_ray)), 64.0);
    
    vec2 uv = frag_uv;
    uv.y = 1.0f - frag_uv[1];
    out_color = vec4(frag_color.rgb * light_color.rgb * (0.5 * ambient + 0.5 * diffuse + 0.5 * specular), 1.0);//texture(samplers[frag_tex_id], uv); 
}
