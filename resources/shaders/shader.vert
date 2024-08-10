#version 450

layout(set = 0, binding = 0) uniform Global {
    mat4 proj;
    mat4 view;
};

layout(push_constant) uniform Constants {
    mat4 model;
};

layout (location = 0) in vec2 vert_position;
layout (location = 1) in vec3 vert_color;

layout (location = 0) out vec3 frag_color;

void main() {
    gl_Position = proj * view * model * vec4(vert_position, 0.0, 1.0);
    frag_color = vert_color;
}
