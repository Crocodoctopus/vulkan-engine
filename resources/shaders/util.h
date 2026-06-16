#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

vec3 rotate_quat(vec3 v, vec4 q) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

VkDrawIndexedIndirectCommand make_draw_cmd(MeshletInstance meshlet, uint padding) {
    VkDrawIndexedIndirectCommand cmd;
    cmd.index_count = meshlet.index_count;
    cmd.first_index = meshlet.first_index;
    cmd.instance_count = 1;
    // vertex_offset carries the object index; first_instance forwards an extra u32 to the vertex stage.
    cmd.vertex_offset = int(meshlet.object_id);
    cmd.first_instance = padding;
    return cmd;
}

#endif
