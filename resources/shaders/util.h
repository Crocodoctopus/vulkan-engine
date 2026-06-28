#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

#include "types.h"

const uint MAX_LODS = 8u;
const uint MESHLET_TAG_BITS = 4u;
const uint MESHLET_TAG_MASK = (1u << MESHLET_TAG_BITS) - 1u;
const float LOD_DISTANCE_BIAS = 2.0;
const float LOD_DISTANCE_OFFSET = 0.25;

vec3 rotate_quat(vec3 v, vec4 q) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void unpack(uint pad0, uint pad1, out uint object_index, out uint meshlet_color_index) {
    object_index = pad0 >> 15;
    meshlet_color_index = pad0 & 15;
}

void pack(out uint pad0, out uint pad1, uint object_index, uint meshlet_color_index) {
    pad0 = (object_index << 15) | (meshlet_color_index & 15);
    pad1 = 0;
}

VkDrawIndexedIndirectCommand make_draw_cmd(Meshlet meshlet, uint scene_index_offset, uint pad0, uint pad1) {
    VkDrawIndexedIndirectCommand cmd;
    cmd.index_count = meshlet.index_count;
    cmd.instance_count = 1;
    cmd.first_index = meshlet.first_index + scene_index_offset;
    // We can encode 2 uints in vertex_offset and first_instance.
    cmd.vertex_offset = int(pad0);
    cmd.first_instance = pad1;
    return cmd;
}

#endif
