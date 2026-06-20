#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

#include "types.h"

const uint MAX_LODS = 8u;
const uint MESHLET_TAG_BITS = 4u;
const uint LOD_ID_BITS = 3u;
const uint MESHLET_TAG_MASK = (1u << MESHLET_TAG_BITS) - 1u;
const uint LOD_ID_MASK = (1u << LOD_ID_BITS) - 1u;
const uint LOD_ID_SHIFT = MESHLET_TAG_BITS;
const uint ACTIVE_MESHLET_ID_SHIFT = LOD_ID_BITS;

vec3 rotate_quat(vec3 v, vec4 q) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

uint pack_draw_payload(uint meshlet_tag, uint lod_id) {
    return (meshlet_tag & MESHLET_TAG_MASK) | ((lod_id & LOD_ID_MASK) << LOD_ID_SHIFT);
}

void unpack_draw_payload(uint payload, out uint meshlet_tag, out uint lod_id) {
    meshlet_tag = payload & MESHLET_TAG_MASK;
    lod_id = (payload >> LOD_ID_SHIFT) & LOD_ID_MASK;
}

uint pack_active_meshlet(uint meshlet_id, uint lod_id) {
    return (meshlet_id << ACTIVE_MESHLET_ID_SHIFT) | (lod_id & LOD_ID_MASK);
}

void unpack_active_meshlet(uint payload, out uint meshlet_id, out uint lod_id) {
    meshlet_id = payload >> ACTIVE_MESHLET_ID_SHIFT;
    lod_id = payload & LOD_ID_MASK;
}

uint choose_lod(float radius, float distance) {
    float ratio = max(distance, 1e-5) / max(radius, 1e-5);
    return uint(clamp(floor(log2(ratio)), 0.0, float(MAX_LODS - 1u)));
}

VkDrawIndexedIndirectCommand make_draw_cmd(MeshletInstance meshlet, uint meshlet_tag, uint lod_id) {
    VkDrawIndexedIndirectCommand cmd;
    cmd.index_count = meshlet.index_count;
    cmd.first_index = meshlet.first_index;
    cmd.instance_count = 1;
    // vertex_offset carries the object index; first_instance carries a 5-bit debug tag and LOD ID.
    cmd.vertex_offset = int(meshlet.object_id);
    cmd.first_instance = pack_draw_payload(meshlet_tag, lod_id);
    return cmd;
}

#endif
