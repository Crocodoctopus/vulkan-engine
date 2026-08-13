use crate::buffer::Buffer;
use crate::core::VulkanCore;
use crate::glsl_types::{GpuIndex, GpuMeshlet, GpuVertex};
use crate::staging::{StagingPool, StagingSpan, Whole};
use ash::vk;
use glam::*;
use itertools::Itertools;
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

pub(crate) const MAX_LODS: usize = 8;

#[derive(Debug)]
pub(crate) struct Meshlet {
    pub center: [f32; 3],
    pub radius: f32,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub cone_apex: [f32; 3],
    pub cone_axis: [f32; 3],
    pub cone_cutoff: f32,
    pub indices: Box<[u8]>,
    pub positions: Box<[[i16; 3]]>,
    pub normals: Box<[[i8; 3]]>,
    pub _texcoords: Box<[[i16; 2]]>,
}

#[derive(Debug)]
pub(crate) struct Mesh {
    pub scale: f32,
    pub radius: f32,
    pub lod_count: u8,
    pub lods: [Box<[Meshlet]>; MAX_LODS],
}

pub(crate) struct GpuMesh {
    pub vertex_buffer: Buffer<[GpuVertex]>,
    pub index_buffer: Buffer<[GpuIndex]>,
    pub meshlet_buffer: Buffer<[GpuMeshlet]>,
    pub meshlet_lod_to_offset: HashMap<u8, Range<u16>>,
}

impl GpuMesh {
    #[allow(dead_code)]
    pub(crate) unsafe fn destroy(self, allocator: &vk_mem::Allocator) {
        self.vertex_buffer.destroy(allocator);
        self.index_buffer.destroy(allocator);
        self.meshlet_buffer.destroy(allocator);
    }
}

pub(crate) struct PendingGpuMesh {
    gpu_mesh: GpuMesh,
    cmd: vk::CommandBuffer,
    staging: StagingSpan,
    semaphore: vk::Semaphore,
}

impl PendingGpuMesh {
    pub(crate) unsafe fn submit(core: &VulkanCore, staging_pool: &StagingPool, mesh: &Mesh) -> Self {
        let device = &core.device;
        let mut vertices = vec![];
        let mut indices = vec![];
        let mut meshlets = vec![];
        let mut meshlet_lod_to_offset = HashMap::new();
        for lod in 0..mesh.lod_count {
            let meshlet_offset = meshlets.len() as u16;

            for meshlet in &mesh.lods[lod as usize] {
                meshlets.push(GpuMeshlet {
                    center: Vec3::from(meshlet.center),
                    radius: meshlet.radius,
                    aabb_min: Vec3::from(meshlet.aabb_min),
                    cone_cutoff: meshlet.cone_cutoff,
                    aabb_max: Vec3::from(meshlet.aabb_max),
                    index_count: meshlet.indices.len() as u32,
                    cone_apex: Vec3::from(meshlet.cone_apex),
                    first_index: indices.len() as u32,
                    cone_axis: Vec3::from(meshlet.cone_axis),
                });

                indices.extend(meshlet.indices.iter().map(|i| *i as u32 + vertices.len() as u32));

                vertices.extend((0..meshlet.positions.len()).map(|i| GpuVertex {
                    position: meshlet.positions[i],
                    normal: meshlet.normals[i],
                    uv: [0, 0],
                }));
            }

            meshlet_lod_to_offset.insert(lod, meshlet_offset..meshlets.len() as u16);
        }

        let staging_len = [
            meshlets.len() * std::mem::size_of::<GpuMeshlet>(),
            indices.len() * std::mem::size_of::<GpuIndex>(),
            vertices.len() * std::mem::size_of::<GpuVertex>(),
        ]
        .into_iter()
        .sum::<usize>() as u64;

        let mut staging = staging_pool.alloc(staging_len);
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(core.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap()[0];
        let semaphore = device
            .create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(
                    &mut vk::SemaphoreTypeCreateInfo::default()
                        .semaphore_type(vk::SemaphoreType::TIMELINE)
                        .initial_value(0),
                ),
                None,
            )
            .unwrap();

        staging.reset();
        device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();

        let gpu_mesh = GpuMesh {
            meshlet_buffer: Buffer::<[GpuMeshlet]>::new(
                &core.allocator,
                meshlets.len() as u32,
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),
            index_buffer: Buffer::<[GpuIndex]>::new(
                &core.allocator,
                indices.len() as u32,
                vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),
            vertex_buffer: Buffer::<[GpuVertex]>::new(
                &core.allocator,
                vertices.len() as u32,
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            ),
            meshlet_lod_to_offset,
        };

        staging_pool.stage(&mut staging, device, cmd, &gpu_mesh.meshlet_buffer, Whole(meshlets.as_slice()));
        staging_pool.stage(&mut staging, device, cmd, &gpu_mesh.index_buffer, Whole(indices.as_slice()));
        staging_pool.stage(&mut staging, device, cmd, &gpu_mesh.vertex_buffer, Whole(vertices.as_slice()));

        device.end_command_buffer(cmd).unwrap();
        device
            .queue_submit2(
                core.graphics_queue,
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(cmd)])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(semaphore)
                        .value(1)
                        .stage_mask(vk::PipelineStageFlags2::TRANSFER)])],
                vk::Fence::null(),
            )
            .unwrap();

        Self { gpu_mesh, cmd, staging, semaphore }
    }

    #[allow(dead_code)]
    pub(crate) fn wait_info(&self) -> vk::SemaphoreSubmitInfo<'static> {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(1)
            .stage_mask(vk::PipelineStageFlags2::TRANSFER)
    }

    pub(crate) unsafe fn wait(&self, core: &VulkanCore) {
        core.device
            .wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&[self.semaphore]).values(&[1]), u64::MAX)
            .unwrap();
    }

    pub(crate) unsafe fn wait_and_unwrap(self, core: &VulkanCore, staging_pool: &StagingPool) -> GpuMesh {
        self.wait(core);
        core.device.free_command_buffers(core.cmd_pool, &[self.cmd]);
        core.device.destroy_semaphore(self.semaphore, None);
        staging_pool.free_span(self.staging);
        self.gpu_mesh
    }
}

pub(crate) fn load_mesh(filename: impl AsRef<Path>) -> Option<Mesh> {
    let filename = filename.as_ref();
    let model = {
        use std::io::BufReader;
        let data = std::fs::read(filename).ok()?;
        let (models, _) =
            tobj::load_obj_buf(&mut BufReader::new(&data[..]), |_| Ok((Vec::new(), HashMap::new()))).unwrap();
        models.into_iter().next()?.mesh
    };
    let lod_count = match filename.file_stem().and_then(|stem| stem.to_str()) {
        Some("viking_room") => 1,
        _ => MAX_LODS,
    };
    println!("Model details ({:?}):", filename);
    println!("  Indices: {}", model.indices.len());
    println!("  Positions: {}", model.positions.len());
    println!("  Normals: {}", model.normals.len());
    println!("  LODs: {}", lod_count);

    // Calculate bounds.
    let scale =
        model.positions.iter().tuples().fold(0f32, |scale, (x, y, z)| scale.max(x.abs()).max(y.abs()).max(z.abs()));

    #[derive(Clone, Copy, Default)]
    struct Vertex {
        position: Vec3,
        normal: Vec3,
        uv: Vec2,
        _color: Vec3,
    }

    impl meshopt::DecodePosition for Vertex {
        fn decode_position(&self) -> [f32; 3] {
            self.position.to_array()
        }
    }

    let indices = model.indices;
    let positions: Vec<Vec3> = model.positions.chunks_exact(3).map(Vec3::from_slice).collect();
    let uvs: Vec<Vec2> = model.texcoords.chunks_exact(2).map(Vec2::from_slice).collect();
    let normals: Vec<Vec3> = if !model.normals.is_empty() {
        model.normals.chunks_exact(3).map(Vec3::from_slice).collect()
    } else {
        // Normals dont exist, and are constructed here:
        let mut normals = vec![Vec3::ZERO; positions.len()];
        for tri in indices.chunks_exact(3) {
            let [i0, i1, i2]: [u32; 3] = tri.try_into().unwrap();
            let i0 = i0 as usize;
            let i1 = i1 as usize;
            let i2 = i2 as usize;
            let p0 = positions[i0];
            let p1 = positions[i1];
            let p2 = positions[i2];
            let face_normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            normals[i0] += face_normal;
            normals[i1] += face_normal;
            normals[i2] += face_normal;
        }
        normals.iter_mut().for_each(|n| *n = n.normalize_or_zero());
        normals
    };

    let vertices: Box<[Vertex]> = (0..model.positions.len() / 3)
        .map(|i| Vertex {
            position: positions[i],
            normal: normals[i],
            uv: uvs.get(i).cloned().unwrap_or_default(),
            _color: Vec3::splat(1.0),
        })
        .collect();

    let radius = positions.iter().fold(0f32, |acc, position| acc.max(position.length())) / scale;
    let triangle_count = indices.len() / 3;
    let mut lod_indices = indices;
    let lods: [Box<[Meshlet]>; MAX_LODS] = std::array::from_fn(|lod| {
        if lod >= lod_count {
            return Vec::new().into_boxed_slice();
        }

        if lod > 0 {
            let target_triangles = (triangle_count >> lod).max(1);
            let target_index_count = target_triangles * 3;
            let simplified = meshopt::simplify_decoder(
                &lod_indices,
                &vertices,
                target_index_count,
                1.0,
                meshopt::SimplifyOptions::None,
                None,
            );
            if simplified.len() >= 3 {
                lod_indices = simplified;
            }
        }

        let mut fetch_indices = lod_indices.clone();
        // Reorder after simplification so LOD content stays stable.
        meshopt::optimize_vertex_cache_in_place(&mut fetch_indices, vertices.len());
        meshopt::optimize_overdraw_in_place_decoder(&mut fetch_indices, &vertices, 1.05);

        let lod_vertices = meshopt::optimize_vertex_fetch(&mut fetch_indices, &vertices);

        let adapter = meshopt::VertexDataAdapter {
            reader: std::io::Cursor::new(unsafe {
                std::slice::from_raw_parts(lod_vertices.as_ptr() as *const u8, size_of::<Vertex>() * lod_vertices.len())
            }),
            vertex_count: lod_vertices.len(),
            vertex_stride: size_of::<Vertex>(),
            position_offset: 0,
        };

        meshopt::build_meshlets(&fetch_indices, &adapter, 64, 124, 0.5)
            .iter()
            .map(|meshlet| {
                let bounds = meshopt::compute_meshlet_bounds_decoder(meshlet, &lod_vertices);
                let (aabb_min, aabb_max) = meshlet.vertices.iter().fold(
                    (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                    |(aabb_min, aabb_max), &i| {
                        let position = lod_vertices[i as usize].position / scale;
                        (aabb_min.min(position), aabb_max.max(position))
                    },
                );
                Meshlet {
                    // Vertex positions are quantized in normalized mesh space, so bounds need
                    // to use the same normalization to stay consistent in shaders.
                    center: (Vec3::from_array(bounds.center) / scale).to_array(),
                    radius: bounds.radius / scale,
                    aabb_min: aabb_min.to_array(),
                    aabb_max: aabb_max.to_array(),
                    cone_apex: (Vec3::from_array(bounds.cone_apex) / scale).to_array(),
                    cone_axis: bounds.cone_axis,
                    cone_cutoff: bounds.cone_cutoff,
                    indices: meshlet.triangles.to_owned().into_boxed_slice(),
                    positions: meshlet
                        .vertices
                        .iter()
                        .map(|&i| (lod_vertices[i as usize].position / scale * 32767.).to_array().map(|e| e as i16))
                        .collect(),
                    normals: meshlet
                        .vertices
                        .iter()
                        .map(|&i| (lod_vertices[i as usize].normal * 127.).to_array().map(|e| e as i8))
                        .collect(),
                    _texcoords: meshlet
                        .vertices
                        .iter()
                        .map(|&i| (lod_vertices[i as usize].uv * 32767.).to_array().map(|e| e as i16))
                        .collect(),
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });

    Some(Mesh { scale, radius, lod_count: lod_count as u8, lods })
}
