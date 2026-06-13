use glam::*;
use itertools::Itertools;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
pub(crate) struct Meshlet {
    pub center: [f32; 3],
    pub radius: f32,
    pub cone_apex: [f32; 3],
    pub cone_axis: [f32; 3],
    pub cone_cutoff: f32,
    pub indices: Box<[u8]>,
    pub positions: Box<[[i16; 3]]>,
    pub normals: Box<[[i8; 3]]>,
    pub _texcoords: Box<[[i16; 2]]>,
}

pub(crate) fn load_mesh(filename: impl AsRef<Path>) -> Option<(f32, Box<[Meshlet]>)> {
    let model = {
        use std::io::BufReader;
        let data = std::fs::read(filename.as_ref()).ok()?;
        let (models, _) =
            tobj::load_obj_buf(&mut BufReader::new(&data[..]), |_| Ok((Vec::new(), HashMap::new()))).unwrap();
        models.into_iter().next()?.mesh
    };
    println!("Model details ({:?}):", filename.as_ref());
    println!("  Indices: {}", model.indices.len());
    println!("  Positions: {}", model.positions.len());
    println!("  Normals: {}", model.normals.len());

    // Calculate bounds.
    let scale =
        model.positions.iter().tuples().fold(0f32, |scale, (x, y, z)| scale.max(x.abs()).max(y.abs()).max(z.abs()));

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

    let mut indices = model.indices;
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

    let mut vertices: Box<[Vertex]> = (0..model.positions.len() / 3)
        .map(|i| Vertex {
            position: positions[i],
            normal: normals[i],
            uv: uvs.get(i).cloned().unwrap_or_default(),
            _color: Vec3::splat(1.0),
        })
        .collect();

    // Optimize index count.
    meshopt::optimize_vertex_cache_in_place(&mut indices, vertices.len());

    // Optimize overdraw.
    meshopt::optimize_overdraw_in_place_decoder(&mut indices, &vertices, 1.05);

    // Optimize vertex fetch.
    meshopt::optimize_vertex_fetch_in_place(&mut indices, &mut vertices);

    let adapter = meshopt::VertexDataAdapter {
        reader: std::io::Cursor::new(unsafe {
            std::slice::from_raw_parts(vertices.as_ptr() as *const u8, size_of::<Vertex>() * vertices.len())
        }),
        vertex_count: vertices.len(),
        vertex_stride: size_of::<Vertex>(),
        position_offset: 0,
    };

    let meshlets = meshopt::build_meshlets(&indices, &adapter, 64, 124, 0.5)
        .iter()
        .map(|meshlet| {
            let bounds = meshopt::compute_meshlet_bounds_decoder(meshlet, &vertices);
            Meshlet {
                // Vertex positions are quantized in normalized mesh space, so bounds need
                // to use the same normalization to stay consistent in shaders.
                center: (Vec3::from_array(bounds.center) / scale).to_array(),
                radius: bounds.radius / scale,
                cone_apex: (Vec3::from_array(bounds.cone_apex) / scale).to_array(),
                cone_axis: bounds.cone_axis,
                cone_cutoff: bounds.cone_cutoff,
                indices: meshlet.triangles.to_owned().into_boxed_slice(),
                positions: meshlet
                    .vertices
                    .iter()
                    .map(|&i| (vertices[i as usize].position / scale * 32767.).to_array().map(|e| e as i16))
                    .collect(),
                normals: meshlet
                    .vertices
                    .iter()
                    .map(|&i| (vertices[i as usize].normal * 127.).to_array().map(|e| e as i8))
                    .collect(),
                _texcoords: meshlet
                    .vertices
                    .iter()
                    .map(|&i| (vertices[i as usize].uv * 32767.).to_array().map(|e| e as i16))
                    .collect(),
            }
        })
        .collect();

    Some((scale, meshlets))
}
