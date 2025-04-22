extern crate ash;
extern crate ash_window;
extern crate glam;
extern crate itertools;
extern crate png;
extern crate tobj;
extern crate vk_mem;
extern crate winit;

mod renderer;
mod staging;
//mod util;

use crate::renderer::*;
use ash::vk;
use glam::*;
use itertools::Itertools;
use std::f32::consts::FRAC_PI_2;
use std::io::BufReader;
use std::mem::size_of;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

use crate::staging::StagingBuffer;

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
struct MeshletRenderGlobal {
    pv: Mat4,
    vertex_buffer: vk::DeviceAddress,
    instance_buffer: vk::DeviceAddress,
    object_buffer: vk::DeviceAddress,
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
struct MeshletCullGlobal {
    view: [f32; 4 * 4],
    camera_position: [f32; 3],
    instances: u32,

    frustum: [f32; 4],

    draw_count_buffer: vk::DeviceAddress,
    meshlet_buffer: vk::DeviceAddress,
    draw_cmd_buffer: vk::DeviceAddress,
    instance_buffer: vk::DeviceAddress,
    object_buffer: vk::DeviceAddress,
}

#[derive(Clone)]
#[repr(C, align(16))]
struct Object {
    model: Mat4,
    texture_id: u32,
}

#[derive(Clone)]
#[repr(C, align(4))]
struct Instance {
    object_id: u32,
}

#[derive(Clone)]
#[repr(C, align(16))]
struct MeshletData {
    // Culling.
    center: [f32; 3],
    radius: f32,
    cone_apex: [f32; 3],
    pad0: f32,
    cone_axis: [f32; 3],
    cone_cutoff: f32,

    // Draw cmd.
    object_id: u32,
    index_count: u32,
    first_index: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Debug, Default)]
struct Vertex {
    position: Vec3,
    u: f32,
    normal: Vec3,
    v: f32,
    color: Vec4,
}

struct Meshlet {
    center: [f32; 3],
    radius: f32,
    cone_apex: [f32; 3],
    cone_axis: [f32; 3],
    cone_cutoff: f32,
    indices: Box<[u8]>,
    positions: Box<[f32]>,
    texcoords: Box<[f32]>,
}

fn main() {
    // File IO.
    /*let (viking_room_tex, viking_room_tex_w, viking_room_tex_h) = {
        let data = include_bytes!("../resources/textures/viking_room.png"); //fs::read("../resources/textures/viking_room.png").unwrap();
        let decoder = png::Decoder::new(&data[..]);
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!(info.buffer_size(), buf.len());
        (buf, info.width, info.height)
    };

    let viking_room_meshlets: Box<[Meshlet]> = {
        let viking_room_model = {
            let data = include_bytes!("../resources/models/viking_room.obj");
            let (viking_room_models, _) =
                tobj::load_obj_buf(&mut BufReader::new(&data[..]), |_| unreachable!()).unwrap();
            viking_room_models.into_iter().next().unwrap().mesh
        };

        let adapter = meshopt::VertexDataAdapter {
            reader: std::io::Cursor::new(unsafe {
                std::slice::from_raw_parts(
                    viking_room_model.positions.as_ptr() as *const u8,
                    viking_room_model.positions.len() * size_of::<f32>(),
                )
            }),
            vertex_count: viking_room_model.positions.len() / 3,
            vertex_stride: 3 * size_of::<f32>(),
            position_offset: 0,
        };

        meshopt::build_meshlets(&viking_room_model.indices, &adapter, 64, 124, 0.25)
            .iter()
            .map(|meshlet| {
                let bounds = meshopt::compute_meshlet_bounds(meshlet, &adapter);
                Meshlet {
                    cone_apex: Vec3::from_array(bounds.cone_apex),
                    cone_axis: Vec3::from_array(bounds.cone_axis),
                    cone_cutoff: bounds.cone_cutoff,
                    indices: meshlet.triangles.to_owned().into_boxed_slice(),
                    positions: meshlet
                        .vertices
                        .iter()
                        .flat_map(|&i: &u32| &viking_room_model.positions[3 * i as usize..][0..3])
                        .copied()
                        .collect(),
                    texcoords: meshlet
                        .vertices
                        .iter()
                        .flat_map(|&i| &viking_room_model.texcoords[2 * i as usize..][0..2])
                        .copied()
                        .collect(),
                }
            })
            .collect()
    };*/

    let bunny_meshlets: Box<[Meshlet]> = {
        let model = {
            let data = include_bytes!("../resources/models/bunny.obj");
            let (models, _) =
                tobj::load_obj_buf(&mut BufReader::new(&data[..]), |_| unreachable!()).unwrap();
            models.into_iter().next().unwrap().mesh
        };

        let mut indices: Box<[u32]> = model.indices.into_boxed_slice();
        let mut positions: Box<[[f32; 3]]> = model
            .positions
            .chunks_exact(3)
            .map(|v| [v[0], v[1], v[2]])
            .collect();

        // Optimize index count.
        meshopt::optimize_vertex_cache_in_place(&mut indices, positions.len());

        // Optimize overdraw.
        meshopt::optimize_overdraw_in_place_decoder(&mut indices, &positions, 1.05);

        // Optimize vertex fetch.
        meshopt::optimize_vertex_fetch_in_place(&mut indices, &mut positions);

        let adapter = meshopt::VertexDataAdapter {
            reader: std::io::Cursor::new(unsafe {
                std::slice::from_raw_parts(
                    positions.as_ptr() as *const u8,
                    3 * positions.len() * size_of::<f32>(),
                )
            }),
            vertex_count: positions.len(),
            vertex_stride: 3 * size_of::<f32>(),
            position_offset: 0,
        };

        meshopt::build_meshlets(&indices, &adapter, 64, 124, 0.5)
            .iter()
            .map(|meshlet| {
                let bounds = meshopt::compute_meshlet_bounds_decoder(meshlet, &positions);
                println!("{:?} vs {:?}", bounds.center, bounds.cone_apex);
                Meshlet {
                    center: bounds.cone_apex,
                    radius: bounds.radius,
                    cone_apex: bounds.cone_apex,
                    cone_axis: bounds.cone_axis,
                    cone_cutoff: bounds.cone_cutoff,
                    indices: meshlet.triangles.to_owned().into_boxed_slice(),
                    positions: meshlet
                        .vertices
                        .iter()
                        .flat_map(|&i: &u32| &positions[i as usize])
                        .copied()
                        .collect(),
                    texcoords: Box::new([]),
                }
            })
            .collect()
    };

    // Create window.
    let (viewport_w, viewport_h) = (1080_u32, 720_u32);
    let mut event_loop = EventLoop::new().expect("Could not create window event loop.");
    #[allow(deprecated)]
    let window = event_loop
        .create_window(
            Window::default_attributes()
                .with_resizable(false)
                .with_inner_size(PhysicalSize::new(viewport_w, viewport_h)),
        )
        .expect("Could not create window.");

    let mut renderer = Renderer::new(viewport_w, viewport_h, &window);

    unsafe {
        let create_shader_module = |src: &[u8]| {
            let shader_module_cinfo = vk::ShaderModuleCreateInfo {
                p_code: src.as_ptr() as _,
                code_size: src.len(),
                ..Default::default()
            };
            renderer
                .device
                .create_shader_module(&shader_module_cinfo, None)
                .unwrap()
        };

        // Global descriptor set.
        let global_set_layout = renderer
            .device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .push_next(
                        &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                            .binding_flags(&[
                                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                            ]),
                    )
                    .bindings(&[
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::ALL),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(1)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1024)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(2)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::VERTEX),
                    ])
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL),
                None,
            )
            .unwrap();

        let (pipeline, pipeline_layout) = {
            let pipeline_layout = renderer
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&[global_set_layout]),
                    None,
                )
                .unwrap();

            let vert_shader = create_shader_module(include_bytes!("shader.vert.spirv"));
            let frag_shader = create_shader_module(include_bytes!("shader.frag.spirv"));

            let pipeline = renderer
                .device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .push_next(
                            &mut vk::PipelineRenderingCreateInfo::default()
                                .color_attachment_formats(&[renderer.surface_format.format])
                                .depth_attachment_format(vk::Format::D32_SFLOAT),
                        )
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .module(vert_shader)
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .name(c"main"),
                            vk::PipelineShaderStageCreateInfo::default()
                                .module(frag_shader)
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .name(c"main"),
                        ])
                        .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                                .primitive_restart_enable(false),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewports(&[vk::Viewport {
                                    x: 0.,
                                    y: 0.,
                                    width: viewport_w as f32,
                                    height: viewport_h as f32,
                                    min_depth: 0.0,
                                    max_depth: 1.0,
                                }])
                                .scissors(&[vk::Rect2D {
                                    offset: vk::Offset2D { x: 0, y: 0 },
                                    extent: vk::Extent2D {
                                        width: viewport_w,
                                        height: viewport_h,
                                    },
                                }]),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .depth_clamp_enable(false)
                                .rasterizer_discard_enable(false)
                                .polygon_mode(vk::PolygonMode::FILL)
                                .line_width(1.0)
                                .cull_mode(vk::CullModeFlags::BACK)
                                .front_face(vk::FrontFace::CLOCKWISE)
                                .depth_bias_enable(false),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .sample_shading_enable(false)
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default()
                                .logic_op_enable(false)
                                .attachments(&[vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                                    .blend_enable(false)]),
                        )
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::default()
                                .depth_test_enable(true)
                                .depth_write_enable(true)
                                .depth_compare_op(vk::CompareOp::LESS),
                        )
                        .layout(pipeline_layout)],
                    None,
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap();

            renderer.device.destroy_shader_module(vert_shader, None);
            renderer.device.destroy_shader_module(frag_shader, None);

            (pipeline, pipeline_layout)
        };

        //
        let comp_set_layout = renderer
            .device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .push_next(
                        &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                            .binding_flags(&[vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND]),
                    )
                    .bindings(&[vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)])
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL),
                None,
            )
            .unwrap();

        let (comp_pipeline, comp_pipeline_layout) = {
            let comp_shader = create_shader_module(include_bytes!("shader.comp.spirv"));

            let pipeline_layout = renderer
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&[comp_set_layout]),
                    None,
                )
                .unwrap();

            let pipeline = renderer
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::default(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .layout(pipeline_layout)
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .name(c"main")
                                .module(comp_shader),
                        )],
                    None,
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap();

            renderer.device.destroy_shader_module(comp_shader, None);

            (pipeline, pipeline_layout)
        };

        let command_pool = renderer
            .device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(renderer.queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .unwrap();

        let command_buffers = renderer
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(4),
            )
            .unwrap();

        let secondary_cmd_buffers = renderer
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::SECONDARY)
                    .command_buffer_count(3),
            )
            .unwrap();

        let graphics_command_buffers = [command_buffers[0], command_buffers[1], command_buffers[2]];
        let staging_command_buffer = command_buffers[3];
        let scene_command_buffers = [
            secondary_cmd_buffers[0],
            secondary_cmd_buffers[1],
            secondary_cmd_buffers[2],
        ];
        drop(command_buffers);
        drop(secondary_cmd_buffers);

        // Synchronization primitives for each frame.
        let image_available: Box<[vk::Semaphore]> = (0..3)
            .map(|_| {
                renderer
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            })
            .collect::<Result<_, _>>()
            .unwrap();
        let render_finished: Box<[vk::Semaphore]> = (0..3)
            .map(|_| {
                renderer
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            })
            .collect::<Result<_, _>>()
            .unwrap();
        let frame_in_flight: Box<[vk::Fence]> = (0..3)
            .map(|_| {
                renderer.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            })
            .collect::<Result<_, _>>()
            .unwrap();

        /*let (viking_room_image, mut viking_room_alloc) = renderer
        .allocator
        .create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D {
                    width: viking_room_tex_w,
                    height: viking_room_tex_h,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .format(vk::Format::R8G8B8A8_UNORM)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .samples(vk::SampleCountFlags::TYPE_1),
            &vk_mem::AllocationCreateInfo {
                ..Default::default()
            },
        )
        .unwrap();

        let viking_room_view = renderer
            .device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(viking_room_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    ),
                None,
            )
            .unwrap();

        let viking_room_sampler = renderer
        .device
        .create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .unnormalized_coordinates(false)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .mip_lod_bias(0.0)
                .min_lod(0.0)
                .max_lod(0.0),
            None,
        )
        .unwrap();*/

        // TODO: delete
        let mut staging_buffer = StagingBuffer::new(10000000, &renderer.allocator);

        let index_count = bunny_meshlets
            .iter()
            .fold(0, |acc, meshlet| acc + meshlet.indices.len());
        let index_buffer: Buffer<u32> = renderer.create_buffer(
            index_count as u32,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let object_buffer: Buffer<Object> = renderer.create_buffer(
            3,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let instance_buffer: Buffer<Instance> = renderer.create_buffer(
            3 * bunny_meshlets.len() as u32,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let meshlet_data_buffer: Buffer<MeshletData> = renderer.create_buffer(
            3 * bunny_meshlets.len() as u32,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let meshlet_cull_global_buffer: Buffer<MeshletCullGlobal> = renderer.create_buffer(
            1,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let indirect_cmd_buffer: Buffer<vk::DrawIndexedIndirectCommand> = renderer.create_buffer(
            3 * bunny_meshlets.len() as u32,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let comp_buffer: Buffer<u32> = renderer.create_buffer(
            1,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let vertex_count = bunny_meshlets
            .iter()
            .fold(0, |acc, meshlet| acc + meshlet.positions.len() / 3);
        let vertex_buffer: Buffer<Vertex> = renderer.create_buffer(
            vertex_count as u32,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let global_buffer: Buffer<MeshletRenderGlobal> = renderer.create_buffer(
            1,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        // Upload vertex buffer data.
        let instances: u32 = {
            // Create a single gigabuffer.
            let mut vertices = vec![];
            let mut indices = vec![];
            let mut meshlet_data = vec![];
            for meshlet in &bunny_meshlets {
                let first_index = indices.len() as u32;
                let index_offset = vertices.len() as u32;

                indices.extend(
                    meshlet
                        .indices
                        .iter()
                        .map(|&index| index as u32 + index_offset),
                );

                vertices.extend((0..meshlet.positions.len() / 3).map(|i| Vertex {
                    position: Vec3::new(
                        meshlet.positions[3 * i],
                        meshlet.positions[3 * i + 1],
                        meshlet.positions[3 * i + 2],
                    ),
                    u: 0., //meshlet.texcoords[2 * i],
                    normal: Vec3::splat(0.0),
                    /*normals: Vec3::new(
                        meshlet.normals[3 * i],
                        meshlet.normals[3 * i + 1],
                        meshlet.normals[3 * i + 2],
                    )*/
                    v: 0., //meshlet.texcoords[2 * i + 1],
                    color: Vec4::splat(1.0),
                }));

                for object_id in 0..1 {
                    meshlet_data.push(MeshletData {
                        center: meshlet.center,
                        radius: meshlet.radius,
                        cone_apex: meshlet.cone_apex,
                        pad0: 0.,
                        cone_axis: meshlet.cone_axis,
                        cone_cutoff: meshlet.cone_cutoff,
                        object_id,
                        index_count: meshlet.indices.len() as u32,
                        first_index,
                    });
                }
            }
            let instances = meshlet_data.len() as u32;

            //
            renderer
                .device
                .reset_command_buffer(staging_command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
            renderer
                .device
                .begin_command_buffer(
                    staging_command_buffer,
                    &vk::CommandBufferBeginInfo::default(),
                )
                .unwrap();

            staging_buffer
                .begin_transfer(&renderer.device, staging_command_buffer)
                .stage_buffer(index_buffer.vk_handle(), 0, indices)
                .stage_buffer(vertex_buffer.vk_handle(), 0, vertices)
                .stage_buffer(meshlet_data_buffer.vk_handle(), 0, meshlet_data)
                /*.stage_image(
                    viking_room_image,
                    viking_room_tex_w,
                    viking_room_tex_h,
                    viking_room_tex
                        .iter()
                        .tuples()
                        .flat_map(|(&x, &y, &z)| [x, y, z, 255]),
                )*/
                .finish();

            renderer
                .device
                .end_command_buffer(staging_command_buffer)
                .unwrap();

            // Create wait fence.
            let wait = renderer
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();

            // Submit.
            renderer
                .device
                .queue_submit(
                    renderer.graphics_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[staging_command_buffer])],
                    wait,
                )
                .unwrap();

            // Wait.
            renderer
                .device
                .wait_for_fences(&[wait], true, u64::MAX)
                .unwrap();
            renderer.device.destroy_fence(wait, None);

            instances
        };

        //
        let descriptor_pool = renderer
            .device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&[vk::DescriptorPoolSize::default().descriptor_count(3)])
                    .max_sets(3)
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                None,
            )
            .unwrap();

        let (global_set, comp_set) = {
            let sets = renderer
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[global_set_layout, comp_set_layout]),
                )
                .unwrap();
            (sets[0], sets[1])
        };

        renderer.device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::default()
                    .dst_set(comp_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(meshlet_cull_global_buffer.vk_handle())
                        .offset(0)
                        .range(vk::WHOLE_SIZE)]),
                //
                vk::WriteDescriptorSet::default()
                    .dst_set(global_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(global_buffer.vk_handle())
                        .offset(0)
                        .range(vk::WHOLE_SIZE)]),
                /*vk::WriteDescriptorSet::default()
                .dst_set(global_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .image_info(&[vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(viking_room_view)
                    .sampler(viking_room_sampler)]),*/
                vk::WriteDescriptorSet::default()
                    .dst_set(global_set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(object_buffer.vk_handle())
                        .offset(0)
                        .range(vk::WHOLE_SIZE)]),
            ],
            &[],
        );

        println!("{}", instances);

        // Pre-record scene command buffers.
        for i in 0..3 {
            let command_buffer = scene_command_buffers[i];
            let image = renderer.swapchain_images[i];
            let color_view = renderer.swapchain_color_views[i];
            let depth_view = renderer.swapchain_depth_views[i];

            // Begin recording.
            renderer
                .device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .inheritance_info(&vk::CommandBufferInheritanceInfo::default()),
                )
                .unwrap();

            // Compute prepass.
            {
                renderer.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    comp_pipeline_layout,
                    0,
                    &[comp_set],
                    &[],
                );

                renderer.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    comp_pipeline,
                );

                renderer
                    .device
                    .cmd_dispatch(command_buffer, instances.div_ceil(64), 1, 1);
            }

            // Convert VK_IMAGE_LAYOUT_UNDEFINED -> VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL.
            renderer.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    vk::ImageMemoryBarrier::default()
                        .image(image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                    vk::ImageMemoryBarrier::default()
                        .image(renderer.depth_image.0)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL),
                ],
            );
            // Begin rendering.
            renderer.device.cmd_begin_rendering(
                command_buffer,
                &vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: viewport_w,
                            height: viewport_h,
                        },
                    })
                    .layer_count(1)
                    .depth_attachment(
                        &vk::RenderingAttachmentInfo::default()
                            .image_view(depth_view)
                            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .clear_value(vk::ClearValue {
                                depth_stencil: vk::ClearDepthStencilValue {
                                    depth: 1.0,
                                    stencil: 0,
                                },
                            }),
                    )
                    .color_attachments(&[vk::RenderingAttachmentInfo::default()
                        .image_view(color_view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        })]),
            );

            // Begin draw calls.
            {
                renderer.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &[global_set],
                    &[],
                );

                renderer.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline,
                );

                renderer.device.cmd_bind_index_buffer(
                    command_buffer,
                    index_buffer.vk_handle(),
                    0,
                    vk::IndexType::UINT32,
                );

                renderer.device.cmd_draw_indexed_indirect_count(
                    command_buffer,
                    indirect_cmd_buffer.vk_handle(),
                    0,
                    comp_buffer.vk_handle(),
                    0,
                    instances,
                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );
            }

            renderer.device.cmd_end_rendering(command_buffer);

            // Convert VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL -> VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.
            renderer.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    vk::ImageMemoryBarrier::default()
                        .image(image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR),
                    vk::ImageMemoryBarrier::default()
                        .image(renderer.depth_image.0)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR),
                ],
            );

            // End recording.
            renderer.device.end_command_buffer(command_buffer).unwrap();
        }

        // "Gameloop"
        //let mut timestamp = 0_u64;
        let mut time = 0_f32;
        let dt = 0.016666_f32;
        // Misc.
        let mut cam_x = 0_f32;
        let cam_y = 0_f32;
        let mut cam_z = 0_f32;
        let mut cam_hr = 0_f32;
        let mut cam_vr = 0_f32;
        let mut w_down = false;
        let mut a_down = false;
        let mut s_down = false;
        let mut d_down = false;
        let mut q_down = false;
        let mut e_down = false;
        for frame in (0..3).cycle() {
            // Input.
            let mut exit = false;
            use winit::platform::pump_events::EventLoopExtPumpEvents;
            #[allow(deprecated)]
            let _status = event_loop.pump_events(Some(std::time::Duration::ZERO), |event, _| {
                match event {
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => exit = true,

                    Event::WindowEvent {
                        event:
                            WindowEvent::KeyboardInput {
                                event:
                                    KeyEvent {
                                        physical_key: PhysicalKey::Code(key),
                                        state,
                                        repeat: false,
                                        ..
                                    },
                                ..
                            },
                        ..
                    } => {
                        // Skip repeats.
                        let var = match key {
                            KeyCode::KeyW => &mut w_down,
                            KeyCode::KeyA => &mut a_down,
                            KeyCode::KeyS => &mut s_down,
                            KeyCode::KeyD => &mut d_down,
                            KeyCode::KeyQ => &mut q_down,
                            KeyCode::KeyE => &mut e_down,
                            _ => return,
                        };

                        match state {
                            ElementState::Pressed => *var = true,
                            ElementState::Released => *var = false,
                        }
                    }

                    // Unhandled.
                    _ => {}
                }
            });

            if exit {
                break;
            }

            // Update.

            // Forward.
            if w_down && !s_down {
                cam_z += dt * cam_hr.cos();
                cam_x -= dt * cam_hr.sin();
            }
            // Backward.
            if !w_down && s_down {
                cam_z -= dt * cam_hr.cos();
                cam_x += dt * cam_hr.sin();
            }

            // Turn left.
            if a_down && !d_down {
                cam_hr -= dt;
            }
            // Turn right.
            if !a_down && d_down {
                cam_hr += dt;
            }

            // Strafe left.
            if q_down && !e_down {
                cam_x += dt * cam_hr.cos();
                cam_z += dt * cam_hr.sin();
            }
            // Strafe right.
            if !q_down && e_down {
                cam_x -= dt * cam_hr.cos();
                cam_z -= dt * cam_hr.sin();
            }

            cam_vr = cam_vr.clamp(-FRAC_PI_2, FRAC_PI_2);

            // Draw.
            let frame_in_flight = frame_in_flight[frame];
            let render_finished = render_finished[frame];
            let image_available = image_available[frame];

            // Wait for next image to become available.
            renderer
                .device
                .wait_for_fences(&[frame_in_flight], true, u64::MAX)
                .unwrap();
            renderer.device.reset_fences(&[frame_in_flight]).unwrap();

            let (image_index, _) = renderer
                .swapchain_device
                .acquire_next_image(
                    renderer.swapchain,
                    u64::MAX,
                    image_available,
                    vk::Fence::null(),
                )
                .unwrap();
            let command_buffer = graphics_command_buffers[image_index as usize];
            let scene_command_buffer = scene_command_buffers[image_index as usize];

            // Reset and record.
            renderer
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
            renderer
                .device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();

            // Transfer some global state data.
            {
                /*let model0 = Mat4::IDENTITY;
                let model1 = model0;
                let model2 = model1;
                let mult = model0;*/
                let model0 = Mat4::from_translation(Vec3::new(0., 0.8, 0.));
                let model1 = Mat4::from_translation(Vec3::new(1., 0.8, 0.));
                let model2 = Mat4::from_translation(Vec3::new(2., 0.8, 0.));
                let mult = Mat4::from_scale(Vec3::splat(2.0))
                    * Mat4::from_rotation_y(time * std::f32::consts::FRAC_PI_2)
                    * Mat4::from_rotation_x(std::f32::consts::PI);
                let object_data = [
                    Object {
                        model: model0 * mult,
                        texture_id: 0,
                    },
                    Object {
                        model: model1 * mult,
                        texture_id: 0,
                    },
                    Object {
                        model: model2 * mult,
                        texture_id: 0,
                    },
                ];

                // Upload global descriptor data & object data.
                let projection = Mat4::perspective_infinite_rh(
                    std::f32::consts::FRAC_PI_4,
                    viewport_w as f32 / viewport_h as f32,
                    0.1,
                    //2.0,
                );
                let view = Mat4::from_rotation_translation(
                    Quat::from_euler(EulerRot::XYZ, -std::f32::consts::FRAC_PI_8, cam_hr, 0.),
                    Vec3::new(0., 0., 0.),
                ) * Mat4::from_translation(Vec3::new(cam_x, cam_y, cam_z));

                // Frustum plane data.
                let normalize_plane = |p: Vec4| p / p.xyz().length();
                let temp = projection.transpose();
                let frustum_x = normalize_plane(temp.w_axis + temp.x_axis);
                let frustum_y = normalize_plane(temp.w_axis + temp.y_axis);
                let frustum = [frustum_x.x, frustum_x.z, frustum_y.y, frustum_y.z];

                staging_buffer
                    .begin_transfer(&renderer.device, command_buffer)
                    .stage_buffer(object_buffer.vk_handle(), 0, object_data)
                    .stage_buffer(
                        global_buffer.vk_handle(),
                        0,
                        [MeshletRenderGlobal {
                            pv: projection * view,
                            vertex_buffer: renderer.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(vertex_buffer.vk_handle()),
                            ),
                            instance_buffer: renderer.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(instance_buffer.vk_handle()),
                            ),
                            object_buffer: renderer.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(object_buffer.vk_handle()),
                            ),
                        }],
                    )
                    .stage_buffer(
                        meshlet_cull_global_buffer.vk_handle(),
                        0,
                        [MeshletCullGlobal {
                            view: view.to_cols_array(),
                            camera_position: [-cam_x, cam_y, -cam_z],
                            instances,
                            frustum,
                            draw_count_buffer: renderer.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(comp_buffer.vk_handle()),
                            ),
                            meshlet_buffer: renderer.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(meshlet_data_buffer.vk_handle()),
                            ),
                            draw_cmd_buffer: renderer.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(indirect_cmd_buffer.vk_handle()),
                            ),
                            instance_buffer: renderer.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(instance_buffer.vk_handle()),
                            ),
                            object_buffer: renderer.device.get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(object_buffer.vk_handle()),
                            ),
                        }],
                    )
                    .stage_buffer(comp_buffer.vk_handle(), 0, [0i32])
                    .finish();
            }

            renderer
                .device
                .cmd_execute_commands(command_buffer, &[scene_command_buffer]);

            renderer.device.end_command_buffer(command_buffer).unwrap();

            // Execute command buffer.
            renderer
                .device
                .queue_submit(
                    renderer.graphics_queue,
                    &[vk::SubmitInfo::default()
                        .wait_semaphores(&[image_available])
                        .signal_semaphores(&[render_finished])
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .command_buffers(&[command_buffer])],
                    frame_in_flight,
                )
                .unwrap();

            // Swap backbuffer.
            renderer
                .swapchain_device
                .queue_present(
                    renderer.present_queue,
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&[render_finished])
                        .swapchains(&[renderer.swapchain])
                        .image_indices(&[image_index]),
                )
                .unwrap();

            //timestamp += 16666;
            time += 0.016666 * 0.1;
            //panic!();
        }

        // Block until the gpu is finished before proceeding to clean up.
        renderer
            .device
            .wait_for_fences(&frame_in_flight, true, u64::MAX)
            .unwrap();

        std::process::abort();
    }
}
