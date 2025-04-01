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
use vk_mem::Alloc;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

use crate::staging::StagingBuffer;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Global {
    proj: Mat4,
    view: Mat4,
    vertex_buffer: vk::DeviceAddress,
}

#[derive(Clone)]
#[repr(C)]
struct Object {
    model: Mat4,
    texture_id: u32,
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
struct Vertex {
    position: Vec3,
    u: f32,
    normal: Vec3,
    v: f32,
    color: Vec4,
}

/*
struct App {
    // Window stuff.
    viewport_w: u32,
    viewport_h: u32,
    event_loop: EventLoop<()>,
    window: Window,

    // Render engine.
    engine: Renderer,

    // Misc.
    vertex_buffer: u16,
    matrix_buffer: u16,
    viking_texture: u16,
}
    */

#[repr(C)]
struct Meshlet {
    indices: Box<[u8]>,
    positions: Box<[f32]>,
    texcoords: Box<[f32]>,
}

fn main() {
    // File IO.
    let (viking_room_tex, viking_room_tex_w, viking_room_tex_h) = {
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
            .map(|meshlet| Meshlet {
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

    let renderer = Renderer::new(viewport_w, viewport_h, &window);

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

        let vert_shader = create_shader_module(include_bytes!("shader.vert.spirv"));
        let frag_shader = create_shader_module(include_bytes!("shader.frag.spirv"));

        let (pipeline, pipeline_layout) = {
            let pipeline_layout = renderer
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&[global_set_layout]),
                    None,
                )
                .unwrap();

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

            (pipeline, pipeline_layout)
        };

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

        let global_set = renderer
            .device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[global_set_layout]),
            )
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

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
            .unwrap()
            .into_boxed_slice();

        let graphics_command_buffers = [command_buffers[0], command_buffers[1], command_buffers[2]];
        let staging_command_buffer = command_buffers[1];
        drop(command_buffers);

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

        let (viking_room_image, mut viking_room_alloc) = renderer
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
            .unwrap();

        // TODO: delete
        let mut staging_buffer = StagingBuffer::new(10000000, &renderer.allocator);

        let index_count = viking_room_meshlets
            .iter()
            .fold(0, |acc, meshlet| acc + meshlet.indices.len());
        let index_buffer = renderer
            .allocator
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size((index_count * size_of::<u32>()) as u64)
                    .usage(vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo::default(),
            )
            .unwrap();

        let object_buffer = renderer
            .allocator
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size((1 * size_of::<Object>()) as u64)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo::default(),
            )
            .unwrap();

        let indirect_cmd_buffer = renderer
            .allocator
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(
                        (viking_room_meshlets.len() * size_of::<vk::DrawIndexedIndirectCommand>())
                            as u64,
                    )
                    .usage(
                        vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo::default(),
            )
            .unwrap();

        let vertex_count = viking_room_meshlets
            .iter()
            .fold(0, |acc, meshlet| acc + meshlet.positions.len() / 3);
        let vertex_buffer = renderer
            .allocator
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size((vertex_count * size_of::<Vertex>()) as u64)
                    .usage(
                        vk::BufferUsageFlags::VERTEX_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo {
                    ..Default::default()
                },
            )
            .unwrap();

        let global_buffer = renderer
            .allocator
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size_of::<Global>() as u64)
                    .usage(
                        vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo::default(),
            )
            .unwrap();

        // Upload vertex buffer data.
        {
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

            // Create a single gigabuffer.
            let mut vertices = vec![];
            let mut indices = vec![];
            let mut indirect_cmds = vec![];
            for meshlet in &viking_room_meshlets {
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
                    u: meshlet.texcoords[2 * i],
                    normal: Vec3::splat(0.0),
                    /*normals: Vec3::new(
                        meshlet.normals[3 * i],
                        meshlet.normals[3 * i + 1],
                        meshlet.normals[3 * i + 2],
                    )*/
                    v: meshlet.texcoords[2 * i + 1],
                    color: Vec4::splat(1.0),
                }));

                indirect_cmds.push(vk::DrawIndexedIndirectCommand {
                    index_count: meshlet.indices.len() as u32,
                    instance_count: 1,
                    first_index,
                    vertex_offset: 0,
                    first_instance: 0,
                });
            }

            staging_buffer
                .begin_transfer(&renderer.device, staging_command_buffer)
                .stage_buffer(index_buffer.0, 0, indices)
                .stage_buffer(vertex_buffer.0, 0, vertices)
                .stage_buffer(indirect_cmd_buffer.0, 0, indirect_cmds)
                .stage_image(
                    viking_room_image,
                    viking_room_tex_w,
                    viking_room_tex_h,
                    viking_room_tex
                        .iter()
                        .tuples()
                        .flat_map(|(&x, &y, &z)| [x, y, z, 255]),
                )
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
            let command_buffer = graphics_command_buffers[frame];
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
            let image = renderer.swapchain_images[image_index as usize];
            let color_view = renderer.swapchain_color_views[image_index as usize];
            let depth_view = renderer.swapchain_depth_views[image_index as usize];

            // Reset and record.
            renderer
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
            renderer
                .device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();

            let model0 = Mat4::from_translation(Vec3::new(0., 0.5, 0.));
            let model1 = Mat4::from_translation(Vec3::new(1., 0.5, 0.));
            let model2 = Mat4::from_translation(Vec3::new(2., 0.5, 0.));
            let mult = Mat4::from_scale(Vec3::splat(0.5))
                * Mat4::from_rotation_y(time * std::f32::consts::FRAC_PI_2)
                * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);
            let objects = [Object {
                model: model0 * mult,
                texture_id: 0,
            }];

            // Upload global descriptor data & object data.
            staging_buffer
                .begin_transfer(&renderer.device, command_buffer)
                .stage_buffer(
                    global_buffer.0,
                    0,
                    std::iter::once_with(|| Global {
                        proj: Mat4::perspective_rh_gl(
                            std::f32::consts::FRAC_PI_4,
                            viewport_w as f32 / viewport_h as f32,
                            0.01,
                            10.0,
                        ),
                        view: Mat4::from_rotation_translation(
                            Quat::from_euler(
                                EulerRot::XYZ,
                                -std::f32::consts::FRAC_PI_8,
                                cam_hr,
                                0.,
                            ),
                            Vec3::new(0., 0., 0.),
                        ) * Mat4::from_translation(Vec3::new(cam_x, cam_y, cam_z)),
                        vertex_buffer: renderer.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.0),
                        ),
                    }),
                )
                .stage_buffer(object_buffer.0, 0, objects)
                .finish();

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

                renderer.device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(global_set)
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .buffer_info(&[vk::DescriptorBufferInfo::default()
                                .buffer(global_buffer.0)
                                .offset(0)
                                .range(vk::WHOLE_SIZE)]),
                        vk::WriteDescriptorSet::default()
                            .dst_set(global_set)
                            .dst_binding(1)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1)
                            .image_info(&[vk::DescriptorImageInfo::default()
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(viking_room_view)
                                .sampler(viking_room_sampler)]),
                        vk::WriteDescriptorSet::default()
                            .dst_set(global_set)
                            .dst_binding(2)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .buffer_info(&[vk::DescriptorBufferInfo::default()
                                .buffer(object_buffer.0)
                                .offset(0)
                                .range(vk::WHOLE_SIZE)]),
                    ],
                    &[],
                );

                renderer.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline,
                );
                renderer.device.cmd_bind_index_buffer(
                    command_buffer,
                    index_buffer.0,
                    0,
                    vk::IndexType::UINT32,
                );
                renderer.device.cmd_draw_indexed_indirect(
                    command_buffer,
                    indirect_cmd_buffer.0,
                    0,
                    viking_room_meshlets.len() as u32,
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

            renderer.device.end_command_buffer(command_buffer).unwrap();

            // Execute command buffer.
            let waits = [image_available];
            let signals = [render_finished];
            let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&waits)
                .signal_semaphores(&signals)
                .wait_dst_stage_mask(&stages)
                .command_buffers(&command_buffers);
            renderer
                .device
                .queue_submit(renderer.graphics_queue, &[submit_info], frame_in_flight)
                .unwrap();

            //
            let waits = [render_finished];
            let swapchains = [renderer.swapchain];
            let images = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&waits)
                .swapchains(&swapchains)
                .image_indices(&images);
            renderer
                .swapchain_device
                .queue_present(renderer.present_queue, &present_info)
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

        // Clean up.
        /*renderer.device.destroy_sampler(viking_room_sampler, None);
        renderer.device.destroy_image_view(viking_room_view, None);
        renderer.allocator.destroy_image(viking_room_image, &mut viking_room_alloc);
        renderer.allocator.destroy_buffer(matrix_buffer, &mut matrix_alloc);
        renderer.allocator.destroy_buffer(position_buffer, &mut position_alloc);
        renderer.allocator.destroy_buffer(uv_buffer, &mut uv_alloc);
        renderer.allocator.destroy_buffer(index_buffer, &mut index_alloc);
        staging_buffer.destroy(&renderer.allocator);
        renderer.allocator.destroy_image(depth_image, &mut depth_alloc);
        drop(renderer.allocator);
        for i in 0..3 {
            device.destroy_fence(frame_in_flight[i], None);
            device.destroy_semaphore(render_finished[i], None);
            device.destroy_semaphore(image_available[i], None); // bleh
            device.destroy_image_view(swapchain_depth_views[i], None);
            device.destroy_image_view(swapchain_color_views[i], None);
        }
        device.destroy_command_pool(command_pool, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_descriptor_set_layout(global_set_layout, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_shader_module(vert_shader, None);
        device.destroy_shader_module(frag_shader, None);
        swapchain_device.destroy_swapchain(swapchain, None);
        device.destroy_device(None);
        surface_instance.destroy_surface(surface, None);
        instance.destroy_instance(None);*/
    }
}
