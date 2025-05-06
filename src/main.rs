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
use glam::*;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

fn main() {
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

    let cwd = std::env::current_dir().unwrap();
    let mut renderer = Renderer::new(cwd, viewport_w, viewport_h, &window);
    let viking_room = renderer
        .load_mesh("resources/models/viking_room.obj")
        .unwrap();
    let sphere = renderer.load_mesh("resources/models/sphere.obj").unwrap();
    let obj0 = renderer
        .create_object(
            viking_room,
            Vec3::new(0.0, 0.5, 0.0),
            1.0,
            Quat::from_euler(EulerRot::XYZ, std::f32::consts::FRAC_PI_2, 0.0, 0.0),
        )
        .unwrap();
    let obj1 = renderer
        .create_object(
            sphere,
            Vec3::new(0.0, 0.0, 0.0),
            0.05,
            Quat::from_euler(EulerRot::XYZ, 0.0, 0.0, 0.0),
        )
        .unwrap();
    /*let bunny_obj2 = renderer
        .create_object(
            bunny_mesh,
            Vec3::new(1.0, 0.4, 0.0),
            Quat::from_euler(EulerRot::XYZ, std::f32::consts::PI, 0.0, 0.0),
            Vec3::splat(2.0),
        )
        .unwrap();
    let bunny_obj3 = renderer
        .create_object(
            bunny_mesh,
            Vec3::new(2.0, 0.4, 0.0),
            Quat::from_euler(EulerRot::XYZ, std::f32::consts::PI, 0.0, 0.0),
            Vec3::splat(2.0),
        )
        .unwrap();*/

    // "Gameloop"
    //let mut timestamp = 0_u64;
    let mut time = 0_f32;
    let dt = 0.016666_f32;
    // Misc.
    let mut w_down = false;
    let mut a_down = false;
    let mut s_down = false;
    let mut d_down = false;
    let mut q_down = false;
    let mut e_down = false;
    loop {
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
        if s_down && !w_down {
            renderer.cam_pos.z += dt * renderer.cam_rot[0].cos();
            renderer.cam_pos.x -= dt * renderer.cam_rot[0].sin();
        }
        // Backward.
        if !s_down && w_down {
            renderer.cam_pos.z -= dt * renderer.cam_rot[0].cos();
            renderer.cam_pos.x += dt * renderer.cam_rot[0].sin();
        }

        // Turn left.
        if a_down && !d_down {
            renderer.cam_rot[0] -= dt;
        }
        // Turn right.
        if !a_down && d_down {
            renderer.cam_rot[0] += dt;
        }

        // Strafe left.
        if e_down && !q_down {
            renderer.cam_pos.x += dt * renderer.cam_rot[0].cos();
            renderer.cam_pos.z += dt * renderer.cam_rot[0].sin();
        }
        // Strafe right.
        if !e_down && q_down {
            renderer.cam_pos.x -= dt * renderer.cam_rot[0].cos();
            renderer.cam_pos.z -= dt * renderer.cam_rot[0].sin();
        }

        //cam_vr = cam_vr.clamp(-FRAC_PI_2, FRAC_PI_2);

        renderer.render(time);

        //timestamp += 16666;
        time += 0.016666 * 0.1;
        //panic!();
    }

    std::process::abort();
}

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
