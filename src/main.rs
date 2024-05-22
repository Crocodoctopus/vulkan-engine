extern crate vulkano;
extern crate winit;

use vulkano::device::{Device, DeviceCreateInfo, Features, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::{Version, VulkanLibrary};
use vulkano::swapchain::{Surface, SurfaceInfo, ColorSpace, Swapchain, SwapchainCreateInfo};
use vulkano::format::Format;
use vulkano::image::ImageUsage;
use winit::event_loop::EventLoop;
use winit::window::Window;
use std::sync::Arc;

fn main() {
    // Create window.
    //use crate::winit::raw_wind::{HasWindowHandle, HasDisplayHandle};
    let event_loop = EventLoop::new().expect("Could not create window event loop.");
    let window = event_loop
        .create_window(Window::default_attributes())
        .expect("Could not create window.");
    let window_handle = Arc::new(window);

    // Create base Vulkan library.
    let vulkano = VulkanLibrary::new().unwrap();

    // List of required Vulkan extensions.
    let extensions = InstanceExtensions {
        khr_surface: true,
        khr_xlib_surface: true,
        ..InstanceExtensions::empty()
    };

    // Create Vulkan instance.
    let instance = Instance::new(
        vulkano,
        InstanceCreateInfo {
            enabled_extensions: extensions,
            ..Default::default()
        },
    )
    .expect("Unable to create Vulkan instance.");

    // Create surface.
    let surface = Surface::from_window(instance.clone(), window_handle).expect("Could not create Vulkan surface.");

    // List of required Device features.
    let features = Features {
        dynamic_rendering: true,
        ..Features::empty()
    };

    // Enumerate physical devices, select one.
    let pdevice = instance
        .enumerate_physical_devices()
        .unwrap()
        .next()
        .expect("Unable to find a physical device.");
    println!(
        "Physical device found: {:?}",
        pdevice.properties().device_name
    );

    //
    let surface_formats = pdevice.surface_formats(&surface, SurfaceInfo::default()).unwrap();
    println!("{:?}", surface_formats);

    // Get queues from device.
    let mut graphics_queue = None;
    //let mut present_queue = None;
    for (index, queue_property) in pdevice.queue_family_properties().iter().enumerate() {
        if queue_property.queue_flags.contains(QueueFlags::GRAPHICS) {
            graphics_queue = Some(index);
        }
    }
    let graphics_queue = graphics_queue.expect("No graphics queue found.");
    //assert!(present_queue.is_some());
    
    // List of required Device extensions.
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    // Create logical device from physical device.
    let (device, mut queues) = Device::new(
        pdevice,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index: graphics_queue as u32,
                ..Default::default()
            }],
            enabled_features: features,
            enabled_extensions: device_extensions,
            ..Default::default()
        },
    )
    .expect("Unable to create a logical device.");

    // Extract queues.
    let graphics_queue = queues.next();
    // let present_queue = queues.next();

    println!("{:?}", graphics_queue);

    // Swapchain?
    let (swapchain, images) = Swapchain::new(device, surface, SwapchainCreateInfo {
        min_image_count: 3,
        image_format: Format::B8G8R8A8_UNORM,
        image_usage: ImageUsage::COLOR_ATTACHMENT,
        image_color_space: ColorSpace::SrgbNonLinear,
        image_extent: [640, 480],
        ..Default::default()
    }).expect("Could not create Vulkan swapchain.");
}
