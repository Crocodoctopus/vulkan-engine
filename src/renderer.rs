use crate::buffer::Buffer;
use crate::core::VulkanCore;
use crate::glsl_types::*;
use crate::image::{Image, ImageView};
use crate::mesh::{GpuMesh, Mesh, PendingGpuMesh, load_mesh};
use crate::pipelines::Pipelines;
use crate::profiling::{PipelineProfiler, ProfileLabel};
use crate::rw_queue::{ResourceQueue, WaitStrategy};
use crate::staging::{StagingPool, StagingSpan, Whole};
use crate::swapchain::Swapchain;
use crate::util::{format_bytes, format_usize_commas, wait_semaphores_any_fallback};
use crate::vk_helpers::*;
use crate::world::{Object, World, WorldDiff};
use ash::vk;
use crossbeam::queue::SegQueue;
use glam::*;
use std::collections::{BTreeMap, HashMap};
use std::mem::offset_of;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use threadpool::ThreadPool;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[derive(Debug)]
pub struct HandleCounter(u32);

#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Handle(u32);

impl Iterator for HandleCounter {
    type Item = Handle;
    fn next(&mut self) -> Option<Self::Item> {
        self.0 += 1;
        return Some(Handle(self.0));
    }
}

pub(crate) type MeshHandle = Handle;
pub(crate) type ObjectHandle = Handle;

#[allow(unused)]
const B: u64 = 1;
#[allow(non_upper_case_globals, unused)]
const KiB: u64 = 1024 * B;
#[allow(non_upper_case_globals, unused)]
const MiB: u64 = 1024 * KiB;
#[allow(non_upper_case_globals, unused)]
const GiB: u64 = 1024 * MiB;

pub(crate) const STAGING_ARENA_SIZE: u64 = 512 * MiB;
pub(crate) const STAGING_FIF_BLOCK_SIZE: u64 = 32 * MiB;

pub(crate) const MAX_FRAMES_IN_FLIGHT: usize = 2;
pub(crate) const VISIBILITY_DEPTH: usize = 2;
pub(crate) const VISIBILITY_RESOURCE_QUEUE_LEN: usize = VISIBILITY_DEPTH + MAX_FRAMES_IN_FLIGHT;

// Dedicated HZB/occlusion descriptor set.
pub(crate) const MAX_HZB_DIMENSION: u32 = 8192;
pub(crate) const MAX_HZB_MIPS: u32 = MAX_HZB_DIMENSION.div_ceil(2).ilog2() + 1;
pub(crate) const HZB_SAMPLED_IMAGE_CAPACITY: u32 = 1 + MAX_HZB_MIPS;
pub(crate) const HZB_STORAGE_IMAGE_CAPACITY: u32 = MAX_HZB_MIPS;

/*Generate index
Plan:
0) Data upload
1) frustum_cull
2) render (only visible)
3) build_hzb
4) occlusion_cull
5) render (late visible)
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum PipelineStage {
    DataUpload,
    FrustumCull,
    EarlyDraw,
    BuildHzb,
    OcclusionCull,
    LateDraw,
    FrameEnd,
}

impl PipelineStage {
    const COUNT: usize = Self::FrameEnd as usize + 1;

    const fn wait_value(self, base: u64) -> u64 {
        base as u64 as u64 + self as u64
    }

    const fn signal_value(self, base: u64) -> u64 {
        self.wait_value(base) + 1
    }
}

impl From<PipelineStage> for ProfileLabel {
    fn from(stage: PipelineStage) -> Self {
        match stage {
            PipelineStage::DataUpload => "data_upload",
            PipelineStage::FrustumCull => "frustum_cull",
            PipelineStage::EarlyDraw => "early_draw",
            PipelineStage::BuildHzb => "build_hzb",
            PipelineStage::OcclusionCull => "occlusion_cull",
            PipelineStage::LateDraw => "late_draw",
            PipelineStage::FrameEnd => "frame_end",
        }
        .into()
    }
}

struct SwapchainState {
    // HZB is per-frame scratch, but the ring only needs to be large enough to
    // cover the live visibility window.
    hzb_descriptor_pool: vk::DescriptorPool,
    hzb_images: [Image; MAX_FRAMES_IN_FLIGHT],
    hzb_build_src_views: [Box<[ImageView]>; MAX_FRAMES_IN_FLIGHT],
    hzb_build_dst_views: [Box<[ImageView]>; MAX_FRAMES_IN_FLIGHT],
    hzb_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    hzb_sampler: vk::Sampler,
    overdraw_images: [Image; MAX_FRAMES_IN_FLIGHT],
    overdraw_views: [ImageView; MAX_FRAMES_IN_FLIGHT],

    render_finished: Box<[vk::Semaphore]>,
    image_acquired_semaphores: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    depth_images: [Image; MAX_FRAMES_IN_FLIGHT],
    depth_views: [ImageView; MAX_FRAMES_IN_FLIGHT],
}

impl SwapchainState {
    unsafe fn new(
        core: &VulkanCore,
        swapchain: &Swapchain,
        pipelines: &Pipelines,
        overdraw_sets: &[vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    ) -> Self {
        let device = &core.device;
        let allocator = &core.allocator;
        let mut hzb_sampler_reduction =
            vk::SamplerReductionModeCreateInfo::default().reduction_mode(vk::SamplerReductionMode::MIN);
        let hzb_sampler = device
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .push_next(&mut hzb_sampler_reduction)
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                    .unnormalized_coordinates(false),
                None,
            )
            .unwrap();
        let staging_cmd_buffer = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(core.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap()[0];

        device.reset_command_buffer(staging_cmd_buffer, vk::CommandBufferResetFlags::empty()).unwrap();
        device.begin_command_buffer(staging_cmd_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

        let hzb_descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&[
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32 * HZB_SAMPLED_IMAGE_CAPACITY),
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32 * HZB_STORAGE_IMAGE_CAPACITY),
                    ])
                    .max_sets(MAX_FRAMES_IN_FLIGHT as u32)
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                None,
            )
            .unwrap();
        let hzb_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT] = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(hzb_descriptor_pool)
                    .set_layouts(&[pipelines.hzb_set_layout; MAX_FRAMES_IN_FLIGHT]),
            )
            .unwrap()
            .try_into()
            .unwrap();

        let vk::Extent2D { width, height, .. } = swapchain.extent;
        if width > MAX_HZB_DIMENSION || height > MAX_HZB_DIMENSION {
            panic!("HZB/occlusion descriptor set only supports up to {MAX_HZB_DIMENSION}; got {width}x{height}");
        }

        // Round the half-res base up to the next power of two so the mip chain is regular.
        let hzb_width = width.div_ceil(2).max(1).next_power_of_two();
        let hzb_height = height.div_ceil(2).max(1).next_power_of_two();
        let mipmaps = u32::max(hzb_width, hzb_height).ilog2() + 1;

        if mipmaps > MAX_HZB_MIPS {
            panic!("HZB mip chain exceeds reserved descriptor range: {mipmaps} mips > {MAX_HZB_MIPS}");
        }

        let create_image =
            |extent: vk::Extent2D, format: vk::Format, usage: vk::ImageUsageFlags, mip_levels: u32| -> Image {
                let create_info = image2d_create_info()
                    .extent(extent3d_from_extent2d(extent))
                    .format(format)
                    .usage(usage)
                    .mip_levels(mip_levels);
                let (image, alloc) =
                    vk_mem::Alloc::create_image(allocator, &create_info, &device_local_alloc()).unwrap();
                Image { image, alloc }
            };

        let create_view = |image: vk::Image,
                           format: vk::Format,
                           aspect: vk::ImageAspectFlags,
                           base_mip_level: u32,
                           level_count: u32|
         -> ImageView {
            let view = device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: aspect,
                            base_mip_level,
                            level_count,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
                .unwrap();
            ImageView { view }
        };

        let hzb_images = std::array::from_fn(|_| {
            create_image(
                vk::Extent2D { width: hzb_width, height: hzb_height },
                vk::Format::R32_SFLOAT,
                vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
                mipmaps,
            )
        });

        let hzb_build_src_views = std::array::from_fn(|slot| {
            (0..mipmaps)
                .into_iter()
                .map(|level| {
                    create_view(hzb_images[slot].image, vk::Format::R32_SFLOAT, vk::ImageAspectFlags::COLOR, level, 1)
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        });

        let hzb_build_dst_views = std::array::from_fn(|slot| {
            (0..mipmaps)
                .into_iter()
                .map(|level| {
                    create_view(hzb_images[slot].image, vk::Format::R32_SFLOAT, vk::ImageAspectFlags::COLOR, level, 1)
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        });

        let render_finished = (0..swapchain.images.len())
            .into_iter()
            .map(|_| device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap())
            .collect();

        let image_acquired_semaphores =
            std::array::from_fn(|_| device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap());

        let depth_images = std::array::from_fn(|_| {
            create_image(
                swapchain.extent,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                1,
            )
        });

        let depth_views = std::array::from_fn(|i| {
            create_view(depth_images[i].image, vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH, 0, 1)
        });

        let overdraw_images = std::array::from_fn(|_| {
            create_image(
                swapchain.extent,
                vk::Format::R32_UINT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
                1,
            )
        });

        let overdraw_views = std::array::from_fn(|i| {
            create_view(overdraw_images[i].image, vk::Format::R32_UINT, vk::ImageAspectFlags::COLOR, 0, 1)
        });

        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            let hzb_src_infos: Box<_> = hzb_build_src_views[slot]
                .iter()
                .map(|image_view| {
                    vk::DescriptorImageInfo::default()
                        .image_view(image_view.view)
                        .sampler(hzb_sampler)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                })
                .collect();

            let hzb_dst_infos: Box<_> = hzb_build_dst_views[slot]
                .iter()
                .map(|image_view| {
                    vk::DescriptorImageInfo::default()
                        .image_view(image_view.view)
                        .image_layout(vk::ImageLayout::GENERAL)
                })
                .collect();

            let depth_info = [vk::DescriptorImageInfo::default()
                .image_view(depth_views[slot].view)
                .sampler(hzb_sampler)
                .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)];

            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(hzb_sets[slot])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(depth_info.len() as u32)
                        .image_info(&depth_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(hzb_sets[slot])
                        .dst_binding(0)
                        .dst_array_element(1)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(hzb_src_infos.len() as u32)
                        .image_info(&hzb_src_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(hzb_sets[slot])
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(hzb_dst_infos.len() as u32)
                        .image_info(&hzb_dst_infos),
                ],
                &[],
            );
        }

        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            let overdraw_info = [vk::DescriptorImageInfo::default()
                .image_view(overdraw_views[slot].view)
                .image_layout(vk::ImageLayout::GENERAL)];

            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(overdraw_sets[slot])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .image_info(&overdraw_info)],
                &[],
            );
        }

        let mut barriers = Vec::new();
        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .image(hzb_images[slot].image)
                    .subresource_range(COLOR_2D_SUBRESOURCE_RANGE.level_count(vk::REMAINING_MIP_LEVELS))
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL),
            );
        }

        barriers.extend((0..MAX_FRAMES_IN_FLIGHT).into_iter().map(|i| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .image(overdraw_images[i].image)
                .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
        }));

        barriers.extend((0..MAX_FRAMES_IN_FLIGHT).into_iter().map(|i| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                )
                .image(depth_images[i].image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
        }));

        device
            .cmd_pipeline_barrier2(staging_cmd_buffer, &vk::DependencyInfo::default().image_memory_barriers(&barriers));

        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            device.cmd_clear_color_image(
                staging_cmd_buffer,
                hzb_images[slot].image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] },
                &[vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mipmaps)
                    .base_array_layer(0)
                    .layer_count(1)],
            );
        }

        let barriers = Vec::from_iter((0..MAX_FRAMES_IN_FLIGHT).map(|slot| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .image(hzb_images[slot].image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(vk::REMAINING_MIP_LEVELS)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        }));

        device
            .cmd_pipeline_barrier2(staging_cmd_buffer, &vk::DependencyInfo::default().image_memory_barriers(&barriers));

        device.end_command_buffer(staging_cmd_buffer).unwrap();
        device
            .queue_submit2(
                *core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default().command_buffer_infos(&[
                    vk::CommandBufferSubmitInfo::default().command_buffer(staging_cmd_buffer)
                ])],
                vk::Fence::null(),
            )
            .unwrap();
        device.queue_wait_idle(*core.graphics_queue.lock().unwrap()).unwrap();
        device.free_command_buffers(core.cmd_pool, &[staging_cmd_buffer]);

        Self {
            hzb_descriptor_pool,
            hzb_images,
            hzb_build_src_views,
            hzb_build_dst_views,
            hzb_sets,
            hzb_sampler,
            overdraw_images,
            overdraw_views,
            render_finished,
            image_acquired_semaphores,
            depth_images,
            depth_views,
        }
    }

    unsafe fn free(self, core: &VulkanCore) {
        let device = &core.device;
        let allocator = &core.allocator;
        let Self {
            hzb_descriptor_pool,
            hzb_images,
            hzb_build_src_views,
            hzb_build_dst_views,
            hzb_sets: _,
            hzb_sampler,
            overdraw_images,
            overdraw_views,
            render_finished,
            image_acquired_semaphores,
            depth_images,
            depth_views,
        } = self;

        device.destroy_sampler(hzb_sampler, None);

        for views in hzb_build_src_views {
            for view in views {
                device.destroy_image_view(view.view, None);
            }
        }
        for views in hzb_build_dst_views {
            for view in views {
                device.destroy_image_view(view.view, None);
            }
        }
        for mut image in hzb_images {
            unsafe { allocator.destroy_image(image.image, &mut image.alloc) }
        }
        for view in overdraw_views {
            device.destroy_image_view(view.view, None);
        }
        for mut image in overdraw_images {
            unsafe { allocator.destroy_image(image.image, &mut image.alloc) }
        }

        for semaphore in render_finished {
            device.destroy_semaphore(semaphore, None);
        }
        for semaphore in image_acquired_semaphores {
            device.destroy_semaphore(semaphore, None);
        }

        for view in depth_views {
            device.destroy_image_view(view.view, None);
        }
        for mut image in depth_images {
            unsafe { allocator.destroy_image(image.image, &mut image.alloc) }
        }

        device.destroy_descriptor_pool(hzb_descriptor_pool, None);
    }
}

struct PendingSceneState {
    // Keep the upload cmd buffer and staging allocation alive until promotion.
    cmd: vk::CommandBuffer,
    staging: StagingSpan,
    scene_states: SceneState,
}

/* Resources that need regeneration when object set changes */
struct SceneState {
    // Scene-wide semi-stable buffers.
    scene_index_buffer: Buffer<[GpuIndex]>,
    scene_object_instance_buffer: Buffer<[GpuObjectInstance]>,

    // FIF local buffers.
    object_instance_buffer: [Buffer<[GpuObjectInstance]>; MAX_FRAMES_IN_FLIGHT],
    indirect_cmd_buffers: [Buffer<GpuDrawCommandBuffer>; MAX_FRAMES_IN_FLIGHT],
    frustum_passing_meshlet_buffers: [Buffer<GpuFrustumPassingMeshletBuffer>; MAX_FRAMES_IN_FLIGHT],
    visibility_buffers: HashMap<ObjectHandle, ResourceQueue<Buffer<[u32]>>>,

    // Bookkeeping
    world: World,
    scene_index_offsets: HashMap<MeshHandle, u32>,
}

impl SceneState {
    unsafe fn free(self, core: &VulkanCore) {
        let device = &core.device;
        let allocator = &core.allocator;
        for buffer in self.frustum_passing_meshlet_buffers {
            buffer.destroy(&allocator);
        }
        self.scene_index_buffer.destroy(&allocator);
        for (_, queue) in self.visibility_buffers {
            for buffer in queue.free(device) {
                buffer.destroy(&allocator);
            }
        }
        self.scene_object_instance_buffer.destroy(&allocator);
        for buffer in self.object_instance_buffer {
            buffer.destroy(&allocator);
        }
        for buffer in self.indirect_cmd_buffers {
            buffer.destroy(&allocator);
        }
    }
}

pub struct Renderer {
    //
    core: Arc<VulkanCore>,

    /* Swapchain data: */
    swapchain: Swapchain,

    /* Profiling: */
    profilers: [PipelineProfiler; MAX_FRAMES_IN_FLIGHT],
    profile_report_samples: u32,

    /* Pipelines: */
    pipelines: Pipelines,

    /* Generic resource containers: */
    cwd: PathBuf,
    resource_counter: HandleCounter,
    objects: BTreeMap<ObjectHandle, Object>,
    meshes: BTreeMap<MeshHandle, Arc<Mesh>>,

    // Some cpu -> gpu resources.
    gpu_meshes: HashMap<MeshHandle, GpuMesh>,
    pending_gpu_meshes: HashMap<MeshHandle, PendingGpuMesh>,

    // 
    thread_pool: ThreadPool,
    graphics_cmd_pools: Arc<SegQueue<vk::CommandPool>>,

    /* Staging: */
    staging: Arc<StagingPool>,

    /* Scene: */
    _descriptor_pools: [vk::DescriptorPool; MAX_FRAMES_IN_FLIGHT],
    _overdraw_descriptor_pools: [vk::DescriptorPool; MAX_FRAMES_IN_FLIGHT],

    // Submit command buffers.
    cmd_buffers: [[vk::CommandBuffer; PipelineStage::COUNT]; MAX_FRAMES_IN_FLIGHT],

    // Reusable FIF staging buffers.
    staging_buffers: [StagingSpan; MAX_FRAMES_IN_FLIGHT],

    // Global & per FIF desciptor sets.
    global_set: vk::DescriptorSet,
    // (frame_sets and overdraw_sets are arrays of samplers)
    frame_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    overdraw_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],

    // Global buffer.
    frame_global_buffers: [Buffer<GpuFrameGlobal>; MAX_FRAMES_IN_FLIGHT],

    // Used for sequencing stages, and other cross-frame syncing.
    pipeline_semaphores: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],

    // Serializes data upload across all FIF slots.
    upload_sync_timeline: vk::Semaphore,
    upload_sync_counter: u64,

    // Scene generation currently associated with each FIF slot.
    fif_scene_generations: [u64; MAX_FRAMES_IN_FLIGHT],

    // Next frame timeline wait value for each slot.
    fif_timeline_waits: [u64; MAX_FRAMES_IN_FLIGHT],

    // Dirty flags for resource regeneration.
    swapchain_states_dirty: bool,
    scene_states_dirty: bool,

    // Swapchain management.
    swapchain_states: SwapchainState,

    // Scene management.
    scene_generation_counter: u64,
    scene_timeline: vk::Semaphore,
    pending_scene_states: BTreeMap<u64, PendingSceneState>,
    scene_states: BTreeMap<u64, SceneState>,

    // Various render state data.
    frame: usize,
    pub cam_pos: Vec3,
    pub cam_rot: Vec2, // YX
    pub overdraw_enabled: bool,
    pub overshade_enabled: bool,
}

const LOD_DISTANCE_BIAS: f32 = 2.0;
const LOD_DISTANCE_OFFSET: f32 = 0.25;

impl Drop for Renderer {
    fn drop(&mut self) {
        panic!("{} dropped implicitly; call explicit renderer shutdown before drop", std::any::type_name::<Self>());
    }
}

impl Renderer {
    pub fn new(
        cwd: impl AsRef<Path>,
        viewport_w: u32,
        viewport_h: u32,
        display: impl HasDisplayHandle + HasWindowHandle,
    ) -> Self {
        unsafe {
            let core = Arc::new(VulkanCore::new(display));
            let device = &core.device;
            let allocator = &core.allocator;

            // Build swapchain from core.
            let swapchain = Swapchain::new(&core, vk::Extent2D { width: viewport_w, height: viewport_h });

            let profilers = std::array::from_fn(|_| PipelineProfiler::new(&core));

            let pipelines = Pipelines::new(&core);

            // Per-frame recorded render buffers.
            let cmd_buffers = std::array::from_fn(|_| {
                device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(core.cmd_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(PipelineStage::COUNT as _),
                    )
                    .unwrap()
                    .try_into()
                    .unwrap()
            });

            // Staging data.
            let staging = Arc::new(StagingPool::new(allocator, STAGING_ARENA_SIZE));
            let staging_buffers = std::array::from_fn(|_| staging.alloc(STAGING_FIF_BLOCK_SIZE));
            let thread_pool =
                ThreadPool::new(std::thread::available_parallelism().map_or(1, |parallelism| parallelism.get()));
            let graphics_cmd_pools = Arc::new(SegQueue::new());

            //
            let global_descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&[
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1024 + (MAX_FRAMES_IN_FLIGHT as u32 * HZB_SAMPLED_IMAGE_CAPACITY)),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::STORAGE_IMAGE)
                                .descriptor_count(1024 + (MAX_FRAMES_IN_FLIGHT as u32 * HZB_STORAGE_IMAGE_CAPACITY)),
                        ])
                        .max_sets(1 + MAX_FRAMES_IN_FLIGHT as u32)
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                    None,
                )
                .unwrap();

            // Generic descriptor pool.
            let descriptor_pools = std::array::from_fn(|_| {
                device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .pool_sizes(&[vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::STORAGE_BUFFER)
                                .descriptor_count(1)])
                            .max_sets(1)
                            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                        None,
                    )
                    .unwrap()
            });

            let overdraw_descriptor_pools = std::array::from_fn(|_| {
                device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .pool_sizes(&[
                                vk::DescriptorPoolSize::default()
                                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                                    .descriptor_count(1),
                                vk::DescriptorPoolSize::default()
                                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                                    .descriptor_count(2),
                            ])
                            .max_sets(1)
                            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                        None,
                    )
                    .unwrap()
            });

            // Timeline semaphores for per-FIF stage sequencing.
            let pipeline_semaphores = std::array::from_fn(|_| {
                device
                    .create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(
                            &mut vk::SemaphoreTypeCreateInfo::default()
                                .semaphore_type(vk::SemaphoreType::TIMELINE)
                                .initial_value(0),
                        ),
                        None,
                    )
                    .unwrap()
            });
            let upload_sync_timeline = device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(
                        &mut vk::SemaphoreTypeCreateInfo::default()
                            .semaphore_type(vk::SemaphoreType::TIMELINE)
                            .initial_value(0),
                    ),
                    None,
                )
                .unwrap();

            // Descriptor sets.
            let global_set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(global_descriptor_pool)
                        .set_layouts(&[pipelines.global_set_layout]),
                )
                .unwrap()[0];

            let frame_sets = std::array::from_fn(|fif| {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pools[fif])
                            .set_layouts(&[pipelines.frame_set_layout]),
                    )
                    .unwrap()[0]
            });

            let overdraw_sets = std::array::from_fn(|fif| {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(overdraw_descriptor_pools[fif])
                            .set_layouts(&[pipelines.overdraw_set_layout]),
                    )
                    .unwrap()[0]
            });

            let frame_global_buffers = std::array::from_fn(|_| {
                Buffer::<GpuFrameGlobal>::new(
                    &allocator,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    vk_mem::MemoryUsage::AutoPreferDevice,
                )
            });
            let mut scene_timeline_type =
                vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
            let scene_timeline = device
                .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut scene_timeline_type), None)
                .unwrap();

            for fif in 0..MAX_FRAMES_IN_FLIGHT {
                let descriptor_buffer_infos = [vk::DescriptorBufferInfo::default()
                    .buffer(frame_global_buffers[fif].vk_handle())
                    .offset(0)
                    .range(vk::WHOLE_SIZE)];

                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(frame_sets[fif])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(&descriptor_buffer_infos)],
                    &[],
                );

                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(overdraw_sets[fif])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(&descriptor_buffer_infos)],
                    &[],
                );
            }

            let swapchain_states = SwapchainState::new(&core, &swapchain, &pipelines, &overdraw_sets);

            //
            Self {
                core,

                swapchain,

                profilers,
                profile_report_samples: 0,

                pipelines,

                cwd: cwd.as_ref().to_owned(),
                resource_counter: HandleCounter(0),
                objects: BTreeMap::new(),
                meshes: BTreeMap::new(),

                gpu_meshes: HashMap::new(),
                pending_gpu_meshes: HashMap::new(),
                thread_pool,
                graphics_cmd_pools,

                staging,

                cmd_buffers,

                staging_buffers,

                _descriptor_pools: descriptor_pools,
                _overdraw_descriptor_pools: overdraw_descriptor_pools,

                global_set,
                frame_sets,
                overdraw_sets,
                frame_global_buffers,

                pipeline_semaphores,
                upload_sync_timeline,
                upload_sync_counter: 0,
                fif_scene_generations: [0; MAX_FRAMES_IN_FLIGHT],
                fif_timeline_waits: [0; MAX_FRAMES_IN_FLIGHT],
                swapchain_states_dirty: false,
                scene_states_dirty: true,

                scene_generation_counter: 0,
                scene_timeline,
                pending_scene_states: BTreeMap::new(),
                scene_states: BTreeMap::new(),
                swapchain_states,

                frame: 0,
                cam_pos: Vec3::new(0., 0., 3.),
                cam_rot: <_>::default(),
                overdraw_enabled: false,
                overshade_enabled: false,
            }
        }
    }

    fn rebuild_swapchain_states_if_dirty(&mut self) {
        if self.swapchain_states_dirty {
            self.swapchain_states_dirty = false;
            unsafe {
                self.core
                    .device
                    .wait_semaphores(
                        &vk::SemaphoreWaitInfo::default()
                            .semaphores(&self.pipeline_semaphores)
                            .values(&self.fif_timeline_waits),
                        u64::MAX,
                    )
                    .unwrap();
            }

            let mut swapchain = unsafe { Swapchain::new(&self.core, self.swapchain.extent) };
            let mut swapchain_states =
                unsafe { SwapchainState::new(&self.core, &swapchain, &self.pipelines, &self.overdraw_sets) };
            std::mem::swap(&mut self.swapchain_states, &mut swapchain_states);
            unsafe {
                swapchain_states.free(&self.core);
            }
            std::mem::swap(&mut self.swapchain, &mut swapchain);
            unsafe {
                swapchain.free(&self.core.device);
            }
        }
    }

    fn promote_completed_gpu_meshes(&mut self) {
        for (mesh_id, pending) in std::mem::take(&mut self.pending_gpu_meshes) {
            match unsafe { pending.try_unwrap(&self.core) } {
                Ok(gpu_mesh) => {
                    self.gpu_meshes.insert(mesh_id, gpu_mesh);
                    self.scene_states_dirty = true;
                }
                Err(pending) => {
                    self.pending_gpu_meshes.insert(mesh_id, pending);
                }
            }
        }
    }

    fn rebuild_scene_if_dirty(&mut self) {
        if !self.scene_states_dirty {
            return;
        }

        let missing_meshes: Vec<_> =
            self.meshes.keys().filter(|mesh_id| !self.gpu_meshes.contains_key(mesh_id)).copied().collect();
        for mesh_id in missing_meshes {
            if self.pending_gpu_meshes.contains_key(&mesh_id) {
                continue;
            }
            let mesh = Arc::clone(self.meshes.get(&mesh_id).unwrap());
            let pending = unsafe {
                PendingGpuMesh::submit(&self.thread_pool, &self.graphics_cmd_pools, &self.core, &self.staging, mesh)
            };
            self.pending_gpu_meshes.insert(mesh_id, pending);
        }

        if self.meshes.keys().any(|mesh_id| !self.gpu_meshes.contains_key(mesh_id)) {
            return;
        }

        self.scene_states_dirty = false;
        unsafe {
            self.build_scene();
        }
    }

    fn promote_completed_scene_states(&mut self) {
        let ready_scene_states: Vec<_> = self
            .pending_scene_states
            .extract_if(.., |generation, _| unsafe {
                self.core.device.get_semaphore_counter_value(self.scene_timeline).unwrap() >= *generation + 1
            })
            .collect();
        for (generation, PendingSceneState { cmd, staging, scene_states }) in ready_scene_states {
            unsafe {
                self.core.device.free_command_buffers(self.core.cmd_pool, &[cmd]);
            }
            unsafe {
                self.staging.free_span(staging);
            }
            self.scene_states.insert(generation, scene_states);
        }
    }

    fn reserve_available_frame_slot(&mut self) -> (usize, u64) {
        unsafe {
            wait_semaphores_any_fallback(&self.core.device, &self.pipeline_semaphores, &self.fif_timeline_waits)
                .unwrap();
        }

        let index = self
            .pipeline_semaphores
            .iter()
            .zip(self.fif_timeline_waits.iter())
            .enumerate()
            .find(|(_, (semaphore, wait))| unsafe {
                self.core.device.get_semaphore_counter_value(**semaphore).unwrap() == **wait
            })
            .unwrap()
            .0;

        let timeline = self.fif_timeline_waits[index];
        self.fif_timeline_waits[index] += PipelineStage::COUNT as u64;
        (index, timeline)
    }

    fn retire_scene_if_unreferenced(&mut self, generation: u64) {
        if self.fif_scene_generations.contains(&generation) {
            return;
        }

        if let Some(scene) = self.scene_states.remove(&generation) {
            unsafe {
                scene.free(&self.core);
            }
        }
    }

    unsafe fn read_and_accumulate_frame_profile(&mut self, frame_index: usize) {
        let read_profile = self.profilers[frame_index]
            .read_and_accumulate(&self.core.device, self.core.physical_device_properties.limits.timestamp_period);
        if !read_profile {
            return;
        }

        self.profile_report_samples += 1;
        if self.profile_report_samples == PipelineProfiler::REPORT_SAMPLES {
            PipelineProfiler::print_report(&self.profilers);
            self.profile_report_samples = 0;
        }
    }

    unsafe fn record_and_submit_data_upload_stage(
        &mut self,
        frame_index: usize,
        frame_timeline_base: u64,
        scene_generation: u64,
        pipeline_semaphore: vk::Semaphore,
        object_dispatch: &mut Vec<(u16, u16)>,
    ) -> Vec<vk::SemaphoreSubmitInfo<'static>> {
        let profiler = &self.profilers[frame_index];
        let data_upload = self.cmd_buffers[frame_index][PipelineStage::DataUpload as usize];
        let upload_sync_wait_value = self.upload_sync_counter;
        self.upload_sync_counter += 1;
        let upload_sync_signal_value = self.upload_sync_counter;
        let swapchain_extent = self.swapchain.extent;
        let visibility_wait_strategy = WaitStrategy {
            semaphore: pipeline_semaphore,
            value: PipelineStage::OcclusionCull.signal_value(frame_timeline_base),
        };

        let SceneState {
            indirect_cmd_buffers,
            visibility_buffers,
            scene_index_offsets,
            scene_object_instance_buffer,
            object_instance_buffer: object_instance_buffers,
            frustum_passing_meshlet_buffers,
            world,
            ..
        } = self.scene_states.get_mut(&scene_generation).unwrap();
        let object_instance_buffer = &object_instance_buffers[frame_index];
        let indirect_cmd_buffer = &indirect_cmd_buffers[frame_index];
        let frustum_passing_meshlet_buffer = &frustum_passing_meshlet_buffers[frame_index];
        let frame_global_buffer = &self.frame_global_buffers[frame_index];

        let mut visibility_resource_waits = Vec::new();
        object_dispatch.reserve(world.objects.len());
        let camera_forward = Vec3::new(
            self.cam_rot[0].sin() * self.cam_rot[1].cos(),
            self.cam_rot[1].sin(),
            -self.cam_rot[0].cos() * self.cam_rot[1].cos(),
        );
        let mut object_data = Vec::with_capacity(world.objects.len());
        for handle in world.objects.keys() {
            let obj = self.objects.get(handle).unwrap();

            let mesh = self.meshes.get(&obj.mesh).unwrap();
            let gpu_mesh = self.gpu_meshes.get(&obj.mesh).unwrap();
            let scene_index_offset = *scene_index_offsets.get(&obj.mesh).unwrap();

            let distance = (self.cam_pos - obj.position).length();
            let object_radius = obj.scale * mesh.scale * mesh.radius;
            let lod_ratio =
                ((distance.max(1e-5) / object_radius.max(1e-5)) - LOD_DISTANCE_OFFSET).max(1e-5) * LOD_DISTANCE_BIAS;
            let lod_id = lod_ratio.log2().floor().clamp(0.0, (mesh.lod_count.saturating_sub(1)) as f32) as u8;
            let meshlet_subrange = gpu_mesh.meshlet_lod_to_offset[&lod_id.min(mesh.lod_count)].clone();

            object_dispatch.push((object_data.len() as u16, meshlet_subrange.end - meshlet_subrange.start));

            let (visibility_buffer, previous_visibility_buffer, waits) = {
                let visibility_queue = visibility_buffers.get_mut(handle).unwrap();
                let previous_visibility_buffer =
                    visibility_queue.read(&self.core.device, visibility_wait_strategy).vk_handle();
                let (visibility_buffer, waits) =
                    visibility_queue.write(&self.core.device, frame_index, visibility_wait_strategy).unwrap();
                (visibility_buffer.vk_handle(), previous_visibility_buffer, waits)
            };
            visibility_resource_waits.extend(waits);

            object_data.push(GpuObjectInstance {
                position: obj.position,
                scale: obj.scale * mesh.scale,
                orientation: obj.orientation,
                vertex_buffer: self.core.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(gpu_mesh.vertex_buffer.vk_handle()),
                ),
                // This BDA is corrected for LOD subrange.
                meshlet_buffer: self.core.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(gpu_mesh.meshlet_buffer.vk_handle()),
                ) + meshlet_subrange.start as u64 * std::mem::size_of::<GpuMeshlet>() as u64,
                visibility_buffer: self
                    .core
                    .device
                    .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(visibility_buffer)),
                previous_visibility_buffer: self.core.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(previous_visibility_buffer),
                ),
                texture_id: 0,
                scene_index_offset,
            });
        }

        // TODO: consider gpu sorting?
        object_dispatch.sort_unstable_by(|a, b| {
            let a_pos = object_data[usize::from(a.0)].position;
            let b_pos = object_data[usize::from(b.0)].position;
            let a_distance = (self.cam_pos - a_pos).length_squared();
            let b_distance = (self.cam_pos - b_pos).length_squared();
            a_distance.total_cmp(&b_distance)
        });

        // Reverse-Z projection: near maps to 1.0, infinity tends toward 0.0.
        let projection = Mat4::perspective_infinite_reverse_rh(
            std::f32::consts::FRAC_PI_6,
            swapchain_extent.width as f32 / swapchain_extent.height as f32,
            0.1,
        );

        let p = camera_forward;
        let view = Mat4::look_to_rh(self.cam_pos, p, Vec3::new(0., 1., 0.));

        // Frustum plane data.
        let normalize_plane = |p: Vec4| p / p.xyz().length();
        let temp = projection.transpose();
        let frustum_x = normalize_plane(temp.w_axis + temp.x_axis);
        let frustum_y = normalize_plane(temp.w_axis + temp.y_axis);
        let frustum = Vec4::from([frustum_x.x, frustum_x.z, frustum_y.y, frustum_y.z]);

        let frame_global = GpuFrameGlobal {
            pv: projection * view,
            proj: projection,
            view,
            camera_position: self.cam_pos.extend(1.0),
            camera_direction: p.extend(0.0),
            light_position: Vec4::new(1.0, 0.0, 0.0, 1.0),
            light_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            frustum,
            screen_info: Vec4::new(swapchain_extent.width as f32, swapchain_extent.height as f32, 0.0, 0.0),
            draw_cmd_buffer: self.core.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(indirect_cmd_buffer.vk_handle()),
            ),
            object_buffer: self.core.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(object_instance_buffer.vk_handle()),
            ),
            frustum_passing_meshlet_buffer: self.core.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(frustum_passing_meshlet_buffer.vk_handle()),
            ),
            occlusion_dispatch: vk::DispatchIndirectCommand { x: 0, y: 1, z: 1 },
        };

        record_cmd_buffer(&self.core.device, data_upload, |cmd| {
            profiler.begin(&self.core.device, cmd, PipelineStage::DataUpload, || {
                // Upload scene data.
                let staging = &mut self.staging_buffers[frame_index];
                staging.reset();
                self.staging.stage(staging, &self.core.device, cmd, &scene_object_instance_buffer, Whole(object_data));
                self.core.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&[vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .buffer(scene_object_instance_buffer.vk_handle())
                        .offset(0)
                        .size(vk::WHOLE_SIZE)]),
                );
                self.core.device.cmd_copy_buffer(
                    cmd,
                    scene_object_instance_buffer.vk_handle(),
                    object_instance_buffer.vk_handle(),
                    &[vk::BufferCopy::default().size(scene_object_instance_buffer.size() as u64)],
                );

                self.staging.stage(staging, &self.core.device, cmd, frame_global_buffer, Whole(frame_global));

                // Set indirect & frustum_passing lens to 0.
                self.core.device.cmd_fill_buffer(
                    cmd,
                    indirect_cmd_buffer.vk_handle(),
                    0,
                    std::mem::size_of::<u32>() as u64,
                    0,
                );
                self.core.device.cmd_fill_buffer(
                    cmd,
                    frustum_passing_meshlet_buffer.vk_handle(),
                    0,
                    std::mem::size_of::<u32>() as u64,
                    0,
                );
            })
        });

        self.core
            .device
            .queue_submit2(
                *self.core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(data_upload)])
                    .wait_semaphore_infos(&[
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::DataUpload.wait_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::TRANSFER),
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.upload_sync_timeline)
                            .value(upload_sync_wait_value)
                            .stage_mask(vk::PipelineStageFlags2::TRANSFER),
                    ])
                    .signal_semaphore_infos(&[
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::DataUpload.signal_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::TRANSFER),
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.upload_sync_timeline)
                            .value(upload_sync_signal_value)
                            .stage_mask(vk::PipelineStageFlags2::TRANSFER),
                    ])],
                vk::Fence::null(),
            )
            .unwrap();

        visibility_resource_waits
    }

    unsafe fn record_and_submit_frustum_cull_stage(
        &self,
        frame_index: usize,
        frame_timeline_base: u64,
        pipeline_semaphore: vk::Semaphore,
        object_dispatch: Vec<(u16, u16)>,
        visibility_resource_waits: Vec<vk::SemaphoreSubmitInfo<'static>>,
    ) {
        let profiler = &self.profilers[frame_index];
        let frustum_cull = self.cmd_buffers[frame_index][PipelineStage::FrustumCull as usize];
        let frame_set = self.frame_sets[frame_index];

        record_cmd_buffer(&self.core.device, frustum_cull, |cmd| {
            profiler.begin(&self.core.device, cmd, PipelineStage::FrustumCull, || {
                self.core.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.frustum_cull_pipeline_layout,
                    0,
                    &[frame_set],
                    &[],
                );

                self.core.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.frustum_cull_pipeline,
                );

                for (object_index, meshlet_count) in object_dispatch {
                    let push_constants = u32::from(object_index) | (u32::from(meshlet_count) << 16);
                    self.core.device.cmd_push_constants(
                        cmd,
                        self.pipelines.frustum_cull_pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &push_constants.to_ne_bytes(),
                    );

                    self.core.device.cmd_dispatch(cmd, (meshlet_count as u32).div_ceil(64), 1, 1);
                }
            })
        });

        let frustum_cmd_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(frustum_cull)];
        let frustum_signal_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(pipeline_semaphore)
            .value(PipelineStage::FrustumCull.signal_value(frame_timeline_base))
            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)];
        let mut frustum_waits = Vec::with_capacity(2 + visibility_resource_waits.len());
        frustum_waits.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(pipeline_semaphore)
                .value(PipelineStage::FrustumCull.wait_value(frame_timeline_base))
                .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER),
        );
        frustum_waits.extend(visibility_resource_waits);
        let frustum_submit_infos = [vk::SubmitInfo2::default()
            .command_buffer_infos(&frustum_cmd_infos)
            .wait_semaphore_infos(&frustum_waits)
            .signal_semaphore_infos(&frustum_signal_infos)];
        self.core
            .device
            .queue_submit2(*self.core.graphics_queue.lock().unwrap(), &frustum_submit_infos, vk::Fence::null())
            .unwrap();
    }

    unsafe fn record_and_submit_early_draw_stage(
        &self,
        frame_index: usize,
        frame_timeline_base: u64,
        scene_generation: u64,
        image_index: u32,
        pipeline_semaphore: vk::Semaphore,
        image_acquired: vk::Semaphore,
        debug_draw_enabled: bool,
    ) {
        let profiler = &self.profilers[frame_index];
        let early_draw = self.cmd_buffers[frame_index][PipelineStage::EarlyDraw as usize];
        let swapchain_extent = self.swapchain.extent;
        let swapchain_image = self.swapchain.images[image_index as usize];
        let swapchain_view = self.swapchain.views[image_index as usize];
        let depth_view = &self.swapchain_states.depth_views[frame_index];
        let overdraw_image = &self.swapchain_states.overdraw_images[frame_index];
        let global_set = self.global_set;
        let frame_set = self.frame_sets[frame_index];
        let overdraw_set = self.overdraw_sets[frame_index];
        let overshade_enabled = self.overshade_enabled;
        let scene = self.scene_states.get(&scene_generation).unwrap();
        let scene_index_buffer = &scene.scene_index_buffer;
        let indirect_cmd_buffer = &scene.indirect_cmd_buffers[frame_index];

        if debug_draw_enabled {
            let swapchain_info = [vk::DescriptorImageInfo::default()
                .image_view(swapchain_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            self.core.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(overdraw_set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(swapchain_info.len() as u32)
                    .image_info(&swapchain_info)],
                &[],
            );
        }

        record_cmd_buffer(&self.core.device, early_draw, |cmd| {
            profiler.begin(&self.core.device, cmd, PipelineStage::EarlyDraw, || {
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        // Reverse-Z clears to the farthest depth value.
                        depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                    });

                let render_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: swapchain_extent.width,
                            height: swapchain_extent.height,
                        },
                    })
                    .layer_count(1)
                    .depth_attachment(&depth_attachment);

                if debug_draw_enabled {
                    self.core.device.cmd_clear_color_image(
                        cmd,
                        overdraw_image.image,
                        vk::ImageLayout::GENERAL,
                        &vk::ClearColorValue { uint32: [0, 0, 0, 0] },
                        &[vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)],
                    );

                    // The overdraw count image was cleared by transfer; make it available to fragment shader atomics.
                    self.core.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .image(overdraw_image.image)
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::GENERAL)]),
                    );

                    self.core.device.cmd_begin_rendering(cmd, &render_info.color_attachments(&[]));
                    self.core.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.overdraw_render_pipeline_layout,
                        0,
                        &[global_set, overdraw_set],
                        &[],
                    );
                    self.core.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        if overshade_enabled {
                            self.pipelines.overshade_render_pipeline
                        } else {
                            self.pipelines.overdraw_render_pipeline
                        },
                    );
                } else {
                    // Swapchain image must move from presentable usage to color attachment usage for the normal render
                    // path.
                    self.core.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .image(swapchain_image)
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)]),
                    );

                    self.core.device.cmd_begin_rendering(
                        cmd,
                        &render_info.color_attachments(&[vk::RenderingAttachmentInfo::default()
                            .image_view(swapchain_view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .clear_value(vk::ClearValue {
                                color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
                            })]),
                    );

                    self.core.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.render_pipeline_layout,
                        0,
                        &[global_set, frame_set],
                        &[],
                    );

                    self.core.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.render_pipeline,
                    );
                }

                self.core.device.cmd_set_viewport(
                    cmd,
                    0,
                    &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: swapchain_extent.width as f32,
                        height: swapchain_extent.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );

                self.core.device.cmd_set_scissor(
                    cmd,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain_extent,
                    }],
                );

                self.core.device.cmd_bind_index_buffer(cmd, scene_index_buffer.vk_handle(), 0, vk::IndexType::UINT32);

                self.core.device.cmd_draw_indexed_indirect_count(
                    cmd,
                    indirect_cmd_buffer.vk_handle(),
                    std::mem::size_of::<GpuIndex>() as u64,
                    indirect_cmd_buffer.vk_handle(),
                    0,
                    indirect_cmd_buffer.len(),
                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );

                self.core.device.cmd_end_rendering(cmd);
            })
        });

        self.core
            .device
            .queue_submit2(
                *self.core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(early_draw)])
                    .wait_semaphore_infos(&[
                        // We can delay this wait until late_draw when overdraw mode is on, but its probably not a
                        // useful optimization.
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::EarlyDraw.wait_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT),
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(image_acquired)
                            .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE),
                    ])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::EarlyDraw.signal_value(frame_timeline_base))
                        .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)])],
                vk::Fence::null(),
            )
            .unwrap();
    }

    unsafe fn record_and_submit_build_hzb_stage(
        &self,
        frame_index: usize,
        frame_timeline_base: u64,
        pipeline_semaphore: vk::Semaphore,
    ) {
        let profiler = &self.profilers[frame_index];
        let build_hzb = self.cmd_buffers[frame_index][PipelineStage::BuildHzb as usize];
        let swapchain_extent = self.swapchain.extent;
        let hzb_base_width = swapchain_extent.width.div_ceil(2).max(1);
        let hzb_base_height = swapchain_extent.height.div_ceil(2).max(1);
        let hzb_image = &self.swapchain_states.hzb_images[frame_index];
        let hzb_set = self.swapchain_states.hzb_sets[frame_index];
        let hzb_build_src_views = &self.swapchain_states.hzb_build_src_views[frame_index];
        let depth_image = &self.swapchain_states.depth_images[frame_index];
        let depth_view = &self.swapchain_states.depth_views[frame_index];

        let depth_info = [vk::DescriptorImageInfo::default()
            .image_view(depth_view.view)
            .sampler(self.swapchain_states.hzb_sampler)
            .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)];
        self.core.device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(hzb_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(depth_info.len() as u32)
                .image_info(&depth_info)],
            &[],
        );

        record_cmd_buffer(&self.core.device, build_hzb, |cmd| {
            profiler.begin(&self.core.device, cmd, PipelineStage::BuildHzb, || {
                // HZB is sampled from the previous frame and rewritten by this frame's reduction passes.
                self.core.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        // Prepare the HZB for writing.
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .image(hzb_image.image)
                            .subresource_range(vk::ImageSubresourceRange {
                                level_count: vk::REMAINING_MIP_LEVELS,
                                ..COLOR_2D_SUBRESOURCE_RANGE
                            })
                            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .new_layout(vk::ImageLayout::GENERAL),
                        // Prepare the depth buffer for sampling @ first HZB reduction.
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(
                                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                            )
                            .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .image(depth_image.image)
                            .subresource_range(DEPTH_2D_SUBRESOURCE_RANGE)
                            .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL),
                    ]),
                );

                let build_hzb = |level: u32| {
                    self.core.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipelines.build_hzb_pipeline_layout,
                        0,
                        &[hzb_set],
                        &[],
                    );

                    self.core.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipelines.build_hzb_pipeline,
                    );

                    let w = hzb_base_width.checked_shr(level).unwrap_or(0).max(1).div_ceil(8);
                    let h = hzb_base_height.checked_shr(level).unwrap_or(0).max(1).div_ceil(8);
                    self.core.device.cmd_dispatch_base(cmd, 0, 0, level, w, h, 1);

                    // Keep each mip level coherent as the reduction chain walks down the pyramid.
                    self.core.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .image(hzb_image.image)
                            .subresource_range(vk::ImageSubresourceRange {
                                base_mip_level: level,
                                level_count: 1,
                                ..COLOR_2D_SUBRESOURCE_RANGE
                            })
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::GENERAL)]),
                    );
                };

                // For the first compute, the src view is the depth buffer, which depends on the depth buffer.
                let mips = hzb_build_src_views.len() as u32;
                for level in 0..mips {
                    build_hzb(level);
                }

                // Return the HZB to sampled-read and the depth buffer to attachment-write for the next render frame.
                self.core.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .image(hzb_image.image)
                            .subresource_range(vk::ImageSubresourceRange {
                                level_count: vk::REMAINING_MIP_LEVELS,
                                ..COLOR_2D_SUBRESOURCE_RANGE
                            })
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .dst_stage_mask(
                                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                            )
                            .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                            .image(depth_image.image)
                            .subresource_range(DEPTH_2D_SUBRESOURCE_RANGE)
                            .old_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
                            .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL),
                    ]),
                );
            })
        });

        self.core
            .device
            .queue_submit2(
                *self.core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(build_hzb)])
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::BuildHzb.wait_value(frame_timeline_base))
                        .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::BuildHzb.signal_value(frame_timeline_base))
                        .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])],
                vk::Fence::null(),
            )
            .unwrap();
    }

    unsafe fn record_and_submit_occlusion_cull_stage(
        &self,
        frame_index: usize,
        frame_timeline_base: u64,
        scene_generation: u64,
        pipeline_semaphore: vk::Semaphore,
    ) {
        let profiler = &self.profilers[frame_index];
        let occlusion_cull = self.cmd_buffers[frame_index][PipelineStage::OcclusionCull as usize];
        let hzb_set = self.swapchain_states.hzb_sets[frame_index];
        let frame_set = self.frame_sets[frame_index];
        let frame_global_buffer = &self.frame_global_buffers[frame_index];
        let scene = self.scene_states.get(&scene_generation).unwrap();
        let indirect_cmd_buffer = &scene.indirect_cmd_buffers[frame_index];

        record_cmd_buffer(&self.core.device, occlusion_cull, |cmd| {
            profiler.begin(&self.core.device, cmd, PipelineStage::OcclusionCull, || {
                // Reuse the indirect buffer for the late list only after early draw has consumed it.
                self.core.device.cmd_fill_buffer(
                    cmd,
                    indirect_cmd_buffer.vk_handle(),
                    0,
                    std::mem::size_of::<u32>() as u64,
                    0,
                );

                self.core.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.occlusion_cull_pipeline_layout,
                    0,
                    &[hzb_set, frame_set],
                    &[],
                );

                self.core.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.occlusion_cull_pipeline,
                );

                self.core.device.cmd_dispatch_indirect(
                    cmd,
                    frame_global_buffer.vk_handle(),
                    offset_of!(GpuFrameGlobal, occlusion_dispatch) as u64,
                );
            })
        });

        self.core
            .device
            .queue_submit2(
                *self.core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(occlusion_cull)])
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::BuildHzb.signal_value(frame_timeline_base))
                        .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::OcclusionCull.signal_value(frame_timeline_base))
                        .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])],
                vk::Fence::null(),
            )
            .unwrap();
    }

    unsafe fn record_and_submit_late_draw_stage(
        &self,
        frame_index: usize,
        frame_timeline_base: u64,
        scene_generation: u64,
        image_index: u32,
        pipeline_semaphore: vk::Semaphore,
        debug_draw_enabled: bool,
    ) {
        let profiler = &self.profilers[frame_index];
        let late_draw = self.cmd_buffers[frame_index][PipelineStage::LateDraw as usize];
        let swapchain_extent = self.swapchain.extent;
        let swapchain_image = self.swapchain.images[image_index as usize];
        let swapchain_view = self.swapchain.views[image_index as usize];
        let depth_view = &self.swapchain_states.depth_views[frame_index];
        let global_set = self.global_set;
        let frame_set = self.frame_sets[frame_index];
        let overdraw_set = self.overdraw_sets[frame_index];
        let scene = self.scene_states.get(&scene_generation).unwrap();
        let scene_index_buffer = &scene.scene_index_buffer;
        let indirect_cmd_buffer = &scene.indirect_cmd_buffers[frame_index];

        record_cmd_buffer(&self.core.device, late_draw, |cmd| {
            profiler.begin(&self.core.device, cmd, PipelineStage::LateDraw, || {
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE);

                let render_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: swapchain_extent.width,
                            height: swapchain_extent.height,
                        },
                    })
                    .layer_count(1)
                    .depth_attachment(&depth_attachment);

                if debug_draw_enabled {
                    self.core.device.cmd_begin_rendering(cmd, &render_info.color_attachments(&[]));
                    self.core.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.overdraw_render_pipeline_layout,
                        0,
                        &[global_set, overdraw_set],
                        &[],
                    );
                    self.core.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.overdraw_render_pipeline,
                    );
                } else {
                    self.core.device.cmd_begin_rendering(
                        cmd,
                        &render_info.color_attachments(&[vk::RenderingAttachmentInfo::default()
                            .image_view(swapchain_view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE)]),
                    );

                    self.core.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.render_pipeline_layout,
                        0,
                        &[global_set, frame_set],
                        &[],
                    );

                    self.core.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.render_pipeline,
                    );
                }

                self.core.device.cmd_set_viewport(
                    cmd,
                    0,
                    &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: swapchain_extent.width as f32,
                        height: swapchain_extent.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );

                self.core.device.cmd_set_scissor(
                    cmd,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain_extent,
                    }],
                );

                self.core.device.cmd_bind_index_buffer(cmd, scene_index_buffer.vk_handle(), 0, vk::IndexType::UINT32);

                self.core.device.cmd_draw_indexed_indirect_count(
                    cmd,
                    indirect_cmd_buffer.vk_handle(),
                    std::mem::size_of::<GpuIndex>() as u64,
                    indirect_cmd_buffer.vk_handle(),
                    0,
                    indirect_cmd_buffer.len(),
                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );

                self.core.device.cmd_end_rendering(cmd);

                if debug_draw_enabled {
                    // In overdraw mode, the swapchain image becomes a storage image for the resolve compute pass.
                    self.core.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .image(swapchain_image)
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                            .new_layout(vk::ImageLayout::GENERAL)]),
                    );

                    self.core.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipelines.overdraw_resolve_pipeline_layout,
                        0,
                        &[global_set, overdraw_set],
                        &[],
                    );
                    self.core.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipelines.overdraw_resolve_pipeline,
                    );
                    self.core.device.cmd_dispatch(
                        cmd,
                        swapchain_extent.width.div_ceil(8),
                        swapchain_extent.height.div_ceil(8),
                        1,
                    );

                    // The compute resolve writes the final swapchain image, so transition it back to presentable usage.
                    self.core.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                            .image(swapchain_image)
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)]),
                    );
                }

                if !debug_draw_enabled {
                    // Hand the swapchain image from color attachment output to presentation.
                    self.core.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                            .image(swapchain_image)
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)]),
                    );
                }
            })
        });

        self.core
            .device
            .queue_submit2(
                *self.core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(late_draw)])
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::OcclusionCull.signal_value(frame_timeline_base))
                        .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::LateDraw.signal_value(frame_timeline_base))
                        .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)])],
                vk::Fence::null(),
            )
            .unwrap();
    }

    unsafe fn record_and_submit_frame_end_stage(
        &self,
        frame_index: usize,
        frame_timeline_base: u64,
        pipeline_semaphore: vk::Semaphore,
        render_finished: vk::Semaphore,
        debug_draw_enabled: bool,
    ) {
        let frame_end = self.cmd_buffers[frame_index][PipelineStage::FrameEnd as usize];

        record_cmd_buffer(&self.core.device, frame_end, |_cmd| {
            // FrameEnd is intentionally empty; it only preserves the stage accounting / timeline structure.
        });

        self.core
            .device
            .queue_submit2(
                *self.core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(frame_end)])
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::LateDraw.signal_value(frame_timeline_base))
                        .stage_mask(match debug_draw_enabled {
                            true => vk::PipelineStageFlags2::TOP_OF_PIPE,
                            false => vk::PipelineStageFlags2::COMPUTE_SHADER,
                        })])
                    .signal_semaphore_infos(&[
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::FrameEnd.signal_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(render_finished)
                            .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
                    ])],
                vk::Fence::null(),
            )
            .unwrap();
    }

    unsafe fn record_and_submit_pipeline_stages(
        &mut self,
        frame_index: usize,
        frame_timeline_base: u64,
        scene_generation: u64,
    ) -> (u32, vk::Semaphore) {
        let pipeline_semaphore = self.pipeline_semaphores[frame_index];
        let image_acquired = self.swapchain_states.image_acquired_semaphores[frame_index];

        let (image_index, _) = self
            .swapchain
            .swapchain_device
            .acquire_next_image(self.swapchain.swapchain, u64::MAX, image_acquired, vk::Fence::null())
            .unwrap();
        let render_finished = self.swapchain_states.render_finished[image_index as usize];
        let debug_draw_enabled = self.overdraw_enabled || self.overshade_enabled;

        self.read_and_accumulate_frame_profile(frame_index);

        // Object-array-relative dispatch order consumed by FrustumCull. Keeping this separate from object_data allows
        // general CPU-side sorting, though the shader forwards frustum-passing meshlets with atomics, so downstream
        // order is not guaranteed, just usually preserved enough to matter for performance.
        let mut object_dispatch = Vec::new();

        // DataUpload stage. Builds CPU frame data, records transfer/reset commands, and serializes staging upload work.
        let visibility_resource_waits = self.record_and_submit_data_upload_stage(
            frame_index,
            frame_timeline_base,
            scene_generation,
            pipeline_semaphore,
            &mut object_dispatch,
        );

        // FrustumCull stage. Records per-object compute dispatches and waits for upload plus visibility resources.
        self.record_and_submit_frustum_cull_stage(
            frame_index,
            frame_timeline_base,
            pipeline_semaphore,
            object_dispatch,
            visibility_resource_waits,
        );

        // EarlyDraw stage. Records the speculative visible-last-frame draw and waits on swapchain acquisition.
        self.record_and_submit_early_draw_stage(
            frame_index,
            frame_timeline_base,
            scene_generation,
            image_index,
            pipeline_semaphore,
            image_acquired,
            debug_draw_enabled,
        );

        // BuildHzb stage. Reduces the early depth buffer into the per-FIF HZB image.
        self.record_and_submit_build_hzb_stage(frame_index, frame_timeline_base, pipeline_semaphore);

        // OcclusionCull stage. Rebuilds the late indirect list from frustum candidates and the freshly built HZB.
        self.record_and_submit_occlusion_cull_stage(
            frame_index,
            frame_timeline_base,
            scene_generation,
            pipeline_semaphore,
        );

        // LateDraw stage. Records newly visible draws and resolves/debug-transitions the swapchain image.
        self.record_and_submit_late_draw_stage(
            frame_index,
            frame_timeline_base,
            scene_generation,
            image_index,
            pipeline_semaphore,
            debug_draw_enabled,
        );

        // FrameEnd stage. Submits an empty command buffer to publish completion and presentation signals.
        self.record_and_submit_frame_end_stage(
            frame_index,
            frame_timeline_base,
            pipeline_semaphore,
            render_finished,
            debug_draw_enabled,
        );

        (image_index, render_finished)
    }

    pub fn render(&mut self, _timestamp: f32) {
        self.frame += 1;

        // Promote completed asynchronous work before rebuilding dependent state.
        self.promote_completed_gpu_meshes();
        self.promote_completed_scene_states();

        // Lazy rebuild global state if neccessary.
        self.rebuild_swapchain_states_if_dirty();
        self.rebuild_scene_if_dirty();

        // A scene may not be promoted yet to use, so return early.
        if self.scene_states.is_empty() {
            return;
        }

        // Grab a FIF slot that isn't doing anything.
        let (frame_index, frame_timeline_base) = self.reserve_available_frame_slot();

        // Update this FIF slot's scene generation.
        let scene_generation = *self.scene_states.last_key_value().unwrap().0;
        let old_generation = std::mem::replace(&mut self.fif_scene_generations[frame_index], scene_generation);

        // Only the scene generation that this FIF slot stopped referencing can become unreferenced.
        self.retire_scene_if_unreferenced(old_generation);

        unsafe {
            let (image_index, render_finished) =
                self.record_and_submit_pipeline_stages(frame_index, frame_timeline_base, scene_generation);

            // Present.
            self.swapchain
                .swapchain_device
                .queue_present(
                    *self.core.present_queue.lock().unwrap(),
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&[render_finished])
                        .swapchains(&[self.swapchain.swapchain])
                        .image_indices(&[image_index]),
                )
                .unwrap();
        }
    }

    unsafe fn build_scene(&mut self) {
        let mut staging = self.staging.alloc(64 * MiB);
        let cmd = self
            .core
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.core.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap()[0];

        let generation = self.scene_generation_counter;
        self.scene_generation_counter = self.scene_generation_counter.wrapping_add(1);
        let timeline_wait_value = generation;
        let timeline_signal_value = generation + 1;

        staging.reset();
        self.core.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        self.core.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();

        let new_world = World {
            objects: self.objects.clone(),
            meshes: self.meshes.keys().copied().collect(),
        };

        let previous_scene = self.scene_states.iter().next_back();
        let previous_world = previous_scene.as_ref().map(|(_, scene)| &scene.world);
        let world_diff = WorldDiff::between(previous_world.unwrap_or(&World::default()), &new_world);
        // TODO: Use world_diff to choose between in-place master-buffer uploads and a new scene generation.
        // For now build_scene remains the full-generation path.
        let added_meshes: Vec<_> = world_diff.added_meshes.iter().copied().collect();
        let removed_meshes: Vec<_> = world_diff.removed_meshes.iter().copied().collect();

        let previous_scene = self.scene_states.iter_mut().next_back();
        let mut previous_scene_visibility_buffers = previous_scene.map(|(_, scene)| &mut scene.visibility_buffers);

        let mut scene_index_offsets = HashMap::with_capacity(new_world.meshes.len());
        let mut scene_index_count = 0u32;
        for mesh_id in &new_world.meshes {
            let index_buffer = &self.gpu_meshes.get(mesh_id).unwrap().index_buffer;
            scene_index_offsets.insert(*mesh_id, scene_index_count);
            scene_index_count += index_buffer.len();
        }

        let scene_index_buffer = Buffer::<[GpuIndex]>::new(
            &self.core.allocator,
            scene_index_count.max(1),
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );

        let scene_index_buffer_barriers: Vec<_> = new_world
            .meshes
            .iter()
            .map(|mesh_id| {
                let index_buffer = &self.gpu_meshes.get(mesh_id).unwrap().index_buffer;
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .buffer(index_buffer.vk_handle())
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
            })
            .collect();

        self.core.device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().buffer_memory_barriers(&scene_index_buffer_barriers),
        );

        for mesh_id in &new_world.meshes {
            let index_buffer = &self.gpu_meshes.get(mesh_id).unwrap().index_buffer;
            let dst_offset = scene_index_offsets[mesh_id] as u64 * std::mem::size_of::<GpuIndex>() as u64;
            self.core.device.cmd_copy_buffer(
                cmd,
                index_buffer.vk_handle(),
                scene_index_buffer.vk_handle(),
                &[vk::BufferCopy::default().src_offset(0).dst_offset(dst_offset).size(index_buffer.size() as u64)],
            );
        }

        let maximum_scene_meshlets = new_world
            .objects
            .values()
            .map(|object| {
                let mesh = self.meshes.get(&object.mesh).unwrap();
                mesh.lods.iter().map(|lod| lod.len() as u32).max().unwrap_or(0)
            })
            .sum::<u32>();

        let mut new_visibility_buffers = HashMap::with_capacity(new_world.objects.len());
        let mut visibility_buffer_bytes = 0usize;
        for object_id in new_world.objects.keys() {
            let object = self.objects.get(object_id).unwrap();
            let mesh = self.meshes.get(&object.mesh).unwrap();
            let maximum_mesh_meshlets = mesh.lods.iter().map(|lod| lod.len()).max().unwrap_or(0);

            let visibility_buffers = (0..VISIBILITY_RESOURCE_QUEUE_LEN)
                .map(|_| {
                    Buffer::<[u32]>::new(
                        &self.core.allocator,
                        maximum_mesh_meshlets as u32,
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST,
                        vk_mem::MemoryUsage::AutoPreferDevice,
                    )
                })
                .collect::<Vec<_>>();
            visibility_buffer_bytes += visibility_buffers.iter().map(|buffer| buffer.size()).sum::<usize>();

            let previous_visibility = previous_scene_visibility_buffers.as_mut().and_then(|buffers| {
                (*buffers).get_mut(object_id).map(|queue| {
                    let buffer = queue.read(
                        &self.core.device,
                        WaitStrategy { semaphore: self.scene_timeline, value: timeline_signal_value },
                    );
                    (buffer.vk_handle(), buffer.size())
                })
            });

            match previous_visibility {
                Some((previous_visibility, previous_visibility_size)) => {
                    for visibility_buffer in &visibility_buffers {
                        let copy_size = previous_visibility_size.min(visibility_buffer.size()) as u64;
                        if copy_size < visibility_buffer.size() as u64 {
                            self.core.device.cmd_fill_buffer(
                                cmd,
                                visibility_buffer.vk_handle(),
                                copy_size,
                                visibility_buffer.size() as u64 - copy_size,
                                0,
                            );
                        }
                        if copy_size > 0 {
                            self.core.device.cmd_copy_buffer(
                                cmd,
                                previous_visibility,
                                visibility_buffer.vk_handle(),
                                &[vk::BufferCopy::default().size(copy_size)],
                            );
                        }
                    }
                }
                None => {
                    for visibility_buffer in &visibility_buffers {
                        self.core.device.cmd_fill_buffer(
                            cmd,
                            visibility_buffer.vk_handle(),
                            0,
                            visibility_buffer.size() as u64,
                            0,
                        );
                    }
                }
            }

            new_visibility_buffers.insert(*object_id, ResourceQueue::new(MAX_FRAMES_IN_FLIGHT, visibility_buffers));
        }

        let object_instance_buffer = Buffer::<[GpuObjectInstance]>::new(
            &self.core.allocator,
            new_world.objects.len() as u32,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        );
        let object_instance_buffers = std::array::from_fn(|_| {
            Buffer::<[GpuObjectInstance]>::new(
                &self.core.allocator,
                new_world.objects.len() as u32,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            )
        });

        let indirect_cmd_buffers = std::array::from_fn(|_| {
            Buffer::<GpuDrawCommandBuffer>::new_trailing(
                &self.core.allocator,
                maximum_scene_meshlets,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferDevice,
            )
        });

        let frustum_passing_meshlet_buffers = std::array::from_fn(|_| {
            Buffer::<GpuFrustumPassingMeshletBuffer>::new_trailing(
                &self.core.allocator,
                maximum_scene_meshlets,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            )
        });

        let total_meshlet_buffer_count = new_world
            .meshes
            .iter()
            .map(|mesh_id| self.gpu_meshes.get(mesh_id).unwrap().meshlet_buffer.len() as usize)
            .sum::<usize>();
        let total_index_count = scene_index_count as usize;
        let total_triangle_count = total_index_count / 3;
        let scene_index_copy_bytes = total_index_count * std::mem::size_of::<GpuIndex>();
        let visibility_buffer_count = new_visibility_buffers.values().map(|queue| queue.len()).sum::<usize>();
        let object_instance_bytes =
            object_instance_buffer.size() + object_instance_buffers.iter().map(|buffer| buffer.size()).sum::<usize>();
        let indirect_cmd_capacity = indirect_cmd_buffers[0].len() as usize;
        let indirect_cmd_bytes = indirect_cmd_buffers.iter().map(|buffer| buffer.size()).sum::<usize>();
        let frustum_passing_capacity = frustum_passing_meshlet_buffers[0].len() as usize;
        let frustum_passing_bytes = frustum_passing_meshlet_buffers.iter().map(|buffer| buffer.size()).sum::<usize>();
        let staging_bytes_used = staging.size() as usize;

        let mesh_upload_bytes = |mesh_id: &MeshHandle| -> usize {
            let gpu_mesh = self.gpu_meshes.get(mesh_id).unwrap();
            gpu_mesh.meshlet_buffer.size() + gpu_mesh.index_buffer.size() + gpu_mesh.vertex_buffer.size()
        };
        let added_mesh_upload_bytes = added_meshes.iter().map(mesh_upload_bytes).sum::<usize>();

        println!(
            "Scene {} created ({} staged):\n  objects = {}\n  meshes = {} (+{}, -{}, {} new mesh payload)",
            generation,
            format_bytes(staging_bytes_used),
            format_usize_commas(new_world.objects.len()),
            format_usize_commas(new_world.meshes.len()),
            format_usize_commas(added_meshes.len()),
            format_usize_commas(removed_meshes.len()),
            format_bytes(added_mesh_upload_bytes),
        );

        for mesh_id in &added_meshes {
            let mesh = self.meshes.get(mesh_id).unwrap();
            let meshlet_count = mesh.lods.iter().map(|lod| lod.len()).sum::<usize>();
            let max_lod_meshlets = mesh.lods.iter().map(|lod| lod.len()).max().unwrap_or(0);
            let gpu_mesh = self.gpu_meshes.get(mesh_id).unwrap();
            println!(
                "    + {:?}: lods = {}, meshlets = {} ({}) max_lod_meshlets = {}, indices = {} ({}), vertices = {} ({}), upload = {}",
                mesh_id,
                mesh.lod_count,
                format_usize_commas(meshlet_count),
                format_bytes(gpu_mesh.meshlet_buffer.size()),
                format_usize_commas(max_lod_meshlets),
                format_usize_commas(gpu_mesh.index_buffer.len() as usize),
                format_bytes(gpu_mesh.index_buffer.size()),
                format_usize_commas(gpu_mesh.vertex_buffer.len() as usize),
                format_bytes(gpu_mesh.vertex_buffer.size()),
                format_bytes(mesh_upload_bytes(mesh_id)),
            );
        }
        for mesh_id in &removed_meshes {
            println!("    - {:?}", mesh_id);
        }
        println!(
            "  meshlet buffers = {} meshlets\n  max visible meshlets = {}\n  scene indices = {} ({} triangles, {} GPU copy)\n  scene buffers = {} objects ({}), {} indirect commands ({}), {} frustum candidates ({})\n  visibility buffers = {} buffers ({})",
            format_usize_commas(total_meshlet_buffer_count),
            format_usize_commas(indirect_cmd_capacity),
            format_usize_commas(total_index_count),
            format_usize_commas(total_triangle_count),
            format_bytes(scene_index_copy_bytes),
            format_usize_commas(new_world.objects.len()),
            format_bytes(object_instance_bytes),
            format_usize_commas(indirect_cmd_capacity),
            format_bytes(indirect_cmd_bytes),
            format_usize_commas(frustum_passing_capacity),
            format_bytes(frustum_passing_bytes),
            format_usize_commas(visibility_buffer_count),
            format_bytes(visibility_buffer_bytes),
        );

        self.core.device.end_command_buffer(cmd).unwrap();
        self.core
            .device
            .queue_submit2(
                *self.core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(cmd)])
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(self.scene_timeline)
                        .value(timeline_wait_value)
                        .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(self.scene_timeline)
                        .value(timeline_signal_value)
                        .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)])],
                vk::Fence::null(),
            )
            .unwrap();

        self.pending_scene_states.insert(
            generation,
            PendingSceneState {
                cmd,
                staging,
                scene_states: SceneState {
                    scene_object_instance_buffer: object_instance_buffer,
                    object_instance_buffer: object_instance_buffers,
                    indirect_cmd_buffers,
                    scene_index_buffer,
                    frustum_passing_meshlet_buffers,
                    visibility_buffers: new_visibility_buffers,
                    scene_index_offsets,
                    world: new_world,
                },
            },
        );
    }

    pub fn create_object(
        &mut self,
        mesh: MeshHandle,
        position: Vec3,
        scale: f32,
        orientation: Quat,
    ) -> Option<ObjectHandle> {
        self.scene_states_dirty = true;
        let handle = self.resource_counter.next().unwrap();
        self.objects.insert(handle, Object { mesh, position, scale, orientation });
        Some(handle)
    }

    pub fn load_mesh(&mut self, filename: impl AsRef<Path>) -> Option<MeshHandle> {
        let mesh = load_mesh(self.cwd.join(filename))?;
        let handle = self.resource_counter.next().unwrap();
        self.meshes.insert(handle, Arc::new(mesh));
        return Some(handle);
    }
}
