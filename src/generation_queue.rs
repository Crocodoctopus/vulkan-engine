use ash::prelude::VkResult;
use ash::vk;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::renderer::MAX_FRAMES_IN_FLIGHT;
use crate::staging::{StagingPool, StagingSpan};

pub(crate) type Generation = u64;

pub(crate) struct Pending {
    pub cmd: vk::CommandBuffer,
    pub staging: StagingSpan,
    pub timeline: vk::Semaphore,
    pub signal_value: Generation,
}

impl Pending {
    unsafe fn free(self, device: &ash::Device, staging: &StagingPool, cmd_pool: vk::CommandPool) {
        device.free_command_buffers(cmd_pool, &[self.cmd]);
        staging.free_span(self.staging);
    }
}

pub(crate) struct GenerationQueue {
    current_generation: Generation,
    next_generation: Generation,
    retired_generations: BTreeSet<Generation>,
    pending_generations: BTreeMap<Generation, Pending>,
    cmd_pool: vk::CommandPool,
    semaphore: vk::Semaphore,
    fif_generations: [Generation; MAX_FRAMES_IN_FLIGHT],
}

impl GenerationQueue {
    pub(crate) unsafe fn new(device: &ash::Device, queue_family_index: u32) -> Self {
        let cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .unwrap();
        let mut timeline_type =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
        let timeline_semaphore =
            device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut timeline_type), None).unwrap();
        Self {
            current_generation: Generation::MAX,
            next_generation: 0,
            retired_generations: BTreeSet::new(),
            pending_generations: BTreeMap::new(),
            cmd_pool,
            semaphore: timeline_semaphore,
            fif_generations: [0; MAX_FRAMES_IN_FLIGHT],
        }
    }

    unsafe fn wait_for_first_pending(&mut self, device: &ash::Device) -> Option<Generation> {
        let Some((&first_pending_generation, pending)) = self.pending_generations.first_key_value() else {
            return None;
        };
        device
            .wait_semaphores(
                &vk::SemaphoreWaitInfo::default().semaphores(&[pending.timeline]).values(&[pending.signal_value]),
                u64::MAX,
            )
            .unwrap();
        Some(first_pending_generation)
    }

    pub(crate) unsafe fn register(
        &mut self,
        device: &ash::Device,
        staging: &StagingPool,
        staging_len: u64,
    ) -> (Generation, &mut Pending) {
        // Retry until we can get a command buffer.
        let cmd_buffer = loop {
            let cmd_buffer = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            );

            match cmd_buffer {
                VkResult::Ok(cmds) => break cmds[0],
                VkResult::Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => {
                    self.wait_for_first_pending(device);
                    self.update(device, staging);
                    continue;
                }
                VkResult::Err(err) => panic!("failed to allocate cmd buffer: {err:?}"),
            }
        };

        // Retry staging allocation until the virtual block has room.
        let staging = loop {
            let staging_buffer = staging.try_alloc(staging_len);

            match staging_buffer {
                Ok(staging_buffer) => break staging_buffer,
                Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                    self.wait_for_first_pending(device);
                    self.update(device, staging);
                    continue;
                }
                Err(err) => panic!("failed to allocate staging buffer: {err:?}"),
            }
        };

        let generation = self.next_generation;
        let pending = Pending {
            cmd: cmd_buffer,
            staging,
            timeline: self.semaphore,
            signal_value: generation + 1,
        };

        self.pending_generations.insert(generation, pending);
        self.next_generation = self.next_generation.wrapping_add(1);
        let pending = self.pending_generations.get_mut(&generation).unwrap();
        (generation, pending)
    }

    unsafe fn update(&mut self, device: &ash::Device, staging: &StagingPool) {
        while let Some((&generation, pending)) = self.pending_generations.first_key_value() {
            if device.get_semaphore_counter_value(pending.timeline).unwrap() < pending.signal_value {
                break;
            }

            let pending = self.pending_generations.remove(&generation).unwrap();
            pending.free(device, staging, self.cmd_pool);
            self.retired_generations.insert(generation);
            self.current_generation = generation;
        }
    }

    pub(crate) unsafe fn next(&mut self, device: &ash::Device, staging: &StagingPool, frame: usize) -> Generation {
        // First frame bootstrap.
        if self.current_generation == Generation::MAX {
            if let Some(generation) = self.wait_for_first_pending(device) {
                self.current_generation = generation;
            }
        }

        self.update(device, staging);
        self.fif_generations[frame % MAX_FRAMES_IN_FLIGHT] = self.current_generation;
        self.current_generation
    }

    pub(crate) unsafe fn retired_scenes(
        &mut self,
        _device: &ash::Device,
        _staging: &StagingPool,
    ) -> impl Iterator<Item = Generation> {
        // Drain retired generations once no FIF slot still references them.
        let fif_generations = self.fif_generations;
        self.retired_generations.extract_if(.., move |generation| !fif_generations.contains(generation))
    }

    pub(crate) unsafe fn free(
        self,
        device: &ash::Device,
        staging: &StagingPool,
    ) -> impl Iterator<Item = Generation> {
        let Self {
            current_generation,
            next_generation: _,
            retired_generations,
            pending_generations,
            cmd_pool,
            semaphore: timeline_semaphore,
            ..
        } = self;
        let pending_generation_keys: Vec<_> = pending_generations.keys().copied().collect();

        if let Some(max_pending_signal_value) = pending_generations.values().map(|pending| pending.signal_value).max() {
            device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&[timeline_semaphore])
                        .values(&[max_pending_signal_value]),
                    u64::MAX,
                )
                .unwrap();
        }

        // Free all pending resources once they are done.
        for (_, pending) in pending_generations {
            pending.free(device, staging, cmd_pool);
        }
        device.destroy_semaphore(timeline_semaphore, None);
        device.destroy_command_pool(cmd_pool, None);

        // Return all generations.
        return retired_generations
            .into_iter()
            .chain(std::iter::once(current_generation))
            .chain(pending_generation_keys);
    }
}
