use ash::vk;
use futures::channel::oneshot;
use futures::executor::ThreadPool;
use std::sync::Arc;

pub(crate) struct Semaphore {
    device: ash::Device,
    semaphore: vk::Semaphore,
}

impl Semaphore {
    pub(crate) unsafe fn new_timeline(device: &ash::Device, initial_value: u64) -> Arc<Self> {
        let semaphore = device
            .create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(
                    &mut vk::SemaphoreTypeCreateInfo::default()
                        .semaphore_type(vk::SemaphoreType::TIMELINE)
                        .initial_value(initial_value),
                ),
                None,
            )
            .unwrap();

        Arc::new(Self { device: device.clone(), semaphore })
    }

    pub(crate) fn vk_handle(&self) -> vk::Semaphore {
        self.semaphore
    }

    pub(crate) unsafe fn wait_timeline_value(&self, value: u64) {
        self.device
            .wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&[self.semaphore]).values(&[value]), u64::MAX)
            .unwrap();
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

pub(crate) struct GpuWorkExecutor {
    pool: ThreadPool,
}

impl GpuWorkExecutor {
    pub(crate) fn new() -> Self {
        Self {
            pool: ThreadPool::new().expect("failed to create GPU work executor"),
        }
    }

    pub(crate) fn wait_timeline_value<T: Send + 'static>(
        &self,
        semaphore: Arc<Semaphore>,
        value: u64,
        output: T,
    ) -> oneshot::Receiver<T> {
        let (tx, rx) = oneshot::channel();
        self.pool.spawn_ok(async move {
            unsafe {
                semaphore.wait_timeline_value(value);
            }
            let _ = tx.send(output);
        });
        rx
    }
}
