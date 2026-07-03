use ash::vk;
use std::collections::VecDeque;

#[derive(Copy, Clone)]
pub(crate) struct WaitStrategy {
    pub semaphore: vk::Semaphore,
    pub value: u64,
}

impl WaitStrategy {
    pub(crate) fn submit_info(self) -> vk::SemaphoreSubmitInfo<'static> {
        // TODO: Thread the actual submit stage through here instead of hard-coding TOP_OF_PIPE.
        vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(self.value)
            .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
    }
}

pub(crate) struct HeadEntry<T> {
    pub t: T,
    pub wait: WaitStrategy,
}

pub(crate) struct TailEntry<T> {
    pub t: T,
    pub waits: Vec<WaitStrategy>,
}

pub(crate) struct ResourceQueue<T> {
    heads: Box<[Option<HeadEntry<T>>]>,
    tails: VecDeque<TailEntry<T>>,
}

impl<T> ResourceQueue<T> {    
    pub(crate) fn new<I>(max_heads: usize, items: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let items = items.into_iter();
        assert!(items.len() > max_heads);
        let tail = items.into_iter().map(|t| TailEntry { t, waits: Vec::new() }).collect();
        let head = (0..max_heads).map(|_| None).collect();
        Self { heads: head, tails: tail }
    }

    pub(crate) fn len(&self) -> usize {
        self.heads.len() + self.tails.len()
    }

    pub(crate) unsafe fn update(&mut self, device: &ash::Device) {
        // Check for head demotes.
        for index in 0..self.heads.len() {
            let head = &mut self.heads[index];
            
            // Extract a head @ index, or skip.
            let Some(HeadEntry { wait: WaitStrategy { semaphore, value }, .. }) = head else {
                continue;
            };

            // If the semaphore isn't ready yet, skip.
            if device.get_semaphore_counter_value(*semaphore).unwrap() < *value {
                continue;
            };

            // Demote.
            let HeadEntry { t, .. } = std::mem::take(head).unwrap();
            self.tails.push_back(TailEntry { t, waits: vec![] });
        }

        // Update tails.
        for entry in &mut self.tails {
            entry.waits.retain(|wait| device.get_semaphore_counter_value(wait.semaphore).unwrap() < wait.value);
        }
    }

    pub(crate) unsafe fn free(mut self, device: &ash::Device) -> impl IntoIterator<Item = T> {
        self.update(device);

        let mut semaphores = vec![];
        let mut values = vec![];
        let mut items = vec![];
        for head in self.heads.into_iter().flatten() {
            semaphores.push(head.wait.semaphore);
            values.push(head.wait.value);
            items.push(head.t);
        }
        for tail in self.tails.into_iter() {
            for wait in tail.waits {
                semaphores.push(wait.semaphore);
                values.push(wait.value);
                
            }
            items.push(tail.t);
        }

        debug_assert_eq!(semaphores.len(), values.len());
        if semaphores.len() > 0 {
            device
                .wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values), u64::MAX)
                .unwrap();
        }

        items   
    }

    pub(crate) unsafe fn read(&mut self, device: &ash::Device, wait_strategy: WaitStrategy) -> &mut T {
        // Cycle.
        self.update(device);

        // NOTE: This is no longer necessary, there *must* be a tail at all times
        // If there is no tails, wait heads until one finishes.
        /*if self.tails.is_empty() {
            self.wait_any_heads(device);

            // Cycle finished head into tails.
            self.update(device);
        }*/

        // By this point, a tail must exist.
        let entry = self.tails.back_mut().unwrap();
        entry.waits.push(wait_strategy);
        &mut entry.t
    }

    pub(crate) unsafe fn write(
        &mut self,
        device: &ash::Device,
        index: usize,
        wait_strategy: WaitStrategy,
    ) -> Option<(&mut T, Vec<vk::SemaphoreSubmitInfo<'static>>)> {
        // Cycle.
        self.update(device);

        // If the head @ index exists, return none.
        if self.heads[index].is_some() {
            return None;
        }

        // A tail must exist.
        let TailEntry { t, waits } = self.tails.pop_front().unwrap();
        self.heads[index] = Some(HeadEntry { t, wait: wait_strategy });
        let submit_infos = waits.into_iter().map(|wait_strategy| wait_strategy.submit_info()).collect::<Vec<_>>();
        // Surely we can make this not ugly.
        Some((&mut self.heads.get_mut(index).unwrap().as_mut().unwrap().t, submit_infos)) 
    }
}
