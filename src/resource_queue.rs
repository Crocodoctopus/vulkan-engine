use ash::vk;
use std::collections::VecDeque;
use std::iter::FromIterator;
use std::rc::Weak;

#[derive(Copy, Clone)]
pub(crate) enum WaitStrategy {
    Binary(vk::Semaphore),
    Timeline(vk::Semaphore, u64),
}

impl WaitStrategy {
    pub(crate) fn submit_info(self) -> vk::SemaphoreSubmitInfo<'static> {
        match self {
            WaitStrategy::Binary(semaphore) => {
                // TODO: Thread the actual submit stage through here instead of hard-coding TOP_OF_PIPE.
                vk::SemaphoreSubmitInfo::default().semaphore(semaphore).stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            }
            WaitStrategy::Timeline(semaphore, value) => vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore)
                .value(value)
                .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE),
        }
    }
}

#[derive(Clone)]
pub(crate) enum FenceRef {
    Strong(vk::Fence),
    Weak(Weak<vk::Fence>),
}

impl From<vk::Fence> for FenceRef {
    fn from(fence: vk::Fence) -> Self {
        Self::Strong(fence)
    }
}

impl From<Weak<vk::Fence>> for FenceRef {
    fn from(fence: Weak<vk::Fence>) -> Self {
        Self::Weak(fence)
    }
}

impl FenceRef {
    fn handle(&self) -> Option<vk::Fence> {
        match self {
            FenceRef::Strong(fence) => Some(*fence),
            FenceRef::Weak(fence) => fence.upgrade().map(|fence| *fence),
        }
    }

    unsafe fn should_drop(&self, device: &ash::Device) -> bool {
        self.handle().map(|f| device.get_fence_status(f).unwrap()).unwrap_or(true)
    }
}

pub(crate) struct HeadEntry<T> {
    pub t: T,
    pub wait: (FenceRef, WaitStrategy),
}

pub(crate) struct TailEntry<T> {
    pub t: T,
    pub waits: Vec<(FenceRef, WaitStrategy)>,
}

pub(crate) struct ResourceQueue<T> {
    head: VecDeque<HeadEntry<T>>,
    tail: VecDeque<TailEntry<T>>,
}

impl<T> ResourceQueue<T> {
    pub(crate) fn new<I>(items: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let items = items.into_iter();
        assert!(items.len() > 0);
        <Self as FromIterator<T>>::from_iter(items)
    }

    pub(crate) fn len(&self) -> usize {
        self.head.len() + self.tail.len()
    }

    fn collect_head_fences(&self) -> Vec<vk::Fence> {
        self.head.iter().filter_map(|entry| entry.wait.0.handle()).collect()
    }

    fn collect_tail_fences(&self) -> Vec<vk::Fence> {
        self.tail
            .iter()
            .flat_map(|entry| entry.waits.iter().filter_map(|(fence, _)| fence.handle()))
            .collect()
    }

    fn collect_all_fences(&self) -> Vec<vk::Fence> {
        let mut fences = self.collect_head_fences();
        fences.extend(self.collect_tail_fences());
        fences
    }

    pub(crate) unsafe fn update(&mut self, device: &ash::Device) {
        for entry in &mut self.tail {
            entry.waits.retain(|(fence, _)| !fence.should_drop(device));
        }

        // Move any completed writes into the readable tail, preserving recency.
        let mut index = self.head.len();
        while index > 0 {
            index -= 1;
            let ready = self
                .head
                .get(index)
                .map(|entry| entry.wait.0.should_drop(device))
                .unwrap_or(false);

            if ready {
                let entry = self.head.remove(index).unwrap();
                self.tail.push_back(TailEntry { t: entry.t, waits: Vec::new() });
            }
        }
    }

    pub(crate) unsafe fn free(mut self, device: &ash::Device) -> impl Iterator<Item = T> {        
        self.update(device);
        
        let fences = self.collect_all_fences();
        if !fences.is_empty() {
            device.wait_for_fences(&fences, true, u64::MAX).unwrap();
        }

        let Self { head, tail } = self;
        head.into_iter().map(|entry| entry.t).chain(tail.into_iter().map(|entry| entry.t))
    }

    pub(crate) unsafe fn read(
        &mut self,
        device: &ash::Device,
        wait_fence: impl Into<FenceRef>,
        wait_strategy: WaitStrategy,
    ) -> &mut T {
        self.update(device);

        if self.tail.is_empty() {
            let fences = self.collect_all_fences();
            assert!(!fences.is_empty());
            device.wait_for_fences(&fences, false, u64::MAX).unwrap();
            self.update(device);
        }

        let entry = self.tail.back_mut().unwrap();
        entry.waits.push((wait_fence.into(), wait_strategy));
        &mut entry.t
    }

    pub(crate) unsafe fn write(
        &mut self,
        device: &ash::Device,
        wait_fence: impl Into<FenceRef>,
        wait_strategy: WaitStrategy,
    ) -> (&mut T, Vec<vk::SemaphoreSubmitInfo<'static>>) {
        self.update(device);

        if self.tail.is_empty() {
            let fences = self.collect_all_fences();
            assert!(!fences.is_empty());
            device.wait_for_fences(&fences, false, u64::MAX).unwrap();
            self.update(device);
        }

        let TailEntry { t, waits } = self.tail.pop_front().unwrap();
        self.head.push_front(HeadEntry { t, wait: (wait_fence.into(), wait_strategy) });
        let entry = self.head.front_mut().unwrap();
        let submit_infos = waits.into_iter().map(|(_, wait_strategy)| wait_strategy.submit_info()).collect::<Vec<_>>();
        (&mut entry.t, submit_infos)
    }
}

impl<T> From<Vec<T>> for ResourceQueue<T> {
    fn from(items: Vec<T>) -> Self {
        <Self as FromIterator<T>>::from_iter(items)
    }
}

impl<T> From<Box<[T]>> for ResourceQueue<T> {
    fn from(items: Box<[T]>) -> Self {
        <Self as FromIterator<T>>::from_iter(items)
    }
}

impl<T, const N: usize> From<[T; N]> for ResourceQueue<T> {
    fn from(items: [T; N]) -> Self {
        <Self as FromIterator<T>>::from_iter(items)
    }
}

impl<T: Copy> From<&[T]> for ResourceQueue<T> {
    fn from(items: &[T]) -> Self {
        <Self as FromIterator<T>>::from_iter(items.iter().copied())
    }
}

impl<T> FromIterator<T> for ResourceQueue<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        let tail = items.into_iter().map(|t| TailEntry { t, waits: Vec::new() }).collect();
        Self { head: VecDeque::new(), tail }
    }
}
