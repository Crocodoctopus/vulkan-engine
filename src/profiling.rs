use crate::renderer::{MAX_FRAMES_IN_FLIGHT, PipelineStage};
use ash::vk;

pub(crate) struct PipelineProfiler {
    query_pool: vk::QueryPool,
    stage_accum_ms: [f32; PipelineStage::FrameEnd as usize],
    total_accum_ms: f32,
    samples: u32,
}

impl PipelineProfiler {
    const QUERIES_PER_FRAME: usize = 2 + PipelineStage::FrameEnd as usize * 2;

    pub(crate) unsafe fn new(device: &ash::Device) -> Self {
        let query_pool = device
            .create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count((MAX_FRAMES_IN_FLIGHT * Self::QUERIES_PER_FRAME) as u32),
                None,
            )
            .unwrap();

        Self {
            query_pool,
            stage_accum_ms: [0.0; PipelineStage::FrameEnd as usize],
            total_accum_ms: 0.0,
            samples: 0,
        }
    }

    const fn query_base(frame_index: usize) -> u32 {
        (frame_index * Self::QUERIES_PER_FRAME) as u32
    }

    const fn stage_query(frame_index: usize, stage: PipelineStage, end: bool) -> u32 {
        Self::query_base(frame_index) + 2 + stage as u32 * 2 + end as u32
    }

    pub(crate) unsafe fn reset_frame(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_index: usize) {
        device.cmd_reset_query_pool(
            cmd,
            self.query_pool,
            Self::query_base(frame_index),
            Self::QUERIES_PER_FRAME as u32,
        );
    }

    pub(crate) unsafe fn write_total_start(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_index: usize) {
        device.cmd_write_timestamp(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            self.query_pool,
            Self::query_base(frame_index),
        );
    }

    pub(crate) unsafe fn write_total_end(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_index: usize) {
        device.cmd_write_timestamp(
            cmd,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            self.query_pool,
            Self::query_base(frame_index) + 1,
        );
    }

    pub(crate) unsafe fn write_stage_start(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_index: usize,
        stage: PipelineStage,
    ) {
        device.cmd_write_timestamp(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            self.query_pool,
            Self::stage_query(frame_index, stage, false),
        );
    }

    pub(crate) unsafe fn write_stage_end(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_index: usize,
        stage: PipelineStage,
    ) {
        device.cmd_write_timestamp(
            cmd,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            self.query_pool,
            Self::stage_query(frame_index, stage, true),
        );
    }

    pub(crate) unsafe fn bootstrap_queries(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        device.cmd_reset_query_pool(cmd, self.query_pool, 0, (MAX_FRAMES_IN_FLIGHT * Self::QUERIES_PER_FRAME) as u32);
        for fif in 0..MAX_FRAMES_IN_FLIGHT {
            self.write_total_start(device, cmd, fif);
            self.write_total_end(device, cmd, fif);
            for stage in 0..PipelineStage::FrameEnd as usize {
                let query = Self::query_base(fif) + 2 + stage as u32 * 2;
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, self.query_pool, query);
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.query_pool, query + 1);
            }
        }
    }

    pub(crate) unsafe fn read_and_accumulate(
        &mut self,
        device: &ash::Device,
        frame_index: usize,
        timestamp_period: f32,
    ) {
        let mut data = [0u64; Self::QUERIES_PER_FRAME];
        device
            .get_query_pool_results(
                self.query_pool,
                Self::query_base(frame_index),
                &mut data,
                vk::QueryResultFlags::TYPE_64,
            )
            .unwrap();

        let to_ms = timestamp_period / 1_000_000.0;
        let elapsed_ms = |start: usize, end: usize| to_ms * (data[end] - data[start]) as f32;

        self.total_accum_ms += elapsed_ms(0, 1);
        for stage in 0..PipelineStage::FrameEnd as usize {
            let query = 2 + stage * 2;
            self.stage_accum_ms[stage] += elapsed_ms(query, query + 1);
        }

        self.samples += 1;
        if self.samples == 300 {
            let scale = 1.0 / self.samples as f32;
            let data_upload_ms = self.stage_accum_ms[PipelineStage::DataUpload as usize] * scale;
            let frustum_cull_ms = self.stage_accum_ms[PipelineStage::FrustumCull as usize] * scale;
            let first_draw_ms = self.stage_accum_ms[PipelineStage::FirstDraw as usize] * scale;
            let build_hzb_ms = self.stage_accum_ms[PipelineStage::BuildHzb as usize] * scale;
            let sum_ms = data_upload_ms + frustum_cull_ms + first_draw_ms + build_hzb_ms;
            let total_ms = self.total_accum_ms * scale;

            println!(
                "timestamp: real = {total_ms:.4}ms (sum = {sum_ms:.4}ms)\n  upload = {data_upload_ms:.4}ms\n  frustum = {frustum_cull_ms:.4}ms\n  draw = {first_draw_ms:.4}ms\n  hzb = {build_hzb_ms:.4}ms"
            );

            self.stage_accum_ms = [0.0; PipelineStage::FrameEnd as usize];
            self.total_accum_ms = 0.0;
            self.samples = 0;
        }
    }
}
