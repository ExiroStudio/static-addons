//! src/runtime/gpu/mod.rs
//! GPU inference module entry point.

pub mod context;
pub mod postprocess;
pub mod preprocess;
pub mod session;

pub struct FramePayload {
    pub epoch: u64,
    pub tier_generation: u32,
    pub width: u32,
    pub height: u32,
    pub bytes: std::sync::Arc<[u8]>,
}

impl FramePayload {
    pub fn new(epoch: u64, gen: u32, w: u32, h: u32, bytes: std::sync::Arc<[u8]>) -> Self {
        Self {
            epoch,
            tier_generation: gen,
            width: w,
            height: h,
            bytes,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DetectionResult {
    pub frame_epoch: u64,
    pub tier_generation: u32,
    pub face: crate::runtime::gpu::postprocess::DecodedFace,
}

impl DetectionResult {
    pub fn is_valid(&self) -> bool {
        self.face.score > 0.0
    }
}
