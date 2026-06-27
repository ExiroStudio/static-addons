//! src/runtime/tracker/state.rs
//! Tracker states and tracked face representation.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackState {
    Lost,
    Acquiring,
    Tracking,
}

#[derive(Clone, Debug)]
pub struct TrackedFace {
    pub position: [f32; 2],
    pub rotation: [f32; 3],
    pub scale: f32,
    pub detector_conf: f32,
    pub pose_conf: f32,
    pub association_score: f32,
    pub state: TrackState,
    pub consecutive_frames: u32,
    pub last_seen: f32, // elapsed time in seconds
    pub is_predicted: bool,
}

impl TrackedFace {
    pub fn new(
        pos: [f32; 2],
        rot: [f32; 3],
        scale: f32,
        det_conf: f32,
        pose_conf: f32,
        now: f32,
        is_predicted: bool,
    ) -> Self {
        Self {
            position: pos,
            rotation: rot,
            scale,
            detector_conf: det_conf,
            pose_conf,
            association_score: 1.0, // initial detection is 1.0
            state: TrackState::Acquiring,
            consecutive_frames: 1,
            last_seen: now,
            is_predicted,
        }
    }

    pub fn total_confidence(&self) -> f32 {
        self.detector_conf * self.pose_conf * self.association_score
    }
}
