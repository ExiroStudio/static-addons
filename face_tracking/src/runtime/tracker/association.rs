//! src/runtime/tracker/association.rs
//! Association scoring for face matching.

use super::state::TrackedFace;
use crate::runtime::pose::output::Pose;

pub struct AssociationConfig {
    pub max_pos_dist: f32, // normalized distance
    pub max_rot_dist: f32, // radians
    pub max_scale_dist: f32,
}

impl Default for AssociationConfig {
    fn default() -> Self {
        Self {
            max_pos_dist: 0.3,   // 30% of frame
            max_rot_dist: 0.5,   // ~30 degrees
            max_scale_dist: 0.2, // 20% scale change
        }
    }
}

pub fn compute_association_score(
    tracked: &TrackedFace,
    new_pose: &Pose,
    config: &AssociationConfig,
) -> f32 {
    // 1. Position score
    let dx = tracked.position[0] - new_pose.position[0];
    let dy = tracked.position[1] - new_pose.position[1];
    let dist = (dx * dx + dy * dy).sqrt();
    let pos_score = 1.0 - (dist / config.max_pos_dist).clamp(0.0, 1.0);

    // 2. Rotation score (average of yaw/pitch/roll diffs)
    let mut rot_diff_total = 0.0;
    for i in 0..3 {
        let diff = (tracked.rotation[i] - new_pose.rotation[i]).abs();
        rot_diff_total += diff;
    }
    let rot_avg_diff = rot_diff_total / 3.0;
    let rot_score = 1.0 - (rot_avg_diff / config.max_rot_dist).clamp(0.0, 1.0);

    // 3. Scale score
    let scale_diff = (tracked.scale - new_pose.scale).abs();
    let scale_score = 1.0 - (scale_diff / config.max_scale_dist).clamp(0.0, 1.0);

    // Final score is a weighted product/average
    // If any dimension is too far, the score drops rapidly
    pos_score * rot_score * scale_score
}
