//! src/runtime/tracker/tracker.rs
//! Main tracker implementation managing identity lifecycle.

use super::association::{compute_association_score, AssociationConfig};
use super::state::{TrackState, TrackedFace};
use crate::runtime::gpu::DetectionResult;
use crate::runtime::pose::output::Pose;

pub struct TrackerMetrics {
    pub state_changes: u64,
    pub lost_events: u64,
    pub avg_track_duration_ms: f32, // Simplified for now
}

pub struct Tracker {
    tracked: Option<TrackedFace>,
    config: AssociationConfig,
    metrics: TrackerMetrics,
    pub lost_timeout_s: f32,
    min_consecutive_frames: u32,
}

impl Tracker {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            tracked: None,
            config: AssociationConfig::default(),
            metrics: TrackerMetrics {
                state_changes: 0,
                lost_events: 0,
                avg_track_duration_ms: 0.0,
            },
            lost_timeout_s: 0.5,
            min_consecutive_frames: 3,
        }
    }

    pub fn update(&mut self, detection: Option<&DetectionResult>, pose: Option<&Pose>, now: f32) {
        match (&mut self.tracked, detection, pose) {
            // Case 1: Already tracking something, and we have a new detection+pose
            (Some(tracked), Some(det), Some(pose)) => {
                let score = compute_association_score(tracked, pose, &self.config);

                if score > 0.0 {
                    // Match found: update state
                    tracked.position = pose.position;
                    tracked.rotation = pose.rotation;
                    tracked.scale = pose.scale;
                    tracked.detector_conf = det.face.score;
                    tracked.pose_conf = pose.confidence;
                    tracked.association_score = score;
                    tracked.last_seen = now;
                    tracked.consecutive_frames += 1;
                    tracked.is_predicted = det.face.is_predicted;

                    if tracked.state == TrackState::Acquiring
                        && tracked.consecutive_frames >= self.min_consecutive_frames
                    {
                        tracked.state = TrackState::Tracking;
                        self.metrics.state_changes += 1;
                    }
                } else {
                    // No match: check for timeout of current track
                    self.check_timeout(now);
                }
            }

            // Case 2: Not tracking anything, and we have a new detection+pose
            (None, _, Some(pose)) => {
                // Potential new face
                self.tracked = Some(TrackedFace::new(
                    pose.position,
                    pose.rotation,
                    pose.scale,
                    detection.map(|d| d.face.score).unwrap_or(1.0),
                    pose.confidence,
                    now,
                    detection.map(|d| d.face.is_predicted).unwrap_or(false),
                ));
                self.metrics.state_changes += 1;
            }

            // Case 3: Tracking something but no new detection/pose this frame
            (Some(_), _, _) => {
                self.check_timeout(now);
            }

            // Case 4: No track, no new pose
            (None, _, _) => {}
        }
    }

    fn check_timeout(&mut self, now: f32) {
        if let Some(tracked) = &self.tracked {
            if now - tracked.last_seen > self.lost_timeout_s {
                self.tracked = None;
                self.metrics.lost_events += 1;
                self.metrics.state_changes += 1;
            }
        }
    }

    pub fn current_face(&self) -> Option<&TrackedFace> {
        self.tracked.as_ref()
    }

    pub fn metrics(&self) -> &TrackerMetrics {
        &self.metrics
    }
}
