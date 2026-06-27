//! src/signals/mod.rs
//! Signal publishing layer.

use crate::host::{HostApi, SignalValue};
use crate::runtime::pose::output::Pose;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedFace {
    pub position: [f32; 2],
    pub rotation: [f32; 3],
    pub scale: f32,
    pub visible: bool,
    pub confidence: f32,
}

impl PublishedFace {
    pub fn invisible() -> Self {
        Self::invisible_with_confidence(0.0)
    }

    pub fn invisible_with_confidence(confidence: f32) -> Self {
        Self {
            position: [0.5, 0.5],
            rotation: [0.0, 0.0, 0.0],
            scale: 0.0,
            visible: false,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    pub fn from_pose(pose: Pose) -> Self {
        Self {
            position: pose.position,
            rotation: pose.rotation,
            scale: pose.scale,
            visible: true,
            confidence: pose.confidence,
        }
    }

    pub fn from_tracked(tracked: &crate::runtime::tracker::TrackedFace) -> Self {
        use crate::runtime::tracker::state::TrackState;

        Self {
            position: tracked.position,
            rotation: tracked.rotation,
            scale: tracked.scale,
            // Only visible if in Tracking state (Acquiring/Lost are invisible)
            visible: tracked.state == TrackState::Tracking,
            confidence: tracked.total_confidence(),
        }
    }
}

#[derive(Default)]
pub struct Publisher {}

impl Publisher {
    pub fn new() -> Self {
        Self {}
    }

    pub fn publish_face_output(&mut self, host: &mut dyn HostApi, face: PublishedFace) {
        // Publish locked signals for v1.0
        host.publish("face.position", SignalValue::Vec2(face.position));
        host.publish("face.rotation", SignalValue::Vec3(face.rotation));
        host.publish("face.scale", SignalValue::F32(face.scale));
        host.publish("face.visible", SignalValue::Bool(face.visible));
        host.publish("face.confidence", SignalValue::F32(face.confidence));
    }
}
