//! Frame-local pose output derived from 5 landmarks.

pub type Vec2 = [f32; 2];
pub type Vec3 = [f32; 3];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub position: Vec2,
    pub rotation: Vec3,
    pub scale: f32,
    pub confidence: f32,
}

impl Pose {
    pub fn is_finite(&self) -> bool {
        self.position[0].is_finite()
            && self.position[1].is_finite()
            && self.rotation.iter().all(|value| value.is_finite())
            && self.scale.is_finite()
            && self.confidence.is_finite()
    }
}
