//! Frame-local landmark output derived from one detector bbox.

pub type Vec2 = [f32; 2];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LandmarkPoint {
    pub position: Vec2,
    pub confidence: f32,
}

impl LandmarkPoint {
    pub fn new(position: Vec2, confidence: f32) -> Self {
        Self {
            position: [position[0].clamp(0.0, 1.0), position[1].clamp(0.0, 1.0)],
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    pub fn is_finite(&self) -> bool {
        self.position[0].is_finite() && self.position[1].is_finite() && self.confidence.is_finite()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LandmarkResult {
    pub left_eye: LandmarkPoint,
    pub right_eye: LandmarkPoint,
    pub nose: LandmarkPoint,
    pub mouth_left: LandmarkPoint,
    pub mouth_right: LandmarkPoint,
    pub bbox: [f32; 4],
    pub detector_confidence: f32,
}

impl LandmarkResult {
    pub fn average_confidence(&self) -> f32 {
        let total = self.left_eye.confidence
            + self.right_eye.confidence
            + self.nose.confidence
            + self.mouth_left.confidence
            + self.mouth_right.confidence;
        (total / 5.0).clamp(0.0, 1.0)
    }

    pub fn eye_midpoint(&self) -> Vec2 {
        midpoint(self.left_eye.position, self.right_eye.position)
    }

    pub fn mouth_midpoint(&self) -> Vec2 {
        midpoint(self.mouth_left.position, self.mouth_right.position)
    }

    pub fn centroid_anchor(&self) -> Vec2 {
        let mut x = 0.0;
        let mut y = 0.0;
        let pts = [
            self.left_eye.position,
            self.right_eye.position,
            self.nose.position,
            self.mouth_left.position,
            self.mouth_right.position,
        ];
        for p in pts.iter() {
            x += p[0];
            y += p[1];
        }
        [x / 5.0, y / 5.0]
    }

    pub fn position_anchor(&self) -> Vec2 {
        // F4: Test 3 anchors: nose vs eye midpoint vs projected face center
        // Using projected face center (centroid) for maximum stability
        self.centroid_anchor()
    }

    pub fn bbox_width(&self) -> f32 {
        (self.bbox[2] - self.bbox[0]).max(0.0)
    }

    pub fn is_finite(&self) -> bool {
        self.bbox.iter().all(|value| value.is_finite())
            && self.detector_confidence.is_finite()
            && self.left_eye.is_finite()
            && self.right_eye.is_finite()
            && self.nose.is_finite()
            && self.mouth_left.is_finite()
            && self.mouth_right.is_finite()
    }
}

fn midpoint(a: Vec2, b: Vec2) -> Vec2 {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}
