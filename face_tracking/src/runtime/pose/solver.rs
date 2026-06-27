//! Stateless geometric pose solve from one landmark set.

use crate::runtime::landmarks::result::{LandmarkResult, Vec2};
use crate::runtime::pose::output::Pose;

const EPSILON: f32 = 1.0e-6;
const EXPECTED_NOSE_RATIO: f32 = 0.4737;
const MAX_YAW_RADIANS: f32 = 1.2;
const MAX_PITCH_RADIANS: f32 = 1.0;

pub struct PoseSolver;

impl PoseSolver {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self
    }

    pub fn solve(&self, landmarks: &LandmarkResult) -> Option<Pose> {
        if !landmarks.is_finite() {
            return None;
        }

        let left_eye = landmarks.left_eye.position;
        let right_eye = landmarks.right_eye.position;
        let nose = landmarks.nose.position;
        let eye_mid = landmarks.eye_midpoint();
        let mouth_mid = landmarks.mouth_midpoint();
        let eye_vec = sub(right_eye, left_eye);
        let inter_eye = length(eye_vec);
        let face_height = length(sub(mouth_mid, eye_mid));
        let bbox_width = landmarks.bbox_width();

        if inter_eye <= EPSILON && bbox_width <= EPSILON {
            return None;
        }
        if face_height <= EPSILON {
            return None;
        }

        let scale = if inter_eye > EPSILON {
            inter_eye
        } else {
            bbox_width
        };
        let roll = eye_vec[1].atan2(eye_vec[0]);

        let nose_offset_x = if inter_eye > EPSILON {
            (nose[0] - eye_mid[0]) / (inter_eye * 0.5)
        } else {
            0.0
        };
        let yaw = (nose_offset_x * 0.85).clamp(-MAX_YAW_RADIANS, MAX_YAW_RADIANS);

        let nose_ratio = ((nose[1] - eye_mid[1]) / face_height).clamp(0.0, 1.0);
        let pitch =
            ((nose_ratio - EXPECTED_NOSE_RATIO) * 1.8).clamp(-MAX_PITCH_RADIANS, MAX_PITCH_RADIANS);

        let position = landmarks.position_anchor();

        let avg_confidence = landmarks.average_confidence();
        let symmetry = if inter_eye > EPSILON {
            let left_len = length(sub(nose, left_eye));
            let right_len = length(sub(nose, right_eye));
            1.0 - ((left_len - right_len).abs() / inter_eye).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mouth_alignment = if inter_eye > EPSILON {
            1.0 - ((mouth_mid[0] - eye_mid[0]).abs() / inter_eye).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let confidence =
            (avg_confidence * symmetry.max(0.0) * mouth_alignment.max(0.0)).clamp(0.0, 1.0);

        let pose = Pose {
            position: [position[0].clamp(0.0, 1.0), position[1].clamp(0.0, 1.0)],
            rotation: [pitch, yaw, roll],
            scale: scale.max(0.0),
            confidence,
        };

        if pose.is_finite() {
            Some(pose)
        } else {
            None
        }
    }
}

fn sub(a: Vec2, b: Vec2) -> Vec2 {
    [a[0] - b[0], a[1] - b[1]]
}

fn length(v: Vec2) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::landmarks::result::{LandmarkPoint, LandmarkResult};

    fn canonical_landmarks() -> LandmarkResult {
        LandmarkResult {
            left_eye: LandmarkPoint::new([0.35, 0.40], 0.95),
            right_eye: LandmarkPoint::new([0.65, 0.40], 0.95),
            nose: LandmarkPoint::new([0.50, 0.55], 0.98),
            mouth_left: LandmarkPoint::new([0.40, 0.72], 0.92),
            mouth_right: LandmarkPoint::new([0.60, 0.72], 0.92),
            bbox: [0.20, 0.20, 0.80, 0.90],
            detector_confidence: 0.95,
        }
    }

    fn rotate(point: Vec2, center: Vec2, angle: f32) -> Vec2 {
        let dx = point[0] - center[0];
        let dy = point[1] - center[1];
        let sin = angle.sin();
        let cos = angle.cos();
        [
            center[0] + dx * cos - dy * sin,
            center[1] + dx * sin + dy * cos,
        ]
    }

    #[test]
    fn symmetric_face_has_near_zero_yaw_and_roll() {
        let solver = PoseSolver::new();
        let pose = solver.solve(&canonical_landmarks()).expect("pose");

        assert!(pose.rotation[1].abs() < 0.05, "yaw should be near zero");
        assert!(pose.rotation[2].abs() < 0.05, "roll should be near zero");
    }

    #[test]
    fn rotated_landmarks_change_roll() {
        let solver = PoseSolver::new();
        let mut landmarks = canonical_landmarks();
        let center = [0.50, 0.55];
        let angle = 0.35;

        landmarks.left_eye.position = rotate(landmarks.left_eye.position, center, angle);
        landmarks.right_eye.position = rotate(landmarks.right_eye.position, center, angle);
        landmarks.nose.position = rotate(landmarks.nose.position, center, angle);
        landmarks.mouth_left.position = rotate(landmarks.mouth_left.position, center, angle);
        landmarks.mouth_right.position = rotate(landmarks.mouth_right.position, center, angle);

        let pose = solver.solve(&landmarks).expect("pose");
        assert!(
            pose.rotation[2].abs() > 0.20,
            "roll should change on rotated landmarks"
        );
    }

    #[test]
    fn eye_distance_changes_scale() {
        let solver = PoseSolver::new();
        let base = solver.solve(&canonical_landmarks()).expect("base pose");

        let mut larger = canonical_landmarks();
        larger.left_eye.position[0] = 0.30;
        larger.right_eye.position[0] = 0.70;
        let larger_pose = solver.solve(&larger).expect("larger pose");

        assert!(
            larger_pose.scale > base.scale,
            "inter-eye distance should drive scale"
        );
    }

    #[test]
    fn random_landmarks_never_emit_nan_or_inf() {
        let solver = PoseSolver::new();
        let mut seed = 7u64;

        for _ in 0..300 {
            let mut next = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 32) as u32 as f32) / (u32::MAX as f32)
            };

            let landmarks = LandmarkResult {
                left_eye: LandmarkPoint::new([next(), next()], next()),
                right_eye: LandmarkPoint::new([next(), next()], next()),
                nose: LandmarkPoint::new([next(), next()], next()),
                mouth_left: LandmarkPoint::new([next(), next()], next()),
                mouth_right: LandmarkPoint::new([next(), next()], next()),
                bbox: [0.1, 0.1, 0.9, 0.9],
                detector_confidence: 0.9,
            };

            if let Some(pose) = solver.solve(&landmarks) {
                assert!(pose.is_finite(), "pose must remain finite");
            }
        }
    }

    #[test]
    fn pose_is_repeatable_for_same_input() {
        let solver = PoseSolver::new();
        let landmarks = canonical_landmarks();

        let first = solver.solve(&landmarks).expect("first pose");
        let second = solver.solve(&landmarks).expect("second pose");

        assert_eq!(first, second);
    }
}
