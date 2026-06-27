//! Real landmark extraction from SCRFD model output.

use crate::runtime::gpu::DetectionResult;
use crate::runtime::landmarks::result::{LandmarkPoint, LandmarkResult};

const DETECTION_CONFIDENCE_MIN: f32 = 0.5;

pub struct LandmarkExtractor;

impl LandmarkExtractor {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self
    }

    pub fn extract(&self, detection: &DetectionResult) -> Option<LandmarkResult> {
        let face = &detection.face;

        if !face.score.is_finite() || face.score < DETECTION_CONFIDENCE_MIN {
            return None;
        }

        if face.bbox.iter().any(|value| !value.is_finite()) {
            return None;
        }

        let x0 = face.bbox[0].clamp(0.0, 1.0);
        let y0 = face.bbox[1].clamp(0.0, 1.0);
        let x1 = face.bbox[2].clamp(0.0, 1.0);
        let y1 = face.bbox[3].clamp(0.0, 1.0);

        let point = |pt: [f32; 2], confidence_scale: f32| {
            LandmarkPoint::new(
                [pt[0].clamp(0.0, 1.0), pt[1].clamp(0.0, 1.0)],
                face.score * confidence_scale,
            )
        };

        Some(LandmarkResult {
            left_eye: point(face.kps[0], 0.96),
            right_eye: point(face.kps[1], 0.96),
            nose: point(face.kps[2], 0.99),
            mouth_left: point(face.kps[3], 0.93),
            mouth_right: point(face.kps[4], 0.93),
            bbox: [x0, y0, x1, y1],
            detector_confidence: face.score,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::gpu::postprocess::DecodedFace;

    #[test]
    fn invalid_detector_returns_none() {
        let extractor = LandmarkExtractor::new();
        let detection = DetectionResult {
            frame_epoch: 1,
            tier_generation: 1,
            face: DecodedFace {
                score: 0.99,
                bbox: [0.2, 0.2, 0.8, 0.8],
                kps: [
                    [0.35, 0.35],
                    [0.65, 0.35],
                    [0.50, 0.50],
                    [0.40, 0.70],
                    [0.60, 0.70],
                ],
                capture_ts: 0,
                infer_ts: 0,
                frame_id: 1,
                is_predicted: false,
            },
        };

        assert!(extractor.extract(&detection).is_none());
    }
}
