//! src/runtime/pipeline/mod.rs
//! Detection submission and publish pipeline.

use crate::host::FrameRef;
use crate::runtime::detector::{
    Detector, DetectorMetrics, DetectorState, GpuDetector, ShutdownOutcome,
};
use crate::runtime::gpu::{DetectionResult, FramePayload};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AcceptedFrame {
    frame_epoch: u64,
    tier_generation: u32,
}

pub struct Pipeline {
    detector: GpuDetector,
    last_submit_time: f32,
    submit_interval: f32,
    latest_detection: Option<DetectionResult>,
    last_accepted_frame: Option<AcceptedFrame>,
    current_frame: Option<AcceptedFrame>,
}

impl Pipeline {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            detector: GpuDetector::init(),
            last_submit_time: 0.0,
            submit_interval: 1.0 / 30.0,
            latest_detection: None,
            last_accepted_frame: None,
            current_frame: None,
        }
    }

    pub fn start(&mut self) {
        self.latest_detection = None;
        self.last_accepted_frame = None;
        self.current_frame = None;
        self.last_submit_time = 0.0;
        self.detector.start();
    }

    pub fn stop(&mut self) -> ShutdownOutcome {
        self.latest_detection = None;
        self.last_accepted_frame = None;
        self.current_frame = None;
        self.last_submit_time = 0.0;
        self.detector.shutdown()
    }

    pub fn metrics(&self) -> DetectorMetrics {
        self.detector.metrics()
    }

    pub fn tick(
        &mut self,
        frame_ref: FrameRef,
        payload: Option<std::sync::Arc<[u8]>>,
        elapsed: f32,
    ) {
        if self.detector.state() == DetectorState::Cold {
            self.detector.start();
        }

        if self.detector.state() == DetectorState::Faulted {
            return;
        }

        self.current_frame = Some(AcceptedFrame {
            frame_epoch: frame_ref.epoch,
            tier_generation: frame_ref.tier_generation,
        });

        let _submitted = if elapsed - self.last_submit_time >= self.submit_interval {
            if let Some(data) = payload {
                let frame = FramePayload::new(
                    frame_ref.epoch,
                    frame_ref.tier_generation,
                    frame_ref.width,
                    frame_ref.height,
                    data,
                );
                self.detector.submit(frame);
                self.last_submit_time = elapsed;
                true
            } else {
                false
            }
        } else {
            false
        };

        self.drain_detector();
    }

    pub fn take_latest_detection(&mut self) -> Option<DetectionResult> {
        self.drain_detector();
        self.latest_detection.take()
    }

    fn drain_detector(&mut self) {
        let Some(frame_ref) = self.current_frame else {
            return;
        };

        if let Some(result) = self.detector.poll() {
            self.accept_result(frame_ref, result);
        }
    }

    fn accept_result(&mut self, frame_ref: AcceptedFrame, result: DetectionResult) {
        if result.tier_generation != frame_ref.tier_generation {
            self.detector.record_dropped();
            return;
        }

        let accepted_frame = AcceptedFrame {
            frame_epoch: result.frame_epoch,
            tier_generation: result.tier_generation,
        };

        if self.last_accepted_frame == Some(accepted_frame) {
            self.detector.record_dropped();
            return;
        }

        if self.latest_detection.is_some() {
            self.detector.record_dropped();
        }

        self.last_accepted_frame = Some(accepted_frame);
        self.latest_detection = Some(result);
    }
}

#[cfg(test)]
impl Pipeline {
    fn inject_result_for_test(&self, result: DetectionResult) {
        self.detector.inject_result_for_test(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{FrameTier, HostApi, MockHost};
    use std::sync::Arc;

    #[test]
    fn pipeline_delivers_latest_visibility() {
        let mut host = MockHost::new();
        host.begin_tick();
        assert!(host.request_frame(FrameTier::R640x360));
        let r1: crate::host::FrameRef = host.read_frame().unwrap();
        let v1: crate::host::FrameView = host.read_frame_view().unwrap();

        host.begin_tick();
        assert!(host.request_frame(FrameTier::R640x360));
        let r2: crate::host::FrameRef = host.read_frame().unwrap();
        let v2: crate::host::FrameView = host.read_frame_view().unwrap();

        let mut pipeline = Pipeline::new();
        pipeline.start();
        pipeline.tick(r1, v1.arc(), 0.0);
        pipeline.tick(r2, v2.arc(), 0.11);

        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(200);
        let detection = loop {
            if let Some(result) = pipeline.take_latest_detection() {
                break result;
            }
            if std::time::Instant::now() >= deadline {
                panic!("detector did not publish output in time");
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        };

        assert_eq!(detection.frame_epoch, r2.epoch);
        assert_eq!(detection.face.score, 0.0);

        let outcome = pipeline.stop();
        assert!(outcome.joined);
    }

    #[test]
    fn duplicate_provenance_publishes_once() {
        let mut pipeline = Pipeline::new();
        pipeline.start();

        let frame = FrameRef {
            epoch: 100,
            tier_generation: 7,
            width: 640,
            height: 360,
            tier: FrameTier::R640x360,
        };
        let payload: Arc<[u8]> = Arc::from(vec![0u8; 640 * 360 * 4].into_boxed_slice());
        let result = DetectionResult {
            frame_epoch: frame.epoch,
            tier_generation: frame.tier_generation,
            face: crate::runtime::gpu::postprocess::DecodedFace {
                score: 0.9,
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
                frame_id: frame.epoch,
                is_predicted: false,
            },
        };

        pipeline.inject_result_for_test(result.clone());
        pipeline.tick(frame, Some(payload.clone()), 0.11);
        assert_eq!(pipeline.take_latest_detection(), Some(result.clone()));

        pipeline.inject_result_for_test(result);
        pipeline.tick(frame, Some(payload), 0.22);
        assert!(
            pipeline.take_latest_detection().is_none(),
            "duplicate provenance must not publish twice"
        );

        let metrics = pipeline.metrics();
        assert!(
            metrics.dropped >= 1,
            "duplicate result should increment dropped metrics"
        );

        let outcome = pipeline.stop();
        assert!(outcome.joined);
    }

    #[test]
    fn shutdown_joins_worker_and_destroys_gpu() {
        let before = GpuDetector::active_workers_for_test();
        let mut pipeline = Pipeline::new();
        pipeline.start();

        let frame = FrameRef {
            epoch: 1,
            tier_generation: 1,
            width: 640,
            height: 360,
            tier: FrameTier::R640x360,
        };
        let payload: Arc<[u8]> = Arc::from(vec![0u8; 640 * 360 * 4].into_boxed_slice());
        pipeline.tick(frame, Some(payload), 0.11);

        let outcome = pipeline.stop();
        assert!(outcome.joined, "worker must join during stop");
        assert!(!outcome.timed_out, "normal shutdown must not time out");
        assert!(
            outcome.gpu_destroyed,
            "gpu context must be destroyed on stop"
        );
        assert_eq!(
            GpuDetector::active_workers_for_test(),
            before,
            "worker count must return to baseline"
        );
    }
}
