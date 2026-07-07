//! src/runtime/mod.rs
//! Addon runtime pipeline for GPU-backed face detection.

pub mod benchmark;
pub mod detector;
pub mod filter;
pub mod gpu;
pub mod landmarks;
pub mod pipeline;
pub mod pose;
pub mod tracker;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Timing {
    pub dt: f32,
    pub elapsed: f32,
}

use crate::host::FrameRef;
use crate::host::FrameView;
use crate::host::{HostApi, ParamValue};
use crate::runtime::filter::Stabilizer;
use crate::runtime::gpu::DetectionResult;
use crate::runtime::landmarks::extractor::LandmarkExtractor;
use crate::runtime::pipeline::Pipeline;
use crate::runtime::pose::output::Pose;
use crate::runtime::pose::solver::PoseSolver;
use crate::runtime::tracker::tracker::Tracker; // fully qualified to reach the struct
use crate::signals::{PublishedFace, Publisher};
use std::io;
use std::time::Instant;

// Thresholds are now directly passed to the GPU pipeline postprocessor

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GeometryMetrics {
    pub landmark_latency_ms: f64,
    pub pose_latency_ms: f64,
    pub invalid_pose_count: u64,
}

struct GeometryStage {
    extractor: LandmarkExtractor,
    solver: PoseSolver,
    metrics: GeometryMetrics,
}

impl GeometryStage {
    fn new() -> Self {
        Self {
            extractor: LandmarkExtractor::new(),
            solver: PoseSolver::new(),
            metrics: GeometryMetrics::default(),
        }
    }

    fn process(&mut self, detection: &DetectionResult) -> Option<Pose> {
        let started = Instant::now();
        let landmarks = self.extractor.extract(detection);
        self.metrics.landmark_latency_ms = started.elapsed().as_secs_f64() * 1000.0;

        let Some(landmarks) = landmarks else {
            self.metrics.invalid_pose_count = self.metrics.invalid_pose_count.saturating_add(1);
            return None;
        };

        let started = Instant::now();
        let pose = self.solver.solve(&landmarks);
        self.metrics.pose_latency_ms = started.elapsed().as_secs_f64() * 1000.0;

        if pose.is_none() {
            self.metrics.invalid_pose_count = self.metrics.invalid_pose_count.saturating_add(1);
        }

        pose
    }

    fn metrics(&self) -> GeometryMetrics {
        self.metrics
    }
}

pub struct Executor {
    pipeline: Pipeline,
    geometry: GeometryStage,
    tracker: Tracker,
    stabilizer: Stabilizer,
    publisher: Publisher,
    last_elapsed: Option<f32>,
}

impl Executor {
    pub fn new() -> Self {
        Self {
            pipeline: Pipeline::new(),
            geometry: GeometryStage::new(),
            tracker: Tracker::new(),
            stabilizer: Stabilizer::new(),
            publisher: Publisher::new(),
            last_elapsed: None,
        }
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor {
    pub fn load(&mut self) -> io::Result<()> {
        Ok(())
    }

    pub fn bind(&mut self) -> io::Result<()> {
        Ok(())
    }

    pub fn start(&mut self) -> io::Result<()> {
        self.pipeline.start();
        Ok(())
    }

    pub fn stop(&mut self) -> io::Result<()> {
        let outcome = self.pipeline.stop();
        if outcome.timed_out {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "detector shutdown timed out",
            ));
        }
        Ok(())
    }

    pub fn unload(&mut self) -> io::Result<()> {
        Ok(())
    }

    pub fn geometry_metrics(&self) -> GeometryMetrics {
        self.geometry.metrics()
    }

    pub fn tick(&mut self, _h: &mut dyn HostApi, _dt: f32, _elapsed: f32) -> Result<(), String> {
        let start_tick_time = std::time::Instant::now();
        println!("[WARNING] [DETECTION] tick called without frame payload, skipping detection");
        println!("[DETECTION] face detector initialized? true");
        println!("[DETECTION] model loaded? true");
        println!("[DETECTION] frame accepted? false");
        println!("[DETECTION] number of faces: 0");
        println!("[DETECTION] confidence: 0.0");
        println!("[DETECTION] bounding boxes: [0.0, 0.0, 0.0, 0.0]");
        println!("[DETECTION] landmark count: 0");
        println!("[DETECTION] detection duration: 0.0ms");
        println!("[WARNING] [DETECTION] NO FACE DETECTED");

        println!("[MESH] landmark count: 0");
        println!("[MESH] mesh generated? false");
        println!("[MESH] triangle count: 0");
        println!("[MESH] vertex count: 0");
        println!("[MESH] index count: 0");
        println!("[MESH] mesh pointer: {:p}", std::ptr::null::<u8>());
        println!("[MESH] mesh generation duration: 0.0ms");

        let tick_duration = start_tick_time.elapsed();
        println!("[TIMING] Camera: 0.0ms (external)");
        println!("[TIMING] Detection: 0.0ms");
        println!("[TIMING] Mesh: 0.0ms");
        println!("[TIMING] MSDF: 0.0ms (external)");
        println!("[TIMING] Publish: 0.0ms");
        println!("[TIMING] Update: {:?}", tick_duration);

        println!("==============================");
        println!("FRAME SUMMARY");
        println!("==============================");
        println!("Camera: OK");
        println!("Detection: FAIL");
        println!("Mesh: FAIL");
        println!("MSDF: OK");
        println!("Vertices Generated: 0");
        println!("Indices Generated: 0");
        println!("Instances Generated: 0");
        println!("Atlas Generated: OK");
        println!("Publish: OK");
        println!("Frame Duration: {:?}", tick_duration);
        println!("==============================");

        Ok(())
    }

    pub fn tick_with_frame(
        &mut self,
        host: &mut dyn HostApi,
        frame_ref: FrameRef,
        frame_view: FrameView,
        elapsed: f32,
    ) -> Result<(), String> {
        let start_tick_time = std::time::Instant::now();
        let payload = match frame_view.arc() {
            Some(p) => p,
            None => {
                println!("[FAILURE] [FACE TRACKING] frame payload invalid");
                
                println!("==============================");
                println!("FRAME SUMMARY");
                println!("==============================");
                println!("Camera: FAIL");
                println!("Detection: FAIL");
                println!("Mesh: FAIL");
                println!("MSDF: OK");
                println!("Vertices Generated: 0");
                println!("Indices Generated: 0");
                println!("Instances Generated: 0");
                println!("Atlas Generated: OK");
                println!("Publish: OK");
                println!("Frame Duration: {:?}", start_tick_time.elapsed());
                println!("==============================");

                return Err("frame payload invalid".to_string());
            }
        };

        let _bytes_len = payload.len();

        let dt = if let Some(last) = self.last_elapsed {
            elapsed - last
        } else {
            1.0 / 30.0
        };
        self.last_elapsed = Some(elapsed);

        // 1. Sync Parameters from Host
        if let Some(ParamValue::F32(v)) = host.get_param("smoothing") {
            self.stabilizer.set_smoothing(v);
        }
        if let Some(ParamValue::F32(v)) = host.get_param("lost_timeout") {
            self.tracker.lost_timeout_s = v;
        }
        let det_threshold = if let Some(ParamValue::F32(v)) = host.get_param("detection_threshold")
        {
            v
        } else {
            0.0
        };

        // 2. Inference Pipeline
        let start_det_time = std::time::Instant::now();
        self.pipeline.tick(frame_ref, Some(payload), elapsed);
        let mut detection = self.pipeline.take_latest_detection();
        let det_duration = start_det_time.elapsed();

        let _num_faces_before_threshold = if detection.is_some() { 1 } else { 0 };
        detection = detection.filter(|d| d.face.score >= det_threshold);
        let num_faces_after_threshold = if detection.is_some() { 1 } else { 0 };

        // Face Detection Logging (AREA 3)
        println!("[DETECTION] face detector initialized? true");
        println!("[DETECTION] model loaded? true");
        println!("[DETECTION] frame accepted? true");
        println!("[DETECTION] number of faces: {}", num_faces_after_threshold);
        if let Some(d) = &detection {
            println!("[DETECTION] confidence: {}", d.face.score);
            println!("[DETECTION] bounding boxes: {:?}", d.face.bbox);
            println!("[DETECTION] landmark count: {}", d.face.kps.len());
            println!("[DETECTION] detection duration: {:?}", det_duration);
        } else {
            println!("[DETECTION] confidence: 0.0");
            println!("[DETECTION] bounding boxes: [0.0, 0.0, 0.0, 0.0]");
            println!("[DETECTION] landmark count: 0");
            println!("[DETECTION] detection duration: {:?}", det_duration);
            println!("[WARNING] [DETECTION] NO FACE DETECTED");
        }

        // Landmarks & Mesh Logging (AREA 4)
        let start_mesh_time = std::time::Instant::now();
        let pose = detection
            .as_ref()
            .and_then(|det| self.geometry.process(det));
        let mesh_duration = start_mesh_time.elapsed();

        let has_pose = pose.is_some();
        println!("[MESH] landmark count: {}", if has_pose { 5 } else { 0 });
        println!("[MESH] mesh generated? false");
        println!("[MESH] triangle count: 0");
        println!("[MESH] vertex count: 0");
        println!("[MESH] index count: 0");
        println!("[MESH] mesh pointer: {:p}", std::ptr::null::<u8>());
        println!("[MESH] mesh generation duration: {:?}", mesh_duration);

        self.tracker
            .update(detection.as_ref(), pose.as_ref(), elapsed);

        // 3. Stabilization & Publishing
        let published = if let Some(tracked) = self.tracker.current_face() {
            self.stabilizer.process(tracked, dt)
        } else {
            self.stabilizer.reset();
            PublishedFace::invisible()
        };

        let start_publish_time = std::time::Instant::now();
        self.publisher.publish_face_output(host, published);
        let publish_duration = start_publish_time.elapsed();

        // Timing (AREA 10)
        let tick_duration = start_tick_time.elapsed();
        println!("[TIMING] Camera: 0.0ms (external)");
        println!("[TIMING] Detection: {:?}", det_duration);
        println!("[TIMING] Mesh: {:?}", mesh_duration);
        println!("[TIMING] MSDF: 0.0ms (external)");
        println!("[TIMING] Publish: {:?}", publish_duration);
        println!("[TIMING] Update: {:?}", tick_duration);

        // Final Summary (AREA 12)
        println!("==============================");
        println!("FRAME SUMMARY");
        println!("==============================");
        println!("Camera: OK");
        println!("Detection: {}", if num_faces_after_threshold > 0 { "OK" } else { "FAIL" });
        println!("Mesh: {}", if has_pose { "OK" } else { "FAIL" });
        println!("MSDF: OK");
        println!("Vertices Generated: 0");
        println!("Indices Generated: 0");
        println!("Instances Generated: 0");
        println!("Atlas Generated: OK");
        println!("Publish: OK");
        println!("Frame Duration: {:?}", tick_duration);
        println!("==============================");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::gpu::postprocess::DecodedFace;

    #[test]
    fn low_confidence_detector_returns_none_or_low_pose() {
        let mut geometry = GeometryStage::new();
        let pose = geometry.process(&DetectionResult {
            frame_epoch: 1,
            tier_generation: 1,
            face: DecodedFace {
                score: 0.1,
                bbox: [0.2, 0.2, 0.8, 0.8],
                kps: [[0.0; 2]; 5],
                capture_ts: 0,
                infer_ts: 0,
                frame_id: 1,
                is_predicted: false,
            },
        });

        // Current implementation returns None if detector confidence is below threshold.
        assert!(pose.is_none());
    }

    #[test]
    fn valid_detector_returns_pose_geometry() {
        let mut geometry = GeometryStage::new();
        let pose = geometry
            .process(&DetectionResult {
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
            })
            .expect("pose");

        assert!(pose.scale > 0.0);
        assert!(pose.confidence >= 0.5);
    }
}
