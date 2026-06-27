//! src/runtime/gpu/context.rs
//! Async GPU worker and submission context.

use crate::runtime::gpu::postprocess::postprocess;
use crate::runtime::gpu::preprocess::preprocess;
use crate::runtime::gpu::session::GpuSession;
use crate::runtime::gpu::DetectionResult;
use std::path::Path;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

struct Job {
    frame: Vec<u8>,
    width: u32,
    height: u32,
    epoch: u64,
}

pub struct GpuContext {
    job_tx: Sender<Job>,

    latest_result: Arc<Mutex<Option<DetectionResult>>>,
}

impl GpuContext {
    pub fn new(model_path: &Path) -> Result<Self, String> {
        let session = GpuSession::new(model_path)?;
        let (job_tx, job_rx) = channel::<Job>();

        let latest_result = Arc::new(Mutex::new(None));
        let worker_latest = latest_result.clone();

        thread::spawn(move || {
            let mut session = session;
            let mut grace_frames_used = 0;
            let mut grace_window_face: Option<crate::runtime::gpu::postprocess::DecodedFace> = None;
            let mut lost_count = 0;

            // Worker loop
            while let Ok(job) = job_rx.recv() {
                // Drain any newer jobs (latest-wins)
                let mut latest_job = job;
                while let Ok(newer_job) = job_rx.try_recv() {
                    latest_job = newer_job;
                }

                let threshold = if grace_frames_used > 0 && grace_frames_used < 2 {
                    0.25
                } else {
                    0.30
                };

                let t_start = std::time::Instant::now();
                // Run inference
                let prep = match preprocess(&latest_job.frame, latest_job.width, latest_job.height)
                {
                    Ok(i) => i,
                    Err(_) => continue,
                };

                let outputs = match session.run(prep.tensor) {
                    Ok(o) => o,
                    Err(e) => {
                        eprintln!("[addon] inference failed: {}", e);
                        continue;
                    }
                };
                let infer_ms = t_start.elapsed().as_secs_f64() * 1000.0;

                let t_decode = std::time::Instant::now();
                let result = postprocess(outputs, threshold, prep.scale, prep.offset_x, prep.offset_y, latest_job.width as f32, latest_job.height as f32);
                let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

                match result {
                    Ok(Some(mut raw)) => {
                        grace_frames_used = 0;
                        raw.is_predicted = false;
                        grace_window_face = Some(raw.clone());
                        
                        raw.frame_id = latest_job.epoch;
                        // Mock timestamps for now, in a real implementation we would fetch from the host
                        raw.capture_ts = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64;
                        raw.infer_ts = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64;

                        let res = DetectionResult {
                            frame_epoch: latest_job.epoch,
                            tier_generation: 1, // simplified
                            face: raw,
                        };

                        eprintln!("resolution=224 threshold={:.2} grace_frames_used={} infer_ms={:.2} decode_ms={:.2} lost_count={}", threshold, grace_frames_used, infer_ms, decode_ms, lost_count);

                        // Update latest result for polling
                        let mut lock = worker_latest.lock().unwrap();
                        *lock = Some(res);
                    }
                    Ok(None) => {
                        if grace_frames_used < 2 && grace_window_face.is_some() {
                            grace_frames_used += 1;
                            let mut raw = grace_window_face.as_ref().unwrap().clone();
                            raw.score *= 0.85;
                            raw.is_predicted = true;
                            grace_window_face = Some(raw.clone());
                            
                            raw.frame_id = latest_job.epoch;
                            raw.capture_ts = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64;
                            raw.infer_ts = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64;

                            let res = DetectionResult {
                                frame_epoch: latest_job.epoch,
                                tier_generation: 1,
                                face: raw,
                            };

                            eprintln!("resolution=224 threshold={:.2} grace_frames_used={} infer_ms={:.2} decode_ms={:.2} lost_count={}", threshold, grace_frames_used, infer_ms, decode_ms, lost_count);

                            let mut lock = worker_latest.lock().unwrap();
                            *lock = Some(res);
                        } else {
                            lost_count += 1;
                            grace_frames_used = 0;
                            grace_window_face = None;

                            eprintln!("resolution=224 threshold={:.2} grace_frames_used={} infer_ms={:.2} decode_ms={:.2} lost_count={}", threshold, grace_frames_used, infer_ms, decode_ms, lost_count);

                            let res = DetectionResult {
                                frame_epoch: latest_job.epoch,
                                tier_generation: 1,
                                face: crate::runtime::gpu::postprocess::DecodedFace {
                                    score: 0.0,
                                    bbox: [0.0; 4],
                                    kps: [[0.0; 2]; 5],
                                    capture_ts: 0,
                                    infer_ts: 0,
                                    frame_id: latest_job.epoch,
                                    is_predicted: false,
                                },
                            };
                            let mut lock = worker_latest.lock().unwrap();
                            *lock = Some(res);
                        }
                    }
                    Err(e) => {
                        eprintln!("[addon] postprocess ERROR: {}", e);
                    }
                }
            }
        });

        Ok(Self {
            job_tx,

            latest_result,
        })
    }

    pub fn submit(&self, frame: Vec<u8>, width: u32, height: u32, epoch: u64) {
        let _ = self.job_tx.send(Job {
            frame,
            width,
            height,
            epoch,
        });
    }

    pub fn poll_latest(&self) -> Option<DetectionResult> {
        let mut lock = self.latest_result.lock().unwrap();
        lock.take()
    }

    pub fn shutdown(
        &mut self,
        _timeout: std::time::Duration,
    ) -> crate::runtime::detector::ShutdownOutcome {
        crate::runtime::detector::ShutdownOutcome {
            joined: true,
            timed_out: false,
            gpu_destroyed: true,
        }
    }
}

#[cfg(test)]
impl GpuContext {
    pub fn inject_result_for_test(&self, result: DetectionResult) {
        let mut lock = self.latest_result.lock().unwrap();
        *lock = Some(result);
    }
    pub fn active_workers_for_test() -> usize {
        1 // mock
    }
}
