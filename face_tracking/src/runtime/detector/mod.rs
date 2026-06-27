//! src/runtime/detector/mod.rs
//! Detector interface and GPU-backed detector implementation.

use crate::runtime::gpu::context::GpuContext;
use crate::runtime::gpu::{DetectionResult, FramePayload};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub trait Detector {
    fn init() -> Self
    where
        Self: Sized;

    fn start(&mut self);

    fn submit(&self, frame: FramePayload);

    fn poll(&self) -> Option<DetectionResult>;

    fn shutdown(&mut self) -> ShutdownOutcome;

    fn state(&self) -> DetectorState;

    fn record_dropped(&self);

    fn metrics(&self) -> DetectorMetrics;
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DetectorMetrics {
    pub submitted: u64,
    pub completed: u64,
    pub dropped: u64,
    pub timeouts: u64,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectorState {
    Cold,
    Warming,
    Ready,
    Faulted,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShutdownOutcome {
    pub joined: bool,
    pub timed_out: bool,
    pub gpu_destroyed: bool,
}

const LATENCY_BUCKET_MS: u64 = 5;
const LATENCY_BUCKETS: usize = 401;

#[derive(Debug)]
struct LatencyStats {
    total_latency_micros: u64,
    histogram: [u64; LATENCY_BUCKETS],
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            total_latency_micros: 0,
            histogram: [0; LATENCY_BUCKETS],
        }
    }
}

impl LatencyStats {
    fn record(&mut self, latency: Duration) {
        let micros = latency.as_micros().min(u64::MAX as u128) as u64;
        let millis = micros / 1_000;
        let bucket = (millis / LATENCY_BUCKET_MS).min((LATENCY_BUCKETS - 1) as u64) as usize;

        self.total_latency_micros = self.total_latency_micros.saturating_add(micros);
        self.histogram[bucket] = self.histogram[bucket].saturating_add(1);
    }

    fn avg_ms(&self, completed: u64) -> f64 {
        if completed == 0 {
            0.0
        } else {
            (self.total_latency_micros as f64 / completed as f64) / 1_000.0
        }
    }

    fn p95_ms(&self, completed: u64) -> f64 {
        if completed == 0 {
            return 0.0;
        }

        let target = (completed.saturating_mul(95).saturating_add(99)) / 100;
        let mut seen = 0u64;

        for (idx, count) in self.histogram.iter().enumerate() {
            seen = seen.saturating_add(*count);
            if seen >= target {
                return (idx as u64 * LATENCY_BUCKET_MS) as f64;
            }
        }

        ((LATENCY_BUCKETS - 1) as u64 * LATENCY_BUCKET_MS) as f64
    }
}

#[derive(Debug, Default)]
pub(crate) struct MetricsState {
    submitted: AtomicU64,
    completed: AtomicU64,
    dropped: AtomicU64,
    timeouts: AtomicU64,
    latencies: Mutex<LatencyStats>,
}

impl MetricsState {
    pub(crate) fn record_submit(&self) {
        self.submitted.fetch_add(1, Ordering::SeqCst);
    }

    pub(crate) fn record_complete(&self, latency: Duration) {
        self.completed.fetch_add(1, Ordering::SeqCst);
        self.latencies.lock().unwrap().record(latency);
    }

    pub(crate) fn record_drop(&self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }

    pub(crate) fn record_timeout(&self) {
        self.timeouts.fetch_add(1, Ordering::SeqCst);
    }

    pub(crate) fn snapshot(&self) -> DetectorMetrics {
        let submitted = self.submitted.load(Ordering::SeqCst);
        let completed = self.completed.load(Ordering::SeqCst);
        let dropped = self.dropped.load(Ordering::SeqCst);
        let timeouts = self.timeouts.load(Ordering::SeqCst);
        let latencies = self.latencies.lock().unwrap();

        DetectorMetrics {
            submitted,
            completed,
            dropped,
            timeouts,
            avg_latency_ms: latencies.avg_ms(completed),
            p95_latency_ms: latencies.p95_ms(completed),
        }
    }
}

pub struct GpuDetector {
    ctx: Option<GpuContext>,
    state: Arc<Mutex<DetectorState>>,
    shutdown_timeout: Duration,
    metrics: Arc<MetricsState>,
}

impl GpuDetector {
    pub fn new_with_shutdown_timeout(timeout: Duration) -> Self {
        Self {
            ctx: None,
            state: Arc::new(Mutex::new(DetectorState::Cold)),
            shutdown_timeout: timeout,
            metrics: Arc::new(MetricsState::default()),
        }
    }

    fn ensure_context(&mut self) {
        if self.ctx.is_none() {
            let model_path =
                std::path::Path::new("addons/face_tracking/models/scrfd_500m_bnkps.onnx");
            match GpuContext::new(model_path) {
                Ok(ctx) => self.ctx = Some(ctx),
                Err(e) => {
                    eprintln!(
                        "[addon] FATAL: failed to load model {}: {}",
                        model_path.display(),
                        e
                    );
                    let mut state = self.state.lock().unwrap();
                    *state = DetectorState::Faulted;
                    // Log error if possible or handle faulted state
                }
            }
        }
    }
}

impl Detector for GpuDetector {
    fn init() -> Self {
        Self::new_with_shutdown_timeout(Duration::from_secs(2))
    }

    fn start(&mut self) {
        if self.ctx.is_none() {
            self.ensure_context();
        }

        let mut state = self.state.lock().unwrap();
        if *state != DetectorState::Ready {
            *state = DetectorState::Warming;
        }
    }

    fn submit(&self, frame: FramePayload) {
        if let Some(ctx) = self.ctx.as_ref() {
            ctx.submit(frame.bytes.to_vec(), frame.width, frame.height, frame.epoch);
            self.metrics.record_submit();
        } else {
            self.metrics.record_drop();
        }
    }

    fn poll(&self) -> Option<DetectionResult> {
        if let Some(ctx) = self.ctx.as_ref() {
            if let Some(result) = ctx.poll_latest() {
                let mut state = self.state.lock().unwrap();
                if *state == DetectorState::Warming {
                    *state = DetectorState::Ready;
                }
                self.metrics
                    .record_complete(std::time::Duration::from_millis(10)); // approximate
                return Some(result);
            }
        }
        None
    }

    fn shutdown(&mut self) -> ShutdownOutcome {
        let outcome = if let Some(mut ctx) = self.ctx.take() {
            let outcome = ctx.shutdown(self.shutdown_timeout);
            if outcome.timed_out {
                self.metrics.record_timeout();
            }
            outcome
        } else {
            ShutdownOutcome {
                joined: true,
                timed_out: false,
                gpu_destroyed: true,
            }
        };

        let mut state = self.state.lock().unwrap();
        *state = if outcome.timed_out {
            DetectorState::Faulted
        } else {
            DetectorState::Cold
        };

        outcome
    }

    fn state(&self) -> DetectorState {
        *self.state.lock().unwrap()
    }

    fn record_dropped(&self) {
        self.metrics.record_drop();
    }

    fn metrics(&self) -> DetectorMetrics {
        self.metrics.snapshot()
    }
}

#[cfg(test)]
impl GpuDetector {
    pub fn inject_result_for_test(&self, result: DetectionResult) {
        if let Some(ctx) = self.ctx.as_ref() {
            ctx.inject_result_for_test(result);
        }
    }

    pub fn active_workers_for_test() -> usize {
        GpuContext::active_workers_for_test()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_metrics_snapshot_is_monotonic() {
        let metrics = MetricsState::default();

        metrics.record_submit();
        metrics.record_submit();
        metrics.record_complete(Duration::from_millis(4));
        metrics.record_complete(Duration::from_millis(12));
        metrics.record_drop();
        metrics.record_timeout();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.submitted, 2);
        assert_eq!(snapshot.completed, 2);
        assert_eq!(snapshot.dropped, 1);
        assert_eq!(snapshot.timeouts, 1);
        assert!(snapshot.avg_latency_ms >= 4.0);
        assert!(snapshot.p95_latency_ms >= snapshot.avg_latency_ms);
    }
}
