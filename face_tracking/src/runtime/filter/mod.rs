//! src/runtime/filter/mod.rs
//! Stabilization pipeline orchestration.

pub mod gate;
pub mod smoother;
pub mod state;

use crate::runtime::tracker::state::{TrackState, TrackedFace};
use crate::signals::PublishedFace;
use gate::VisibilityGate;
use smoother::{DampedSmoother, EmaFilter, OneEuroFilter};
use state::FilterState;

pub struct Stabilizer {
    state: FilterState,
    gate: VisibilityGate,
    pos_filter: OneEuroFilter,
    rot_filter: DampedSmoother,
    scale_filter: EmaFilter,
    conf_filter: EmaFilter,
}

impl Stabilizer {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            state: FilterState::default(),
            gate: VisibilityGate::new(3, 2),
            pos_filter: OneEuroFilter::new(1.0, 0.01),
            rot_filter: DampedSmoother::new(15.0),
            scale_filter: EmaFilter::new(0.4, 0.1),
            conf_filter: EmaFilter::new(0.2, 0.2),
        }
    }

    pub fn set_smoothing(&mut self, smoothing: f32) {
        // smoothing 1.0 -> more stable, less jitter
        // smoothing 0.0 -> raw, high jitter

        // One Euro beta: high beta = more jitter.
        // We want beta to be LOW when smoothing is HIGH.
        let beta = 0.1 * (1.0 - smoothing).max(0.001);
        self.pos_filter.beta = beta;
        self.pos_filter.min_cutoff = 1.0;

        // Damped omega: low omega = more stable/slower.
        // We want omega to be LOW when smoothing is HIGH.
        let omega = 30.0 * (1.1 - smoothing).max(0.1);
        self.rot_filter.omega = omega;
    }

    pub fn process(&mut self, tracked: &TrackedFace, dt: f32) -> PublishedFace {
        // 1. Handle Reset Rules
        match tracked.state {
            TrackState::Lost => {
                self.state.reset();
                return PublishedFace::invisible();
            }
            TrackState::Acquiring => {
                // Pre-warm filters with current raw values but remain invisible
                self.apply_filters(tracked, dt);
            }
            TrackState::Tracking => {
                // Apply stabilization
                self.apply_filters(tracked, dt);
            }
        }

        // 2. Apply Visibility Hysteresis
        let is_tracked = tracked.state == TrackState::Tracking;
        let (next_visible, next_counter) = self.gate.update(
            self.state.visible,
            is_tracked,
            self.state.hysteresis_counter,
        );
        self.state.visible = next_visible;
        self.state.hysteresis_counter = next_counter;

        // 3. Construct Output
        // Note: Even if invisible, we return the filtered position for a "warm" start
        // but the 'visible' flag in PublishedFace determines if it's sent to the engine.

        // Confidence cap: filtered <= raw
        let raw_conf = tracked.total_confidence();
        let filtered_conf = self.state.confidence.value.min(raw_conf);

        // Position: Bypass smoothing completely (instant position)
        let published_position = [tracked.position[0], tracked.position[1]];

        // Rotation: Export smoothed signal
        let published_rotation = [
            self.state.rot[0].x,
            self.state.rot[1].x,
            self.state.rot[2].x,
        ];

        eprintln!(
            "pitch={:.3} yaw={:.3} roll={:.3} conf={:.2} is_predicted={}",
            published_rotation[0], published_rotation[1], published_rotation[2],
            filtered_conf, tracked.is_predicted
        );

        PublishedFace {
            position: published_position,
            rotation: published_rotation,
            scale: self.state.scale.value,
            visible: self.state.visible,
            confidence: filtered_conf,
        }
    }

    fn apply_filters(&mut self, tracked: &TrackedFace, dt: f32) {
        // Position
        self.state.pos[0].x_prev = Some(self.pos_filter.update(
            &mut self.state.pos[0],
            tracked.position[0],
            dt,
        ));
        self.state.pos[1].x_prev = Some(self.pos_filter.update(
            &mut self.state.pos[1],
            tracked.position[1],
            dt,
        ));

        // Rotation
        for i in 0..3 {
            self.rot_filter
                .update(&mut self.state.rot[i], tracked.rotation[i], dt);
        }

        // Scale
        self.scale_filter
            .update(&mut self.state.scale, tracked.scale);

        // Confidence
        self.conf_filter
            .update(&mut self.state.confidence, tracked.total_confidence());
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }
}
