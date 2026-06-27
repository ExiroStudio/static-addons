//! src/runtime/filter/smoother.rs
//! Mathematical filter implementations.

use std::f32::consts::PI;

// --- One Euro Filter (Position) ---

#[derive(Clone, Debug, Default)]
pub struct OneEuroState {
    pub x_prev: Option<f32>,
    pub dx_prev: f32,
}

impl OneEuroState {
    pub fn reset(&mut self) {
        self.x_prev = None;
        self.dx_prev = 0.0;
    }
}

pub struct OneEuroFilter {
    pub min_cutoff: f32,
    pub beta: f32,
    pub d_cutoff: f32,
}

impl OneEuroFilter {
    pub fn new(min_cutoff: f32, beta: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff: 1.0,
        }
    }

    pub fn update(&self, state: &mut OneEuroState, x: f32, dt: f32) -> f32 {
        if dt <= 0.0 {
            return x;
        }
        let freq = 1.0 / dt;

        let x_prev = match state.x_prev {
            Some(v) => v,
            None => {
                state.x_prev = Some(x);
                return x;
            }
        };

        let dx = (x - x_prev) * freq;
        let edx = low_pass(dx, state.dx_prev, alpha(freq, self.d_cutoff));
        state.dx_prev = edx;

        let cutoff = self.min_cutoff + self.beta * edx.abs();
        let ex = low_pass(x, x_prev, alpha(freq, cutoff));
        state.x_prev = Some(ex);

        ex
    }
}

// --- Critically Damped Smoothing (Rotation) ---

#[derive(Clone, Debug, Default)]
pub struct DampedState {
    pub x: f32,
    pub v: f32,
    pub initialized: bool,
}

impl DampedState {
    pub fn reset(&mut self) {
        self.initialized = false;
        self.v = 0.0;
    }
}

pub struct DampedSmoother {
    pub omega: f32,
}

impl DampedSmoother {
    pub fn new(omega: f32) -> Self {
        Self { omega }
    }

    pub fn update(&self, state: &mut DampedState, target: f32, dt: f32) -> f32 {
        if !state.initialized {
            state.x = target;
            state.v = 0.0;
            state.initialized = true;
            return target;
        }

        // Angle-safe wrapping
        let mut diff = target - state.x;
        while diff > PI {
            diff -= 2.0 * PI;
        }
        while diff < -PI {
            diff += 2.0 * PI;
        }

        let accel = self.omega * self.omega * diff - 2.0 * self.omega * state.v;
        state.v += accel * dt;
        state.x += state.v * dt;

        // Keep state.x in -PI..PI
        while state.x > PI {
            state.x -= 2.0 * PI;
        }
        while state.x < -PI {
            state.x += 2.0 * PI;
        }

        state.x
    }
}

// --- EMA Filter (Scale & Confidence) ---

#[derive(Clone, Debug, Default)]
pub struct EmaState {
    pub value: f32,
    pub initialized: bool,
}

impl EmaState {
    pub fn reset(&mut self) {
        self.initialized = false;
    }
}

pub struct EmaFilter {
    pub alpha_grow: f32,
    pub alpha_shrink: f32,
}

impl EmaFilter {
    pub fn new(grow: f32, shrink: f32) -> Self {
        Self {
            alpha_grow: grow,
            alpha_shrink: shrink,
        }
    }

    pub fn update(&self, state: &mut EmaState, x: f32) -> f32 {
        if !state.initialized {
            state.value = x;
            state.initialized = true;
            return x;
        }

        let a = if x > state.value {
            self.alpha_grow
        } else {
            self.alpha_shrink
        };
        state.value = a * x + (1.0 - a) * state.value;
        state.value
    }
}

// --- Helpers ---

fn alpha(freq: f32, cutoff: f32) -> f32 {
    let tau = 1.0 / (2.0 * PI * cutoff);
    1.0 / (1.0 + freq * tau)
}

fn low_pass(x: f32, x_prev: f32, alpha: f32) -> f32 {
    alpha * x + (1.0 - alpha) * x_prev
}
