//! src/runtime/filter/state.rs
//! Filter state storage for temporal stabilization.

use super::smoother::{DampedState, EmaState, OneEuroState};

#[derive(Clone, Debug, Default)]
pub struct FilterState {
    pub pos: [OneEuroState; 2],
    pub rot: [DampedState; 3],
    pub scale: EmaState,
    pub confidence: EmaState,
    pub hysteresis_counter: i32,
    pub visible: bool,
}

impl FilterState {
    pub fn reset(&mut self) {
        for p in &mut self.pos {
            p.reset();
        }
        for r in &mut self.rot {
            r.reset();
        }
        self.scale.reset();
        self.confidence.reset();
        self.hysteresis_counter = 0;
        self.visible = false;
    }
}
