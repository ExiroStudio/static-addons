//! src/runtime/filter/gate.rs
//! Visibility hysteresis logic.

pub struct VisibilityGate {
    enter_threshold: i32,
    leave_threshold: i32,
}

impl VisibilityGate {
    pub fn new(enter: i32, leave: i32) -> Self {
        Self {
            enter_threshold: enter,
            leave_threshold: leave,
        }
    }

    /// Update visibility state based on tracked status.
    /// Returns (new_visible, new_counter)
    pub fn update(&self, currently_visible: bool, tracked: bool, counter: i32) -> (bool, i32) {
        let mut next_counter = counter;
        let mut next_visible = currently_visible;

        if tracked {
            if !currently_visible {
                next_counter += 1;
                if next_counter >= self.enter_threshold {
                    next_visible = true;
                    next_counter = 0;
                }
            } else {
                next_counter = 0; // Reset counter while stable
            }
        } else if currently_visible {
            next_counter += 1;
            if next_counter >= self.leave_threshold {
                next_visible = false;
                next_counter = 0;
            }
        } else {
            next_counter = 0;
        }

        (next_visible, next_counter)
    }
}
