//! src/runtime/tracker/mod.rs
//! Identity tracking module.

pub mod association;
pub mod state;
#[allow(clippy::module_inception)]
pub mod tracker;

pub use state::{TrackState, TrackedFace};
pub use tracker::{Tracker, TrackerMetrics};
