//! src/host/mod.rs
//! HostApi abstractions for external addons.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameTier {
    R160x90,
    R320x180,
    R640x360,
    FullRes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRef {
    pub epoch: u64,
    pub tier_generation: u32,
    pub width: u32,
    pub height: u32,
    pub tier: FrameTier,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SignalValue {
    Bool(bool),
    F32(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    F32(f32),
    Bool(bool),
}

/// A borrow-like payload view. Metadata is copy-safe across ticks; payload
/// bytes expire deterministically at the next tick boundary.
#[derive(Clone)]
pub struct FrameView {
    bytes: std::sync::Arc<[u8]>,
    valid_tick: u64,
    current_tick: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl FrameView {
    pub fn bytes(&self) -> Option<&[u8]> {
        let current = self.current_tick.load(std::sync::atomic::Ordering::SeqCst);
        if current != self.valid_tick {
            None
        } else {
            Some(&self.bytes)
        }
    }

    pub fn arc(&self) -> Option<std::sync::Arc<[u8]>> {
        let current = self.current_tick.load(std::sync::atomic::Ordering::SeqCst);
        if current != self.valid_tick {
            None
        } else {
            Some(self.bytes.clone())
        }
    }

    /// Construct a FrameView from raw bytes. The view is immediately valid —
    /// `bytes()` and `arc()` will return `Some` until the next tick boundary
    /// (which for FFI-bridged addons is managed by the caller).
    pub fn from_raw(bytes: std::sync::Arc<[u8]>) -> Self {
        let tick = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        Self {
            bytes,
            valid_tick: 0,
            current_tick: tick,
        }
    }
}

/// The capability surface an addon sees.
pub trait HostApi {
    /// Subscribe to camera frames at `tier`.
    fn request_frame(&mut self, tier: FrameTier) -> bool;

    /// Switch the subscribed tier without invalidating the current tick.
    fn change_frame_tier(&mut self, tier: FrameTier) -> bool;

    /// Sample the latest frame of the subscribed tier.
    fn read_frame(&mut self) -> Option<FrameRef>;

    /// Publish a signal value to the host.
    fn publish(&mut self, name: &str, value: SignalValue);

    /// Read a parameter from the host.
    fn get_param(&self, name: &str) -> Option<ParamValue>;

    /// Get current timing.
    fn timing(&self) -> crate::runtime::Timing;
}

// Re-export Timing for convenience if needed, but it's usually defined elsewhere.
// I'll ensure I have a Timing struct available.

/// A mock implementation of HostApi for Build B.
/// It preserves the strict tick pinning and latest-wins semantics.
pub struct MockHost {
    subscribed_tier: Option<FrameTier>,
    pending_tier: Option<FrameTier>,
    frame_epoch: u64,
    tier_generation: u32,
    tick_generation: std::sync::Arc<std::sync::atomic::AtomicU64>,
    pinned_frame: Option<FrameRef>,
    pinned_view: Option<FrameView>,
    requested_this_tick: bool,
    timing: crate::runtime::Timing,
}

impl Default for MockHost {
    fn default() -> Self {
        Self::new()
    }
}

impl MockHost {
    pub fn new() -> Self {
        Self {
            subscribed_tier: None,
            pending_tier: None,
            frame_epoch: 0,
            tier_generation: 0,
            tick_generation: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            pinned_frame: None,
            pinned_view: None,
            requested_this_tick: false,
            timing: crate::runtime::Timing::default(),
        }
    }

    /// Begin a new tick: apply pending tier changes, clear the pinned metadata,
    /// and invalidate any existing payload views.
    pub fn begin_tick(&mut self) {
        self.tick_generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if let Some(tier) = self.pending_tier.take() {
            if self.subscribed_tier != Some(tier) {
                self.subscribed_tier = Some(tier);
                self.tier_generation = self.tier_generation.saturating_add(1);
            }
        }
        self.pinned_frame = None;
        self.pinned_view = None;
        self.requested_this_tick = false;
    }

    fn dimensions_for_tier(tier: FrameTier) -> (u32, u32) {
        match tier {
            FrameTier::R160x90 => (160, 90),
            FrameTier::R320x180 => (320, 180),
            FrameTier::R640x360 => (640, 360),
            FrameTier::FullRes => (1280, 720),
        }
    }

    pub fn read_frame_view(&mut self) -> Option<FrameView> {
        if let Some(view) = self.pinned_view.clone() {
            return Some(view);
        }

        // No view unless the addon requested a frame this tick.
        if !self.requested_this_tick {
            return None;
        }

        let tier = self.subscribed_tier?;
        let size = match tier {
            FrameTier::R160x90 => 160 * 90 * 4,
            FrameTier::R320x180 => 320 * 180 * 4,
            FrameTier::R640x360 => 640 * 360 * 4,
            FrameTier::FullRes => 1280 * 720 * 4,
        };
        let bytes = vec![0u8; size].into_boxed_slice();
        let view = FrameView {
            bytes: std::sync::Arc::from(bytes),
            valid_tick: self
                .tick_generation
                .load(std::sync::atomic::Ordering::SeqCst),
            current_tick: self.tick_generation.clone(),
        };
        self.pinned_view = Some(view.clone());
        Some(view)
    }
}

impl HostApi for MockHost {
    fn request_frame(&mut self, tier: FrameTier) -> bool {
        if self.subscribed_tier != Some(tier) {
            self.subscribed_tier = Some(tier);
            self.tier_generation = self.tier_generation.saturating_add(1);
        }
        self.requested_this_tick = true;
        true
    }

    fn change_frame_tier(&mut self, tier: FrameTier) -> bool {
        if self.subscribed_tier == Some(tier) {
            return true;
        }
        self.pending_tier = Some(tier);
        true
    }

    fn read_frame(&mut self) -> Option<FrameRef> {
        if let Some(frame) = self.pinned_frame {
            return Some(frame);
        }

        // No frame unless the addon requested a frame this tick.
        if !self.requested_this_tick {
            return None;
        }

        let tier = self.subscribed_tier?;
        let (width, height) = Self::dimensions_for_tier(tier);
        self.frame_epoch = self.frame_epoch.saturating_add(1);

        let frame = FrameRef {
            epoch: self.frame_epoch,
            tier_generation: self.tier_generation,
            width,
            height,
            tier,
        };
        self.pinned_frame = Some(frame);
        Some(frame)
    }

    fn publish(&mut self, _name: &str, _value: SignalValue) {}

    fn get_param(&self, _name: &str) -> Option<ParamValue> {
        None
    }

    fn timing(&self) -> crate::runtime::Timing {
        self.timing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rss_kb() -> u64 {
        let contents = std::fs::read_to_string("/proc/self/status").unwrap();
        for line in contents.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                return rest
                    .trim()
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
            }
        }
        0
    }

    #[test]
    fn first_frame_is_available_after_request() {
        let mut host = MockHost::new();
        host.begin_tick();
        assert!(host.request_frame(FrameTier::R640x360));

        let frame = host.read_frame().expect("first frame delivered");
        assert!(frame.width > 0);
        assert!(frame.height > 0);
        assert!(frame.epoch > 0);
        assert!(frame.tier_generation > 0);
    }

    #[test]
    fn stable_tick_returns_same_frame_metadata() {
        let mut host = MockHost::new();
        host.begin_tick();
        assert!(host.request_frame(FrameTier::R640x360));

        let a = host.read_frame().unwrap();
        let b = host.read_frame().unwrap();
        let c = host.read_frame().unwrap();

        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn tier_switch_mid_tick_is_visible_next_tick_only() {
        let mut host = MockHost::new();
        host.begin_tick();
        assert!(host.request_frame(FrameTier::R640x360));
        let first = host.read_frame().unwrap();

        assert!(host.change_frame_tier(FrameTier::R320x180));
        let same = host.read_frame().unwrap();
        assert_eq!(first, same, "current tick remains pinned to the same frame");

        host.begin_tick();
        assert!(
            host.read_frame().is_none(),
            "new tick must pin only after request_frame"
        );
        assert!(host.request_frame(FrameTier::R320x180));
        let next = host.read_frame().unwrap();

        assert_eq!(next.tier, FrameTier::R320x180);
        assert_ne!(first.epoch, next.epoch);
        assert_ne!(first.tier_generation, next.tier_generation);
    }

    #[test]
    fn stale_frame_handle_is_safe_across_ticks() {
        let mut host = MockHost::new();
        host.begin_tick();
        assert!(host.request_frame(FrameTier::R640x360));
        let old = host.read_frame().unwrap();

        host.begin_tick();
        assert!(host.request_frame(FrameTier::R640x360));
        let new = host.read_frame().unwrap();

        assert_eq!(old.width, 640);
        assert_eq!(old.height, 360);
        assert!(old.epoch < new.epoch);
    }

    #[test]
    fn tick_boundary_reuse_invalidate_old_payload() {
        let mut host = MockHost::new();
        host.begin_tick();
        assert!(host.request_frame(FrameTier::R640x360));
        let old_metadata = host.read_frame().unwrap();
        let old_view = host.read_frame_view().unwrap();

        host.begin_tick();
        assert!(host.request_frame(FrameTier::R640x360));
        let _new_metadata = host.read_frame().unwrap();
        let new_view = host.read_frame_view().unwrap();

        assert_eq!(old_metadata.width, 640);
        assert_eq!(old_metadata.height, 360);
        assert!(old_metadata.epoch > 0);
        assert!(
            old_view.bytes().is_none(),
            "old payload view must be invalid after tick"
        );
        assert!(
            new_view.bytes().is_some(),
            "new payload view must remain valid within tick"
        );
    }

    #[test]
    fn long_run_frame_acquisition_stays_stable_and_memory_small() {
        let mut host = MockHost::new();
        let start_rss = rss_kb();
        let mut prev_epoch = 0;
        let mut frame_misses = 0;

        for tick in 0..300 {
            host.begin_tick();
            assert!(host.request_frame(FrameTier::R640x360));
            let frame = match host.read_frame() {
                Some(frame) => frame,
                None => {
                    frame_misses += 1;
                    continue;
                }
            };

            assert!(frame.epoch > prev_epoch);
            prev_epoch = frame.epoch;

            if tick == 150 {
                assert!(host.change_frame_tier(FrameTier::R320x180));
            }
        }

        let end_rss = rss_kb();
        let growth_kb = end_rss.saturating_sub(start_rss);
        eprintln!(
            "long_run: start={}kB, end={}kB, growth={}kB, misses={}",
            start_rss, end_rss, growth_kb, frame_misses
        );
        assert_eq!(frame_misses, 0, "no frame misses during 300 ticks");
        assert!(
            growth_kb < 5 * 1024,
            "RSS growth too large: {}kB",
            growth_kb
        );
    }
}
