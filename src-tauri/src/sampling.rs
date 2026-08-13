//! Adaptive frame-sampling scheduler for the recognition loop.
//!
//! Replaces the former fixed `TARGET_FRAME_INTERVAL`.  Rules:
//!
//! - a fresh start or a clear / large change resumes fast sampling
//! - repeated unchanged frames gradually back off to avoid pointless VLM
//!   requests on a static desktop
//! - a persistent high-motion scene (video, scrolling, cursor flashes) raises
//!   the sampling interval while still occasionally taking a look
//! - model / service errors use exponential backoff so the local server is
//!   never hammered while it is struggling

use std::time::Duration;

/// Minimum interval right after a meaningful change (fast re-sampling).
pub const FAST_INTERVAL: Duration = Duration::from_millis(350);
/// First relaxed cadence while the screen is mostly idle.
pub const NORMAL_INTERVAL: Duration = Duration::from_secs(2);
/// Longest gap between cheap screen-change probes on a static desktop.
///
/// A probe is still kept so a newly opened dialog, an error notification, or
/// another meaningful change wakes recognition without requiring a restart.
pub const MAX_INTERVAL: Duration = Duration::from_secs(20);

/// Frames classified as "high motion" when this many coarse cells change.
pub const HIGH_MOTION_CELLS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Motion {
    /// No meaningful change detected.
    Idle,
    /// Localised, possibly transient change (cursor blink, typing caret).
    Small,
    /// Broad change spanning several regions (popup, new window, layout).
    Significant,
    /// Persistent, everywhere-changing scene (video / scroll / animation).
    HighActivity,
}

#[derive(Debug)]
pub struct AdaptiveSampler {
    fast: Duration,
    normal: Duration,
    max: Duration,
    idle_rounds: u32,
    consecutive_motion: u32,
    consecutive_dynamic: u32,
    error_rounds: u32,
}

impl Default for AdaptiveSampler {
    fn default() -> Self {
        Self {
            fast: FAST_INTERVAL,
            normal: NORMAL_INTERVAL,
            max: MAX_INTERVAL,
            idle_rounds: 0,
            consecutive_motion: 0,
            consecutive_dynamic: 0,
            error_rounds: 0,
        }
    }
}

impl AdaptiveSampler {
    /// Duration to wait until the next capture/reasoning round for the last
    /// observed motion class.
    pub fn next_interval(&self, motion: Motion, changed_cells: usize) -> Duration {
        match motion {
            Motion::Significant => self.fast,
            Motion::Small => self.fast,
            Motion::Idle => self.idle_interval(),
            Motion::HighActivity => self.dynamic_interval(changed_cells),
        }
    }

    fn idle_interval(&self) -> Duration {
        match self.idle_rounds {
            // Back off quickly: once two consecutive captures show no visual
            // change, there is no reason to keep resizing and hashing the
            // same desktop several times per second.
            0..=1 => self.normal,
            2..=3 => self.normal.mul_f64(2.5),
            4..=5 => self.normal.mul_f64(5.0),
            6..=7 => self.normal.mul_f64(7.5),
            _ => self.max,
        }
    }

    fn dynamic_interval(&self, changed_cells: usize) -> Duration {
        // High-motion scenes still deserve a look every few seconds so a
        // late popup / error toast is not missed entirely.  The heavier the
        // motion, the longer the gap so GPU throughput stays with the real
        // user workload.
        let grace = if changed_cells >= 8 { 5 } else { 3 };
        self.normal.mul_f64(grace as f64).min(self.max)
    }

    /// Call after a frame with no meaningful change.
    pub fn note_idle(&mut self) {
        self.idle_rounds = self.idle_rounds.saturating_add(1);
        self.consecutive_motion = 0;
        self.consecutive_dynamic = 0;
        self.error_rounds = 0;
    }

    /// Call on any change (small or significant): resume fast sampling.
    pub fn note_change(&mut self) {
        self.idle_rounds = 0;
        self.consecutive_motion = self.consecutive_motion.saturating_add(1);
        self.consecutive_dynamic = 0;
        self.error_rounds = 0;
    }

    /// Call when several cells changed persistently: raises the gap.
    pub fn note_dynamic(&mut self) {
        self.consecutive_dynamic = self.consecutive_dynamic.saturating_add(1);
        self.consecutive_motion = self.consecutive_motion.saturating_add(1);
        self.idle_rounds = 0;
        self.error_rounds = 0;
    }

    /// True once a high-motion scene has persisted for a few frames and should
    /// be serviced with longer gaps instead of per-frame requests.
    pub fn is_dynamic_scene(&self) -> bool {
        self.consecutive_dynamic >= 3
    }

    /// Call after a model / service error: exponential backoff.
    pub fn note_error(&mut self) {
        self.error_rounds = self.error_rounds.saturating_add(1);
    }

    /// Call after a successful model response.
    pub fn note_success(&mut self) {
        self.error_rounds = 0;
    }

    pub fn error_backoff(&self) -> Duration {
        let exponent = self.error_rounds.min(5);
        self.fast.mul_f64(2_f64.powi(exponent as i32)).min(self.max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_screen_quickly_reaches_low_frequency_probes() {
        let mut sampler = AdaptiveSampler::default();

        for _ in 0..8 {
            sampler.note_idle();
        }

        assert_eq!(
            sampler.next_interval(Motion::Idle, 0),
            MAX_INTERVAL,
            "a stable screen should only be checked periodically"
        );
    }

    #[test]
    fn meaningful_change_restores_fast_sampling() {
        let mut sampler = AdaptiveSampler::default();
        for _ in 0..8 {
            sampler.note_idle();
        }

        sampler.note_change();

        assert_eq!(sampler.next_interval(Motion::Significant, 4), FAST_INTERVAL);
    }
}
