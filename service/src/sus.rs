//! Suspicion scoring: every flag carries a weight, context multiplies it, a
//! threshold flags it.
//!
//! The behavior NN says *what* a body is doing (a flag); this turns that into
//! *does it matter here, now*. Deliberately a declarative config, not a model —
//! the same `pacing` is benign at a storefront at noon and an alert at 3 a.m.,
//! and that judgment should be tunable per site in seconds, not retrained.
//!
//! ```text
//! sus = max(flag_weight[behavior], reason_weight[reason])
//!         × (night ? night_mult : 1)
//!         × (in_zone ? zone_mult : 1)         # clamped to [0, 1]
//! alert = sus >= threshold
//! ```
//!
//! Loaded from the JSON at `SUS_POLICY` (see `sus_policy.example.json`).
//! Absent → scoring is simply off (additive; callers see no sus fields).
use std::collections::HashMap;

use serde::Deserialize;

fn one() -> f32 { 1.0 }
fn default_threshold() -> f32 { 0.7 }

#[derive(Deserialize, Clone, Debug, Default)]
pub struct SusPolicy {
    /// NN behavior flag -> base weight (walking 0.0 … pickup 0.5).
    #[serde(default)]
    pub flag_weight: HashMap<String, f32>,
    /// Rule-engine reason -> base weight (intrusion 0.95, loitering 0.7 …).
    /// The track's score takes the MAX of its flag weight and reason weight.
    #[serde(default)]
    pub reason_weight: HashMap<String, f32>,
    #[serde(default = "one")]
    pub night_mult: f32,
    #[serde(default = "one")]
    pub zone_mult: f32,
    #[serde(default = "default_threshold")]
    pub threshold: f32,
}

impl SusPolicy {
    pub fn load(path: &str) -> Option<Self> {
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
    }

    /// (sus_score in [0,1], alert). Pure — unit-tested without any model.
    pub fn score(
        &self, behavior: Option<&str>, reason: &str, night: bool, in_zone: bool,
    ) -> (f32, bool) {
        let fw = behavior
            .and_then(|b| self.flag_weight.get(b))
            .copied()
            .unwrap_or(0.0);
        let rw = self.reason_weight.get(reason).copied().unwrap_or(0.0);
        let mut s = fw.max(rw);
        if night {
            s *= self.night_mult;
        }
        if in_zone {
            s *= self.zone_mult;
        }
        let s = s.clamp(0.0, 1.0);
        (s, s >= self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> SusPolicy {
        SusPolicy {
            flag_weight: [("walking", 0.0), ("pacing", 0.5), ("pickup", 0.5)]
                .iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            reason_weight: [("intrusion", 0.95), ("loitering", 0.7)]
                .iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            night_mult: 2.0,
            zone_mult: 1.5,
            threshold: 0.7,
        }
    }

    #[test]
    fn walking_is_never_suspicious_on_its_own() {
        let (s, alert) = policy().score(Some("walking"), "walk_by", false, false);
        assert_eq!(s, 0.0);
        assert!(!alert);
    }

    #[test]
    fn pacing_alerts_only_with_context() {
        let p = policy();
        // daytime, open area: 0.5 — under threshold, informational.
        let (day, day_alert) = p.score(Some("pacing"), "walk_by", false, false);
        assert!((day - 0.5).abs() < 1e-6);
        assert!(!day_alert);
        // same pacing, at night: 0.5 × 2.0 = 1.0 — alert.
        let (night, night_alert) = p.score(Some("pacing"), "walk_by", true, false);
        assert!((night - 1.0).abs() < 1e-6);
        assert!(night_alert);
    }

    #[test]
    fn reason_and_flag_take_the_max() {
        // intrusion reason (0.95) dominates a benign walking flag (0.0).
        let (s, alert) = policy().score(Some("walking"), "intrusion", false, false);
        assert!((s - 0.95).abs() < 1e-6);
        assert!(alert);
    }

    #[test]
    fn score_is_clamped_to_one() {
        // 0.95 × 2.0 (night) × 1.5 (zone) = 2.85 -> clamped to 1.0.
        let (s, _) = policy().score(None, "intrusion", true, true);
        assert_eq!(s, 1.0);
    }
}
