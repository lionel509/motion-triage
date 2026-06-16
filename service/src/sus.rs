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
fn default_review_at() -> f32 { 0.40 }
fn default_alert_at() -> f32 { 0.85 }
fn default_conf_floor() -> f32 { 0.50 }

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
    /// Legacy single line for the boolean `sus_alert` field (score >= threshold).
    /// Kept for back-compat; the three-way `route` uses review_at/alert_at below.
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    /// Below this the event is not worth anyone's time → dismiss.
    #[serde(default = "default_review_at")]
    pub review_at: f32,
    /// At/above this AND the NN is confident → fire now (no VLM round-trip).
    #[serde(default = "default_alert_at")]
    pub alert_at: f32,
    /// A behavior-NN classification below this confidence is "unsure": a would-be
    /// alert is demoted to the VLM (quality check) instead of firing blind.
    #[serde(default = "default_conf_floor")]
    pub conf_floor: f32,
}

/// Three-way routing outcome (string-serialized in the verdict).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Route { Alert, Vlm, Dismiss }

impl Route {
    pub fn as_str(self) -> &'static str {
        match self { Route::Alert => "alert", Route::Vlm => "vlm", Route::Dismiss => "dismiss" }
    }
    /// Severity order for aggregating many tracks into one event route.
    pub fn rank(self) -> u8 { match self { Route::Dismiss => 0, Route::Vlm => 1, Route::Alert => 2 } }
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

    /// Route a graded track three ways from its sus score + the NN's confidence
    /// in the behavior it classified (None when no behavior NN ran — then we
    /// trust the score alone).
    ///
    /// - `sus >= alert_at` **and** the NN is sure → `Alert` (fire now).
    /// - unsure NN, or `review_at <= sus < alert_at` → `Vlm` (manager checks).
    /// - `sus < review_at` → `Dismiss` (a street walker scores ~0 here).
    pub fn route(&self, sus: f32, behavior_conf: Option<f32>) -> Route {
        let sure = behavior_conf.map_or(true, |c| c >= self.conf_floor);
        if sus >= self.alert_at && sure { Route::Alert }
        else if sus >= self.review_at { Route::Vlm }
        else { Route::Dismiss }
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
            review_at: 0.40,
            alert_at: 0.85,
            conf_floor: 0.50,
        }
    }

    #[test]
    fn route_bands_relevance_then_confidence() {
        let p = policy();
        // street walker: sus 0 → dismiss, no matter how sure the box was.
        assert_eq!(p.route(0.0, Some(0.99)), Route::Dismiss);
        // ambiguous loiter (0.7) → manager checks.
        assert_eq!(p.route(0.70, None), Route::Vlm);
        // high relevance + confident NN → fire now.
        assert_eq!(p.route(0.95, Some(0.90)), Route::Alert);
        // high relevance but the NN is unsure → demote to VLM, don't fire blind.
        assert_eq!(p.route(0.95, Some(0.30)), Route::Vlm);
        // high relevance, no behavior NN ran (hard geometry) → trust the score.
        assert_eq!(p.route(0.95, None), Route::Alert);
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
