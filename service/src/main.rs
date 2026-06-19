//! YOLO behavior-triage (Steps B+C, + service): detect people on the Apple GPU
//! (CoreML), track them across a clip's frames, classify the trajectory, and
//! return DISMISS (walk-by) / ALERT (loiter) / ESCALATE (concerning → grid VLM).
//!
//! Two modes:
//!   spike <model> <frames-dir|image>   # CLI: run + print (dev/testing)
//!   spike serve [port]                 # HTTP: POST /triage multipart frames
//!
//! The Python pipeline extracts the clip frames (it has ffmpeg) and POSTs them
//! to /triage; this service does the ML on the Apple GPU and returns the verdict.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use axum::{extract::{DefaultBodyLimit, Multipart, State}, routing::{get, post}, Json, Router};
use image::imageops::FilterType;
use ndarray::Array4;
#[cfg(target_os = "macos")]
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::Serialize;

mod behavior;
mod draw;
mod identity;
mod pose;
mod sus;

const IMGSZ: u32 = 640;
const CONF: f32 = 0.30;
const IOU_NMS: f32 = 0.45;
const IOU_TRACK: f32 = 0.30;
const PERSON_CLASS: usize = 0;
// COCO vehicle classes a person can arrive/leave in: car, motorcycle, bus,
// truck. (bicycle/1 is excluded — you don't "vanish into" a bike.) yolo11m
// already emits these in the same forward pass; we just stop discarding them.
const VEHICLE_CLASSES: [usize; 4] = [2, 3, 5, 7];
// A person briefly occluded (walks behind a shelf, another person, a sign) or
// momentarily mis-detected used to break into TWO tracks at MAX_GAP=3 — and a
// fragmented track reads as a fake reversal (pacing) or a mid-frame vanish
// (intrusion). Hold tracks across a longer gap; the center-distance fallback
// below also re-associates fast movers whose boxes don't overlap frame-to-frame.
const MAX_GAP: usize = 8;
// Center-distance fallback: when no box OVERLAPS (fast walker at 4-6 fps, or a
// post-occlusion reappearance shifted over), match the nearest unused same-class
// detection whose center is within this many of the track's last box-heights.
const TRACK_DIST_BH: f32 = 1.2;

// Phantom / distant-figure gates for the "soft" behaviors — the low-value,
// FP-prone trajectory reads (loitering, pacing, u_turn, direction_change,
// erratic, sudden_stop, approach). Without them these fire on (a) low-confidence
// mis-detections on static clutter — camera housings, power bricks, cables,
// furniture, foliage (operator FPs 942e680a / af68f624 / 85c9d48d: a desk-
// facing indoor cam, a porch cam) — and (b) tiny far-off figures + passing
// traffic (47c64e81 / 59878b21: a distant street pedestrian). They must clear
// a real-person confidence AND a minimum apparent size; loitering additionally
// rejects a perfectly frozen box (a fixed object, not a person). HIGH-VALUE
// reasons (running, intrusion, camera_approach, scaling, crouch, vehicle,
// multi_person, zone/line) stay UNGATED so a genuine low-confidence threat (e.g.
// a person in poor light fleeing or breaking in) still alerts. The soft family
// also feeds the server-side pacing/prolonged_loitering upgrade (lanes.py
// _refine_track_reason), which has no conf/size of its own — gating at the
// source here is what stops a jittery clutter box escaping as direction_change→
// pacing. NOTE: extends the originally-approved loiter/pacing scope to the
// sibling soft reads, because gating loiter/pacing alone left that escape hole.
const SOFT_MIN_CONF: f32 = 0.45;   // peak track confidence (base CONF is 0.30)
const SOFT_MIN_HFRAC: f32 = 0.08;  // box height ≥ 8% of frame height
const DEAD_BOX_MOTION: f32 = 0.006; // total center path / frame-diagonal
const DEAD_BOX_SCALE: f32 = 0.02;   // apparent-size change across the clip

fn is_vehicle(c: usize) -> bool { VEHICLE_CLASSES.contains(&c) }

/// A person box is taller-than-wide to roughly square; a box much WIDER than
/// tall is a horizontal object (railing, counter edge, shelf lip) the detector
/// mislabeled — a recurring false-track source on the store cams. Vehicles are
/// legitimately wide, so this only screens the person class.
fn implausible_person(d: &Det) -> bool {
    d.cls == PERSON_CLASS && (d.x2 - d.x1) > 2.2 * (d.y2 - d.y1).max(1.0)
}

#[derive(Clone, Copy, Debug)]
struct Det { x1: f32, y1: f32, x2: f32, y2: f32, score: f32, cls: usize }
impl Det {
    fn cx(&self) -> f32 { (self.x1 + self.x2) / 2.0 }
    fn cy(&self) -> f32 { (self.y1 + self.y2) / 2.0 }
    fn h(&self) -> f32 { self.y2 - self.y1 }
}

struct Track { id: usize, cls: usize, last_frame: usize, first_frame: usize, first: Det, last: Det, centers: Vec<(f32, f32)>, heights: Vec<f32>,
    // Frame index of each centers/heights entry — lets the pose pass attach
    // keypoints to the exact frames this track was seen in (gaps excluded).
    frame_idxs: Vec<usize>,
    // v3 identity: the matched box and its clothing-color signature per seen
    // frame (parallel to centers/heights/frame_idxs). `boxes` lets the identity
    // audit re-crop the original pixels for body/face embedding; `color_sigs`
    // backs the real-time association veto + the reversal split. Empty when
    // identity gating is off (additive — nothing downstream reads them then).
    boxes: Vec<Det>,
    color_sigs: Vec<identity::ColorSig>,
    // Set when this track was carved out of a merged one by the reversal split
    // audit: the original track id + which signal (color/body/face) decided the
    // two halves were different people. Both None on ordinary tracks.
    split_from: Option<usize>,
    split_by: Option<String> }

// ---- Operator-drawn zones / lines (door + restricted-area analytics) --------
// Coordinates arrive NORMALIZED to fractions of the frame ([0,1]) in the POST
// `zones` field, so the triage service is resolution-independent: the Python
// side divides the source-keyframe pixels by the clip's native dimensions, and
// here we multiply back by the actual decoded frame size. A person's footpoint
// (bottom-centre of the box) is what's tested — that's where they stand.
#[derive(Clone, Default, serde::Deserialize)]
struct ZonesMeta {
    #[serde(default)] zones: Vec<ZoneDef>,
    #[serde(default)] lines: Vec<LineDef>,
}
#[derive(Clone, serde::Deserialize)]
struct ZoneDef { #[allow(dead_code)] name: String, polygon: Vec<[f32; 2]> }
#[derive(Clone, serde::Deserialize)]
struct LineDef { #[allow(dead_code)] name: String, a: [f32; 2], b: [f32; 2],
    #[serde(default)] in_direction: String }

/// Ray-casting point-in-polygon (polygon vertices in frame pixels).
fn point_in_poly(p: (f32, f32), poly: &[(f32, f32)]) -> bool {
    if poly.len() < 3 { return false; }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if (yi > p.1) != (yj > p.1)
            && p.0 < (xj - xi) * (p.1 - yi) / (yj - yi + 1e-9) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Which side of directed line a→b the point sits on (sign of the cross product).
fn side(a: (f32, f32), b: (f32, f32), p: (f32, f32)) -> f32 {
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
}

/// Does segment prev→curr cross line a→b? Returns +1 (left→right), -1 (right→
/// left), or 0 (no crossing). Sign convention matches the Python `crosses_line`.
fn crosses_line(prev: (f32, f32), curr: (f32, f32), a: (f32, f32), b: (f32, f32)) -> i32 {
    let (s1, s2) = (side(a, b, prev), side(a, b, curr));
    if (s1 > 0.0) == (s2 > 0.0) { return 0; }            // both on same side → no cross
    // also require the crossing point to fall within the drawn segment
    let (s3, s4) = (side(prev, curr, a), side(prev, curr, b));
    if (s3 > 0.0) == (s4 > 0.0) { return 0; }
    // +1 when leaving the positive side (e.g. left→right across a downward line)
    if s1 > s2 { 1 } else { -1 }
}

fn iou(a: &Det, b: &Det) -> f32 {
    let xx1 = a.x1.max(b.x1); let yy1 = a.y1.max(b.y1);
    let xx2 = a.x2.min(b.x2); let yy2 = a.y2.min(b.y2);
    let inter = (xx2 - xx1).max(0.0) * (yy2 - yy1).max(0.0);
    let aa = (a.x2 - a.x1) * (a.y2 - a.y1);
    let ab = (b.x2 - b.x1) * (b.y2 - b.y1);
    inter / (aa + ab - inter + 1e-6)
}

fn preprocess(img: &image::RgbImage) -> (Array4<f32>, f32, f32, f32) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let scale = (IMGSZ as f32 / w).min(IMGSZ as f32 / h);
    let nw = (w * scale).round() as u32;
    let nh = (h * scale).round() as u32;
    let resized = image::imageops::resize(img, nw, nh, FilterType::Triangle);
    let pad_x = ((IMGSZ - nw) / 2) as f32;
    let pad_y = ((IMGSZ - nh) / 2) as f32;
    let mut t = Array4::<f32>::from_elem((1, 3, IMGSZ as usize, IMGSZ as usize), 114.0 / 255.0);
    for (x, y, px) in resized.enumerate_pixels() {
        let (cx, cy) = ((x + pad_x as u32) as usize, (y + pad_y as u32) as usize);
        t[[0, 0, cy, cx]] = px[0] as f32 / 255.0;
        t[[0, 1, cy, cx]] = px[1] as f32 / 255.0;
        t[[0, 2, cy, cx]] = px[2] as f32 / 255.0;
    }
    (t, scale, pad_x, pad_y)
}

fn detect(session: &Session, img: &image::RgbImage) -> Result<Vec<Det>> {
    let (input, scale, px, py) = preprocess(img);
    let outputs = session.run(ort::inputs!["images" => input.view()]?)?;
    let out = outputs["output0"].try_extract_tensor::<f32>()?;
    let shape = out.shape();

    // YOLO26 exports an NMS-free end-to-end head: [1, N, 6] rows of
    // [x1, y1, x2, y2, score, cls], already decoded + deduped (top-N) in 640px
    // letterbox space. YOLO11 exports the raw grid: [1, 4+nc, anchors] needing
    // argmax + NMS (the path below). Branch on shape so both models work.
    if shape.len() == 3 && shape[2] == 6 {
        let mut dets = Vec::new();
        for i in 0..shape[1] {
            let score = out[[0, i, 4]];
            if score < CONF { continue; }
            let cls = out[[0, i, 5]].round() as usize;
            if cls != PERSON_CLASS && !is_vehicle(cls) { continue; }
            let (x1, y1, x2, y2) = (out[[0, i, 0]], out[[0, i, 1]], out[[0, i, 2]], out[[0, i, 3]]);
            dets.push(Det {
                x1: (x1 - px) / scale, y1: (y1 - py) / scale,
                x2: (x2 - px) / scale, y2: (y2 - py) / scale, score, cls,
            });
        }
        dets.retain(|d| !implausible_person(d));
        return Ok(dets); // model already applied NMS
    }

    let na = shape[2];
    let nc = shape[1] - 4;
    let mut dets = Vec::new();
    for a in 0..na {
        // Argmax over all classes (one pass), then keep only person + the vehicle
        // classes a person can arrive/leave in. We used to read PERSON_CLASS only
        // and discard everything else; yolo11m emits cars/trucks here too.
        let mut cls = 0usize;
        let mut score = out[[0, 4, a]];
        for c in 1..nc {
            let s = out[[0, 4 + c, a]];
            if s > score { score = s; cls = c; }
        }
        if score < CONF { continue; }
        if cls != PERSON_CLASS && !is_vehicle(cls) { continue; }
        let (cx, cy, w, h) = (out[[0, 0, a]], out[[0, 1, a]], out[[0, 2, a]], out[[0, 3, a]]);
        dets.push(Det {
            x1: ((cx - w / 2.0) - px) / scale, y1: ((cy - h / 2.0) - py) / scale,
            x2: ((cx + w / 2.0) - px) / scale, y2: ((cy + h / 2.0) - py) / scale, score, cls,
        });
    }
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep: Vec<Det> = Vec::new();
    // Class-aware NMS: a person standing in front of a car must not suppress the
    // car (or vice-versa) — only same-class boxes compete.
    'o: for d in dets {
        for k in &keep { if k.cls == d.cls && iou(&d, k) > IOU_NMS { continue 'o; } }
        keep.push(d);
    }
    keep.retain(|d| !implausible_person(d));
    Ok(keep)
}

/// Recent heading of a track from its last few centers (average step). None
/// until there are ≥2 points. Appearance-free — pure trajectory.
fn track_velocity(centers: &[(f32, f32)]) -> Option<(f32, f32)> {
    let n = centers.len();
    if n < 2 { return None; }
    let k = n.min(3);
    let a = centers[n - k];
    let b = centers[n - 1];
    Some(((b.0 - a.0) / (k - 1) as f32, (b.1 - a.1) / (k - 1) as f32))
}

/// Is a candidate step strongly OPPOSITE the track's recent heading? Both the
/// heading and the step must be real motion (≥ 0.3 box-heights) so a stationary
/// jitter never vetoes; then a cosine < -0.3 means ~>110° reversal — physically
/// implausible for one person at speed, i.e. a different person.
fn motion_incompatible(vel: Option<(f32, f32)>, step: (f32, f32), bh: f32) -> bool {
    let Some(v) = vel else { return false };
    let vm = (v.0 * v.0 + v.1 * v.1).sqrt();
    let sm = (step.0 * step.0 + step.1 * step.1).sqrt();
    let floor = 0.30 * bh.max(1.0);
    if vm < floor || sm < floor { return false; }
    (v.0 * step.0 + v.1 * step.1) / (vm * sm) < -0.3
}

/// Per-frame color signatures parallel to `frames` (one entry per detection).
/// Empty inner vecs (or `gating=false`) disable the appearance veto, restoring
/// the exact geometry-only behavior.
fn track(frames: &[Vec<Det>], sigs: &[Vec<identity::ColorSig>],
         cfg: &identity::IdentityConfig, gating: bool) -> Vec<Track> {
    let empty: Vec<identity::ColorSig> = Vec::new();
    let sig_at = |fi: usize, di: usize| -> Option<identity::ColorSig> {
        sigs.get(fi).unwrap_or(&empty).get(di).copied()
    };
    let mut tracks: Vec<Track> = Vec::new();
    let mut next_id = 0usize;
    for (fi, dets) in frames.iter().enumerate() {
        let mut used = vec![false; dets.len()];
        for t in tracks.iter_mut() {
            if fi - t.last_frame > MAX_GAP { continue; }
            // Running clothing signature of this track (mean of seen frames).
            // Only meaningful when gating + we have sigs; cheap over a few frames.
            let tmean = if gating && !t.color_sigs.is_empty() {
                Some(identity::color_mean(&t.color_sigs))
            } else { None };
            // Veto: an outfit clearly unlike the track's own is a DIFFERENT
            // person — refuse the association so they spawn their own track
            // instead of being stitched into a fake reversal. Lenient on strong
            // box overlap (geometry is trustworthy there); strict otherwise.
            let vetoed = |di: usize, overlap: f32| -> bool {
                let (Some(tm), Some(cs)) = (tmean, sig_at(fi, di)) else { return false };
                if overlap >= cfg.strong_iou || tm.weak || cs.weak { return false; }
                identity::color_distance(&tm, &cs) > cfg.veto_color_dist
            };
            let mut best = (IOU_TRACK, usize::MAX);
            for (di, d) in dets.iter().enumerate() {
                if used[di] || d.cls != t.cls { continue; } // never merge a person into a car track
                let s = iou(&t.last, d);
                if s > best.0 && !vetoed(di, s) { best = (s, di); }
            }
            // Fallback: no box overlapped (fast mover / post-occlusion shift).
            // Take the nearest unused same-class det within TRACK_DIST_BH of the
            // last box-height — re-links the track instead of spawning a new one
            // (which would read as a reversal/vanish). Distance, not IoU, so it
            // survives a frame-to-frame jump bigger than a box. The veto is
            // STRICT here (overlap=0): this fallback is exactly the path that
            // stitches person-A-exits to person-B-enters.
            if best.1 == usize::MAX {
                let bh = (t.last.y2 - t.last.y1).max(1.0);
                let (lcx, lcy) = (t.last.cx(), t.last.cy());
                // Appearance-FREE motion veto (works at night/IR where color is
                // weak): a track moving one way can't instantly claim a detection
                // requiring the opposite heading — that's two people, not one
                // reversing person. Gated to the distance-fallback only, so a real
                // pacer (who reverses but keeps overlapping boxes → IoU match) is
                // untouched. This is the night-robust half of the un-merge.
                let tvel = track_velocity(&t.centers);
                let mut bd = TRACK_DIST_BH * bh;
                for (di, d) in dets.iter().enumerate() {
                    if used[di] || d.cls != t.cls { continue; }
                    let step = (d.cx() - lcx, d.cy() - lcy);
                    if gating && motion_incompatible(tvel, step, bh) { continue; }
                    let dist = (step.0 * step.0 + step.1 * step.1).sqrt();
                    if dist < bd && !vetoed(di, 0.0) { bd = dist; best.1 = di; }
                }
            }
            if best.1 != usize::MAX {
                let d = dets[best.1];
                used[best.1] = true;
                t.last = d; t.last_frame = fi;
                t.centers.push((d.cx(), d.cy())); t.heights.push(d.h());
                t.frame_idxs.push(fi);
                t.boxes.push(d);
                t.color_sigs.push(sig_at(fi, best.1).unwrap_or_default());
            }
        }
        for (di, d) in dets.iter().enumerate() {
            if !used[di] {
                tracks.push(Track { id: next_id, cls: d.cls, last_frame: fi, first_frame: fi,
                    first: *d, last: *d, centers: vec![(d.cx(), d.cy())], heights: vec![d.h()],
                    frame_idxs: vec![fi], boxes: vec![*d],
                    color_sigs: vec![sig_at(fi, di).unwrap_or_default()],
                    split_from: None, split_by: None });
                next_id += 1;
            }
        }
    }
    tracks
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum Decision { Dismiss, Alert, Escalate }
impl Decision {
    fn as_str(&self) -> &'static str {
        match self { Decision::Dismiss => "dismiss", Decision::Alert => "alert", Decision::Escalate => "escalate" }
    }
}

#[derive(Serialize)]
struct TrackVerdict { id: usize, n: usize, straightness: f32, span: f32, dwell_frac: f32,
    // Temporal window this track occupies within the clip, as fractions of the
    // clip duration: start = first_frame/total, end = (last_frame+1)/total, so
    // (end_frac - start_frac) == dwell_frac. Lets the dashboard trim the VLM's
    // input to the seconds the subject is actually present instead of the whole
    // clip. Additive; synthetic event-level rows report 0.0/0.0 (no window).
    start_frac: f32,
    end_frac: f32,
    // Absolute dwell estimate in seconds: dwell_frac × the clip window length
    // (clip_seconds in the POSTed context, default 60). The operator reads this
    // directly — <10s = passing through (ignore), ~minute = loitering. Additive;
    // present whenever clip_seconds is known.
    #[serde(skip_serializing_if = "Option::is_none")]
    dwell_s: Option<f32>,
    decision: String, reason: String,
    // 17-COCO-joint skeletons, one entry per frame the track was seen (null
    // where pose found no match) — populated only when POSE_MODEL is set.
    // Additive per docs/API.md: absent entirely when pose is off, so existing
    // consumers see an unchanged shape.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    keypoints: Vec<Option<Vec<[f32; 3]>>>,
    // Behavior-NN classification of the skeleton trajectory (BEHAVIOR_MODEL
    // env). Additive like keypoints: absent when the NN is off or the track
    // had too little pose signal to classify.
    #[serde(skip_serializing_if = "Option::is_none")]
    behavior: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    behavior_conf: Option<f32>,
    // Suspicion score (sus.rs): flag/reason weight × night × zone, [0,1].
    // Present only when SUS_POLICY is set. sus_alert = score >= threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    sus_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sus_alert: Option<bool>,
    // Three-way routing for this track: "alert" (fire now) | "vlm" (manager
    // quality-checks the unsure middle) | "dismiss". Additive; only emitted once
    // routing has run. The event-level route is the most severe across tracks.
    #[serde(skip_serializing_if = "Option::is_none")]
    route: Option<String>,
    // v3 identity (additive): present only on tracks the reversal-split audit
    // carved out of a merged track — `split_from` is the original track id and
    // `decided_by` is which signal (color/body/face) judged them distinct.
    #[serde(skip_serializing_if = "Option::is_none")]
    identity: Option<IdentityInfo> }

#[derive(Serialize, Clone)]
struct IdentityInfo { split_from: usize, decided_by: String }

// Event-level union of every real track's temporal window, in clip-duration
// fractions. The dashboard converts these to seconds against the actual clip
// length and trims the VLM's input (dense frames / clip-as-video) to this span
// instead of the whole padded clip. Absent when no real track had a window.
#[derive(Serialize)]
struct ActiveWindow { start_frac: f32, end_frac: f32 }

/// Is this person box at/adjacent to any vehicle box? True on real overlap, or
/// when the person's center sits within a vehicle box expanded by ~8% of the
/// frame diagonal — i.e. they're stepping in/out, not necessarily overlapping it
/// yet. Vehicle boxes are static context (one per car track); this is cheap.
fn near_vehicle(p: &Det, vehicles: &[Det], frame_diag: f32) -> bool {
    let gap = frame_diag * 0.08;
    let (pcx, pcy) = (p.cx(), p.cy());
    vehicles.iter().any(|v| {
        iou(p, v) > 0.05
            || (pcx > v.x1 - gap && pcx < v.x2 + gap && pcy > v.y1 - gap && pcy < v.y2 + gap)
    })
}

/// Did the person move TOWARD a vehicle before vanishing? (operator 2026-06-09:
/// "if the person box disappears moving toward the car box, they got in"). Finds
/// the vehicle nearest where the person ended and requires they got meaningfully
/// closer to it across the track. This is what separates "walked up to a parked
/// car and got in" from a background person who merely ends up near PASSING
/// traffic (the 70b6aa50 false positive — a car driving by on the street).
fn approached_vehicle(t: &Track, vehicles: &[Det], frame_diag: f32) -> bool {
    if t.centers.len() < 3 { return false; }
    let last = *t.centers.last().unwrap();
    let first = t.centers[0];
    let nearest = vehicles.iter().min_by(|a, b| {
        let da = (last.0 - a.cx()).hypot(last.1 - a.cy());
        let db = (last.0 - b.cx()).hypot(last.1 - b.cy());
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some(v) = nearest else { return false };
    let (vx, vy) = (v.cx(), v.cy());
    let first_d = (first.0 - vx).hypot(first.1 - vy);
    let last_d = (last.0 - vx).hypot(last.1 - vy);
    first_d - last_d > frame_diag * 0.10 // got closer by >10% of the frame diagonal
}

/// Mirror of approached_vehicle: did the person move AWAY from a vehicle after
/// appearing (got out of the car and walked off)? Nearest vehicle to the FIRST
/// position; require they ended meaningfully farther from it. Separates a real
/// "got out of a car and walked into view" from a background person who merely
/// appears near parked cars / passing traffic on a street camera.
fn departed_vehicle(t: &Track, vehicles: &[Det], frame_diag: f32) -> bool {
    if t.centers.len() < 3 { return false; }
    let first = t.centers[0];
    let last = *t.centers.last().unwrap();
    let nearest = vehicles.iter().min_by(|a, b| {
        let da = (first.0 - a.cx()).hypot(first.1 - a.cy());
        let db = (first.0 - b.cx()).hypot(first.1 - b.cy());
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some(v) = nearest else { return false };
    let (vx, vy) = (v.cx(), v.cy());
    let first_d = (first.0 - vx).hypot(first.1 - vy);
    let last_d = (last.0 - vx).hypot(last.1 - vy);
    last_d - first_d > frame_diag * 0.10 // moved away by >10% of the frame diagonal
}

/// Goal (2026-06-08): YOLO fires the alerts; the VLM is a last resort. So the
/// classifier outputs DISMISS (clear walk-by) or ALERT (loiter / intrusion /
/// approach / any real person that isn't a clean pass-by). It does NOT escalate
/// — the VLM is only reached via the pipeline's fail-safe when this service is
/// unreachable. Reasons drive the alert text + task_category.
fn classify(t: &Track, total_frames: usize, w: f32, h: f32, vehicles: &[Det]) -> (Decision, TrackVerdict) {
    let frame_diag = (w * w + h * h).sqrt();
    let n = t.centers.len();
    let mut path = 0.0f32;
    for win in t.centers.windows(2) {
        path += ((win[1].0 - win[0].0).powi(2) + (win[1].1 - win[0].1).powi(2)).sqrt();
    }
    let first = t.centers[0];
    let last = *t.centers.last().unwrap();
    let net = ((last.0 - first.0).powi(2) + (last.1 - first.1).powi(2)).sqrt();
    let straightness = if path > 1.0 { net / path } else { 1.0 };
    let span = net / frame_diag;
    let frames_seen = (t.last_frame - t.first_frame + 1).max(1);
    let dwell_frac = frames_seen as f32 / total_frames as f32;
    // Clip-fraction window this track spans (parallels dwell_frac; gives the
    // dashboard the *position* of the dwell, not just its length).
    let start_frac = if total_frames > 0 { t.first_frame as f32 / total_frames as f32 } else { 0.0 };
    let end_frac = if total_frames > 0 {
        ((t.last_frame + 1) as f32 / total_frames as f32).min(1.0)
    } else { 0.0 };
    let avg_h = t.heights.iter().sum::<f32>() / n as f32;
    let close = avg_h / h > 0.30; // person taller than ~30% of frame = near camera
    let motion = path / frame_diag; // total center path, frame-normalized

    // ---- Perspective-invariant motion (operator bugs, 2026-06-08) -----------
    // Frame-fraction motion misreads a straight walk TOWARD/AWAY from the camera
    // (center pinned while the box grows/shrinks) and a DISTANT crosser (few
    // pixels but many strides) as "barely moved" → false loitering. Measure
    // travel in the person's OWN body-lengths, and read box-scale change as
    // depth motion.
    let span_h = avg_h.max(h * 0.04);             // floor: a tiny far box can't fake a walker
    let bodylen_travel = path / span_h;           // total path in body-heights
    let bodylen_per_frame = bodylen_travel / frames_seen as f32;
    let h_min = t.heights.iter().cloned().fold(f32::INFINITY, f32::min).max(1.0);
    let h_max = t.heights.iter().cloned().fold(0.0_f32, f32::max);
    let scale_change = (h_max - h_min) / h_min;   // 0 = box never changed apparent size
    let h_first = t.heights[0].max(1.0);
    let h_last = *t.heights.last().unwrap();
    let height_trend = (h_last - h_first) / h_first; // >0 grew (toward), <0 shrank (away)

    // per-step center deltas (reused by several behaviors)
    let steps: Vec<(f32, f32)> = t.centers.windows(2)
        .map(|wd| (wd[1].0 - wd[0].0, wd[1].1 - wd[0].1)).collect();
    let step_mag = |v: &(f32, f32)| (v.0 * v.0 + v.1 * v.1).sqrt();

    // ---- Heading: 1st-half vs 2nd-half (direction_change + u_turn) ----------
    let mid = n / 2;
    let v1 = (t.centers[mid].0 - first.0, t.centers[mid].1 - first.1);
    let v2 = (last.0 - t.centers[mid].0, last.1 - t.centers[mid].1);
    let m1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
    let m2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();
    let move_thresh = frame_diag * 0.05;
    let cos = if m1 > move_thresh && m2 > move_thresh {
        (v1.0 * v2.0 + v1.1 * v2.1) / (m1 * m2)
    } else { 1.0 };
    let turned = cos < 0.5;                        // heading changed > ~60°
    let u_turn = m1 > frame_diag * 0.10 && m2 > frame_diag * 0.06 && cos <= -0.6; // ~180° in-and-out

    // ---- Phantom / distant-figure gates (operator FPs 2026-06-09) -----------
    // Loitering + pacing otherwise fire on low-confidence mis-detections of
    // inanimate clutter and on tiny far-off figures. Require a real-person
    // confidence and a minimum apparent box size for BOTH; additionally reject a
    // perfectly frozen box (a fixed object — never moves, never changes size) as
    // loitering. A real loiterer still drifts/jitters more than DEAD_BOX_MOTION.
    let track_conf = t.first.score.max(t.last.score); // peak of the endpoints
    let conf_ok = track_conf >= SOFT_MIN_CONF;
    let size_ok = avg_h / h >= SOFT_MIN_HFRAC;
    let soft_ok = conf_ok && size_ok; // credible + close-enough subject for a soft read
    let dead_box = motion < DEAD_BOX_MOTION && scale_change < DEAD_BOX_SCALE;

    // ---- Pacing: ≥2 real heading reversals, lots of legwork, small net ------
    let move_floor = frame_diag * 0.03;
    let mut reversals = 0;
    for i in 1..steps.len() {
        let (a, b) = (steps[i - 1], steps[i]);
        let (ma, mb) = (step_mag(&a), step_mag(&b));
        if ma > move_floor && mb > move_floor && (a.0 * b.0 + a.1 * b.1) / (ma * mb) < -0.3 {
            reversals += 1;
        }
    }
    let pacing = n >= 6 && reversals >= 2 && motion >= 0.30 && span < 0.25
        && soft_ok;

    // ---- Running: sustained fast gait, in body-lengths (cadence-tunable) ----
    let running = n >= 4 && bodylen_per_frame > 1.5 && straightness > 0.5;

    // ---- Sudden stop: clearly moving 1st half, frozen 2nd half --------------
    let half = mid.min(steps.len());
    let path1: f32 = steps[..half].iter().map(step_mag).sum();
    let path2: f32 = steps[half..].iter().map(step_mag).sum();
    let v1n = (path1 / half.max(1) as f32) / frame_diag;
    let v2n = (path2 / steps.len().saturating_sub(half).max(1) as f32) / frame_diag;
    let sudden_stop = n >= 6 && v1n >= 0.012 && v2n <= 0.004 && v1n >= 3.0 * v2n;

    // ---- Camera approach / tamper: box grows until it fills the frame -------
    let final_fill = (t.last.y2 - t.last.y1) / h;
    let camera_approach = n >= 4 && height_trend > 0.0 && (h_last / h_first) >= 2.0 && final_fill >= 0.60;

    // ---- Loitering (FIXED): stands still in-plane AND in depth, sustained ---
    let cluster_ok = {
        let (mut nx, mut xx, mut ny, mut xy) = (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
        for &(cx, cy) in &t.centers { nx = nx.min(cx); xx = xx.max(cx); ny = ny.min(cy); xy = xy.max(cy); }
        (xx - nx) / w < 0.12 && (xy - ny) / h < 0.12
    };
    let is_loiter = bodylen_travel < 1.5 && scale_change < 0.30
        && dwell_frac >= 0.40 && frames_seen >= 6 && cluster_ok
        && soft_ok && !dead_box;

    // ---- Scaling / climbing: rises in frame while apparent size holds -------
    let dy_up = first.1 - last.1;                  // +ve = moved UP the frame
    let dx_abs = (last.0 - first.0).abs();
    let scaling = n >= 4 && dy_up >= 0.20 * h && dy_up >= 2.0 * dx_abs
        && (h_last / h_first) >= 0.8 && motion >= 0.12;

    // ---- Crouch / drop: box short-and-wide, or collapses without receding ---
    let aspect_last = (t.last.x2 - t.last.x1) / (t.last.y2 - t.last.y1).max(1.0);
    let collapsed = (h_last / h_max) <= 0.6 && last.1 >= first.1 - 0.05 * h;
    let crouch = n >= 4 && dwell_frac >= 0.30 && t.last.y1 > 2.0
        && (aspect_last >= 0.9 || (collapsed && avg_h / h >= 0.18));

    // ---- Erratic / wandering: covered ground on a low-straightness path -----
    let erratic = n >= 6 && motion >= 0.25 && straightness <= 0.45 && dwell_frac >= 0.40;

    // ---- Intrusion (FIXED): vanished WELL INSIDE the frame, mid-motion ------
    // Test the real LAST BOX EDGES (not the center) + extrapolate velocity, so a
    // tall box exiting the BOTTOM (toward camera) or a small box exiting the TOP
    // (receding) reads as LEAVING. Require prior motion + a near target + a
    // non-receding trend so a stationary dropout is never "intrusion".
    let lb = t.last;
    let bh = lb.y2 - lb.y1;
    let k = n.min(4);
    let (vx, vy) = if k >= 2 {
        let a = t.centers[n - k];
        ((last.0 - a.0) / (k - 1) as f32, (last.1 - a.1) / (k - 1) as f32)
    } else { (0.0, 0.0) };
    let look = (MAX_GAP + 1) as f32;
    let m_side = w * 0.05;
    let m_top = h * 0.10;
    let m_bot = h * 0.04 + 0.15 * bh;             // grows with box height — the perspective fix
    let exits_edge =
           lb.x1 <= m_side     || (lb.x1 + vx * look) <= 0.0
        || lb.x2 >= w - m_side || (lb.x2 + vx * look) >= w
        || lb.y1 <= m_top      || (lb.y1 + vy * look) <= 0.0
        || lb.y2 >= h - m_bot  || (lb.y2 + vy * look) >= h;
    let interior = lb.x1 > w * 0.10 && lb.x2 < w - w * 0.10
        && lb.y1 > h * 0.10 && lb.y2 < h - m_bot;
    let speed = (vx * vx + vy * vy).sqrt() / frame_diag;
    let receding = height_trend < -0.20;
    let ended_early = t.last_frame + (MAX_GAP + 1) < total_frames;
    let moved = motion > 0.10 && span > 0.05 && speed > 0.008;
    let real_target = avg_h / h > 0.22;
    let vanished_mid = ended_early && !exits_edge && interior && moved && real_target && !receding;

    // ---- Transient: short-lived near-stationary dropout (NOT a real event) --
    let transient = dwell_frac < 0.25 && motion < 0.06 && span < 0.04;

    // ---- Vehicle interactions (kept; person ends at a car = got in, etc.) ---
    let started_late = t.first_frame > 2;
    // Getting INTO a car = vanish MID-FRAME at the car, having moved toward it.
    // `interior` (last box well inside ALL edges) separates "got in a car" from
    // "walked out of the frame" (operator 2026-06-09: people leaving the frame
    // were being marked entered_vehicle). Position-based, so it doesn't fire on
    // the velocity of someone walking toward a car the way exits_edge would.
    let entered_vehicle = ended_early
        && interior
        && near_vehicle(&t.last, vehicles, frame_diag)
        && approached_vehicle(t, vehicles, frame_diag);
    // Mirror of entered_vehicle: appeared MID-FRAME at a car and walked AWAY from
    // it = got out of a car. A person who appears at the frame edge (walked into
    // view) near parked cars / passing traffic is NOT arriving by vehicle.
    let first_interior = first.0 > w * 0.10 && first.0 < w - w * 0.10
        && first.1 > h * 0.10 && first.1 < h - h * 0.10;
    let arrived_by_vehicle = started_late
        && first_interior
        && near_vehicle(&t.first, vehicles, frame_diag)
        && departed_vehicle(t, vehicles, frame_diag);

    // ---- Approach (refined): came closer AND stayed (not walking under) -----
    let approach = close && motion < 0.18 && dwell_frac >= 0.45
        && height_trend >= 0.0 && lb.y2 < h - m_bot;

    // Collect every behavior that fired; the track's reason is the highest-
    // severity ALERT (reason_severity), so a low-priority context (e.g.
    // arrived_by_vehicle) never masks a genuine threat on the same track.
    // DISMISS reasons stand only when nothing alerted; `transient`/`edge_exit`
    // make the two operator-reported false positives explicit dismissals.
    // Box CLIPPING at a frame edge corrupts both box geometry (height/aspect →
    // crouch/scaling) AND the box CENTER (the centroid of a truncated box jumps
    // inward, faking running / turns / stops / wander). The decision picks the
    // highest-severity alert BEFORE the edge_exit dismiss, so EVERY reason that
    // reads center-trajectory or box-shape must be gated on !exits_edge or a
    // plain walk-OUT masks the correct edge_exit dismiss (edge-fragility audit
    // 2026-06-09; c6163728 / 70b6aa50 / 73dc129c were all frame-exits).
    // ROBUST against edge-exits (left ungated on !exits_edge): camera_approach
    // (needs 2x grow + 60% fill), pacing (needs 2 reversals + small span),
    // loitering (tight cluster + low travel), multi_person / converging (need 2
    // distinct tracks) — a single walk-out can't fake those. intrusion/entered/
    // arrived carry their own gates. SEPARATELY, loitering + pacing now also
    // require conf_ok + size_ok (and loitering rejects a dead_box) so they don't
    // fire on clutter mis-detections or tiny distant figures (FPs 2026-06-09).
    let mut alerts: Vec<&'static str> = Vec::new();
    if entered_vehicle { alerts.push("entered_vehicle"); }
    if camera_approach { alerts.push("camera_approach"); }
    // running is NOT gated on !exits_edge: someone FLEEING is high-value and
    // exits by definition. Its bodylen_per_frame>1.5 bar already rejects normal
    // walk-outs (only a rare clip-jump artifact can fake it).
    if running { alerts.push("running"); }
    // Soft trajectory reads (u_turn/direction_change/sudden_stop/erratic/approach,
    // + pacing/loitering above) also require soft_ok so a low-confidence clutter
    // mis-detection or a tiny distant figure can't fire them — and, crucially,
    // can't escape to the server's pacing/prolonged_loitering upgrade.
    if u_turn && !exits_edge && soft_ok { alerts.push("u_turn"); }
    else if turned && !exits_edge && soft_ok { alerts.push("direction_change"); }
    if pacing { alerts.push("pacing"); }
    if vanished_mid { alerts.push("intrusion"); }
    if is_loiter { alerts.push("loitering"); }
    if sudden_stop && !exits_edge && soft_ok { alerts.push("sudden_stop"); }
    if scaling && !exits_edge { alerts.push("scaling"); }
    if crouch && !exits_edge { alerts.push("crouch"); }
    if erratic && !exits_edge && soft_ok { alerts.push("erratic"); }
    if approach && !exits_edge && soft_ok { alerts.push("approach"); }
    if arrived_by_vehicle { alerts.push("arrived_by_vehicle"); }

    let (decision, reason): (Decision, &'static str) = if n < 3 {
        (Decision::Dismiss, "blip") // too few points to judge a trajectory
    } else if transient {
        // Short near-stationary dropout (occlusion / flicker / foliage). Too little
        // reliable signal to label ANY behavior — dismiss before the alert ladder
        // so a noisy box can't fake crouch / intrusion / etc. (operator FP).
        (Decision::Dismiss, "transient")
    } else if let Some(best) = alerts.iter().copied().max_by_key(|&r| reason_severity(r)) {
        (Decision::Alert, best)
    } else if exits_edge {
        (Decision::Dismiss, "edge_exit") // walked out of view (any edge, incl top/bottom)
    } else {
        (Decision::Dismiss, "walk_by") // moving through / past = normal
    };
    (decision, TrackVerdict { id: t.id, n, straightness, span, dwell_frac, start_frac, end_frac, dwell_s: None,
        decision: decision.as_str().into(), reason: reason.into(),
        keypoints: Vec::new(), behavior: None, behavior_conf: None,
        sus_score: None, sus_alert: None, route: None, identity: None })
}

#[derive(Serialize)]
struct TriageResult { decision: String, route: String, reason: String, detect_ms: u64, frames: usize,
    // Raw NN one-liner for the operator (NO prose): person count, per-person
    // behavior + confidence% + dwell estimate + route. Built by nn_summary().
    summary: String,
    // Union of all real tracks' temporal windows (clip-fraction). The dashboard
    // trims the VLM's clip/frames to this span. Absent when no real track had a
    // window (synthetic-only / no persons) → caller uses the full clip.
    #[serde(skip_serializing_if = "Option::is_none")]
    active_window: Option<ActiveWindow>,
    tracks: Vec<TrackVerdict> }

/// Compact raw-NN summary the operator reads directly — no sentence-building.
/// e.g. "2 ppl | #1 pacing 87% ~52s →alert | #2 walk_by 3% ~4s →dismiss | +multi_person →vlm"
/// Real tracks (id != usize::MAX) carry behavior/conf/dwell; synthetic
/// event-level verdicts (multi-person, zone/line) are appended as "+reason →route".
fn nn_summary(verdicts: &[TrackVerdict]) -> String {
    let people: Vec<&TrackVerdict> = verdicts.iter().filter(|v| v.id != usize::MAX).collect();
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("{} ppl", people.len()));
    for (i, v) in people.iter().enumerate() {
        // What the NN actually said: the classified behavior if present, else the
        // geometric reason. Confidence% = behavior_conf when classified, else the
        // sus score (both already in [0,1]); "--" when neither ran.
        let label = v.behavior.as_deref().unwrap_or(&v.reason);
        let conf = v.behavior_conf.or(v.sus_score)
            .map(|c| format!("{}%", (c * 100.0).round() as i32))
            .unwrap_or_else(|| "--%".into());
        let dwell = v.dwell_s.map(|s| format!(" ~{}s", s.round() as i32)).unwrap_or_default();
        let route = v.route.as_deref().unwrap_or(&v.decision);
        parts.push(format!("#{} {} {}{} \u{2192}{}", i + 1, label, conf, dwell, route));
    }
    for v in verdicts.iter().filter(|v| v.id == usize::MAX) {
        let route = v.route.as_deref().unwrap_or(&v.decision);
        parts.push(format!("+{} \u{2192}{}", v.reason, route));
    }
    parts.join(" | ")
}

/// Route ONE person track: urgent break-in shapes (severity ≥ 8) fire now; the
/// rest go through the relevance×confidence bands; a geometry/behavior alert the
/// policy didn't grade still floors at VLM so a flagged event never silently drops.
fn route_track(decision: &Decision, reason: &str, sus_score: Option<f32>,
               behavior_conf: Option<f32>, pol: Option<&sus::SusPolicy>) -> sus::Route {
    if reason_severity(reason) >= 8 { return sus::Route::Alert; }
    let banded = match (pol, sus_score) {
        (Some(p), Some(s)) => p.route(s, behavior_conf),
        _ => sus::Route::Dismiss,
    };
    if banded == sus::Route::Dismiss && *decision == Decision::Alert { sus::Route::Vlm } else { banded }
}

fn reason_severity(r: &str) -> u8 {
    match r {
        "camera_approach" => 10,
        "intrusion" => 9,
        "door_entry" => 8,
        "zone_intrusion" => 8,
        "entered_vehicle" => 7,
        "line_cross" => 6,
        "pacing" => 6,
        "running" => 6,
        "u_turn" => 5,
        "scaling" => 5,
        "crouch" => 5,
        "converging" => 5,
        "direction_change" => 4,
        "sudden_stop" => 4,
        "multi_person" => 4,
        "loitering" => 3,
        "approach" => 3,
        "erratic" => 2,
        "arrived_by_vehicle" => 1,
        _ => 0,
    }
}

/// Event-level behaviors that need the WHOLE track set, not one track: multiple
/// distinct people present at once, and two people converging. classify() is
/// strictly per-track, so these are computed here over all person tracks.
/// (Thin spatial-only wrapper over `event_level_id`; used by tests + callers
/// that have no identity signatures.)
#[allow(dead_code)]
fn event_level(tracks: &[Track], w: f32, h: f32) -> Option<&'static str> {
    event_level_id(tracks, w, h, None, &identity::IdentityConfig::default())
}

/// As `event_level`, but when per-track identity signatures are supplied (v3),
/// a pair that identity confirms is the SAME person (one fragmented by an
/// occlusion) is NOT counted as two people — it can neither raise `multi_person`
/// nor `converging`. `id_sigs` is aligned to the `tracks` slice (None per entry
/// where no signature was built). Falls back to spatial-only when absent.
fn event_level_id(tracks: &[Track], w: f32, h: f32,
                  id_sigs: Option<&[Option<identity::IdSig>]>,
                  id_cfg: &identity::IdentityConfig) -> Option<&'static str> {
    let frame_diag = (w * w + h * h).sqrt();
    let ppl: Vec<(usize, &Track)> = tracks.iter().enumerate()
        .filter(|(_, t)| t.cls == PERSON_CLASS && t.centers.len() >= 3
            && (t.heights.iter().sum::<f32>() / t.heights.len() as f32) / h >= 0.12)
        .collect();
    if ppl.len() < 2 { return None; }
    let dist = |p: (f32, f32), q: (f32, f32)| ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt();
    let same_person = |ia: usize, ib: usize| -> bool {
        match id_sigs {
            Some(sigs) => match (sigs.get(ia).and_then(|s| s.as_ref()),
                                 sigs.get(ib).and_then(|s| s.as_ref())) {
                (Some(a), Some(b)) =>
                    identity::identity_match(a, b, id_cfg) == identity::IdMatch::Same,
                _ => false,
            },
            None => false,
        }
    };
    let (mut multi, mut converging) = (false, false);
    for i in 0..ppl.len() {
        for j in (i + 1)..ppl.len() {
            let ((ia, a), (ib, b)) = (ppl[i], ppl[j]);
            // must be alive in the same window (MAX_GAP already bridges dropouts)
            if a.first_frame.max(b.first_frame) > a.last_frame.min(b.last_frame) { continue; }
            // identity says it's one person split across an occlusion → not two
            if same_person(ia, ib) { continue; }
            let (af, al) = (*a.centers.first().unwrap(), *a.centers.last().unwrap());
            let (bf, bl) = (*b.centers.first().unwrap(), *b.centers.last().unwrap());
            let (d_start, d_end) = (dist(af, bf), dist(al, bl));
            // spatially distinct somewhere → two real people, not one split ID
            if d_start >= 0.10 * frame_diag || d_end >= 0.10 * frame_diag {
                multi = true;
                if d_start >= 0.20 * frame_diag && d_end <= 0.10 * frame_diag
                    && (d_start - d_end) >= 0.12 * frame_diag {
                    converging = true;
                }
            }
        }
    }
    if converging { Some("converging") } else if multi { Some("multi_person") } else { None }
}

// ---- v3 identity: reversal-split audit ------------------------------------
// The geometry tracker can stitch person-A-exits to person-B-enters into one
// track that reverses direction — a false pacing/u_turn/direction_change. When
// a track carries one of those reasons, split it at its turn point and ask the
// fused identity check whether the two halves are the SAME person; if they're a
// confidently DIFFERENT person, hand back two tracks (each re-classifies to a
// clean walk_by). Same/Unknown leaves the track whole — a real pacer's halves
// are the same person, so this never fragments one.

fn is_reversal_reason(r: &str) -> bool {
    matches!(r, "pacing" | "u_turn" | "direction_change")
}

/// The frame index at which the track's first-half and second-half headings
/// point most oppositely (cos most negative) — the crossover. Requires a real
/// reversal (cos < -0.3) and both halves ≥ 3 points.
fn reversal_split_index(t: &Track) -> Option<usize> {
    let c = &t.centers;
    let n = c.len();
    if n < 6 { return None; }
    let (mut best_k, mut best_cos) = (None, -0.3f32);
    for k in 2..=n - 3 {
        let v1 = (c[k].0 - c[0].0, c[k].1 - c[0].1);
        let v2 = (c[n - 1].0 - c[k].0, c[n - 1].1 - c[k].1);
        let m1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
        let m2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();
        if m1 < 1e-3 || m2 < 1e-3 { continue; }
        let cos = (v1.0 * v2.0 + v1.1 * v2.1) / (m1 * m2);
        if cos < best_cos { best_cos = cos; best_k = Some(k); }
    }
    best_k
}

/// Build the fused appearance signature of one track segment `[lo, hi]`: color
/// from the segment's per-frame sigs (always); body (OSNet) + face (ArcFace from
/// pose keypoints) from the segment's LARGEST box (best crop) when those models
/// are loaded and the crop clears the size/confidence gates.
#[allow(clippy::too_many_arguments)]
fn seg_idsig(t: &Track, lo: usize, hi: usize, frames: &[image::RgbImage],
             pose_frames: &[Vec<pose::PoseDet>], reid: Option<&identity::ReidCtx>,
             face: Option<&identity::FaceCtx>, cfg: &identity::IdentityConfig) -> identity::IdSig {
    let hi = hi.min(t.centers.len().saturating_sub(1));
    let mut sig = identity::IdSig::default();
    if t.color_sigs.len() > hi {
        sig.color = identity::color_mean(&t.color_sigs[lo..=hi]);
    }
    if t.boxes.len() <= hi || t.frame_idxs.len() <= hi { return sig; }
    // representative frame = the largest box in [lo, hi]
    let (mut best_k, mut best_h) = (lo, -1.0f32);
    for k in lo..=hi {
        if t.heights[k] > best_h { best_h = t.heights[k]; best_k = k; }
    }
    let bx = t.boxes[best_k];
    let fi = t.frame_idxs[best_k];
    let Some(frame) = frames.get(fi) else { return sig };
    if let Some(r) = reid {
        if bx.h() >= cfg.reid_min_box_h {
            sig.body = r.embed(frame, &bx).ok();
        }
    }
    if let Some(fc) = face {
        let pd = pose_frames.get(fi).and_then(|dets| {
            let (cx, cy) = (bx.cx(), bx.cy());
            dets.iter()
                .map(|p| (((p.x1 + p.x2) / 2.0 - cx).powi(2) + ((p.y1 + p.y2) / 2.0 - cy).powi(2), p))
                .filter(|(d2, _)| *d2 < (0.6 * bx.h()).powi(2))
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .map(|(_, p)| p)
        });
        if let Some(p) = pd {
            sig.face = fc.embed_from_kps(frame, &p.kps, cfg.face_kp_conf);
        }
    }
    sig
}

/// Whole-track appearance signature (for the event-level same-person merge).
fn track_idsig(t: &Track, frames: &[image::RgbImage], pose_frames: &[Vec<pose::PoseDet>],
               reid: Option<&identity::ReidCtx>, face: Option<&identity::FaceCtx>,
               cfg: &identity::IdentityConfig) -> identity::IdSig {
    seg_idsig(t, 0, t.centers.len().saturating_sub(1), frames, pose_frames, reid, face, cfg)
}

/// Partition a track at `k` (the pivot belongs to BOTH halves so each is a valid
/// trajectory) into two new tracks tagged with their origin + deciding signal.
fn split_track(t: &Track, k: usize, id_a: usize, id_b: usize, by: &str) -> (Track, Track) {
    let n = t.centers.len();
    let mk = |lo: usize, hi: usize, id: usize| -> Track {
        let frame_idxs = t.frame_idxs[lo..=hi].to_vec();
        let boxes = t.boxes[lo..=hi].to_vec();
        Track {
            id, cls: t.cls,
            first_frame: frame_idxs[0], last_frame: *frame_idxs.last().unwrap(),
            first: boxes[0], last: *boxes.last().unwrap(),
            centers: t.centers[lo..=hi].to_vec(),
            heights: t.heights[lo..=hi].to_vec(),
            frame_idxs, boxes,
            color_sigs: t.color_sigs[lo..=hi].to_vec(),
            split_from: Some(t.id), split_by: Some(by.to_string()),
        }
    };
    (mk(0, k, id_a), mk(k, n - 1, id_b))
}

/// Expand any reversal-shaped person track the identity check judges to be two
/// different people into two tracks; pass everything else through unchanged.
#[allow(clippy::too_many_arguments)]
fn split_reversal_tracks(tracks: Vec<Track>, frames: &[image::RgbImage],
                         pose_frames: &[Vec<pose::PoseDet>], reid: Option<&identity::ReidCtx>,
                         face: Option<&identity::FaceCtx>, cfg: &identity::IdentityConfig,
                         w: f32, h: f32, total_frames: usize) -> Vec<Track> {
    let mut next_id = tracks.iter().map(|t| t.id).max().map_or(0, |m| m + 1);
    let mut out: Vec<Track> = Vec::with_capacity(tracks.len());
    for t in tracks {
        let n = t.centers.len();
        let consistent = t.boxes.len() == n && t.color_sigs.len() == n && t.frame_idxs.len() == n;
        if t.cls != PERSON_CLASS || n < 6 || !consistent {
            out.push(t);
            continue;
        }
        let reason = classify(&t, total_frames, w, h, &[]).1.reason;
        if !is_reversal_reason(&reason) {
            out.push(t);
            continue;
        }
        let Some(k) = reversal_split_index(&t) else { out.push(t); continue; };
        let a = seg_idsig(&t, 0, k, frames, pose_frames, reid, face, cfg);
        let b = seg_idsig(&t, k, n - 1, frames, pose_frames, reid, face, cfg);
        let (verdict, by) = identity::match_explain(&a, &b, cfg);
        if verdict != identity::IdMatch::Different {
            out.push(t);
            continue;
        }
        let (ta, tb) = split_track(&t, k, next_id, next_id + 1, by);
        next_id += 2;
        out.push(ta);
        out.push(tb);
    }
    out
}

/// Door / restricted-area analytics: did any PERSON track enter a drawn zone or
/// cross a drawn line? The footpoint (bottom-centre of the box, where the person
/// stands) is tested. `meta` coords are frame fractions; we scale by (w, h).
/// Returns the single most-severe reason fired across all tracks.
fn zone_line_reason(tracks: &[Track], w: f32, h: f32, meta: &ZonesMeta) -> Option<&'static str> {
    if meta.zones.is_empty() && meta.lines.is_empty() { return None; }
    let bump = |r: &'static str, cur: &mut Option<&'static str>| {
        if cur.map_or(0, |c| reason_severity(c)) < reason_severity(r) { *cur = Some(r); }
    };
    let mut hit: Option<&'static str> = None;
    for t in tracks.iter().filter(|t| t.cls == PERSON_CLASS && t.centers.len() >= 2) {
        let foot: Vec<(f32, f32)> = t.centers.iter().zip(t.heights.iter())
            .map(|(&(cx, cy), &hb)| (cx, cy + hb / 2.0)).collect();
        // zones: ended INSIDE a zone they were OUTSIDE of at the start = entry
        for z in &meta.zones {
            if z.polygon.len() < 3 { continue; }
            let poly: Vec<(f32, f32)> = z.polygon.iter().map(|p| (p[0] * w, p[1] * h)).collect();
            if point_in_poly(*foot.last().unwrap(), &poly) && !point_in_poly(foot[0], &poly) {
                bump("zone_intrusion", &mut hit);
            }
        }
        // lines: net crossing direction → operator semantics (in_direction)
        for l in &meta.lines {
            let (a, b) = ((l.a[0] * w, l.a[1] * h), (l.b[0] * w, l.b[1] * h));
            let net: i32 = foot.windows(2).map(|win| crosses_line(win[0], win[1], a, b)).sum();
            if net == 0 { continue; }
            let geo_in = net > 0;
            let is_in = if l.in_direction == "b_to_a" { !geo_in } else { geo_in };
            bump(if is_in { "door_entry" } else { "line_cross" }, &mut hit);
        }
    }
    hit
}

/// Loaded behavior NN + its routing policy. Built once at startup from
/// BEHAVIOR_MODEL / BEHAVIOR_ALERT_CLASSES / BEHAVIOR_MIN_CONF.
struct BehaviorCtx {
    session: Mutex<Session>,
    meta: behavior::ModelMeta,
    /// Classes allowed to ESCALATE a track to alert (CSV env). Empty by
    /// default — with only walking/standing trained, the NN is informational
    /// and the rule engine keeps alert authority. Populate as alert-grade
    /// classes (crouching, climbing, …) earn their stripes in shadow.
    alert_classes: Vec<String>,
    min_conf: f32,
}

impl BehaviorCtx {
    fn from_env() -> Result<Option<Self>> {
        let Ok(model) = std::env::var("BEHAVIOR_MODEL") else { return Ok(None) };
        if model.trim().is_empty() { return Ok(None); }
        let meta = behavior::load_meta(&model);
        anyhow::ensure!(!meta.classes.is_empty(),
            "BEHAVIOR_MODEL set but {model}.classes.json is missing/empty — \
             the service refuses to guess class names");
        anyhow::ensure!(meta.channels == behavior::CHANNELS,
            "behavior model trained with {} channels, this build expects {} — \
             retrain or upgrade in lockstep", meta.channels, behavior::CHANNELS);
        let alert_classes: Vec<String> = std::env::var("BEHAVIOR_ALERT_CLASSES")
            .unwrap_or_default()
            .split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
            .collect();
        let min_conf: f32 = std::env::var("BEHAVIOR_MIN_CONF").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(0.8);
        Ok(Some(Self { session: Mutex::new(build_session(&model)?), meta, alert_classes, min_conf }))
    }
}

/// Should this NN classification escalate the track to alert? Pure so the
/// routing rule is unit-testable without a model.
fn behavior_escalates(class: &str, conf: f32, alert_classes: &[String], min_conf: f32) -> bool {
    conf >= min_conf && alert_classes.iter().any(|c| c == class)
}

/// Minimum pose-matched frames before the NN gets a say — below this the
/// sequence is mostly padding and the logits are prior, not signal.
const BEHAVIOR_MIN_POSE_FRAMES: usize = 4;

/// Is the track's footpoint (bottom-centre of its last box) inside any
/// operator zone? Used as the `in_zone` context multiplier for sus scoring.
fn foot_in_zone(foot: (f32, f32), w: f32, h: f32, meta: &ZonesMeta) -> bool {
    meta.zones.iter().any(|z| {
        let poly: Vec<(f32, f32)> =
            z.polygon.iter().map(|p| (p[0] * w, p[1] * h)).collect();
        point_in_poly(foot, &poly)
    })
}

#[allow(clippy::too_many_arguments)]
fn run_triage(session: &Session, frames: &[image::RgbImage], meta: &ZonesMeta,
              pose_session: Option<&Session>, beh: Option<&BehaviorCtx>,
              night: bool, sus_policy: Option<&sus::SusPolicy>,
              clip_seconds: f32,
              reid: Option<&identity::ReidCtx>, face: Option<&identity::FaceCtx>,
              id_cfg: &identity::IdentityConfig, id_on: bool) -> Result<TriageResult> {
    if frames.is_empty() {
        return Ok(TriageResult { decision: "dismiss".into(), route: "dismiss".into(), reason: "no_person".into(), detect_ms: 0, frames: 0, summary: "0 ppl".into(), active_window: None, tracks: vec![] });
    }
    let (w, h) = (frames[0].width() as f32, frames[0].height() as f32);
    let t0 = Instant::now();
    let mut per_frame = Vec::with_capacity(frames.len());
    // v3: per-detection clothing-color signature, computed in the SAME pass we
    // already decode each frame and detect — no extra image I/O. Off (empty)
    // when identity gating is disabled, restoring exact geometry-only behavior.
    let mut per_sig: Vec<Vec<identity::ColorSig>> = Vec::with_capacity(frames.len());
    for f in frames {
        let dets = detect(session, f)?;
        let sigs = if id_on {
            dets.iter().map(|d| identity::color_sig(f, d)).collect()
        } else { Vec::new() };
        per_sig.push(sigs);
        per_frame.push(dets);
    }
    let det_ms = t0.elapsed().as_millis() as u64;
    // Optional pose pass (POSE_MODEL env): full-frame inference once per
    // frame — cost is independent of person count. Keypoints are carried
    // data only; behavior verdicts below are untouched by them. Computed BEFORE
    // the identity split so the face embedder can borrow a track's keypoints.
    let pose_frames: Vec<Vec<pose::PoseDet>> = match pose_session {
        Some(ps) => {
            let mut v = Vec::with_capacity(frames.len());
            for f in frames { v.push(pose::detect_pose(ps, f, pose::POSE_CONF)?); }
            v
        }
        None => Vec::new(),
    };
    let tracks = track(&per_frame, &per_sig, id_cfg, id_on);
    // v3 reversal-split audit: a track classified as pacing/u_turn/direction_change
    // is the prime "two people merged into one reversing path" suspect. Split it
    // and, if the two halves are a DIFFERENT person by clothing/body/face, emit
    // them as two tracks (each re-classifies to a plain walk_by). Same/Unknown →
    // left untouched, so a genuine single-person pacer is never fragmented.
    let tracks = if id_on {
        split_reversal_tracks(tracks, frames, &pose_frames, reid, face, id_cfg, w, h, frames.len())
    } else { tracks };
    // Vehicle tracks are context, not events: one representative box each (they
    // barely move). We classify only the people, passing the cars as context.
    let vehicles: Vec<Det> = tracks.iter().filter(|t| is_vehicle(t.cls)).map(|t| t.last).collect();
    let mut verdicts = Vec::new();
    let mut decisions = Vec::new();
    for t in &tracks {
        if t.cls != PERSON_CLASS { continue; } // cars/trucks are context only
        let (mut d, mut v) = classify(t, frames.len(), w, h, &vehicles);
        // Attach skeletons: for each frame this track was seen in, the pose
        // detection whose center is nearest the track's center there (within
        // 0.6 box-heights — persons rarely overlap that tightly; a miss is a
        // null entry, never a wrong skeleton).
        if !pose_frames.is_empty() {
            let assoc: Vec<Option<&pose::PoseDet>> = t.frame_idxs.iter().enumerate()
                .map(|(k, &fi)| {
                    let (cx, cy) = t.centers[k];
                    let h_box = t.heights[k];
                    pose_frames.get(fi).and_then(|dets| {
                        dets.iter()
                            .map(|p| {
                                let pcx = (p.x1 + p.x2) / 2.0;
                                let pcy = (p.y1 + p.y2) / 2.0;
                                (((pcx - cx).powi(2) + (pcy - cy).powi(2)).sqrt(), p)
                            })
                            .filter(|(dist, _)| *dist < 0.6 * h_box)
                            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                            .map(|(_, p)| p)
                    })
                })
                .collect();
            v.keypoints = assoc.iter()
                .map(|o| o.map(|p| p.kps.iter().map(|&(x, y, c)| [x, y, c]).collect()))
                .collect();
            // Behavior NN (BEHAVIOR_MODEL env): featurize the pose trajectory
            // exactly like the training corpus — the matched pose detection
            // supplies BOTH box and skeleton (same box source training used),
            // and unseen frames inside the track span become explicit gaps.
            if let Some(bc) = beh {
                if let (Some(&lo), Some(&hi)) = (t.frame_idxs.first(), t.frame_idxs.last()) {
                    let mut bframes: Vec<Option<behavior::TrackFrame>> = vec![None; hi - lo + 1];
                    let mut present = 0usize;
                    for (k, &fi) in t.frame_idxs.iter().enumerate() {
                        if let Some(p) = assoc[k] {
                            let mut kps = [[0f32; 3]; pose::NUM_KP];
                            for (j, &(x, y, c)) in p.kps.iter().enumerate().take(pose::NUM_KP) {
                                kps[j] = [x, y, c];
                            }
                            bframes[fi - lo] = Some(behavior::TrackFrame {
                                bbox: [p.x1, p.y1, p.x2, p.y2], kps });
                            present += 1;
                        }
                    }
                    if present >= BEHAVIOR_MIN_POSE_FRAMES {
                        let (flat, _vis) = behavior::normalize_track(
                            &bframes, bc.meta.seq_len, bc.meta.min_joint_conf);
                        let probs = {
                            let sess = bc.session.lock().unwrap();
                            behavior::classify(&sess, &flat, bc.meta.seq_len)
                        };
                        if let Ok(probs) = probs {
                            if let Some((ci, &p)) = probs.iter().enumerate()
                                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)) {
                                let cls = bc.meta.classes.get(ci).cloned()
                                    .unwrap_or_else(|| format!("class{ci}"));
                                if behavior_escalates(&cls, p, &bc.alert_classes, bc.min_conf)
                                    && d != Decision::Alert {
                                    d = Decision::Alert;
                                    v.decision = "alert".into();
                                    v.reason = format!("behavior_{cls}");
                                }
                                v.behavior = Some(cls);
                                v.behavior_conf = Some((p * 1000.0).round() / 1000.0);
                            }
                        }
                    }
                }
            }
        }
        // Suspicion score: the flag/reason weight, multiplied by context
        // (night + whether the footpoint sits in an operator zone). Additive —
        // only populated when a SUS_POLICY is loaded.
        if let Some(pol) = sus_policy {
            let foot = (t.last.cx(), t.last.y2);
            let (s, a) = pol.score(v.behavior.as_deref(), &v.reason, night,
                                   foot_in_zone(foot, w, h, meta));
            v.sus_score = Some((s * 1000.0).round() / 1000.0);
            v.sus_alert = Some(a);
        }
        // Three-way route: relevance × confidence, with the urgent override.
        v.route = Some(route_track(&d, &v.reason, v.sus_score, v.behavior_conf, sus_policy)
            .as_str().into());
        // Absolute dwell estimate the operator reads directly (frames are sampled
        // across the clip window, so dwell_frac × clip_seconds ≈ seconds present).
        v.dwell_s = Some((v.dwell_frac * clip_seconds * 10.0).round() / 10.0);
        // v3: report when this track was carved out of a merged one by the audit.
        if let Some(orig) = t.split_from {
            v.identity = Some(IdentityInfo {
                split_from: orig,
                decided_by: t.split_by.clone().unwrap_or_else(|| "color".into()),
            });
        }
        decisions.push(d); verdicts.push(v);
    }
    // Event-level signals (multiple people / converging) over the full track set.
    // Soft signals → the manager (VLM) judges; two people near each other isn't
    // an emergency on its own. v3: per-track identity sigs let a single person
    // fragmented by an occlusion NOT be miscounted as two, and let two unrelated
    // clean walk-bys settle as a recorded multi_person without an escalation.
    let id_sigs: Option<Vec<Option<identity::IdSig>>> = if id_on {
        Some(tracks.iter().map(|t| {
            if t.cls != PERSON_CLASS || t.centers.is_empty() { return None; }
            Some(track_idsig(t, frames, &pose_frames, reid, face, id_cfg))
        }).collect())
    } else { None };
    if let Some(reason) = event_level_id(&tracks, w, h, id_sigs.as_deref(), id_cfg) {
        // multi_person of all-dismiss walk-bys is benign (record, don't escalate);
        // any non-dismiss member, or a converging pair, still goes to the VLM.
        let any_live = verdicts.iter().any(|v| v.id != usize::MAX
            && v.route.as_deref().is_some_and(|r| r != "dismiss"));
        let mp_route = if reason == "multi_person" && !any_live { "dismiss" } else { "vlm" };
        if mp_route != "dismiss" { decisions.push(Decision::Alert); }
        verdicts.push(TrackVerdict { id: usize::MAX, n: 0, straightness: 0.0, span: 0.0,
            dwell_frac: 0.0, start_frac: 0.0, end_frac: 0.0, dwell_s: None, decision: "alert".into(), reason: reason.into(),
            keypoints: Vec::new(), behavior: None, behavior_conf: None,
            sus_score: None, sus_alert: None, route: Some(mp_route.into()), identity: None });
    }
    // Zone / door analytics (operator-drawn zones + lines, if any were POSTed).
    // The operator drew that boundary deliberately → crossing it IS the alert.
    if let Some(reason) = zone_line_reason(&tracks, w, h, meta) {
        decisions.push(Decision::Alert);
        verdicts.push(TrackVerdict { id: usize::MAX, n: 0, straightness: 0.0, span: 0.0,
            dwell_frac: 0.0, start_frac: 0.0, end_frac: 0.0, dwell_s: None, decision: "alert".into(), reason: reason.into(),
            keypoints: Vec::new(), behavior: None, behavior_conf: None,
            sus_score: None, sus_alert: None, route: Some("alert".into()), identity: None });
    }
    let event = if decisions.iter().any(|d| *d == Decision::Alert) { Decision::Alert } else { Decision::Dismiss };
    // Event route = the most severe track route (alert > vlm > dismiss).
    let event_route = verdicts.iter()
        .filter_map(|v| v.route.as_deref())
        .map(|r| match r { "alert" => sus::Route::Alert, "vlm" => sus::Route::Vlm, _ => sus::Route::Dismiss })
        .max_by_key(|r| r.rank())
        .unwrap_or(if event == Decision::Alert { sus::Route::Alert } else { sus::Route::Dismiss });
    // Event reason = highest-severity reason among tracks that aren't dismissed,
    // so the headline reason matches what actually drove the route.
    let reason = verdicts.iter()
        .filter(|v| v.route.as_deref() != Some("dismiss") && v.decision == "alert")
        .max_by_key(|v| reason_severity(&v.reason))
        .map(|v| v.reason.clone())
        .unwrap_or_else(|| "none".into());
    let summary = nn_summary(&verdicts);
    // Event-level active window = union over real tracks (exclude synthetic
    // id==usize::MAX rows and degenerate 0-span rows). The dashboard trims the
    // VLM's input to this span; None → no real window → caller uses full clip.
    let active_window = {
        let reals: Vec<&TrackVerdict> = verdicts.iter()
            .filter(|v| v.id != usize::MAX && v.end_frac > v.start_frac)
            .collect();
        if reals.is_empty() {
            None
        } else {
            let start = reals.iter().map(|v| v.start_frac).fold(1.0f32, f32::min);
            let end = reals.iter().map(|v| v.end_frac).fold(0.0f32, f32::max);
            Some(ActiveWindow { start_frac: start, end_frac: end })
        }
    };
    Ok(TriageResult { decision: event.as_str().into(), route: event_route.as_str().into(),
        reason, detect_ms: det_ms, frames: frames.len(), summary, active_window, tracks: verdicts })
}

fn build_session(model: &str) -> Result<Session> {
    let builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?;
    // CoreML (Apple GPU / ANE) on macOS — production. Elsewhere ort's default
    // CPU EP, so the service builds and runs on non-Mac dev boxes too.
    #[cfg(target_os = "macos")]
    let builder =
        builder.with_execution_providers([CoreMLExecutionProvider::default().build()])?;
    Ok(builder.commit_from_file(model)?)
}

fn load_frames(path: &str) -> Result<Vec<image::RgbImage>> {
    let p = Path::new(path);
    let mut files: Vec<_> = if p.is_dir() {
        std::fs::read_dir(p)?.filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| matches!(p.extension().and_then(|s| s.to_str()), Some("jpg" | "jpeg" | "png"))).collect()
    } else { vec![p.to_path_buf()] };
    files.sort();
    files.iter().map(|f| Ok(image::open(f).with_context(|| format!("open {f:?}"))?.to_rgb8())).collect()
}

// ---- HTTP service ----
#[derive(Clone)]
struct AppState {
    session: Arc<Mutex<Session>>,
    // Optional pose session (POSE_MODEL env). None = service behaves exactly
    // as before pose existed; Some = keypoints attach to person tracks.
    pose: Option<Arc<Mutex<Session>>>,
    // Optional behavior NN (BEHAVIOR_MODEL env). Requires pose to be on —
    // it classifies the skeleton trajectories pose produces.
    behavior: Option<Arc<BehaviorCtx>>,
    // Optional suspicion policy (SUS_POLICY env). None = no sus fields emitted.
    sus: Option<Arc<sus::SusPolicy>>,
    // v3 identity. `id_on` gates the always-free clothing-color tracker veto +
    // reversal split (default on; IDENTITY_GATING=0 reverts to exact prior
    // behavior). `reid`/`face` are the optional OSNet/ArcFace embedders.
    id_on: bool,
    id_cfg: identity::IdentityConfig,
    reid: Option<Arc<identity::ReidCtx>>,
    face: Option<Arc<identity::FaceCtx>>,
}

async fn health() -> &'static str { "ok" }

async fn triage_endpoint(State(st): State<AppState>, mut mp: Multipart) -> Json<serde_json::Value> {
    let mut frame_bytes: Vec<Vec<u8>> = Vec::new();
    let mut meta = ZonesMeta::default();
    let mut night = false;
    // Clip window length the frames were sampled from (seconds). Drives the
    // dwell_s estimate. Default 60 (the validated capture window).
    let mut clip_seconds: f32 = 60.0;
    while let Ok(Some(field)) = mp.next_field().await {
        let name = field.name().map(|s| s.to_string());
        let Ok(b) = field.bytes().await else { continue };
        match name.as_deref() {
            Some("zones") => {
                // Optional JSON sidecar: operator-drawn zones/lines (frame fractions).
                if let Ok(m) = serde_json::from_slice::<ZonesMeta>(&b) { meta = m; }
            }
            Some("context") => {
                // Optional {"night": bool} — the caller knows its site's local
                // time; we don't guess timezone from UTC frames.
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&b) {
                    night = v.get("night").and_then(|x| x.as_bool()).unwrap_or(false);
                    if let Some(cs) = v.get("clip_seconds").and_then(|x| x.as_f64()) {
                        if cs > 0.0 { clip_seconds = cs as f32; }
                    }
                }
            }
            _ => frame_bytes.push(b.to_vec()),
        }
    }
    let res = tokio::task::spawn_blocking(move || -> Result<TriageResult> {
        let mut frames = Vec::with_capacity(frame_bytes.len());
        for b in &frame_bytes {
            frames.push(image::load_from_memory(b)?.to_rgb8());
        }
        let s = st.session.lock().unwrap();
        let pose_guard = st.pose.as_ref().map(|p| p.lock().unwrap());
        run_triage(&s, &frames, &meta, pose_guard.as_deref(), st.behavior.as_deref(),
                   night, st.sus.as_deref(), clip_seconds,
                   st.reid.as_deref(), st.face.as_deref(), &st.id_cfg, st.id_on)
    }).await;
    // On ANY failure, fail safe → escalate (let the VLM decide rather than drop).
    match res {
        Ok(Ok(r)) => Json(serde_json::to_value(r).unwrap()),
        Ok(Err(e)) => Json(serde_json::json!({"decision":"escalate","error":e.to_string()})),
        Err(e) => Json(serde_json::json!({"decision":"escalate","error":format!("join: {e}")})),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ort::init().commit()?;

    if args.get(1).map(String::as_str) == Some("serve") {
        let model = std::env::var("TRIAGE_MODEL").unwrap_or_else(|_| "models/yolo11m.onnx".into());
        let port: u16 = args.get(2).and_then(|p| p.parse().ok())
            .or_else(|| std::env::var("TRIAGE_PORT").ok().and_then(|p| p.parse().ok())).unwrap_or(8091);
        let session = Arc::new(Mutex::new(build_session(&model)?));
        // Optional pose model: set POSE_MODEL=<path>.onnx to attach 17-joint
        // skeletons to person tracks in /triage responses (additive field).
        let pose = match std::env::var("POSE_MODEL") {
            Ok(p) if !p.trim().is_empty() => Some(Arc::new(Mutex::new(build_session(&p)?))),
            _ => None,
        };
        let pose_on = pose.is_some();
        let behavior = BehaviorCtx::from_env()?.map(Arc::new);
        if behavior.is_some() && !pose_on {
            anyhow::bail!("BEHAVIOR_MODEL requires POSE_MODEL — the NN \
                           classifies skeleton trajectories, which pose produces");
        }
        let beh_banner = match &behavior {
            Some(b) => format!("behavior on ({} classes{})", b.meta.classes.len(),
                if b.alert_classes.is_empty() { ", informational".to_string() }
                else { format!(", alerts on {:?}", b.alert_classes) }),
            None => "behavior off".to_string(),
        };
        // Optional suspicion policy: SUS_POLICY=<path>.json adds a per-track
        // sus_score + sus_alert (flag/reason weight × night × zone).
        let sus = match std::env::var("SUS_POLICY") {
            Ok(p) if !p.trim().is_empty() => Some(Arc::new(
                sus::SusPolicy::load(&p).ok_or_else(
                    || anyhow::anyhow!("SUS_POLICY set but {p} is missing/invalid JSON"))?)),
            _ => None,
        };
        let sus_banner = match &sus {
            Some(s) => format!("sus on (threshold {})", s.threshold),
            None => "sus off".to_string(),
        };
        // v3 identity: clothing-color tracker veto + reversal split are ON by
        // default (the whole point — stop merging two people into fake pacing);
        // IDENTITY_GATING=0 reverts to exact geometry-only behavior. Body Re-ID
        // and face are optional embedders, env-gated like pose/behavior.
        let id_on = std::env::var("IDENTITY_GATING").map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);
        let id_cfg = identity::IdentityConfig::from_env();
        let reid = if id_on { identity::ReidCtx::from_env()?.map(Arc::new) } else { None };
        let face = if id_on { identity::FaceCtx::from_env()?.map(Arc::new) } else { None };
        let id_banner = if id_on {
            format!("identity on (veto {:.2}, reid {}, face {})", id_cfg.veto_color_dist,
                if reid.is_some() { "on" } else { "off" }, if face.is_some() { "on" } else { "off" })
        } else { "identity off".to_string() };
        let app = Router::new()
            .route("/health", get(health))
            .route("/triage", post(triage_endpoint))
            // 16 frames of full-res 4 MP JPEG fit comfortably; axum's 2 MB
            // default rejects multi-frame posts mid-stream (curl error 55).
            .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
            .with_state(AppState { session, pose, behavior, sus, id_on, id_cfg, reid, face });
        let addr = format!("0.0.0.0:{port}");
        println!("motion-triage serving on {addr} (model {model}, pose {}, {}, {}, {})",
                 if pose_on { "on" } else { "off" }, beh_banner, sus_banner, id_banner);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
        return Ok(());
    }

    // Pose spike: run a YOLO-pose model, print per-joint detail, and write
    // annotated frames (boxes + 17-joint skeletons) — the "can it see limbs
    // from this camera" check.
    if args.get(1).map(String::as_str) == Some("pose") {
        let model = args.get(2).map(String::as_str)
            .context("usage: spike pose <pose-model.onnx> <frames-dir|image> [out-dir]")?;
        let input = args.get(3).map(String::as_str)
            .context("usage: spike pose <pose-model.onnx> <frames-dir|image> [out-dir]")?;
        let out_dir = args.get(4).map(String::as_str).unwrap_or("pose_out");
        return pose::run_pose_spike(model, input, out_dir);
    }

    // Behavior spike: classify exported corpus tracks with a trained
    // behavior ONNX (the NN the Python training loop produces).
    if args.get(1).map(String::as_str) == Some("behavior") {
        let model = args.get(2).map(String::as_str)
            .context("usage: spike behavior <behavior.onnx> <corpus.jsonl> [max-lines]")?;
        let corpus = args.get(3).map(String::as_str)
            .context("usage: spike behavior <behavior.onnx> <corpus.jsonl> [max-lines]")?;
        let max = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10);
        return behavior::run_behavior_spike(model, corpus, max);
    }

    // CLI mode. IDENTITY_GATING env toggles the v3 layer so the same binary can
    // run identity OFF (old geometry-only behavior) vs ON (v3) for A/B testing.
    let id_on = std::env::var("IDENTITY_GATING")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false")).unwrap_or(true);
    let model = args.get(1).map(String::as_str).unwrap_or("models/yolo11m.onnx");
    let input = args.get(2).map(String::as_str).unwrap_or("testdata/clip");
    let session = build_session(model)?;
    let frames = load_frames(input)?;
    let r = run_triage(&session, &frames, &ZonesMeta::default(), None, None, false, None, 60.0,
                       None, None, &identity::IdentityConfig::default(), id_on)?;
    println!("{} frames · detect {} ms ({:.1} ms/frame, CoreML) · {} track(s) · identity {}",
        r.frames, r.detect_ms, r.detect_ms as f64 / r.frames.max(1) as f64, r.tracks.len(),
        if id_on { "ON" } else { "OFF" });
    for v in &r.tracks {
        let id_note = v.identity.as_ref()
            .map(|i| format!(" [split from #{} by {}]", i.split_from, i.decided_by))
            .unwrap_or_default();
        println!("  track#{:<2} seen {:<2}  straightness {:.2}  span {:.2}  dwell {:.0}%  → {} ({}){}",
            v.id, v.n, v.straightness, v.span, v.dwell_frac * 100.0, v.decision, v.reason, id_note);
    }
    println!("\n  ▶ EVENT: decision={} route={} reason={}", r.decision, r.route, r.reason);
    println!("  ▶ SUMMARY: {}", r.summary);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn behavior_routing_only_escalates_configured_classes() {
        let allow = vec!["crouching".to_string(), "climbing".to_string()];
        // Configured class above threshold escalates…
        assert!(behavior_escalates("crouching", 0.93, &allow, 0.8));
        // …below threshold doesn't…
        assert!(!behavior_escalates("crouching", 0.62, &allow, 0.8));
        // …and unconfigured classes never do, however confident. With the
        // default EMPTY allow-list the NN is purely informational.
        assert!(!behavior_escalates("walking", 0.99, &allow, 0.8));
        assert!(!behavior_escalates("walking", 0.99, &[], 0.8));
    }

    const W: f32 = 1920.0;
    const H: f32 = 1080.0;

    fn det(cx: f32, cy: f32, h: f32, cls: usize) -> Det {
        let w = h * 0.4; // rough person aspect; for cars we pass an explicit h
        Det { x1: cx - w / 2.0, y1: cy - h / 2.0, x2: cx + w / 2.0, y2: cy + h / 2.0, score: 0.9, cls }
    }

    fn person_track(first_frame: usize, centers: Vec<(f32, f32)>, ph: f32) -> Track {
        let n = centers.len();
        let first = det(centers[0].0, centers[0].1, ph, PERSON_CLASS);
        let lc = *centers.last().unwrap();
        let last = det(lc.0, lc.1, ph, PERSON_CLASS);
        let boxes: Vec<Det> = centers.iter().map(|&(cx, cy)| det(cx, cy, ph, PERSON_CLASS)).collect();
        Track { id: 0, cls: PERSON_CLASS, first_frame, last_frame: first_frame + n - 1,
            first, last, heights: vec![ph; n],
            frame_idxs: (0..n).map(|k| first_frame + k).collect(),
            boxes, color_sigs: vec![identity::ColorSig::default(); n],
            split_from: None, split_by: None, centers }
    }

    // A person who walks across and out the right side with NO car present is a
    // normal walk-by → dismiss.
    #[test]
    fn walk_to_edge_without_car_is_dismissed() {
        let centers = vec![(900.0, 650.0), (1100.0, 670.0), (1300.0, 690.0), (1500.0, 700.0), (1650.0, 700.0)];
        let t = person_track(0, centers, 120.0);
        let (d, v) = classify(&t, 16, W, H, &[]);
        assert_eq!(d, Decision::Dismiss, "reason was {}", v.reason);
        // Walked out the right edge → the specific "edge_exit" dismiss (still no
        // alert); "walk_by" remains the dismiss for a through-frame pass.
        assert_eq!(v.reason, "edge_exit");
    }

    // The SAME trajectory, but now a car sits where the person vanishes → they
    // got in. The car alone flips dismiss → entered_vehicle (the requested case).
    #[test]
    fn walk_into_a_car_is_entered_vehicle() {
        let centers = vec![(900.0, 650.0), (1100.0, 670.0), (1300.0, 690.0), (1500.0, 700.0), (1650.0, 700.0)];
        let t = person_track(0, centers, 120.0);
        let car = det(1620.0, 700.0, 360.0, 2); // COCO car at the exit point
        let (d, v) = classify(&t, 16, W, H, &[car]);
        assert_eq!(d, Decision::Alert, "reason was {}", v.reason);
        assert_eq!(v.reason, "entered_vehicle");
    }

    // 70b6aa50 regression: a person who ends NEAR a car but never moved TOWARD it
    // (here: near-stationary beside passing traffic) must NOT be entered_vehicle.
    #[test]
    fn near_a_car_without_approaching_is_not_entered_vehicle() {
        let centers = vec![(1600.0, 700.0), (1612.0, 702.0), (1598.0, 699.0), (1606.0, 701.0)];
        let t = person_track(0, centers, 120.0);
        let car = det(1620.0, 700.0, 360.0, 2); // car right where they vanish
        let (_d, v) = classify(&t, 16, W, H, &[car]);
        assert_ne!(v.reason, "entered_vehicle", "did not approach the car");
    }

    // Operator 2026-06-09: people LEAVING THE FRAME were marked entered_vehicle.
    // Walking out the edge (box ends hard against the boundary) past a car is a
    // frame-exit, not a car-entry — must read as edge_exit, not entered_vehicle.
    #[test]
    fn walking_out_of_frame_past_a_car_is_not_entered_vehicle() {
        let centers = vec![(900.0, 700.0), (1200.0, 700.0), (1500.0, 700.0), (1800.0, 700.0), (1895.0, 700.0)];
        let t = person_track(0, centers, 120.0); // ends hard against the right edge
        let car = det(1850.0, 700.0, 360.0, 2);
        let (_d, v) = classify(&t, 16, W, H, &[car]);
        assert_ne!(v.reason, "entered_vehicle", "left the frame, not entered the car");
    }

    // A person who appears at a car partway through the clip and walks off got
    // OUT of it → arrival by vehicle.
    #[test]
    fn appearing_at_a_car_is_arrived_by_vehicle() {
        let centers = vec![(1600.0, 700.0), (1400.0, 680.0), (1100.0, 650.0), (850.0, 640.0), (700.0, 640.0), (650.0, 640.0)];
        let t = person_track(6, centers, 120.0); // first_frame > 2 → appeared late
        let car = det(1650.0, 700.0, 360.0, 2);
        let (d, v) = classify(&t, 16, W, H, &[car]);
        assert_eq!(d, Decision::Alert, "reason was {}", v.reason);
        assert_eq!(v.reason, "arrived_by_vehicle");
    }

    // Variable-height track builder (heights drive the perspective signals:
    // scale_change, height_trend, camera_approach, intrusion's receding gate).
    fn person_track_h(first_frame: usize, centers: Vec<(f32, f32)>, heights: Vec<f32>) -> Track {
        let n = centers.len();
        let first = det(centers[0].0, centers[0].1, heights[0], PERSON_CLASS);
        let lc = *centers.last().unwrap();
        let last = det(lc.0, lc.1, *heights.last().unwrap(), PERSON_CLASS);
        let boxes: Vec<Det> = centers.iter().zip(heights.iter())
            .map(|(&(cx, cy), &hh)| det(cx, cy, hh, PERSON_CLASS)).collect();
        Track { id: 0, cls: PERSON_CLASS, first_frame, last_frame: first_frame + n - 1,
            first, last, heights,
            frame_idxs: (0..n).map(|k| first_frame + k).collect(),
            boxes, color_sigs: vec![identity::ColorSig::default(); n],
            split_from: None, split_by: None, centers }
    }

    fn reason_of(t: &Track, total: usize) -> String {
        classify(t, total, W, H, &[]).1.reason
    }

    // A person standing in one spot (tiny jitter, constant size) for most of the
    // clip → loitering.
    #[test]
    fn standing_still_is_loitering() {
        let c = vec![(900.0, 600.0), (905.0, 598.0), (898.0, 602.0), (902.0, 599.0),
                     (900.0, 601.0), (903.0, 600.0), (899.0, 598.0), (901.0, 600.0)];
        let t = person_track_h(0, c, vec![200.0; 8]);
        assert_eq!(reason_of(&t, 16), "loitering");
    }

    // FIX (operator): walking STRAIGHT TOWARD the camera — center barely moves but
    // the box doubles in height — must NOT be loitering. scale_change vetoes it.
    #[test]
    fn walk_toward_camera_is_not_loitering() {
        let c = vec![(960.0, 500.0), (960.0, 508.0), (960.0, 517.0), (960.0, 526.0),
                     (960.0, 535.0), (960.0, 544.0), (960.0, 552.0), (960.0, 560.0)];
        let h = vec![90.0, 107.0, 124.0, 141.0, 158.0, 175.0, 193.0, 210.0];
        assert_eq!(reason_of(&person_track_h(0, c, h), 16), "walk_by");
    }

    // FIX (operator clip c3ef0a8c): a DISTANT person walking a straight line covers
    // few pixels but many body-lengths — must be walk_by, not loitering.
    #[test]
    fn distant_straight_walk_is_not_loitering() {
        let c = vec![(300.0, 150.0), (360.0, 152.0), (420.0, 154.0), (480.0, 156.0),
                     (540.0, 158.0), (600.0, 159.0), (660.0, 160.0), (720.0, 160.0)];
        let t = person_track_h(0, c, vec![60.0; 8]);
        assert_eq!(reason_of(&t, 16), "walk_by");
    }

    // A person who walks up and vanishes WELL INSIDE the frame (interior box, was
    // moving, near, not receding) → intrusion.
    #[test]
    fn interior_vanish_is_intrusion() {
        let c = vec![(600.0, 500.0), (670.0, 510.0), (740.0, 525.0), (810.0, 535.0),
                     (880.0, 550.0), (950.0, 560.0)];
        let h = vec![260.0, 272.0, 284.0, 296.0, 308.0, 320.0];
        assert_eq!(reason_of(&person_track_h(0, c, h), 16), "intrusion");
    }

    // FIX (operator): receding out the TOP (box shrinks, center rises to the edge)
    // is LEAVING, not intrusion.
    #[test]
    fn top_exit_receding_is_not_intrusion() {
        let c = vec![(900.0, 400.0), (896.0, 350.0), (892.0, 300.0), (888.0, 250.0),
                     (884.0, 200.0), (880.0, 150.0)];
        let h = vec![200.0, 174.0, 148.0, 122.0, 96.0, 70.0];
        let r = reason_of(&person_track_h(0, c, h), 16);
        assert!(r == "edge_exit" || r == "walk_by", "got {r}");
    }

    // FIX (operator): walking TOWARD the camera and out the BOTTOM (tall box, feet
    // hit the bottom edge) is LEAVING, not intrusion.
    #[test]
    fn bottom_exit_toward_camera_is_not_intrusion() {
        let c = vec![(950.0, 600.0), (955.0, 680.0), (960.0, 760.0), (965.0, 840.0),
                     (970.0, 900.0), (975.0, 940.0)];
        let h = vec![200.0, 260.0, 330.0, 410.0, 480.0, 520.0];
        assert_ne!(reason_of(&person_track_h(0, c, h), 16), "intrusion");
    }

    // FIX (testdata track#0): a 5-frame near-stationary dropout that just ENDS is a
    // transient, never an intrusion.
    #[test]
    fn short_stationary_dropout_is_transient() {
        let c = vec![(500.0, 500.0), (503.0, 498.0), (499.0, 501.0), (502.0, 500.0), (500.0, 499.0)];
        let t = person_track_h(0, c, vec![200.0; 5]);
        let r = reason_of(&t, 30);
        assert!(r == "transient", "got {r}");
    }

    // Sustained fast straight motion → running.
    #[test]
    fn fast_straight_motion_is_running() {
        let c = vec![(200.0, 600.0), (520.0, 600.0), (840.0, 600.0), (1160.0, 600.0),
                     (1480.0, 600.0), (1700.0, 600.0)];
        let t = person_track_h(0, c, vec![150.0; 6]);
        assert_eq!(reason_of(&t, 16), "running");
    }

    // Repeated back-and-forth with ~zero net displacement → pacing (outranks the
    // single-turn u_turn / direction_change).
    #[test]
    fn back_and_forth_is_pacing() {
        let c = vec![(500.0, 600.0), (900.0, 600.0), (500.0, 600.0), (900.0, 600.0),
                     (500.0, 600.0), (900.0, 600.0), (500.0, 600.0)];
        let t = person_track_h(0, c, vec![150.0; 7]);
        assert_eq!(reason_of(&t, 16), "pacing");
    }

    // ---- Phantom / distant-figure gates (operator FPs 2026-06-09) -----------

    // Desk-facing indoor cam / porch cam: a clutter mis-detection ("person" on a power
    // brick, camera housing, shoes) reads as standing still → loitering. A real
    // person detects well above CONF; require SOFT_MIN_CONF so a weak box can't.
    #[test]
    fn low_confidence_standing_box_is_not_loitering() {
        let c = vec![(900.0, 600.0), (905.0, 598.0), (898.0, 602.0), (902.0, 599.0),
                     (900.0, 601.0), (903.0, 600.0), (899.0, 598.0), (901.0, 600.0)];
        let mut t = person_track_h(0, c, vec![200.0; 8]);
        t.first.score = 0.35; t.last.score = 0.35; // below SOFT_MIN_CONF (0.45)
        let r = reason_of(&t, 16);
        assert_ne!(r, "loitering", "weak detection must not loiter; got {r}");
    }

    // Street cam: a tiny far-off pedestrian (box < 8% of frame height) standing
    // still must NOT be loitering — too small to be a credible loiter subject.
    #[test]
    fn tiny_distant_standing_box_is_not_loitering() {
        let c = vec![(300.0, 150.0), (305.0, 148.0), (298.0, 152.0), (302.0, 149.0),
                     (300.0, 151.0), (303.0, 150.0), (299.0, 148.0), (301.0, 150.0)];
        let t = person_track_h(0, c, vec![40.0; 8]); // 40/1080 = 3.7% of frame
        let r = reason_of(&t, 16);
        assert_ne!(r, "loitering", "distant speck must not loiter; got {r}");
    }

    // A perfectly frozen box (never moves, never changes size) is a fixed object,
    // not a person standing still — a real loiterer drifts more than DEAD_BOX.
    #[test]
    fn frozen_box_is_not_loitering() {
        let c = vec![(900.0, 600.0); 8];
        let t = person_track_h(0, c, vec![200.0; 8]);
        let (d, v) = classify(&t, 16, W, H, &[]);
        assert_ne!(v.reason, "loitering", "frozen object must not loiter; got {}", v.reason);
        assert_eq!(d, Decision::Dismiss, "got {} ({})", v.decision, v.reason);
    }

    // f6167ffb: a jittery clutter mis-detection swinging back and forth must NOT
    // fire pacing — and, with the soft family gated, must NOT escape as u_turn /
    // direction_change / erratic either (those feed the server pacing upgrade).
    #[test]
    fn low_confidence_back_and_forth_is_dismissed() {
        let c = vec![(500.0, 600.0), (900.0, 600.0), (500.0, 600.0), (900.0, 600.0),
                     (500.0, 600.0), (900.0, 600.0), (500.0, 600.0)];
        let mut t = person_track_h(0, c, vec![150.0; 7]);
        t.first.score = 0.35; t.last.score = 0.35;
        let (d, v) = classify(&t, 16, W, H, &[]);
        assert_eq!(d, Decision::Dismiss, "weak jitter must dismiss; got {} ({})", v.decision, v.reason);
    }

    // The same back-and-forth by a TINY distant figure (size gate) also dismisses
    // across the whole soft family — not pacing, not u_turn, not direction_change.
    #[test]
    fn tiny_distant_back_and_forth_is_dismissed() {
        let c = vec![(180.0, 150.0), (420.0, 150.0), (180.0, 150.0), (420.0, 150.0),
                     (180.0, 150.0), (420.0, 150.0), (180.0, 150.0)];
        let t = person_track_h(0, c, vec![40.0; 7]); // 3.7% of frame height
        let (d, v) = classify(&t, 16, W, H, &[]);
        assert_eq!(d, Decision::Dismiss, "distant jitter must dismiss; got {} ({})", v.decision, v.reason);
    }

    // Walked clearly, then froze for the rest of the clip → sudden_stop.
    #[test]
    fn walk_then_freeze_is_sudden_stop() {
        let c = vec![(300.0, 600.0), (500.0, 600.0), (700.0, 600.0), (900.0, 600.0),
                     (905.0, 600.0), (908.0, 600.0), (906.0, 600.0), (907.0, 600.0)];
        let t = person_track_h(0, c, vec![150.0; 8]);
        assert_eq!(reason_of(&t, 16), "sudden_stop");
    }

    // Box grows until it fills the frame → camera_approach / tamper (top severity).
    #[test]
    fn box_fills_frame_is_camera_approach() {
        let c = vec![(960.0, 500.0), (960.0, 515.0), (960.0, 530.0), (960.0, 545.0),
                     (960.0, 555.0), (960.0, 560.0)];
        let h = vec![150.0, 260.0, 400.0, 540.0, 640.0, 720.0];
        assert_eq!(reason_of(&person_track_h(0, c, h), 16), "camera_approach");
    }

    // Two spatially-distinct people alive at once → event-level multi_person.
    #[test]
    fn two_people_at_once_is_multi_person() {
        let a = person_track_h(0, vec![(300.0, 600.0), (310.0, 605.0), (320.0, 610.0), (330.0, 615.0)], vec![200.0; 4]);
        let b = person_track_h(0, vec![(1500.0, 600.0), (1490.0, 605.0), (1480.0, 610.0), (1470.0, 615.0)], vec![200.0; 4]);
        let tracks = vec![a, b];
        assert_eq!(event_level(&tracks, W, H), Some("multi_person"));
    }

    #[test]
    fn point_in_poly_basic() {
        let sq = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        assert!(point_in_poly((50.0, 50.0), &sq));
        assert!(!point_in_poly((150.0, 50.0), &sq));
    }

    #[test]
    fn line_crossing_direction() {
        let (a, b) = ((960.0, 0.0), (960.0, 1080.0)); // vertical line, drawn downward
        assert_eq!(crosses_line((400.0, 550.0), (1500.0, 550.0), a, b), 1);  // left→right = in
        assert_eq!(crosses_line((1500.0, 550.0), (400.0, 550.0), a, b), -1); // right→left = out
        assert_eq!(crosses_line((100.0, 550.0), (200.0, 550.0), a, b), 0);   // never reaches line
    }

    // Footpoint starts OUTSIDE a drawn zone (left half) and ends INSIDE it
    // (right half) → zone_intrusion.
    #[test]
    fn entering_a_zone_is_zone_intrusion() {
        let t = person_track_h(0, vec![(400.0, 450.0), (800.0, 450.0), (1200.0, 450.0), (1500.0, 450.0)], vec![200.0; 4]);
        let meta = ZonesMeta {
            zones: vec![ZoneDef { name: "yard".into(),
                polygon: vec![[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]] }],
            lines: vec![],
        };
        assert_eq!(zone_line_reason(&[t], W, H, &meta), Some("zone_intrusion"));
    }

    // Footpoint crosses an operator line left→right with in_direction a_to_b →
    // door_entry (an "in" crossing).
    #[test]
    fn crossing_a_line_inward_is_door_entry() {
        let t = person_track_h(0, vec![(400.0, 450.0), (800.0, 450.0), (1200.0, 450.0), (1500.0, 450.0)], vec![200.0; 4]);
        let meta = ZonesMeta {
            zones: vec![],
            lines: vec![LineDef { name: "gate".into(), a: [0.5, 0.0], b: [0.5, 1.0],
                in_direction: "a_to_b".into() }],
        };
        assert_eq!(zone_line_reason(&[t], W, H, &meta), Some("door_entry"));
    }

    // ---- v3 identity: appearance-gated tracking + reversal split ------------

    /// A real clothing-color signature for a solid-color person crop.
    fn outfit(rgb: [u8; 3]) -> identity::ColorSig {
        let img = image::RgbImage::from_pixel(120, 300, image::Rgb(rgb));
        identity::color_sig(&img, &Det { x1: 0.0, y1: 0.0, x2: 120.0, y2: 300.0, score: 0.9, cls: 0 })
    }

    // THE reported bug. Person A (red shirt) walks left→right and exits; person
    // B (blue shirt) enters near the same spot going right→left. Geometry alone
    // stitches them into ONE reversing track (fake pacing). The clothing veto
    // keeps them apart → two clean tracks.
    #[test]
    fn opposite_direction_two_people_are_not_merged() {
        let red = outfit([200, 30, 30]);
        let blue = outfit([30, 40, 200]);
        // one detection per frame: A exits right (f0-3), B enters and goes left (f4-7)
        let xs_a = [1500.0, 1600.0, 1700.0, 1850.0];
        let xs_b = [1700.0, 1500.0, 1300.0, 1100.0];
        let mut per_frame: Vec<Vec<Det>> = Vec::new();
        let mut per_sig: Vec<Vec<identity::ColorSig>> = Vec::new();
        for &x in &xs_a {
            per_frame.push(vec![det(x, 600.0, 200.0, PERSON_CLASS)]);
            per_sig.push(vec![red]);
        }
        for &x in &xs_b {
            per_frame.push(vec![det(x, 600.0, 200.0, PERSON_CLASS)]);
            per_sig.push(vec![blue]);
        }
        let cfg = identity::IdentityConfig::default();
        // Geometry only (gating off): the merge happens → ONE reversing track.
        let merged = track(&per_frame, &per_sig, &cfg, false);
        assert_eq!(merged.len(), 1, "without the veto the two should merge");
        // With the clothing veto: two distinct people → TWO tracks.
        let split = track(&per_frame, &per_sig, &cfg, true);
        assert_eq!(split.len(), 2, "different outfits must not be stitched together");
    }

    // The reversal-split audit: a track already merged into a u_turn/pacing whose
    // two halves are different outfits is carved back into two tracks (each then
    // re-classifies to a plain walk_by).
    #[test]
    fn reversal_split_separates_two_outfits() {
        let red = outfit([200, 30, 30]);
        let blue = outfit([30, 40, 200]);
        // out (right) then back, decelerating to a stop mid-frame so it reads as
        // a real interior reversal, not a walk-off-the-edge. First half red shirt
        // (person A), second half blue shirt (person B).
        let c = vec![(700.0, 600.0), (900.0, 600.0), (1100.0, 600.0), (1200.0, 600.0),
                     (1100.0, 600.0), (1000.0, 600.0), (980.0, 600.0), (975.0, 600.0)];
        let mut t = person_track_h(0, c, vec![200.0; 8]);
        t.color_sigs = vec![red, red, red, red, blue, blue, blue, blue];
        assert!(is_reversal_reason(&classify(&t, 16, W, H, &[]).1.reason),
            "merged track should read as a reversal, got {}", classify(&t, 16, W, H, &[]).1.reason);
        let cfg = identity::IdentityConfig::default();
        let out = split_reversal_tracks(vec![t], &[], &[], None, None, &cfg, W, H, 16);
        assert_eq!(out.len(), 2, "two outfits across the reversal must split");
        assert!(out.iter().all(|s| s.split_from == Some(0)));
        // each half is now a one-directional walk → dismissed
        for s in &out {
            let r = classify(s, 16, W, H, &[]).1;
            assert_eq!(r.decision, "dismiss", "split half was {} ({})", r.decision, r.reason);
        }
    }

    // Guard against OVER-splitting: a genuine single person pacing (same outfit
    // throughout) must NOT be split — it stays one track / one pacing reason.
    #[test]
    fn same_outfit_pacer_is_not_split() {
        let red = outfit([200, 30, 30]);
        let c = vec![(500.0, 600.0), (900.0, 600.0), (500.0, 600.0), (900.0, 600.0),
                     (500.0, 600.0), (900.0, 600.0), (500.0, 600.0)];
        let mut t = person_track_h(0, c, vec![150.0; 7]);
        t.color_sigs = vec![red; 7];
        assert_eq!(classify(&t, 16, W, H, &[]).1.reason, "pacing");
        let cfg = identity::IdentityConfig::default();
        let out = split_reversal_tracks(vec![t], &[], &[], None, None, &cfg, W, H, 16);
        assert_eq!(out.len(), 1, "one person pacing must stay whole");
        assert_eq!(classify(&out[0], 16, W, H, &[]).1.reason, "pacing");
    }

    // Color too weak to judge (tiny/no crop) → Unknown → never split (fail safe).
    #[test]
    fn weak_color_pacer_is_not_split() {
        let c = vec![(500.0, 600.0), (900.0, 600.0), (500.0, 600.0), (900.0, 600.0),
                     (500.0, 600.0), (900.0, 600.0), (500.0, 600.0)];
        let t = person_track_h(0, c, vec![150.0; 7]); // default (weak) color_sigs
        let cfg = identity::IdentityConfig::default();
        let out = split_reversal_tracks(vec![t], &[], &[], None, None, &cfg, W, H, 16);
        assert_eq!(out.len(), 1, "weak color is Unknown → must not split");
    }
}
