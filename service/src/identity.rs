//! v3 — appearance-based identity, so two DIFFERENT people are never merged
//! into one reversing track (the false "pacing / u_turn / direction_change /
//! loitering" the geometry-only tracker produces when person A exits and person
//! B enters near the same spot).
//!
//! Three signals, fused with graceful degradation — strongest-available wins:
//!   1. **Clothing color** — pure-Rust HSV histogram of the shirt + pants bands.
//!      Always on, no model, microseconds. The workhorse: gates the tracker's
//!      frame-to-frame association in real time.
//!   2. **Body Re-ID** — OSNet x1.0 512-d embedding (env `REID_MODEL`). Run only
//!      on the few ambiguous tracks, not every detection, to protect the budget.
//!   3. **Face** — ArcFace `w600k_r50` 512-d embedding (env `FACE_MODEL`),
//!      cropped+aligned from the YOLO-pose facial keypoints we already compute
//!      (no separate face detector). Opportunistic: surveillance faces are often
//!      too small, so it simply abstains and body/color carry the decision.
//!
//! `identity_match` fuses them into `Same | Different | Unknown`; **Unknown fails
//! toward NOT splitting** so a genuine single-person pacer is never fragmented.
//!
//! Pure functions (color, fusion) are unit-tested here without any model file.

use std::sync::Mutex;

use anyhow::{Context, Result};
use image::RgbImage;
use ndarray::Array4;
#[cfg(target_os = "macos")]
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};

use crate::Det;

// ---- Clothing-color signature (HSV histogram, two body bands) ---------------

const NH: usize = 10; // hue bins
const NS: usize = 2; // saturation bins (muted / vivid) for colored pixels
const NG: usize = 3; // value bins for low-saturation (gray/black/white) pixels
/// Per-band histogram length. Kept ≤ 32 so `[f32; BINS]` derives Default/Copy.
const BINS: usize = NH * NS + NG; // 23

/// Coarse appearance fingerprint of one person crop: separate color histograms
/// for the upper body (shirt) and lower body (pants). Resolution-independent.
#[derive(Clone, Copy)]
pub struct ColorSig {
    upper: [f32; BINS],
    lower: [f32; BINS],
    /// Too small / too few pixels to trust — callers must not veto/split on it.
    pub weak: bool,
}

impl Default for ColorSig {
    fn default() -> Self {
        ColorSig { upper: [0.0; BINS], lower: [0.0; BINS], weak: true }
    }
}

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (r, g, b) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d <= 1e-6 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / d) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    let s = if max <= 1e-6 { 0.0 } else { d / max };
    (h, s, max)
}

/// Histogram bin index for one pixel — colored pixels bin by (hue, sat),
/// low-saturation pixels (black/white/gray clothing) bin by brightness.
fn bin_index(h: f32, s: f32, v: f32) -> usize {
    if v < 0.12 || s < 0.18 {
        NH * NS + (((v * NG as f32) as usize).min(NG - 1))
    } else {
        let hbin = ((h / 360.0 * NH as f32) as usize).min(NH - 1);
        let sbin = if s < 0.5 { 0 } else { 1 };
        hbin * NS + sbin
    }
}

fn normalize(hist: &mut [f32; BINS]) -> f32 {
    let sum: f32 = hist.iter().sum();
    if sum > 0.0 {
        for x in hist.iter_mut() {
            *x /= sum;
        }
    }
    sum
}

/// Histogram intersection distance in [0,1]: 0 = identical, 1 = disjoint.
fn hist_dist(a: &[f32; BINS], b: &[f32; BINS]) -> f32 {
    let inter: f32 = a.iter().zip(b.iter()).map(|(x, y)| x.min(*y)).sum();
    (1.0 - inter).clamp(0.0, 1.0)
}

/// Combined upper+lower distance (shirt weighted a touch more — torsos are more
/// discriminative than legs at a distance).
pub fn color_distance(a: &ColorSig, b: &ColorSig) -> f32 {
    0.55 * hist_dist(&a.upper, &b.upper) + 0.45 * hist_dist(&a.lower, &b.lower)
}

/// Build a `ColorSig` from a person box. Samples the torso band (15–55% of box
/// height) and the legs band (55–90%), trimming 10% off each side to avoid
/// background. Subsamples for speed; marks `weak` when the crop is tiny.
pub fn color_sig(img: &RgbImage, d: &Det) -> ColorSig {
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let x1 = d.x1.max(0.0);
    let y1 = d.y1.max(0.0);
    let x2 = d.x2.min(iw - 1.0);
    let y2 = d.y2.min(ih - 1.0);
    let bw = x2 - x1;
    let bh = y2 - y1;
    let mut sig = ColorSig::default();
    if bw < 6.0 || bh < 24.0 {
        return sig; // weak (default)
    }
    let xa = (x1 + 0.10 * bw) as u32;
    let xb = (x2 - 0.10 * bw) as u32;
    let up_a = (y1 + 0.15 * bh) as u32;
    let up_b = (y1 + 0.55 * bh) as u32;
    let lo_a = up_b;
    let lo_b = (y1 + 0.90 * bh) as u32;
    // Subsample stride scaled to box size: ~1k samples per band regardless of
    // crop resolution.
    let stride = ((bw.min(bh) / 24.0) as u32).max(1);
    let accumulate = |ya: u32, yb: u32, hist: &mut [f32; BINS]| {
        let mut y = ya;
        while y < yb && y < img.height() {
            let mut x = xa;
            while x < xb && x < img.width() {
                let p = img.get_pixel(x, y);
                let (h, s, v) = rgb_to_hsv(p[0], p[1], p[2]);
                hist[bin_index(h, s, v)] += 1.0;
                x += stride;
            }
            y += stride;
        }
    };
    accumulate(up_a, up_b, &mut sig.upper);
    accumulate(lo_a, lo_b, &mut sig.lower);
    let n_up = normalize(&mut sig.upper);
    let n_lo = normalize(&mut sig.lower);
    sig.weak = n_up < 30.0 || n_lo < 30.0 || bh < 48.0;
    sig
}

/// Mean of several per-frame color sigs → a stable running track signature.
/// `weak` only if EVERY contributing sig was weak.
pub fn color_mean(sigs: &[ColorSig]) -> ColorSig {
    let mut out = ColorSig::default();
    let strong: Vec<&ColorSig> = sigs.iter().filter(|s| !s.weak).collect();
    let pool: &[&ColorSig] = if strong.is_empty() {
        // fall back to all (still produces a histogram, just flagged weak)
        return sigs.first().copied().unwrap_or_default();
    } else {
        &strong
    };
    for s in pool {
        for i in 0..BINS {
            out.upper[i] += s.upper[i];
            out.lower[i] += s.lower[i];
        }
    }
    normalize(&mut out.upper);
    normalize(&mut out.lower);
    out.weak = false;
    out
}

// ---- Fused identity ---------------------------------------------------------

/// One person-segment's full appearance: color always, body/face when a model
/// was loaded and the crop was good enough.
#[derive(Clone, Default)]
pub struct IdSig {
    pub color: ColorSig,
    pub body: Option<Vec<f32>>, // L2-normalized 512-d
    pub face: Option<Vec<f32>>, // L2-normalized 512-d
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IdMatch {
    Same,
    Different,
    Unknown,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Tunable thresholds + crop gates. Conservative defaults bias toward NOT
/// splitting. Every field is overridable via env (see `from_env`).
#[derive(Clone, Copy, Debug)]
pub struct IdentityConfig {
    /// Tracker veto: reject a same-track association whose color distance to the
    /// track's running signature exceeds this (when boxes don't strongly overlap).
    pub veto_color_dist: f32,
    /// IoU above which two boxes are "clearly the same person" → never veto.
    pub strong_iou: f32,
    pub color_same: f32,
    pub color_diff: f32,
    pub body_same: f32,
    pub body_diff: f32,
    pub face_same: f32,
    pub face_diff: f32,
    /// Min person-box height (px) before we spend an OSNet body embedding on it.
    pub reid_min_box_h: f32,
    /// Min eye/keypoint confidence before we trust the pose-derived face crop.
    pub face_kp_conf: f32,
}

impl Default for IdentityConfig {
    fn default() -> Self {
        IdentityConfig {
            veto_color_dist: 0.55,
            strong_iou: 0.50,
            color_same: 0.22,
            color_diff: 0.55,
            body_same: 0.72,
            body_diff: 0.50,
            face_same: 0.45,
            face_diff: 0.25,
            reid_min_box_h: 60.0,
            face_kp_conf: 0.40,
        }
    }
}

fn env_f32(key: &str, cur: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(cur)
}

impl IdentityConfig {
    pub fn from_env() -> Self {
        let mut c = IdentityConfig::default();
        c.veto_color_dist = env_f32("ID_VETO_COLOR_DIST", c.veto_color_dist);
        c.strong_iou = env_f32("ID_STRONG_IOU", c.strong_iou);
        c.color_same = env_f32("ID_COLOR_SAME", c.color_same);
        c.color_diff = env_f32("ID_COLOR_DIFF", c.color_diff);
        c.body_same = env_f32("ID_BODY_SAME", c.body_same);
        c.body_diff = env_f32("ID_BODY_DIFF", c.body_diff);
        c.face_same = env_f32("ID_FACE_SAME", c.face_same);
        c.face_diff = env_f32("ID_FACE_DIFF", c.face_diff);
        c.reid_min_box_h = env_f32("ID_REID_MIN_BOX_H", c.reid_min_box_h);
        c.face_kp_conf = env_f32("ID_FACE_KP_CONF", c.face_kp_conf);
        c
    }
}

/// Fuse the three signals, strongest-available first. A signal that is present
/// but ambiguous (neither clearly same nor clearly different) falls through to
/// the next; if nothing is decisive the verdict is `Unknown`.
pub fn identity_match(a: &IdSig, b: &IdSig, c: &IdentityConfig) -> IdMatch {
    match_explain(a, b, c).0
}

/// Like `identity_match` but also returns which signal decided ("face" / "body"
/// / "color" / "none") — surfaced in the verdict so an operator knows what split
/// two tracks apart.
pub fn match_explain(a: &IdSig, b: &IdSig, c: &IdentityConfig) -> (IdMatch, &'static str) {
    if let (Some(fa), Some(fb)) = (&a.face, &b.face) {
        let s = cosine(fa, fb);
        if s >= c.face_same {
            return (IdMatch::Same, "face");
        }
        if s <= c.face_diff {
            return (IdMatch::Different, "face");
        }
    }
    if let (Some(ba), Some(bb)) = (&a.body, &b.body) {
        let s = cosine(ba, bb);
        if s >= c.body_same {
            return (IdMatch::Same, "body");
        }
        if s <= c.body_diff {
            return (IdMatch::Different, "body");
        }
    }
    if !a.color.weak && !b.color.weak {
        let d = color_distance(&a.color, &b.color);
        if d <= c.color_same {
            return (IdMatch::Same, "color");
        }
        if d >= c.color_diff {
            return (IdMatch::Different, "color");
        }
    }
    (IdMatch::Unknown, "none")
}

// ---- Optional ONNX embedders (env-gated, mirror the pose/behavior pattern) ---

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

fn build_session(model: &str) -> Result<Session> {
    let builder = Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;
    #[cfg(target_os = "macos")]
    let builder = builder.with_execution_providers([CoreMLExecutionProvider::default().build()])?;
    Ok(builder.commit_from_file(model)?)
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    for x in v.iter_mut() {
        *x /= norm;
    }
    v
}

/// Crop a person box to an `out_w × out_h` RGB image, clamped to bounds.
fn crop_resize(img: &RgbImage, d: &Det, out_w: u32, out_h: u32) -> Option<RgbImage> {
    let x1 = d.x1.max(0.0) as u32;
    let y1 = d.y1.max(0.0) as u32;
    let x2 = (d.x2.min(img.width() as f32 - 1.0)) as u32;
    let y2 = (d.y2.min(img.height() as f32 - 1.0)) as u32;
    if x2 <= x1 + 2 || y2 <= y1 + 2 {
        return None;
    }
    let sub = image::imageops::crop_imm(img, x1, y1, x2 - x1, y2 - y1).to_image();
    Some(image::imageops::resize(&sub, out_w, out_h, image::imageops::FilterType::Triangle))
}

/// Body Re-ID embedder (OSNet x1.0). Input 1×3×256×128, ImageNet-normalized RGB.
pub struct ReidCtx {
    session: Mutex<Session>,
    input: String,
    output: String,
}

impl ReidCtx {
    pub fn from_env() -> Result<Option<Self>> {
        let Ok(p) = std::env::var("REID_MODEL") else { return Ok(None) };
        if p.trim().is_empty() {
            return Ok(None);
        }
        let session = build_session(&p).with_context(|| format!("load REID_MODEL {p}"))?;
        let input = session.inputs.first().map(|i| i.name.clone()).unwrap_or_else(|| "image".into());
        let output = session.outputs.first().map(|o| o.name.clone()).unwrap_or_else(|| "embedding".into());
        Ok(Some(ReidCtx { session: Mutex::new(session), input, output }))
    }

    pub fn embed(&self, img: &RgbImage, d: &Det) -> Result<Vec<f32>> {
        let crop = crop_resize(img, d, 128, 256).context("reid crop empty")?;
        let mut t = Array4::<f32>::zeros((1, 3, 256, 128));
        for (x, y, px) in crop.enumerate_pixels() {
            for ch in 0..3 {
                t[[0, ch, y as usize, x as usize]] =
                    (px[ch] as f32 / 255.0 - IMAGENET_MEAN[ch]) / IMAGENET_STD[ch];
            }
        }
        // Hold the lock across extraction — the output tensor borrows the
        // session — and collect into an owned Vec before releasing it.
        let raw: Vec<f32> = {
            let sess = self.session.lock().unwrap();
            let outputs = sess.run(ort::inputs![self.input.as_str() => t.view()]?)?;
            let out = outputs[self.output.as_str()].try_extract_tensor::<f32>()?;
            out.iter().copied().collect()
        };
        Ok(l2_normalize(raw))
    }
}

/// Face embedder (ArcFace `w600k_r50`). Input 1×3×112×112, RGB, (x−127.5)/127.5,
/// aligned from the two eye keypoints. Abstains (None) when the face is too small
/// or the eye keypoints aren't confident enough.
pub struct FaceCtx {
    session: Mutex<Session>,
    input: String,
    output: String,
    min_eye_px: f32,
}

// Canonical ArcFace eye positions in the 112×112 aligned crop.
const ARC_LEFT_EYE: (f32, f32) = (38.2946, 51.6963);
const ARC_RIGHT_EYE: (f32, f32) = (73.5318, 51.5014);

impl FaceCtx {
    pub fn from_env() -> Result<Option<Self>> {
        let Ok(p) = std::env::var("FACE_MODEL") else { return Ok(None) };
        if p.trim().is_empty() {
            return Ok(None);
        }
        let session = build_session(&p).with_context(|| format!("load FACE_MODEL {p}"))?;
        let input = session.inputs.first().map(|i| i.name.clone()).unwrap_or_else(|| "input.1".into());
        let output = session.outputs.first().map(|o| o.name.clone()).unwrap_or_else(|| "683".into());
        Ok(Some(FaceCtx { session: Mutex::new(session), input, output, min_eye_px: 10.0 }))
    }

    /// Align + embed from 17 COCO keypoints. Uses l_eye(1), r_eye(2). Returns
    /// None unless both eyes clear `kp_conf` and are far enough apart to be a
    /// usable face.
    pub fn embed_from_kps(
        &self,
        img: &RgbImage,
        kps: &[(f32, f32, f32); crate::pose::NUM_KP],
        kp_conf: f32,
    ) -> Option<Vec<f32>> {
        let leye = kps[1];
        let reye = kps[2];
        if leye.2 < kp_conf || reye.2 < kp_conf {
            return None;
        }
        let (dx, dy) = (reye.0 - leye.0, reye.1 - leye.1);
        let eye_px = (dx * dx + dy * dy).sqrt();
        if eye_px < self.min_eye_px {
            return None;
        }
        let aligned = self.align(img, (leye.0, leye.1), (reye.0, reye.1))?;
        let mut t = Array4::<f32>::zeros((1, 3, 112, 112));
        for (x, y, px) in aligned.enumerate_pixels() {
            for ch in 0..3 {
                t[[0, ch, y as usize, x as usize]] = (px[ch] as f32 - 127.5) / 127.5;
            }
        }
        let raw: Vec<f32> = {
            let sess = self.session.lock().unwrap();
            let outputs = sess.run(ort::inputs![self.input.as_str() => t.view()].ok()?).ok()?;
            let out = outputs[self.output.as_str()].try_extract_tensor::<f32>().ok()?;
            out.iter().copied().collect()
        };
        Some(l2_normalize(raw))
    }

    /// 2-point similarity warp mapping the canonical eyes → the detected eyes,
    /// sampling a 112×112 aligned RGB crop (bilinear). Janky-but-works.
    fn align(&self, img: &RgbImage, leye: (f32, f32), reye: (f32, f32)) -> Option<RgbImage> {
        // canonical eye vector → detected eye vector gives scale s + rotation θ.
        let cv = (ARC_RIGHT_EYE.0 - ARC_LEFT_EYE.0, ARC_RIGHT_EYE.1 - ARC_LEFT_EYE.1);
        let dv = (reye.0 - leye.0, reye.1 - leye.1);
        let cn = (cv.0 * cv.0 + cv.1 * cv.1).sqrt();
        let dn = (dv.0 * dv.0 + dv.1 * dv.1).sqrt();
        if cn < 1e-3 || dn < 1e-3 {
            return None;
        }
        let s = dn / cn;
        let theta = dv.1.atan2(dv.0) - cv.1.atan2(cv.0);
        let (ct, st) = (theta.cos() * s, theta.sin() * s);
        // R = [[ct,-st],[st,ct]]; t = leye - R*ARC_LEFT_EYE  (maps canonical→src).
        let tx = leye.0 - (ct * ARC_LEFT_EYE.0 - st * ARC_LEFT_EYE.1);
        let ty = leye.1 - (st * ARC_LEFT_EYE.0 + ct * ARC_LEFT_EYE.1);
        let (iw, ih) = (img.width(), img.height());
        let mut out = RgbImage::new(112, 112);
        for dy in 0..112u32 {
            for dx in 0..112u32 {
                let sx = ct * dx as f32 - st * dy as f32 + tx;
                let sy = st * dx as f32 + ct * dy as f32 + ty;
                if sx < 0.0 || sy < 0.0 || sx >= (iw - 1) as f32 || sy >= (ih - 1) as f32 {
                    continue; // leave black (edge of frame)
                }
                let (x0, y0) = (sx.floor() as u32, sy.floor() as u32);
                let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
                let p00 = img.get_pixel(x0, y0);
                let p10 = img.get_pixel(x0 + 1, y0);
                let p01 = img.get_pixel(x0, y0 + 1);
                let p11 = img.get_pixel(x0 + 1, y0 + 1);
                let mut rgb = [0u8; 3];
                for ch in 0..3 {
                    let top = p00[ch] as f32 * (1.0 - fx) + p10[ch] as f32 * fx;
                    let bot = p01[ch] as f32 * (1.0 - fx) + p11[ch] as f32 * fx;
                    rgb[ch] = (top * (1.0 - fy) + bot * fy).round() as u8;
                }
                out.put_pixel(dx, dy, image::Rgb(rgb));
            }
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(w, h, image::Rgb(rgb))
    }

    fn full_box(img: &RgbImage) -> Det {
        Det { x1: 0.0, y1: 0.0, x2: img.width() as f32, y2: img.height() as f32, score: 0.9, cls: 0 }
    }

    #[test]
    fn same_color_is_near_different_color_is_far() {
        let red = solid(120, 300, [200, 30, 30]);
        let red2 = solid(120, 300, [205, 35, 25]);
        let blue = solid(120, 300, [30, 40, 200]);
        let sr = color_sig(&red, &full_box(&red));
        let sr2 = color_sig(&red2, &full_box(&red2));
        let sb = color_sig(&blue, &full_box(&blue));
        assert!(!sr.weak && !sb.weak);
        let same = color_distance(&sr, &sr2);
        let diff = color_distance(&sr, &sb);
        assert!(same < 0.15, "same outfit should be near, got {same}");
        assert!(diff > 0.55, "red vs blue should be far, got {diff}");
    }

    #[test]
    fn tiny_crop_is_weak() {
        let img = solid(8, 20, [100, 100, 100]);
        assert!(color_sig(&img, &full_box(&img)).weak);
    }

    fn sig_from(c: ColorSig) -> IdSig {
        IdSig { color: c, body: None, face: None }
    }

    #[test]
    fn fusion_color_only_decides_when_no_models() {
        let red = solid(120, 300, [200, 30, 30]);
        let blue = solid(120, 300, [30, 40, 200]);
        let cfg = IdentityConfig::default();
        let a = sig_from(color_sig(&red, &full_box(&red)));
        let b = sig_from(color_sig(&blue, &full_box(&blue)));
        assert_eq!(identity_match(&a, &b, &cfg), IdMatch::Different);
        let a2 = sig_from(color_sig(&red, &full_box(&red)));
        assert_eq!(identity_match(&a, &a2, &cfg), IdMatch::Same);
    }

    #[test]
    fn fusion_face_outranks_body_and_color() {
        let cfg = IdentityConfig::default();
        // Color says DIFFERENT, body says DIFFERENT, but a confident face match
        // says SAME → face wins.
        let red = solid(120, 300, [200, 30, 30]);
        let blue = solid(120, 300, [30, 40, 200]);
        let face = l2_normalize(vec![1.0, 0.0, 0.0, 0.0]);
        let a = IdSig {
            color: color_sig(&red, &full_box(&red)),
            body: Some(l2_normalize(vec![1.0, 0.0, 0.0])),
            face: Some(face.clone()),
        };
        let b = IdSig {
            color: color_sig(&blue, &full_box(&blue)),
            body: Some(l2_normalize(vec![0.0, 1.0, 0.0])), // orthogonal → body "different"
            face: Some(face),                               // identical → face "same"
        };
        assert_eq!(identity_match(&a, &b, &cfg), IdMatch::Same);
    }

    #[test]
    fn fusion_body_used_when_face_absent() {
        let cfg = IdentityConfig::default();
        let red = solid(120, 300, [200, 30, 30]);
        let blue = solid(120, 300, [30, 40, 200]);
        // Body identical → Same, even though color is different and no face.
        let emb = l2_normalize(vec![0.3, 0.5, 0.8]);
        let a = IdSig { color: color_sig(&red, &full_box(&red)), body: Some(emb.clone()), face: None };
        let b = IdSig { color: color_sig(&blue, &full_box(&blue)), body: Some(emb), face: None };
        assert_eq!(identity_match(&a, &b, &cfg), IdMatch::Same);
    }

    #[test]
    fn fusion_unknown_when_color_weak_and_no_models() {
        let cfg = IdentityConfig::default();
        let a = sig_from(ColorSig::default()); // weak
        let b = sig_from(ColorSig::default());
        assert_eq!(identity_match(&a, &b, &cfg), IdMatch::Unknown);
    }
}
