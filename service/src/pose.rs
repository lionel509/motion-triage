//! YOLO-pose inference: person boxes + 17 COCO keypoints per person.
//!
//! Handles BOTH exported head layouts, branching on output shape exactly like
//! the detector path in `main.rs`:
//!   * raw grid  `[1, 56, anchors]` — yolo11*-pose: 4 box (cxcywh) + 1 person
//!     score + 17×(x,y,conf), needs NMS;
//!   * end-to-end rows `[1, N, 56|57]` — yolo26*-pose: `[x1,y1,x2,y2,score,
//!     (cls,) 17×(x,y,conf)]`, already top-N/deduped (51 keypoint cols from
//!     the right; prefix is whatever remains, so both 5- and 6-col prefixes
//!     decode).
//!
//! Coordinates come out in SOURCE-frame pixels (the letterbox is undone with
//! the same scale/pad the preprocessor produced). Decode functions are pure
//! over `ArrayViewD` so the unit tests drive them with synthetic tensors —
//! no ort session or model file needed.

use anyhow::{ensure, Context, Result};
use image::RgbImage;
use ndarray::ArrayViewD;
use ort::session::Session;

/// Person-box confidence floor for the pose path. Deliberately permissive —
/// the corpus/training side wants marginal detections WITH their confidences;
/// consumers filter downstream.
pub const POSE_CONF: f32 = 0.25;
const POSE_IOU_NMS: f32 = 0.45;
pub const NUM_KP: usize = 17;

/// COCO-17 joint order (yolo*-pose output order).
pub const JOINT_NAMES: [&str; NUM_KP] = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
];

/// Standard COCO-17 skeleton edges (joint-index pairs) for rendering.
pub const SKELETON: [(usize, usize); 19] = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
];

#[derive(Clone, Debug)]
pub struct PoseDet {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    /// 17 × (x, y, conf), same pixel space as the box.
    pub kps: [(f32, f32, f32); NUM_KP],
}

fn iou(a: &PoseDet, b: &PoseDet) -> f32 {
    let xx1 = a.x1.max(b.x1);
    let yy1 = a.y1.max(b.y1);
    let xx2 = a.x2.min(b.x2);
    let yy2 = a.y2.min(b.y2);
    let inter = (xx2 - xx1).max(0.0) * (yy2 - yy1).max(0.0);
    let aa = (a.x2 - a.x1) * (a.y2 - a.y1);
    let ab = (b.x2 - b.x1) * (b.y2 - b.y1);
    inter / (aa + ab - inter + 1e-6)
}

/// Raw-grid head: `[1, 56, anchors]` in 640-letterbox space.
pub fn decode_grid(out: &ArrayViewD<'_, f32>, conf: f32) -> Vec<PoseDet> {
    let shape = out.shape();
    let na = shape[2];
    let mut dets = Vec::new();
    for a in 0..na {
        let score = out[[0, 4, a]];
        if score < conf {
            continue;
        }
        let (cx, cy, w, h) = (out[[0, 0, a]], out[[0, 1, a]], out[[0, 2, a]], out[[0, 3, a]]);
        let mut kps = [(0.0f32, 0.0f32, 0.0f32); NUM_KP];
        for (j, kp) in kps.iter_mut().enumerate() {
            *kp = (out[[0, 5 + 3 * j, a]], out[[0, 6 + 3 * j, a]], out[[0, 7 + 3 * j, a]]);
        }
        dets.push(PoseDet {
            x1: cx - w / 2.0, y1: cy - h / 2.0,
            x2: cx + w / 2.0, y2: cy + h / 2.0,
            score, kps,
        });
    }
    dets
}

/// End-to-end head: `[1, N, C]` rows, keypoints are the LAST 51 columns —
/// robust to both the 5-col (`x1,y1,x2,y2,score`) and 6-col (`…,cls`)
/// prefixes seen across exports.
pub fn decode_e2e(out: &ArrayViewD<'_, f32>, conf: f32) -> Vec<PoseDet> {
    let shape = out.shape();
    let (n, cols) = (shape[1], shape[2]);
    let prefix = cols - 3 * NUM_KP; // 5 or 6
    let mut dets = Vec::new();
    for i in 0..n {
        let score = out[[0, i, 4]];
        if score < conf {
            continue;
        }
        let mut kps = [(0.0f32, 0.0f32, 0.0f32); NUM_KP];
        for (j, kp) in kps.iter_mut().enumerate() {
            *kp = (
                out[[0, i, prefix + 3 * j]],
                out[[0, i, prefix + 3 * j + 1]],
                out[[0, i, prefix + 3 * j + 2]],
            );
        }
        dets.push(PoseDet {
            x1: out[[0, i, 0]], y1: out[[0, i, 1]],
            x2: out[[0, i, 2]], y2: out[[0, i, 3]],
            score, kps,
        });
    }
    dets
}

/// Greedy NMS (person-only model → single class). Harmless on the e2e head.
pub fn nms(mut dets: Vec<PoseDet>) -> Vec<PoseDet> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep: Vec<PoseDet> = Vec::new();
    'o: for d in dets {
        for k in &keep {
            if iou(&d, k) > POSE_IOU_NMS {
                continue 'o;
            }
        }
        keep.push(d);
    }
    keep
}

/// Undo the letterbox: map box corners AND every keypoint back to
/// source-frame pixels with the preprocessor's (scale, pad_x, pad_y).
pub fn unletterbox(dets: Vec<PoseDet>, scale: f32, px: f32, py: f32) -> Vec<PoseDet> {
    dets.into_iter()
        .map(|mut d| {
            d.x1 = (d.x1 - px) / scale;
            d.y1 = (d.y1 - py) / scale;
            d.x2 = (d.x2 - px) / scale;
            d.y2 = (d.y2 - py) / scale;
            for kp in d.kps.iter_mut() {
                kp.0 = (kp.0 - px) / scale;
                kp.1 = (kp.1 - py) / scale;
            }
            d
        })
        .collect()
}

/// Run pose on one frame: letterbox → session → shape-branched decode →
/// NMS → back to source pixels.
pub fn detect_pose(session: &Session, img: &RgbImage, conf: f32) -> Result<Vec<PoseDet>> {
    let (input, scale, px, py) = crate::preprocess(img);
    let outputs = session.run(ort::inputs!["images" => input.view()]?)?;
    let out = outputs["output0"].try_extract_tensor::<f32>()?;
    let shape = out.shape();
    ensure!(shape.len() == 3, "unexpected pose output rank: {shape:?}");
    let dets = if shape[1] == 4 + 1 + 3 * NUM_KP && shape[2] > 57 {
        decode_grid(&out, conf)
    } else if shape[2] == 56 || shape[2] == 57 {
        decode_e2e(&out, conf)
    } else {
        anyhow::bail!("unrecognized pose head shape {shape:?} (expected [1,56,A] grid or [1,N,56|57] e2e)");
    };
    Ok(unletterbox(nms(dets), scale, px, py))
}

/// CLI: `spike pose <model.onnx> <frames-dir|image> [out-dir]` — detect limbs
/// on real frames, print per-joint detail, write annotated JPEGs.
pub fn run_pose_spike(model: &str, input: &str, out_dir: &str) -> Result<()> {
    let session = crate::build_session(model)?;
    let frames = crate::load_frames(input)?;
    ensure!(!frames.is_empty(), "no frames found at {input}");
    std::fs::create_dir_all(out_dir).with_context(|| format!("mkdir {out_dir}"))?;
    println!("pose spike · model {model} · {} frame(s) → {out_dir}/", frames.len());

    let t0 = std::time::Instant::now();
    let mut total_persons = 0usize;
    for (i, frame) in frames.iter().enumerate() {
        let dets = detect_pose(&session, frame, POSE_CONF)?;
        total_persons += dets.len();
        let mut canvas = frame.clone();
        for d in &dets {
            crate::draw::draw_pose_overlay(&mut canvas, d);
        }
        let path = format!("{out_dir}/frame{i:03}.jpg");
        canvas.save(&path).with_context(|| format!("save {path}"))?;

        let best = dets.iter().cloned().reduce(|a, b| if a.score >= b.score { a } else { b });
        match &best {
            Some(b) => {
                let mean_kc: f32 =
                    b.kps.iter().map(|k| k.2).sum::<f32>() / NUM_KP as f32;
                println!(
                    "  frame {i:03}: {} person(s) · best box {:.2} · mean joint conf {:.2}",
                    dets.len(), b.score, mean_kc,
                );
            }
            None => println!("  frame {i:03}: 0 persons"),
        }
        // Per-joint table for the first frame's best detection — the
        // "which limbs can it actually see from here" readout.
        if i == 0 {
            if let Some(b) = best {
                println!("    per-joint confidence (best person, frame 000):");
                for (j, name) in JOINT_NAMES.iter().enumerate() {
                    let (x, y, c) = b.kps[j];
                    let bar = if c >= 0.5 { "##" } else if c >= 0.25 { "# " } else { ". " };
                    println!("      {bar} {name:<11} {c:>5.2}  ({x:>6.1}, {y:>6.1})");
                }
            }
        }
    }
    let ms = t0.elapsed().as_millis();
    println!(
        "\n  ▶ {} person detection(s) over {} frame(s) · {} ms ({:.1} ms/frame) · annotated frames in {out_dir}/",
        total_persons, frames.len(), ms, ms as f64 / frames.len().max(1) as f64,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    /// Grid tensor [1,56,A] with one strong anchor.
    fn grid_with_one(cx: f32, cy: f32, w: f32, h: f32, score: f32) -> ArrayD<f32> {
        let mut t = ArrayD::<f32>::zeros(ndarray::IxDyn(&[1, 56, 4]));
        let a = 2; // arbitrary anchor column
        t[[0, 0, a]] = cx;
        t[[0, 1, a]] = cy;
        t[[0, 2, a]] = w;
        t[[0, 3, a]] = h;
        t[[0, 4, a]] = score;
        for j in 0..NUM_KP {
            t[[0, 5 + 3 * j, a]] = 10.0 + j as f32;
            t[[0, 6 + 3 * j, a]] = 20.0 + j as f32;
            t[[0, 7 + 3 * j, a]] = 0.9;
        }
        t
    }

    #[test]
    fn grid_decode_box_and_kps() {
        let t = grid_with_one(100.0, 200.0, 40.0, 80.0, 0.8);
        let dets = decode_grid(&t.view(), 0.3);
        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        assert!((d.x1 - 80.0).abs() < 1e-4 && (d.y1 - 160.0).abs() < 1e-4);
        assert!((d.x2 - 120.0).abs() < 1e-4 && (d.y2 - 240.0).abs() < 1e-4);
        assert!((d.kps[3].0 - 13.0).abs() < 1e-4);
        assert!((d.kps[16].1 - 36.0).abs() < 1e-4);
        assert!((d.kps[0].2 - 0.9).abs() < 1e-4);
    }

    #[test]
    fn grid_decode_respects_conf_floor() {
        let t = grid_with_one(100.0, 200.0, 40.0, 80.0, 0.1);
        assert!(decode_grid(&t.view(), 0.3).is_empty());
    }

    /// e2e tensor [1,N,cols] with one row; kpts fill the LAST 51 cols.
    fn e2e_with_one(cols: usize, score: f32) -> ArrayD<f32> {
        let mut t = ArrayD::<f32>::zeros(ndarray::IxDyn(&[1, 3, cols]));
        let prefix = cols - 3 * NUM_KP;
        let i = 1;
        t[[0, i, 0]] = 50.0;
        t[[0, i, 1]] = 60.0;
        t[[0, i, 2]] = 150.0;
        t[[0, i, 3]] = 260.0;
        t[[0, i, 4]] = score;
        for j in 0..NUM_KP {
            t[[0, i, prefix + 3 * j]] = 100.0 + j as f32;
            t[[0, i, prefix + 3 * j + 1]] = 200.0 + j as f32;
            t[[0, i, prefix + 3 * j + 2]] = 0.7;
        }
        t
    }

    #[test]
    fn e2e_decode_57_cols_with_cls() {
        let t = e2e_with_one(57, 0.8);
        let dets = decode_e2e(&t.view(), 0.3);
        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        assert!((d.x1 - 50.0).abs() < 1e-4 && (d.y2 - 260.0).abs() < 1e-4);
        assert!((d.kps[0].0 - 100.0).abs() < 1e-4);
        assert!((d.kps[16].1 - 216.0).abs() < 1e-4);
    }

    #[test]
    fn e2e_decode_56_cols_without_cls() {
        let t = e2e_with_one(56, 0.8);
        let dets = decode_e2e(&t.view(), 0.3);
        assert_eq!(dets.len(), 1);
        assert!((dets[0].kps[0].0 - 100.0).abs() < 1e-4);
    }

    #[test]
    fn nms_suppresses_overlap() {
        let mk = |score: f32| PoseDet {
            x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0, score,
            kps: [(0.0, 0.0, 0.0); NUM_KP],
        };
        let kept = nms(vec![mk(0.9), mk(0.8)]);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn unletterbox_maps_box_and_kps() {
        let mut d = PoseDet {
            x1: 110.0, y1: 220.0, x2: 210.0, y2: 420.0, score: 0.9,
            kps: [(160.0, 320.0, 0.8); NUM_KP],
        };
        d.kps[0] = (110.0, 220.0, 0.8);
        let out = unletterbox(vec![d], 2.0, 10.0, 20.0);
        let d = &out[0];
        assert!((d.x1 - 50.0).abs() < 1e-4 && (d.y1 - 100.0).abs() < 1e-4);
        assert!((d.x2 - 100.0).abs() < 1e-4 && (d.y2 - 200.0).abs() < 1e-4);
        assert!((d.kps[0].0 - 50.0).abs() < 1e-4 && (d.kps[0].1 - 100.0).abs() < 1e-4);
        assert!((d.kps[1].0 - 75.0).abs() < 1e-4 && (d.kps[1].1 - 150.0).abs() < 1e-4);
        assert!((d.kps[0].2 - 0.8).abs() < 1e-6, "conf must be untouched");
    }
}
