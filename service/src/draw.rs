//! Minimal raster overlays on `RgbImage` — boxes, joints, skeleton edges.
//!
//! Hand-rolled (Bresenham lines, scanned circles) instead of pulling in
//! `imageproc`: ~80 lines buys zero new dependencies in a service that ships
//! to customer-adjacent boxes. Every write is bounds-checked, so partially
//! off-frame detections draw their visible part instead of panicking.

use image::{Rgb, RgbImage};

use crate::pose::{PoseDet, NUM_KP, SKELETON};

const BOX_COLOR: Rgb<u8> = Rgb([0, 255, 96]);      // green
const EDGE_COLOR: Rgb<u8> = Rgb([0, 196, 255]);    // cyan
const JOINT_HI: Rgb<u8> = Rgb([0, 255, 96]);       // conf ≥ 0.5
const JOINT_LO: Rgb<u8> = Rgb([255, 165, 0]);      // 0.25 ≤ conf < 0.5
const KP_DRAW_CONF: f32 = 0.25;                    // below: joint not drawn

fn put(img: &mut RgbImage, x: i64, y: i64, c: Rgb<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, c);
    }
}

fn line(img: &mut RgbImage, x0: i64, y0: i64, x1: i64, y1: i64, c: Rgb<u8>) {
    // Bresenham.
    let (mut x0, mut y0) = (x0, y0);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        put(img, x0, y0, c);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn thick_line(img: &mut RgbImage, x0: i64, y0: i64, x1: i64, y1: i64, c: Rgb<u8>) {
    // 2px: the line + a 1px offset along the minor axis.
    line(img, x0, y0, x1, y1, c);
    if (x1 - x0).abs() >= (y1 - y0).abs() {
        line(img, x0, y0 + 1, x1, y1 + 1, c);
    } else {
        line(img, x0 + 1, y0, x1 + 1, y1, c);
    }
}

fn rect(img: &mut RgbImage, x1: i64, y1: i64, x2: i64, y2: i64, c: Rgb<u8>) {
    for t in 0..2i64 {
        line(img, x1 + t, y1 + t, x2 - t, y1 + t, c);
        line(img, x1 + t, y2 - t, x2 - t, y2 - t, c);
        line(img, x1 + t, y1 + t, x1 + t, y2 - t, c);
        line(img, x2 - t, y1 + t, x2 - t, y2 - t, c);
    }
}

fn filled_circle(img: &mut RgbImage, cx: i64, cy: i64, r: i64, c: Rgb<u8>) {
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                put(img, cx + dx, cy + dy, c);
            }
        }
    }
}

/// Box + skeleton edges + per-joint dots (colored by confidence) for one
/// person. Joints below `KP_DRAW_CONF` are skipped; an edge draws only when
/// BOTH its joints are drawable — no phantom limbs on occluded sides.
pub fn draw_pose_overlay(img: &mut RgbImage, det: &PoseDet) {
    rect(
        img,
        det.x1 as i64, det.y1 as i64,
        det.x2 as i64, det.y2 as i64,
        BOX_COLOR,
    );
    for &(a, b) in SKELETON.iter() {
        let (ax, ay, ac) = det.kps[a];
        let (bx, by, bc) = det.kps[b];
        if ac >= KP_DRAW_CONF && bc >= KP_DRAW_CONF {
            thick_line(img, ax as i64, ay as i64, bx as i64, by as i64, EDGE_COLOR);
        }
    }
    for j in 0..NUM_KP {
        let (x, y, c) = det.kps[j];
        if c < KP_DRAW_CONF {
            continue;
        }
        let color = if c >= 0.5 { JOINT_HI } else { JOINT_LO };
        filled_circle(img, x as i64, y as i64, 3, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offscreen_detection_does_not_panic() {
        let mut img = RgbImage::new(64, 64);
        let det = PoseDet {
            x1: -50.0, y1: -50.0, x2: 200.0, y2: 200.0, score: 0.9,
            kps: [(-10.0, 500.0, 0.9); NUM_KP],
        };
        draw_pose_overlay(&mut img, &det); // bounds-checked writes only
    }

    #[test]
    fn drawing_marks_pixels() {
        let mut img = RgbImage::new(64, 64);
        let mut kps = [(0.0f32, 0.0f32, 0.0f32); NUM_KP]; // conf 0 → no joints drawn
        kps[0] = (32.0, 32.0, 0.9);
        let det = PoseDet { x1: 8.0, y1: 8.0, x2: 56.0, y2: 56.0, score: 0.9, kps };
        draw_pose_overlay(&mut img, &det);
        assert_eq!(*img.get_pixel(8, 8), Rgb([0, 255, 96]), "box corner drawn");
        assert_eq!(*img.get_pixel(32, 32), Rgb([0, 255, 96]), "joint dot drawn");
    }
}
