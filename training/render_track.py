"""Render corpus tracks as animated skeleton GIFs — the labeling tool.

Labeling real footage by re-watching clips is slow and drags raw video
around. This renders each exported track as a stick-figure animation
instead: **you can label "walking / running / crouching / reaching" from
the skeleton alone**, no faces, no pixels — faster AND privacy-clean
(the GIFs can leave the box; the footage never has to).

::

    python training/render_track.py --corpus corpus.jsonl --out-dir gifs/ \
        --labels-template gifs/labels.csv

Workflow: open the GIF folder, watch, fill the second column of
labels.csv, feed it to train.py / autotrain.py. A few hundred labels is
an evening, not a project.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# COCO-17 skeleton edges (matches service/src/pose.rs SKELETON).
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]
KP_DRAW_CONF = 0.25

BG = (16, 18, 24)
BOX = (0, 200, 90)
EDGE = (0, 170, 230)
JOINT_HI = (0, 230, 110)
JOINT_LO = (240, 160, 0)
GAP_TEXT = (140, 140, 150)


def _render_frame(draw_mod, size, frame, scale):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(img)
    if frame is None:
        d.text((10, 10), "no detection", fill=GAP_TEXT)
        return img
    box = frame["box"]
    d.rectangle(
        [box["x1"] * scale, box["y1"] * scale, box["x2"] * scale, box["y2"] * scale],
        outline=BOX, width=2,
    )
    kps = frame.get("kps") or frame.get("keypoints") or []
    pts = [(k[0] * scale, k[1] * scale, k[2]) for k in kps]
    for a, b in SKELETON:
        if a < len(pts) and b < len(pts) and pts[a][2] >= KP_DRAW_CONF and pts[b][2] >= KP_DRAW_CONF:
            d.line([pts[a][:2], pts[b][:2]], fill=EDGE, width=2)
    for x, y, c in pts:
        if c < KP_DRAW_CONF:
            continue
        color = JOINT_HI if c >= 0.5 else JOINT_LO
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
    return img


def render_track(rec: dict, out_path: Path, *, height: int = 360, fps: int = 6) -> bool:
    """One corpus line → one GIF. Returns False when the line has no frames."""
    from PIL import ImageDraw  # noqa: F401 — fail early if Pillow is absent
    frames = rec.get("frames")
    if not frames:
        return False
    fw = int(rec.get("frame_w") or 960)
    fh = int(rec.get("frame_h") or 540)
    scale = height / fh
    size = (max(64, round(fw * scale)), height)
    imgs = [_render_frame(None, size, f, scale) for f in frames]
    imgs[0].save(
        out_path, save_all=True, append_images=imgs[1:],
        duration=int(1000 / fps), loop=0,
    )
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--labels-template", default=None,
                   help="also write a sample_id,label CSV to fill in")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    n = 0
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if args.limit and n >= args.limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(rec.get("sample_id") or rec.get("event_id") or f"line{n}")
            ti = rec.get("track_index", 0)
            gif = out / f"{sid}_t{ti}.gif"
            if render_track(rec, gif, height=args.height, fps=args.fps):
                rows.append(sid)
                n += 1
    if args.labels_template:
        seen = set()
        with open(args.labels_template, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["# sample_id", "label  (fill in: walking/running/standing/crouching/...)"])
            for sid in rows:
                if sid not in seen:
                    seen.add(sid)
                    w.writerow([sid, ""])
    print(f"rendered {n} GIF(s) -> {out}"
          + (f" + template {args.labels_template}" if args.labels_template else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
