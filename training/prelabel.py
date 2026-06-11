"""Heuristic pre-labels for the easy motion classes — walking vs standing.

Bootstrap, not ground truth. Box-center travel (in box-heights, the same
scale-free unit the model's gdx/gdy channels use) separates the two easy
classes with wide conservative bands; everything between the bands stays
BLANK for a human to fill from the skeleton GIFs. The blanks and the
corrections are where the model earns its keep — a model trained only on
these auto-labels has learned little more than a displacement threshold.

::

    # 1. look at the real distribution first, sanity-check the bands:
    python training/prelabel.py --corpus data/corpus_real.jsonl --stats

    # 2. write the labels file (blank = human decides from the GIF):
    python training/prelabel.py --corpus data/corpus_real.jsonl \
        --out data/labels.csv

The output is ``sample_id,label`` — exactly what ``train.py --labels-csv``
eats (it skips blank labels), and reviewable side-by-side with
``render_track.py``'s GIFs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def track_motion_stats(rec: dict) -> dict | None:
    """Scale-free motion summary of one corpus track.

    Returns None when the track has <2 present frames (no motion to
    measure). All distances are in median-box-height units; rates are
    per second using the corpus's ``fps_sampled``.
    """
    frames = rec.get("frames") or []
    fps = float(rec.get("fps_sampled") or 4.0)
    pts: list[tuple[int, float, float, float]] = []  # (idx, cx, cy, h)
    vis_fracs: list[float] = []
    for i, f in enumerate(frames):
        if not f:
            continue
        b = f["box"]
        h = max(1.0, float(b["y2"]) - float(b["y1"]))
        pts.append((i, (float(b["x1"]) + float(b["x2"])) / 2.0,
                    (float(b["y1"]) + float(b["y2"])) / 2.0, h))
        kps = f.get("kps") or f.get("keypoints") or []
        if kps:
            vis_fracs.append(sum(1 for k in kps if k[2] >= 0.25) / len(kps))
    if len(pts) < 2:
        return None
    med_h = sorted(p[3] for p in pts)[len(pts) // 2]
    path = 0.0
    for (i0, x0, y0, h0), (i1, x1, y1, _) in zip(pts, pts[1:]):
        # Per-step scale: the box height at the step's start — robust to
        # the person approaching/leaving the camera mid-track.
        path += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / max(1.0, h0)
    net = ((pts[-1][1] - pts[0][1]) ** 2 + (pts[-1][2] - pts[0][2]) ** 2) ** 0.5 / med_h
    visible_s = (pts[-1][0] - pts[0][0]) / fps
    return {
        "path_bh": path,
        "net_bh": net,
        "visible_s": visible_s,
        "path_rate": path / max(0.25, visible_s),
        "n_present": len(pts),
        "med_h_frac": med_h / max(1.0, float(rec.get("frame_h") or 1)),
        # Pose quality: mean fraction of joints above draw-confidence. A
        # stationary box with a degenerate skeleton is usually a false
        # detection — auto-labeling it "standing" would poison the class.
        "kp_vis": (sum(vis_fracs) / len(vis_fracs)) if vis_fracs else 0.0,
    }


def classify(st: dict, a: argparse.Namespace) -> str:
    if st["kp_vis"] < a.min_kp_vis:
        return ""  # degenerate skeleton — human decides from the GIF
    if st["net_bh"] >= a.walk_net and st["path_rate"] >= a.walk_rate:
        return "walking"
    # Pacing: lots of movement, no travel — the kid milling by the truck,
    # back-and-forth in front of a door. Without this class, in-place
    # activity collapses into "standing" (real-footage finding 2026-06-10).
    if (st["path_rate"] >= a.pace_rate and st["net_bh"] <= a.pace_net
            and st["visible_s"] >= a.stand_min_s):
        return "pacing"
    if (st["path_rate"] <= a.stand_rate and st["net_bh"] <= a.stand_net
            and st["visible_s"] >= a.stand_min_s):
        return "standing"
    return ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", default=None, help="labels CSV to write")
    p.add_argument("--stats", action="store_true",
                   help="print the motion distribution and exit")
    # Conservative bands — calibrate against --stats on YOUR corpus before
    # trusting them. The gap between the bands is deliberate.
    p.add_argument("--walk-net", type=float, default=1.0,
                   help="walking: net displacement ≥ this (box-heights)")
    p.add_argument("--walk-rate", type=float, default=0.25,
                   help="walking: path rate ≥ this (box-heights/s)")
    p.add_argument("--stand-rate", type=float, default=0.10,
                   help="standing: path rate ≤ this")
    p.add_argument("--stand-net", type=float, default=0.40,
                   help="standing: net displacement ≤ this")
    p.add_argument("--stand-min-s", type=float, default=3.0,
                   help="standing: visible at least this long (s)")
    p.add_argument("--pace-rate", type=float, default=0.20,
                   help="pacing: path rate ≥ this with low net travel")
    p.add_argument("--pace-net", type=float, default=0.90,
                   help="pacing: net displacement ≤ this")
    p.add_argument("--min-kp-vis", type=float, default=0.5,
                   help="auto-label only when this mean fraction of joints "
                        "is visible (degenerate skeletons stay blank)")
    a = p.parse_args()

    rows: list[tuple[str, dict]] = []
    with open(a.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            st = track_motion_stats(rec)
            if st is None:
                continue
            rows.append((str(rec.get("sample_id")), st))
    if not rows:
        print("no usable tracks in corpus", file=sys.stderr)
        return 1

    if a.stats:
        def pct(vals: list[float], q: float) -> float:
            s = sorted(vals)
            return s[min(len(s) - 1, int(q * len(s)))]
        for key in ("path_rate", "net_bh", "visible_s", "med_h_frac"):
            vals = [st[key] for _, st in rows]
            print(f"{key:<12} p10 {pct(vals, .10):6.2f}  p25 {pct(vals, .25):6.2f}  "
                  f"p50 {pct(vals, .50):6.2f}  p75 {pct(vals, .75):6.2f}  "
                  f"p90 {pct(vals, .90):6.2f}  p99 {pct(vals, .99):6.2f}")
        print(f"tracks: {len(rows)}")
        return 0

    if not a.out:
        print("error: --out required (or use --stats)", file=sys.stderr)
        return 2
    n = {"walking": 0, "standing": 0, "pacing": 0, "": 0}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# sample_id", "label  (auto-prelabeled; BLANKS need a human; "
                    "fix anything the GIF disagrees with)"])
        for sid, st in rows:
            lb = classify(st, a)
            n[lb] += 1
            w.writerow([sid, lb])
    print(f"wrote {out}: {n['walking']} walking · {n['standing']} standing · "
          f"{n['pacing']} pacing · {n['']} blank (human review) of {len(rows)} tracks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
