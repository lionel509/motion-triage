"""Generate a SYNTHETIC behavior corpus — smoke fuel for the training loop.

Emits corpus-shaped JSONL (same line shape as the real
``export_pose_dataset.py`` output) plus a ``labels.csv``, with four
classes whose keypoint dynamics are mechanically distinct:

  * walking   — box translates steadily; ankles alternate (gait swing)
  * running   — same shape, ~3× the speed and stride amplitude
  * standing  — stationary box, small jitter
  * crouching — box height collapses ~40%; hips drop toward the ankles

This exists so the WHOLE pipeline — train.py → ONNX → the Rust service's
`spike behavior` — can be proven end-to-end before a single real clip is
labeled. It is NOT a substitute for real data: a model trained here
learns synthetic physics, nothing about your cameras. Throw it away the
moment staged/real labels exist.

::

    python training/gen_synth_corpus.py --out /tmp/synth
    python training/train.py --corpus /tmp/synth/corpus.jsonl \
        --labels-csv /tmp/synth/labels.csv --epochs 15 \
        --out /tmp/synth/behavior.onnx
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

T = 16          # frames per track
FRAME_W, FRAME_H = 960, 540


def _person_kps(rng, x1: float, y1: float, h: float, phase: float,
                stride: float, crouch: float) -> list[list[float]]:
    """17 plausible COCO joints inside a box at (x1,y1) height h.
    ``phase``/``stride`` drive the gait swing; ``crouch`` ∈ [0,1] drops
    the upper body toward the legs."""
    w = 0.4 * h
    cx = x1 + w / 2
    j = lambda fx, fy: [cx + fx * w, y1 + fy * h, round(0.6 + 0.3 * rng.random(), 3)]  # noqa: E731
    import math
    swing = math.sin(phase) * stride            # ankle/wrist alternation
    drop = 0.25 * crouch                         # upper-body vertical drop
    kps = [
        j(0.0, 0.06 + drop),                     # nose
        j(-0.05, 0.05 + drop), j(0.05, 0.05 + drop),   # eyes
        j(-0.10, 0.06 + drop), j(0.10, 0.06 + drop),   # ears
        j(-0.22, 0.20 + drop), j(0.22, 0.20 + drop),   # shoulders
        j(-0.28, 0.35 + drop), j(0.28, 0.35 + drop),   # elbows
        [cx - 0.25 * w + swing * w, y1 + (0.48 + drop * 0.8) * h, 0.8],  # l wrist
        [cx + 0.25 * w - swing * w, y1 + (0.48 + drop * 0.8) * h, 0.8],  # r wrist
        j(-0.12, 0.52 + drop * 0.6), j(0.12, 0.52 + drop * 0.6),         # hips
        j(-0.14, 0.75), j(0.14, 0.75),                                   # knees
        [cx - 0.12 * w + swing * w, y1 + 0.97 * h, 0.85],                # l ankle
        [cx + 0.12 * w - swing * w, y1 + 0.97 * h, 0.85],                # r ankle
    ]
    # mild jitter everywhere — clean synthetic joints train a brittle net
    return [[round(x + rng.gauss(0, 1.0), 2),
             round(y + rng.gauss(0, 1.0), 2), c] for x, y, c in kps]


def _track(rng, behavior: str) -> list[dict]:
    import math
    x = rng.uniform(80, 300)
    y = rng.uniform(120, 200)
    h = rng.uniform(160, 240)
    frames = []
    for t in range(T):
        if behavior == "walking":
            x += rng.uniform(6, 10)
            phase, stride, crouch, hh = t * 0.9, 0.18, 0.0, h
        elif behavior == "running":
            x += rng.uniform(22, 30)
            phase, stride, crouch, hh = t * 1.8, 0.30, 0.0, h
        elif behavior == "standing":
            x += rng.gauss(0, 0.8)
            phase, stride, crouch, hh = 0.0, 0.0, 0.0, h + rng.gauss(0, 1.0)
        else:  # crouching — height collapses over the first half, stays down
            k = min(1.0, t / (T / 2))
            hh = h * (1.0 - 0.4 * k)
            y_t = y + (h - hh)                  # feet stay planted
            phase, stride, crouch = 0.0, 0.0, k
            frames.append(_mk_frame(rng, t, x, y_t, hh, phase, stride, crouch))
            continue
        frames.append(_mk_frame(rng, t, x, y, hh, phase, stride, crouch))
    return frames


def _mk_frame(rng, t, x1, y1, h, phase, stride, crouch) -> dict:
    import math  # noqa: F401 — phase math lives in _person_kps
    w = 0.4 * h
    return {
        "i": t,
        "box": {"x1": round(x1, 2), "y1": round(y1, 2),
                "x2": round(x1 + w, 2), "y2": round(y1 + h, 2)},
        "conf": round(0.6 + 0.3 * rng.random(), 3),
        "kps": _person_kps(rng, x1, y1, h, phase, stride, crouch),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--per-class", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = ["walking", "running", "standing", "crouching"]

    corpus = out / "corpus.jsonl"
    labels = out / "labels.csv"
    n = 0
    with corpus.open("w", encoding="utf-8") as cf, \
            labels.open("w", newline="", encoding="utf-8") as lf:
        writer = csv.writer(lf)
        writer.writerow(["# sample_id", "label  (SYNTHETIC smoke data)"])
        for behavior in classes:
            for k in range(args.per_class):
                sid = f"synth_{behavior}_{k:03d}"
                frames = _track(rng, behavior)
                # a few gap frames, like real occlusion
                for gi in rng.sample(range(T), k=rng.choice([0, 1, 2])):
                    frames[gi] = None
                cf.write(json.dumps({
                    "sample_id": sid,
                    "track_index": 0,
                    "frame_source": "clip",
                    "frame_count": T,
                    "frame_w": FRAME_W, "frame_h": FRAME_H,
                    "model_name": "synthetic",
                    "hour_utc": rng.choice([3, 4, 11, 15, 21, 23]),
                    "dow_utc": rng.randrange(7),
                    "frames": frames,
                    "labels": {"disposition": None,
                               "operator_disposition_correct": None,
                               "task_category": None, "llm_verdict": None,
                               "operator_correction": None},
                }) + "\n")
                writer.writerow([sid, behavior])
                n += 1
    print(f"wrote {n} synthetic tracks -> {corpus} + {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
