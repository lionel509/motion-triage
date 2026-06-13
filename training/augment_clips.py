"""Multiply clips into angle + lighting variants — before pose extraction.

The behavior NN is skeleton-based, so augmentation belongs HERE, on the
video fed to the pose model — not on the skeletons. Rotating the frame
changes the joint geometry the pose model sees (simulating different camera
mounting angles); relighting changes joint *confidence* (simulating night /
glare / backlight), which is exactly where pose quality — and the night
metric — was weakest.

Each input clip becomes several variants; run clips_to_corpus over the
output dir afterward. Tagged filenames keep the timestamp suffix so the
corpus still parses hour_utc, and keep variants as distinct sample_ids.

::

    python training/augment_clips.py --clips data/clips/meva_door \
        --out data/clips/door_aug --variants rot-10,rot10,dark,bright,lowlight

Variants:
  rot<deg>   rotate by <deg> degrees (e.g. rot-10, rot15)
  bright     brighten (gain 1.4)
  dark       darken (gain 0.5)
  lowlight   darken + gamma + mild noise — night-mode simulation
  contrast   CLAHE local-contrast (helps backlit / washed scenes)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
TS_RE = re.compile(r"(__\d{8}T\d{6}Z)")


def _apply(frame, kind: str, cv2, np):
    if kind.startswith("rot"):
        deg = float(kind[3:])
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if kind == "bright":
        return cv2.convertScaleAbs(frame, alpha=1.4, beta=15)
    if kind == "dark":
        return cv2.convertScaleAbs(frame, alpha=0.5, beta=-10)
    if kind == "lowlight":
        d = cv2.convertScaleAbs(frame, alpha=0.45, beta=-12)
        inv = 1.0 / 1.6
        lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype="uint8")
        d = cv2.LUT(d, lut)
        noise = np.random.default_rng(0).integers(-8, 9, d.shape, dtype="int16")
        return np.clip(d.astype("int16") + noise, 0, 255).astype("uint8")
    if kind == "contrast":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    raise ValueError(f"unknown variant {kind}")


def augment_clip(path: Path, out_dir: Path, variants: list[str], cv2, np) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        return 0
    # Keep the timestamp suffix; insert the variant tag before it so the
    # corpus still reads hour_utc and each variant is a unique sample_id.
    stem = path.stem
    m = TS_RE.search(stem)
    base, ts = (stem[:m.start()], m.group(1)) if m else (stem, "")
    n = 0
    for v in variants:
        dest = out_dir / f"{base}-{v}{ts}.mp4"
        if dest.exists():
            n += 1
            continue
        vw = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fr in frames:
            vw.write(_apply(fr, v, cv2, np))
        vw.release()
        n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variants", default="rot-12,rot12,bright,lowlight",
                   help="comma-separated; see module docstring")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    import cv2
    import numpy as np

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    clips = sorted(f for f in Path(args.clips).iterdir() if f.suffix.lower() in VIDEO_EXTS)
    if args.limit:
        clips = clips[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, c in enumerate(clips, 1):
        total += augment_clip(c, out, variants, cv2, np)
        if i % 20 == 0 or i == len(clips):
            print(f"  [{i}/{len(clips)}] {total} variants written")
    print(f"done: {total} variant clips from {len(clips)} source clips "
          f"({len(variants)} variants each) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
