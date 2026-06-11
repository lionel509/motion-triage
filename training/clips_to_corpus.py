"""Video clips → pose-track corpus JSONL. The no-dashboard entry point.

The training pipeline (``dataset.py`` → ``train.py`` → ``autotrain.py``)
eats a corpus of per-track 17-keypoint sequences. Deployments that run the
full dashboard export that corpus from their database; everyone else — and
every bootstrap — starts from a folder of short clips. This closes that
gap: point it at a directory of videos, get a ``corpus.jsonl`` the rest of
the pipeline consumes unchanged.

::

    python training/clips_to_corpus.py --clips data/clips/batch1 \
        --out data/corpus.jsonl --device cpu

Per clip: sample frames at ``--fps``, run YOLO-pose on each, associate
detections across frames with a gap-tolerant greedy-IoU tracker, and emit
one JSONL line per track. ``sample_id`` is ``{clip-stem}-t{n}`` — unique
per TRACK, so two people crossing one clip can carry different labels in
``labels.csv``.

Filename convention (optional): ``{anything}__{YYYYMMDDTHHMMSSZ}.mp4``.
When the suffix is present it supplies ``hour_utc``/``dow_utc`` (the
night-slice metric needs them); otherwise file mtime is used. Clip pullers
should pseudonymize the ``{anything}`` part — this script never needs to
know what the clip was or where it came from.

Privacy: the output contains boxes + skeletons only, never pixels. Keep
both clips and corpus out of the public repo regardless (``data/`` is
gitignored).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
TS_RE = re.compile(r"__(\d{8}T\d{6}Z)$")


def iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / max(1e-6, area_a + area_b - inter)


def greedy_iou_tracks(
    frames: list[list[dict]], *, iou_threshold: float = 0.3, max_gap: int = 6,
) -> list[dict[int, dict]]:
    """Associate per-frame detections into tracks.

    ``frames[i]`` is the detection list for sampled frame ``i``. Returns one
    ``{frame_idx: detection}`` map per track. Greedy best-IoU matching
    against each track's last seen box; a track survives ``max_gap``
    frames unseen before it's closed. Same association family as the
    service's runtime tracker — close enough for corpus building, where
    a rare ID switch costs one noisy label, not an alert.
    """
    tracks: list[dict] = []  # {"last": det, "last_idx": int, "frames": {idx: det}}
    for fi, dets in enumerate(frames):
        open_tracks = [t for t in tracks if fi - t["last_idx"] <= max_gap]
        used: set[int] = set()
        # Highest-confidence detections claim their track first.
        for det in sorted(dets, key=lambda d: -d.get("det_conf", 0.0)):
            best, best_iou = None, iou_threshold
            for ti, t in enumerate(open_tracks):
                if ti in used:
                    continue
                v = iou(det["box"], t["last"]["box"])
                if v > best_iou:
                    best, best_iou = ti, v
            if best is not None:
                t = open_tracks[best]
                used.add(best)
                t["last"], t["last_idx"] = det, fi
                t["frames"][fi] = det
            else:
                tracks.append({"last": det, "last_idx": fi, "frames": {fi: det}})
    return [t["frames"] for t in tracks]


def clip_timestamp(path: Path) -> datetime:
    m = TS_RE.search(path.stem)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc)
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def extract_clip(
    path: Path, model, *, target_fps: float, max_frames: int,
    imgsz: int, conf: float, device: str, min_box_frac: float,
) -> tuple[list[list[dict]], int, int, float]:
    """Run pose on sampled frames → (per-frame detections, w, h, sampled_fps)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(src_fps / max(0.1, target_fps)))
    sampled_fps = src_fps / step

    per_frame: list[list[dict]] = []
    w = h = 0
    src_i = 0
    while len(per_frame) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if src_i % step:
            src_i += 1
            continue
        src_i += 1
        h, w = frame.shape[:2]
        res = model.predict(frame, imgsz=imgsz, conf=conf, classes=[0],
                            device=device, verbose=False)[0]
        dets: list[dict] = []
        if res.boxes is not None and len(res.boxes):
            boxes = res.boxes.xyxy.cpu().tolist()
            confs = res.boxes.conf.cpu().tolist()
            kpts = (res.keypoints.data.cpu().tolist()
                    if res.keypoints is not None else [[]] * len(boxes))
            for (x1, y1, x2, y2), c, kp in zip(boxes, confs, kpts):
                if (y2 - y1) < min_box_frac * h:
                    continue  # speck — night-mode hallucination territory
                dets.append({
                    "box": {"x1": round(x1, 1), "y1": round(y1, 1),
                            "x2": round(x2, 1), "y2": round(y2, 1)},
                    "det_conf": round(c, 4),
                    "kps": [[round(x, 1), round(y, 1), round(kc, 3)]
                            for x, y, kc in kp],
                })
        per_frame.append(dets)
    cap.release()
    if n_src and not per_frame:
        raise RuntimeError("no decodable frames")
    return per_frame, w, h, sampled_fps


def track_to_record(
    track: dict[int, dict], *, sample_id: str, track_index: int,
    w: int, h: int, sampled_fps: float, ts: datetime,
) -> dict:
    lo, hi = min(track), max(track)
    frames: list[dict | None] = []
    for fi in range(lo, hi + 1):
        det = track.get(fi)
        if det is None:
            frames.append(None)
        else:
            frames.append({"frame_index": fi, "box": det["box"],
                           "kps": det["kps"], "det_conf": det["det_conf"]})
    return {
        "sample_id": sample_id,
        "track_index": track_index,
        "frames": frames,
        "n_frames": len(frames),
        "frame_w": w,
        "frame_h": h,
        "fps_sampled": round(sampled_fps, 3),
        "frame_source": "clip",
        "hour_utc": ts.hour,
        "dow_utc": ts.weekday(),
        "labels": {},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips", required=True, help="directory of video files")
    p.add_argument("--out", required=True, help="corpus JSONL to write")
    p.add_argument("--model", default="yolo11n-pose.pt")
    p.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    # Defaults tuned on real storefront clips: a walk-through is often only
    # 1.5–2.5 s in frame, so sample densely (6 fps) and keep any track seen
    # for ≥1 s (6 frames). 96 frames covers a full 15 s clip at 6 fps.
    p.add_argument("--fps", type=float, default=6.0, help="sampling rate")
    p.add_argument("--max-frames", type=int, default=96, help="per clip")
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--max-gap", type=int, default=6,
                   help="sampled frames a track survives unseen")
    p.add_argument("--min-frames", type=int, default=6,
                   help="drop tracks seen in fewer sampled frames")
    p.add_argument("--min-box-frac", type=float, default=0.05,
                   help="drop boxes shorter than this fraction of frame height")
    p.add_argument("--limit", type=int, default=0, help="clips to process (0=all)")
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="process clips where index %% N == I (e.g. 0/4) — "
                        "run N processes on N shards and concatenate the "
                        "output JSONLs; cap each process's torch threads "
                        "(OMP_NUM_THREADS) or they fight for every core")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("error: ultralytics required — pip install ultralytics", file=sys.stderr)
        return 2

    clips = sorted(
        f for f in Path(args.clips).iterdir()
        if f.suffix.lower() in VIDEO_EXTS
    )
    if args.shard:
        try:
            i_s, n_s = args.shard.split("/")
            shard_i, shard_n = int(i_s), int(n_s)
            assert 0 <= shard_i < shard_n
        except (ValueError, AssertionError):
            print(f"error: --shard must be I/N with 0 <= I < N, got {args.shard!r}",
                  file=sys.stderr)
            return 2
        clips = [c for ci, c in enumerate(clips) if ci % shard_n == shard_i]
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        print(f"error: no videos under {args.clips}", file=sys.stderr)
        return 2

    model = YOLO(args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_tracks = n_empty = n_err = 0
    with out_path.open("w", encoding="utf-8") as out:
        for ci, clip in enumerate(clips, 1):
            try:
                per_frame, w, h, sfps = extract_clip(
                    clip, model, target_fps=args.fps, max_frames=args.max_frames,
                    imgsz=args.imgsz, conf=args.conf, device=args.device,
                    min_box_frac=args.min_box_frac,
                )
            except Exception as e:  # noqa: BLE001 — skip bad files, keep going
                n_err += 1
                print(f"  [{ci}/{len(clips)}] {clip.name}: SKIP ({e})", file=sys.stderr)
                continue
            tracks = greedy_iou_tracks(
                per_frame, iou_threshold=args.iou_threshold, max_gap=args.max_gap)
            kept = 0
            ts = clip_timestamp(clip)
            for ti, tr in enumerate(t for t in tracks if len(t) >= args.min_frames):
                rec = track_to_record(
                    tr, sample_id=f"{clip.stem}-t{ti}", track_index=ti,
                    w=w, h=h, sampled_fps=sfps, ts=ts)
                out.write(json.dumps(rec) + "\n")
                kept += 1
            n_tracks += kept
            if not kept:
                n_empty += 1
            if ci % 20 == 0 or ci == len(clips):
                print(f"  [{ci}/{len(clips)}] tracks so far: {n_tracks}")

    print(f"done: {n_tracks} track(s) from {len(clips)} clip(s) -> {out_path}"
          f"  ({n_empty} clips with no usable person track, {n_err} unreadable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
