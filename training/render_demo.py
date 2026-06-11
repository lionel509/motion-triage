"""Annotated demo video — SEE what the service sees.

Runs the full chain densely over one clip (pose every frame, IoU tracking,
behavior NN per track) and writes a video with boxes, 17-joint skeletons,
limb edges, and the NN's verdict stamped above each person. The same math
the live service runs, drawn frame by frame.

::

    python training/render_demo.py --clip data/clips/x.mp4 \
        --model models/behavior.onnx --out data/demo/x_annotated.mp4

Optionally --stills data/demo/stills writes a few annotated PNGs (handy
for docs / sharing without video).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clips_to_corpus import greedy_iou_tracks  # noqa: E402
from dataset import track_to_tensor  # noqa: E402
from infer_mlx import forward_numpy, load_sidecars  # noqa: E402

# COCO-17 skeleton edges (matches service/src/pose.rs SKELETON).
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]
KP_CONF = 0.25
# BGR (cv2): box green, edges cyan-blue, joints green/orange like the service.
BOX = (90, 200, 0)
EDGE = (230, 170, 0)
JOINT_HI = (110, 230, 0)
JOINT_LO = (0, 160, 240)
LABEL_BG = (24, 18, 16)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True)
    p.add_argument("--model", required=True, help="behavior .onnx (sidecars next to it)")
    p.add_argument("--pose-model", default="yolo11n-pose.pt")
    p.add_argument("--out", required=True, help="annotated .mp4 to write")
    p.add_argument("--stills", default=None, help="also dump a few PNG stills here")
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--step", type=int, default=1, help="process every Nth frame")
    p.add_argument("--min-frames", type=int, default=6)
    args = p.parse_args()

    import cv2
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(args.pose_model)
    weights, meta = load_sidecars(args.model)
    classes = meta["classes"]

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        print(f"error: cannot open {args.clip}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    # Pass 1 — pose on every Nth frame, keep the raw frames for drawing.
    frames_bgr, dets_per_frame = [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % args.step:
            i += 1
            continue
        i += 1
        res = model.predict(fr, imgsz=args.imgsz, conf=args.conf, classes=[0],
                            device=args.device, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes):
            kpts = (res.keypoints.data.cpu().tolist()
                    if res.keypoints is not None else [[]] * len(res.boxes))
            for (x1, y1, x2, y2), c, kp in zip(
                    res.boxes.xyxy.cpu().tolist(), res.boxes.conf.cpu().tolist(), kpts):
                dets.append({"box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                             "det_conf": c, "kps": kp})
        frames_bgr.append(fr)
        dets_per_frame.append(dets)
    cap.release()
    if not frames_bgr:
        print("error: no frames decoded", file=sys.stderr)
        return 2

    # Pass 2 — track + classify each track with the behavior NN.
    tracks = greedy_iou_tracks(dets_per_frame)
    verdicts: list[tuple[dict, str, float]] = []  # (frames-map, label, conf)
    for tr in tracks:
        if len(tr) < args.min_frames:
            continue
        lo, hi = min(tr), max(tr)
        span = [tr.get(fi) and {"box": tr[fi]["box"], "kps": tr[fi]["kps"]} or None
                for fi in range(lo, hi + 1)]
        x, _vis = track_to_tensor(span, seq_len=int(meta.get("seq_len", 16)),
                                  min_joint_conf=float(meta.get("min_joint_conf", 0.0)))
        logits = forward_numpy(weights, np.asarray([x], dtype=np.float32))
        e = np.exp(logits - logits.max())
        probs = (e / e.sum())[0]
        ci = int(probs.argmax())
        verdicts.append((tr, classes[ci], float(probs[ci])))

    # Pass 3 — draw.
    h, w = frames_bgr[0].shape[:2]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps / args.step, (w, h))
    th = max(2, round(h / 540))  # line thickness scales with resolution
    still_idxs = set()
    if args.stills and verdicts:
        tr0 = verdicts[0][0]
        ks = sorted(tr0)
        still_idxs = {ks[len(ks) // 4], ks[len(ks) // 2], ks[(3 * len(ks)) // 4]}
        Path(args.stills).mkdir(parents=True, exist_ok=True)

    for fi, fr in enumerate(frames_bgr):
        for ti, (tr, label, conf) in enumerate(verdicts):
            det = tr.get(fi)
            if det is None:
                continue
            b = det["box"]
            x1, y1, x2, y2 = (int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]))
            cv2.rectangle(fr, (x1, y1), (x2, y2), BOX, th)
            pts = [(int(k[0]), int(k[1]), k[2]) for k in det["kps"]]
            for a, bb in SKELETON:
                if a < len(pts) and bb < len(pts) \
                        and pts[a][2] >= KP_CONF and pts[bb][2] >= KP_CONF:
                    cv2.line(fr, pts[a][:2], pts[bb][:2], EDGE, th)
            for x, y, c in pts:
                if c < KP_CONF:
                    continue
                cv2.circle(fr, (x, y), th + 2,
                           JOINT_HI if c >= 0.5 else JOINT_LO, -1)
            text = f"#{ti} {label} {conf:.2f}"
            scale = max(0.6, h / 900)
            (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
            ty = max(tht + 8, y1 - 8)
            cv2.rectangle(fr, (x1 - 2, ty - tht - 6), (x1 + tw + 6, ty + 4), LABEL_BG, -1)
            cv2.putText(fr, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), th, cv2.LINE_AA)
        vw.write(fr)
        if fi in still_idxs:
            cv2.imwrite(str(Path(args.stills) / f"{out_path.stem}_f{fi:03d}.png"), fr)
    vw.release()
    n_lab = len(verdicts)
    print(f"wrote {out_path} ({len(frames_bgr)} frames, {n_lab} classified track(s): "
          + ", ".join(f"{l} {c:.2f}" for _, l, c in verdicts) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
