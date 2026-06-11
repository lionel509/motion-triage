"""Heuristic keypoint features — the v1 behavior reads and the NN's baseline.

Pure functions over one track's keypoint sequence (the ``frames`` list of a
corpus JSONL line). Designed for noisy outdoor CCTV skeletons:

* every feature masks low-confidence joints instead of trusting them — an
  occluded wrist is HALLUCINATED more often than missing (phantom-reach
  guard), and night/IR keypoints jitter;
* everything is normalized to the subject's own box height, so a distant
  walker and a near one read the same;
* gap frames (``None``) are skipped, not interpolated — interpolation
  invents motion.

No third-party deps (stdlib math only) so the module runs anywhere the
corpus lands. If the trained temporal model can't beat these features, it
isn't trained yet.

COCO-17 joint order (yolo*-pose / RTMPose): 0 nose, 1-2 eyes, 3-4 ears,
5-6 shoulders, 7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees,
15-16 ankles.
"""
from __future__ import annotations

import math
from typing import Any

KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_ANKLE, KP_R_ANKLE = 15, 16

# Below this, a joint is treated as ABSENT. Tuned permissive for night/IR;
# revisit against docs/NIGHT_POSE_FINDINGS.md once the night spike runs.
MIN_JOINT_CONF = 0.30


Frame = dict[str, Any] | None  # one corpus track frame ({"box","conf","kps"} | None)


def _joint(frame: Frame, idx: int) -> tuple[float, float] | None:
    """(x, y) of one joint, or None when the frame is a gap / the joint
    is low-confidence."""
    if frame is None:
        return None
    kps = frame.get("kps") or frame.get("keypoints")
    if not kps or idx >= len(kps):
        return None
    x, y, c = kps[idx]
    if c < MIN_JOINT_CONF:
        return None
    return (float(x), float(y))


def _mid(frame: Frame, i: int, j: int) -> tuple[float, float] | None:
    a, b = _joint(frame, i), _joint(frame, j)
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _box_h(frame: Frame) -> float | None:
    if frame is None:
        return None
    box = frame.get("box")
    if not box:
        return None
    h = float(box["y2"]) - float(box["y1"])
    return h if h > 1.0 else None


def hip_vertical_velocity(frames: list[Frame]) -> float | None:
    """Mean signed vertical hip motion per present-frame step, in
    box-heights (+ = moving DOWN the image: drop/crouch; − = rising:
    climb). The single strongest climb/fall cue. None when hips were
    confidently visible in <2 frames."""
    pts: list[tuple[int, tuple[float, float], float]] = []
    for i, f in enumerate(frames):
        hip = _mid(f, KP_L_HIP, KP_R_HIP)
        h = _box_h(f)
        if hip is not None and h is not None:
            pts.append((i, hip, h))
    if len(pts) < 2:
        return None
    deltas = []
    for (i0, p0, h0), (i1, p1, h1) in zip(pts, pts[1:]):
        steps = max(1, i1 - i0)
        href = (h0 + h1) / 2.0
        deltas.append((p1[1] - p0[1]) / href / steps)
    return sum(deltas) / len(deltas)


def posture_height_ratio(frames: list[Frame]) -> float | None:
    """Median (nose→ankle vertical extent) / box height. Standing ≈ 0.85+,
    crouch/crawl ≪ that. Distance-robust (both terms scale together).
    None when nose+ankles never co-visible."""
    ratios: list[float] = []
    for f in frames:
        nose = _joint(f, KP_NOSE)
        ankle = _mid(f, KP_L_ANKLE, KP_R_ANKLE)
        h = _box_h(f)
        if nose is None or ankle is None or h is None:
            continue
        ratios.append(abs(ankle[1] - nose[1]) / h)
    if not ratios:
        return None
    ratios.sort()
    return ratios[len(ratios) // 2]


def max_arm_extension(frames: list[Frame]) -> float | None:
    """Max (wrist distance from shoulder midpoint) / box height across the
    track, both arms. Reaching toward a lock/door/camera reads ≳ 0.45;
    arms-at-sides ≈ 0.25. Confidence-masked per joint — a phantom wrist
    on an occluded arm never fires this."""
    best: float | None = None
    for f in frames:
        sho = _mid(f, KP_L_SHOULDER, KP_R_SHOULDER)
        h = _box_h(f)
        if sho is None or h is None:
            continue
        for w_idx in (KP_L_WRIST, KP_R_WRIST):
            w = _joint(f, w_idx)
            if w is None:
                continue
            d = math.hypot(w[0] - sho[0], w[1] - sho[1]) / h
            if best is None or d > best:
                best = d
    return best


def stride_periodicity(frames: list[Frame]) -> float | None:
    """Zero-crossing rate of the ankle-separation signal (normalized to
    box height): real gait oscillates (≈0.2–0.6 crossings/frame at
    1.6 fps clip sampling); standing-with-jitter ≈ 0. Separates a real
    walker from a stationary jitterer when the box barely moves.
    None when ankles co-visible in <4 frames."""
    sep: list[float] = []
    for f in frames:
        la, ra = _joint(f, KP_L_ANKLE), _joint(f, KP_R_ANKLE)
        h = _box_h(f)
        if la is None or ra is None or h is None:
            continue
        sep.append((la[0] - ra[0]) / h)
    if len(sep) < 4:
        return None
    mean = sum(sep) / len(sep)
    centered = [s - mean for s in sep]
    crossings = sum(
        1 for a, b in zip(centered, centered[1:])
        if (a > 0) != (b > 0) and (abs(a) > 0.02 or abs(b) > 0.02)
    )
    return crossings / (len(centered) - 1)


def head_direction(frames: list[Frame]) -> dict[str, Any]:
    """Coarse head direction from the face keypoints — no extra model.

    ``head_yaw``: mean of (nose.x − shoulder-midpoint.x) / shoulder-width
    over frames where nose + both shoulders are confidently visible.
    Sign is SCREEN-relative: negative = subject's head turned toward
    screen-left, positive = screen-right, ≈0 = facing the camera.

    ``head_facing``: "camera" | "left" | "right" | "away" | None.
    "away" is inferred when the shoulders are repeatedly visible but the
    nose almost never is — you don't see a nose on the back of a head.
    (Ear-only asymmetry could refine profile views; deliberately not
    over-claimed here — yaw from 2D keypoints is coarse by nature.)
    """
    yaws: list[float] = []
    shoulders_seen = 0
    nose_seen = 0
    for f in frames:
        ls, rs = _joint(f, KP_L_SHOULDER), _joint(f, KP_R_SHOULDER)
        if ls is None or rs is None:
            continue
        shoulders_seen += 1
        nose = _joint(f, KP_NOSE)
        if nose is None:
            continue
        nose_seen += 1
        width = abs(ls[0] - rs[0])
        if width < 2.0:  # edge-on shoulders — yaw from x-offset is meaningless
            continue
        mid_x = (ls[0] + rs[0]) / 2.0
        yaws.append((nose[0] - mid_x) / width)
    if shoulders_seen == 0:
        return {"head_yaw": None, "head_facing": None}
    if nose_seen / shoulders_seen < 0.3:
        return {"head_yaw": None, "head_facing": "away"}
    if not yaws:
        return {"head_yaw": None, "head_facing": None}
    yaw = sum(yaws) / len(yaws)
    facing = "camera" if abs(yaw) < 0.25 else ("left" if yaw < 0 else "right")
    return {"head_yaw": yaw, "head_facing": facing}


def visibility(frames: list[Frame]) -> float:
    """Fraction of frames where the person was detected at all — a
    confidence weight for every other feature (a track seen 3/16 frames
    should sway decisions less than one seen 16/16)."""
    if not frames:
        return 0.0
    return sum(1 for f in frames if f is not None) / len(frames)


def extract_features(frames: list[Frame]) -> dict[str, float | None]:
    """All v1 features for one track. None = not computable (joints never
    confidently visible) — callers must treat None as ABSTAIN, never 0:
    'we couldn't see the arms' is not 'the arms were at rest'."""
    out: dict[str, float | None] = {
        "hip_vertical_velocity": hip_vertical_velocity(frames),
        "posture_height_ratio": posture_height_ratio(frames),
        "max_arm_extension": max_arm_extension(frames),
        "stride_periodicity": stride_periodicity(frames),
        "visibility": visibility(frames),
    }
    out.update(head_direction(frames))  # head_yaw + head_facing
    return out
