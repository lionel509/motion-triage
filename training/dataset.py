"""Corpus JSONL → training tensors.

Loads the pseudonymized track corpus emitted by the deployment's
``export_pose_dataset.py`` and produces fixed-shape sequences for a
temporal model (ST-GCN++ via PYSKL, or anything that eats
``[T, V=17, C=4]``).

Channels per joint: ``(x, y, conf, v, gdx, gdy)`` where
  * ``x, y`` are **box-relative** — translated to the box top-left and
    scaled by box height, so a distant person and a near one produce the
    same numbers (the network must learn behavior, not camera distance);
  * ``conf`` is the per-joint confidence, passed THROUGH to the model —
    don't silently zero noisy joints, let the model learn to discount
    them (and mask explicitly below ``min_joint_conf``);
  * ``v`` is per-joint velocity (frame-to-frame delta of box-relative
    position), 0 at gaps and t=0;
  * ``gdx, gdy`` are GLOBAL motion — the box center's frame-to-frame
    delta in box-height units, replicated across joints. Box-relative
    x/y are translation-invariant by design, which erases walk-across-
    frame travel entirely (walking vs standing then rests on gait
    alone); these two channels put global travel back in, scale-
    normalized so near and far walkers read the same. 0 at gaps/t0.

Gap frames (person not detected) are zero-filled with conf=0 — an
explicit "not seen" the model can learn, never interpolated motion.

Stdlib-only on purpose (no numpy/torch import at module level): the
loader runs anywhere, and the tensor conversion is a one-liner at the
call site (``np.asarray(sample.x)``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

V = 17  # COCO joints
C = 6   # x, y, conf, velocity, global-dx, global-dy


@dataclass
class Sample:
    """One track, model-ready."""

    sample_id: str
    x: list[list[list[float]]]          # [T][V][C]
    labels: dict[str, Any] = field(default_factory=dict)
    hour_utc: int | None = None
    dow_utc: int | None = None
    frame_source: str = "clip"
    visibility: float = 0.0             # fraction of frames present


def _box_relative(kp: list[float], box: dict[str, Any]) -> tuple[float, float, float]:
    """(x, y, conf) with x/y translated to box top-left and scaled by box
    height. Height (not width) is the scale: it's the stable body axis."""
    h = max(1.0, float(box["y2"]) - float(box["y1"]))
    return (
        (float(kp[0]) - float(box["x1"])) / h,
        (float(kp[1]) - float(box["y1"])) / h,
        float(kp[2]),
    )


def track_to_tensor(
    frames: list[dict[str, Any] | None],
    *,
    seq_len: int = 16,
    min_joint_conf: float = 0.0,
) -> tuple[list[list[list[float]]], float]:
    """One corpus ``frames`` list → ``[seq_len][V][C]`` + visibility.

    Shorter tracks are right-padded with gap frames; longer ones are
    uniformly subsampled. ``min_joint_conf`` zero-masks joints below the
    threshold (set from the deployment's night findings; 0.0 = keep all
    and let the conf channel carry the information).
    """
    # Uniform subsample / pad to seq_len.
    if len(frames) > seq_len:
        idxs = [round(i * (len(frames) - 1) / (seq_len - 1)) for i in range(seq_len)]
        frames = [frames[i] for i in idxs]
    elif len(frames) < seq_len:
        frames = list(frames) + [None] * (seq_len - len(frames))

    out: list[list[list[float]]] = []
    prev: list[tuple[float, float] | None] = [None] * V
    prev_center: tuple[float, float] | None = None
    present = 0
    for f in frames:
        row: list[list[float]] = []
        if f is None:
            row = [[0.0] * C for _ in range(V)]
            prev = [None] * V
            prev_center = None
        else:
            present += 1
            kps = f.get("kps") or f.get("keypoints") or []
            box = f["box"]
            h = max(1.0, float(box["y2"]) - float(box["y1"]))
            cx = (float(box["x1"]) + float(box["x2"])) / 2.0
            cy = (float(box["y1"]) + float(box["y2"])) / 2.0
            # Global motion: box-center travel in box-heights (per-person
            # info, replicated across joints; the conv mixes channels anyway).
            if prev_center is None:
                gdx, gdy = 0.0, 0.0
            else:
                gdx = (cx - prev_center[0]) / h
                gdy = (cy - prev_center[1]) / h
            prev_center = (cx, cy)
            for j in range(V):
                if j < len(kps):
                    x, y, c = _box_relative(kps[j], box)
                    if c < min_joint_conf:
                        # Joint invisible — zero its local channels but KEEP
                        # the global motion: the person's travel is known
                        # even when one limb isn't.
                        row.append([0.0, 0.0, 0.0, 0.0, gdx, gdy])
                        prev[j] = None
                        continue
                    p = prev[j]
                    v = 0.0 if p is None else ((x - p[0]) ** 2 + (y - p[1]) ** 2) ** 0.5
                    row.append([x, y, c, v, gdx, gdy])
                    prev[j] = (x, y)
                else:
                    row.append([0.0, 0.0, 0.0, 0.0, gdx, gdy])
                    prev[j] = None
        out.append(row)
    return out, present / max(1, len(frames))


def load_corpus(
    path: str | Path,
    *,
    seq_len: int = 16,
    min_joint_conf: float = 0.0,
    min_visibility: float = 0.25,
    labeled_only: bool = False,
) -> Iterator[Sample]:
    """Stream Samples from an export JSONL.

    ``min_visibility`` drops tracks the person was barely present in —
    a 2-of-16-frames blip carries more noise than signal. Raw export
    lines (``--raw``, ``frames_raw``) are skipped: tensorization needs
    associated tracks.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{line_no}: invalid JSON — {e}") from e
            frames = rec.get("frames")
            if frames is None:
                continue  # --raw line; not tensorizable
            labels = rec.get("labels") or {}
            if labeled_only and not any(v is not None for v in labels.values()):
                continue
            x, vis = track_to_tensor(
                frames, seq_len=seq_len, min_joint_conf=min_joint_conf,
            )
            if vis < min_visibility:
                continue
            yield Sample(
                sample_id=str(rec.get("sample_id") or rec.get("event_id") or line_no),
                x=x,
                labels=labels,
                hour_utc=rec.get("hour_utc"),
                dow_utc=rec.get("dow_utc"),
                frame_source=rec.get("frame_source") or "clip",
                visibility=vis,
            )


def is_night(sample: Sample, *, dawn_utc: int = 10, dusk_utc: int = 0) -> bool | None:
    """Crude UTC night split for slice-wise eval (None = unknown hour).

    Defaults assume US-Eastern sites (≈ dawn 06:00 ET = 10 UTC, dusk
    20:00 ET = 0 UTC); pass site-appropriate bounds rather than trusting
    these. The corpus deliberately carries UTC — do the honest local
    conversion downstream where the site's timezone is known.
    """
    if sample.hour_utc is None:
        return None
    h = sample.hour_utc
    if dawn_utc <= dusk_utc:
        return not (dawn_utc <= h < dusk_utc)
    return dusk_utc <= h < dawn_utc
