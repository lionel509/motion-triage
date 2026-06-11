# training/ — behavior model pipeline (Python, offline)

This directory is the **offline** half of motion-triage: it consumes the
pseudonymized keypoint corpus exported by the deployment's
`export_pose_dataset.py` and produces the temporal behavior model that the
service will eventually run via ONNX. Nothing here runs in the event path.

**No Ultralytics imports in this directory** — it consumes exported keypoint
JSONL only, so it stays free of AGPL code regardless of which detector
produced the skeletons.

## Pipeline

```
corpus JSONL ─▶ dataset.py ─▶ [T × 17 × C] tensors ─▶ (train) ─▶ ONNX ─▶ service
                    │
                    └▶ features.py — heuristic keypoint features (v1 behavior
                       reads + the NN's forever-baseline)
```

## Corpus line shape (from export_pose_dataset.py)

One JSON line per person track:

```json
{"sample_id": "…", "track_index": 0, "frame_source": "clip",
 "frame_count": 16, "frame_w": 960, "frame_h": 540,
 "hour_utc": 3, "dow_utc": 6,
 "frames": [{"i": 0, "box": {"x1":…}, "conf": 0.81,
             "kps": [[x, y, c], "… 17 joints"]} , null, "…"],
 "labels": {"disposition": "false_alarm",
            "operator_disposition_correct": false,
            "task_category": "loitering_suspicious",
            "llm_verdict": "suspicious", "operator_correction": null}}
```

`null` frames are gaps (occlusion / person out of frame). Per-joint
confidences are kept; **mask, don't trust, low-confidence joints** — an
occluded wrist is hallucinated more often than missing.

## v1 — heuristics (`features.py`, no training data needed)

Hand-crafted features computed per track, designed to tolerate jittery
night/IR keypoints:

- hip vertical velocity (climb / fall signal)
- posture-height ratio (crouch vs stand, robust to distance)
- arm-extension (reach toward a lock/door)
- stride periodicity (real gait vs stationary jitter)

These ship value with zero training data and remain the sanity baseline
forever: **if the trained model can't beat the heuristics, it isn't trained
yet.**

## v2 — temporal model (when the corpus is ready)

- **When:** a few hundred labeled sequences per behavior class. The store
  cameras accumulate "normal" passively; stage the rare classes after hours
  (lingering, reaching, creeping, climbing) under the real cameras — that's
  how NTU was built too, and staged-by-you skeletons generalize because the
  model never sees pixels.
- **What:** ST-GCN++ via [PYSKL](https://github.com/kennymckormick/pyskl)
  (Apache-2.0) as the baseline architecture. Input `[T=16, V=17, C=4]`
  (x, y, conf, per-joint velocity), normalized box-relative (`dataset.py`).
- **Framing:** lean anomaly-over-normal rather than closed-set threat
  classification — real intrusions are rare, the long tail is unbounded, and
  "this isn't normal for this hour" routes to escalation without needing to
  name the threat. Condition normalcy on hour/day (else "anomalous" just
  means "it's 3am").
- **Discipline:** train → run in **shadow mode** against the heuristics →
  review disagreements → only then let it gate anything. Keep a frozen
  golden eval set; never let the feedback loop touch it.
- **Night:** evaluate the night slice separately (the corpus carries
  `hour_utc`). Night is the highest-value, lowest-quality-keypoints regime —
  an aggregate metric will lie about it.
- **Deployment:** export to ONNX; the service runs it next to the detector.

## Files

- `dataset.py` — corpus JSONL → normalized tensors + label mapping; gap and
  confidence masking; day/night slicing helpers.
- `features.py` — the heuristic keypoint features (v1).
