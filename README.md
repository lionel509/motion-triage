# motion-triage

Real-time human-motion analysis as a standalone service. Frames in, behavioral
verdict out — `detect → track → pose → behavior`. Pixels in, verdicts out.

Runs as its own process behind a thin API, so it can power a closed product
without entangling it: the caller speaks HTTP/gRPC and never links the model
code. That process boundary is deliberate — it's what keeps copyleft on this
side of the wire.

## What it does

Given a short sequence of frames from a single camera, motion-triage:

1. **Detects** people and vehicles per frame (YOLO).
2. **Tracks** them across frames (IoU tracker, tolerant to short gaps / occlusion).
3. **Estimates pose** — 17-keypoint COCO skeletons per person (RTMPose).
4. **Classifies behavior** from the skeleton trajectories — loitering, pacing,
   running, intrusion, approach, climbing, crouch, … — and returns
   `ALERT` / `DISMISS` with a reason.

It is camera-geometry aware (operator-drawn zones / lines) and perspective-aware:
motion is measured in the subject's own body-lengths, not raw pixels, so a
distant walker and a near loiterer are never confused.

## Why it exists

Every ingredient is open — YOLO, RTMPose, ST-GCN. The *assembled, deployable*
service — one that runs unattended on real outdoor CCTV and emits trustworthy
behavior verdicts — is not. Commercial platforms (Ambient.ai, Spot AI, Verkada)
keep it closed; research repos stop at benchmark numbers or frontal-webcam demos
their own authors warn don't survive the real world. This is the missing middle.

The hard part was never the models. It's the unglamorous engineering: tracking
through occlusion, pose quality at 30+ ft on a 2 MP night-mode sensor, ID
switches when two people cross, and not crying wolf at 3 a.m. Validated on real
storefront cameras at production angles, not webcam demos.

## Architecture

```
frames ─▶ detect ─▶ track ─▶ pose ─▶ behavior ─▶ { decision, reason, tracks }
          (YOLO)    (IoU)    (RTMPose)  (rules → ST-GCN)
```

The behavior layer ships in two stages, same API for both:

- **v1 — heuristics:** hand-crafted keypoint features (hip vertical velocity,
  posture-height ratio, arm-extension angle, stride regularity) feeding a rule
  engine. No training data required; ships value on day one and stays the
  sanity baseline forever.
- **v2 — learned:** a small temporal model (ST-GCN++) trained on logged keypoint
  sequences from real deployments. If it can't beat the heuristics, it isn't
  trained yet.

## The contract

[`docs/API.md`](docs/API.md) is the only thing callers depend on. Detector,
pose model, language, and behavior model can all change behind it without any
caller noticing. A gRPC mirror lives in [`proto/motion.proto`](proto/motion.proto).

## Status

🚧 Early — contract first. Implementation lands in phases:

- [x] Service contract (`docs/API.md`, `proto/motion.proto`)
- [x] detect → track → behavior service core (`service/` — seeded from the
      production triage service; CoreML on macOS, CPU EP elsewhere, so it
      builds and runs cross-platform)
- [x] pose inference in Rust (`service/src/pose.rs` — both YOLO-pose head
      layouts) + skeleton/box rendering (`service/src/draw.rs`) +
      `spike pose` CLI (annotated frames + per-joint confidence readout)
- [x] behavior-NN inference in Rust (`service/src/behavior.rs` — ONNX via
      ort; featurization parity-tested against the Python trainer) +
      `spike behavior` CLI
- [x] training pipeline (`training/train.py` → ONNX + sidecars) and
      auto-retraining with a promotion gate (`training/autotrain.py`)
- [x] heuristic behavior features (v1) — `training/features.py` (incl.
      head-direction from face keypoints)
- [x] corpus loader / tensorization — `training/dataset.py`
- [x] keypoints attached to `/triage` API responses (`POSE_MODEL` env;
      per-frame pose, center-proximity association to person tracks —
      additive field, absent when pose is off)
- [x] global-motion feature channels (C=6) — fixes walking-vs-standing
      under box-relative translation invariance (synthetic walking F1
      0.55 → 1.00)
- [x] skeleton-GIF labeling tool (`training/render_track.py`) +
      [data playbook](docs/DATA_PLAYBOOK.md)
- [x] clips → corpus direct path (`training/clips_to_corpus.py` — any
      folder of short videos → pose-track corpus, no dashboard needed) +
      heuristic walking/standing prelabeler (`training/prelabel.py`)
- [x] portable inference (`training/infer_mlx.py`) — the trained model
      runs via Rust/ort (ONNX; CPU or CoreML), pure **numpy** (anywhere),
      or **MLX** (Apple-silicon GPU), all from the same weight sidecars;
      backends cross-verified per-sample
- [x] behavior NN v0 trained on real storefront tracks (walking vs
      standing; displacement-bootstrap labels — see the playbook for why
      human-reviewed labels are still the point)
- [ ] behavior NN trained on REAL labeled corpus (current model is
      synthetic-smoke only; follow `docs/DATA_PLAYBOOK.md`)
- [ ] BEHAVIOR_MODEL wired into live verdict routing (shadow-eval first)

## License

AGPL-3.0. The detection layer wraps Ultralytics YOLO (AGPL), so the whole
service inherits it. Callers that talk to it across a process boundary
(HTTP / gRPC) are **not** derivative works and carry no AGPL obligation — that
boundary is the entire point of shipping this as a service. Swap the detector
for a permissive one (RTMDet / D-FINE / RF-DETR) and this can be relicensed
Apache-2.0. Full text in [LICENSE](LICENSE).
