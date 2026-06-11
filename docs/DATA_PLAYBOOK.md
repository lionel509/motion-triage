# Data playbook — from cameras to a trained behavior model

The pipeline is built and proven end-to-end (synthetic). What remains is
**data**, and data is an operations task, not an engineering one. This is
the runbook.

Two ways in, identical from step 4 onward:

- **Path A — dashboard deployment** (steps 1–3): the event pipeline logs
  skeletons passively; export from the database.
- **Path B — a folder of clips** (step 3b): no dashboard, no database.
  Pull short clips from anywhere (an NVR, a phone, staged sessions) and
  run pose extraction directly. This is also the fastest bootstrap for a
  deployment that hasn't enabled shadow logging yet.

## Path B — straight from clips (no dashboard required)

```bash
# clips named {anything}__{YYYYMMDDTHHMMSSZ}.mp4 get correct night-slice
# hours; any other name falls back to file mtime. Pseudonymize the
# {anything} part at pull time — this repo never needs to know the source.
python training/clips_to_corpus.py --clips data/clips/batch1 \
    --out data/corpus.jsonl --device cpu        # mps/cuda if you have it

# eyeball the motion distribution, then auto-prelabel the easy classes
# (walking/standing, conservative bands, ambiguous left BLANK):
python training/prelabel.py --corpus data/corpus.jsonl --stats
python training/prelabel.py --corpus data/corpus.jsonl --out data/labels.csv
```

Then continue at step 4 (render GIFs, review the prelabels, fill the
blanks) and step 5 (train). Pre-labels are a bootstrap, not ground truth:
a model trained ONLY on them has learned a displacement threshold — the
human-reviewed blanks and corrections are what make it a behavior model.

## 1. Turn on corpus collection (Mac, ~10 minutes)

```bash
cd services/python/server && alembic upgrade head     # applies 0066 (event_pose_frames)
# in the server env:
POSE_LOGGING_ENABLED=true POSE_LOGGING_DEVICE=mps     # shadow lane — zero effect on verdicts
```

From that moment every gated person-event logs 17-joint skeletons. Leave it
on; it accumulates passively.

## 2. Night spike (same evening)

Replay a handful of night + day events through
`dashboard.workers.pose_logger.process_event(<event_id>)`, inspect rows,
and write `docs/NIGHT_POSE_FINDINGS.md`: the box-size floor where night
keypoints stay usable, and per-joint conf behavior on IR. This number
calibrates export filters and the future inference gate. **Do this before
trusting any night skeletons.**

## 3. Staged sessions (the rare classes — store, after hours)

Natural footage gives you thousands of "walking" and near-zero of
everything you actually alert on. Stage the rest under the REAL cameras —
real angles, real IR, real compression (that's the whole advantage; NTU
was acted too). The model never sees pixels, only skeletons, so acted
sequences generalize.

Shot list per behavior — **15–20 reps × 2–3 cameras × day AND night**:

| class            | direction notes |
|------------------|---------------|
| walking          | through frame, varied paths/speeds (mostly free from natural footage) |
| running          | both directions, including diagonal toward/away from camera |
| standing         | phone-checking, waiting — include subtle weight shifts |
| loitering/pacing | 30–60 s dwell, back-and-forth, perimeter hugging |
| crouching        | tying-shoe vs hiding-low — label them the SAME class until the model earns finer splits |
| reaching         | toward door handles / locks / shelves, both arms |
| climbing         | fence/gate mock — SAFETY FIRST, a step ladder sells the skeleton fine |

Mark session start/end times; you'll filter the export to those windows.

## 4. Export → render → label (an evening, not a project)

```bash
# on the box with the DB:
.venv/bin/python server/scripts/export_pose_dataset.py \
    --from 2026-06-12T00:00:00 --labeled-only --out corpus.jsonl
# (staged sessions: drop --labeled-only and filter by --from/--to windows)

# anywhere (GIFs are skeleton-only — privacy-clean to move around):
python training/render_track.py --corpus corpus.jsonl \
    --out-dir gifs/ --labels-template gifs/labels.csv
```

Watch the GIFs, fill the second CSV column. Rules of thumb: label what the
BODY does, not what you infer ("crouching", not "hiding"); skip ambiguous
tracks (blank = skipped — silence beats noise labels); aim for ≥200/class
before expecting the NN to beat the heuristics.

## 5. Train / auto-retrain

```bash
python training/train.py --corpus corpus.jsonl --labels-csv gifs/labels.csv \
    --out models/behavior.onnx
# or let the loop own it (cron/launchd cadence):
python training/autotrain.py --corpus corpus.jsonl --labels-csv gifs/labels.csv \
    --models-dir models/
```

The promotion gate only replaces the champion when macro-F1 improves on a
≥20-sample validation split. Watch the **night macro-F1** line separately —
the aggregate will lie about the regime that matters.

## 6. Deploy to the service

```bash
BEHAVIOR_MODEL=models/behavior_latest.onnx   # (wiring lands with the shadow-eval step)
POSE_MODEL=models/yolo11n-pose.onnx ./supervise.sh
```

The service runs the ONNX via ort (CoreML on macOS, CPU elsewhere). For
ad-hoc runs without the service — a notebook, a Linux box, the Mac GPU —
`training/infer_mlx.py` executes the same weights from the `.weights.npz`
sidecar: `--backend numpy` anywhere, `--backend mlx` on Apple silicon
(`pip install mlx`; add `--verify 32` on first run — it hard-fails if MLX
disagrees with the numpy reference).

Run the NN in **shadow** against the rule engine first; review
disagreements; only then let it route. If it can't beat
`training/features.py`'s heuristics, it isn't trained yet — collect more.

## Standing rules

- Night data is logged, marked, never filtered at capture.
- The corpus carries `event_id`-free pseudonymized samples once exported
  (`--salt`); raw exports never leave the box.
- Skeleton data never joins identity stores (faces/Re-ID) — that line is
  the difference between action classification and gait recognition.
