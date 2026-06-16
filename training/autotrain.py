"""Auto-retraining loop: retrain when the corpus grows, promote only if better.

Stdlib-only orchestrator around ``train.py``::

    # one shot (cron / launchd / Task Scheduler owns the cadence):
    python training/autotrain.py --corpus corpus.jsonl --labels-csv labels.csv
    # or self-looping:
    python training/autotrain.py --corpus corpus.jsonl --labels-csv labels.csv --every 86400

Each round:
  1. Count corpus lines; skip unless ≥ ``--min-new-lines`` new since the
     last round (no point retraining on the same data).
  2. Run ``train.py`` to a temp output.
  3. PROMOTION GATE: the candidate replaces the champion only when its
     val macro-F1 beats the best ever recorded AND the val set is big
     enough to mean anything (``--min-val``). A model that wins on 8
     validation samples didn't win anything.
  4. On promote: copy to ``behavior_v{N}.onnx`` (+ sidecars) and refresh
     ``behavior_latest.onnx`` — the stable path the Rust service points at.

State lives in ``<models-dir>/autotrain_state.json``. Delete it to reset
the champion (e.g. after a corpus schema change).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

TRAIN = Path(__file__).resolve().parent / "train.py"


def _count_lines(paths: list[str]) -> int:
    n = 0
    for p in paths:
        try:
            with open(p, "rb") as f:
                n += sum(1 for _ in f)
        except OSError:
            pass
    return n


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 — missing/corrupt state = fresh start
        return {"corpus_lines": 0, "best_macro_f1": 0.0, "version": 0}


def run_round(args) -> bool:
    """One retrain round. Returns True when a new champion was promoted."""
    models = Path(args.models_dir)
    models.mkdir(parents=True, exist_ok=True)
    state_path = models / "autotrain_state.json"
    state = _load_state(state_path)

    lines = _count_lines(args.corpus)
    new = lines - int(state.get("corpus_lines", 0))
    if new < args.min_new_lines and not args.force:
        print(f"[autotrain] {lines} corpus lines (+{new} new) < "
              f"--min-new-lines {args.min_new_lines} — skipping (use --force)")
        return False

    candidate = models / "behavior_candidate.onnx"
    cmd = [sys.executable, str(TRAIN),
           "--out", str(candidate),
           "--epochs", str(args.epochs),
           "--seq-len", str(args.seq_len)]
    for c in args.corpus:
        cmd += ["--corpus", c]
    if args.labels_csv:
        cmd += ["--labels-csv", args.labels_csv]
    if args.weak_from_task_category:
        cmd += ["--weak-from-task-category"]
    if args.augment:
        cmd += ["--augment"]
    print(f"[autotrain] training: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"[autotrain] train.py failed (exit {proc.returncode}) — champion unchanged")
        return False

    metrics = json.loads(Path(f"{candidate}.metrics.json").read_text())
    f1, n_val = float(metrics["macro_f1"]), int(metrics["n_val"])
    best = float(state.get("best_macro_f1", 0.0))
    print(f"[autotrain] candidate macro-F1 {f1:.4f} (val n={n_val}) vs champion {best:.4f}")

    if n_val < args.min_val:
        print(f"[autotrain] HELD: val set too small (n={n_val} < {args.min_val}) "
              "— a win on a tiny val set is noise, not progress")
    elif f1 <= best:
        print("[autotrain] HELD: candidate does not beat the champion")
    else:
        version = int(state.get("version", 0)) + 1
        for suffix in ("", ".classes.json", ".metrics.json"):
            shutil.copy2(f"{candidate}{suffix}",
                         models / f"behavior_v{version}.onnx{suffix}")
            shutil.copy2(f"{candidate}{suffix}",
                         models / f"behavior_latest.onnx{suffix}")
        state.update({"best_macro_f1": f1, "version": version})
        print(f"[autotrain] PROMOTED -> behavior_v{version}.onnx (+ behavior_latest.onnx)")

    state["corpus_lines"] = lines
    state_path.write_text(json.dumps(state, indent=2))
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", action="append", required=True)
    p.add_argument("--labels-csv", default=None)
    p.add_argument("--weak-from-task-category", action="store_true")
    p.add_argument("--augment", action="store_true",
                   help="forward --augment to train.py (flip + rotation + "
                        "conf-dropout skeleton augmentation)")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--min-new-lines", type=int, default=50)
    p.add_argument("--min-val", type=int, default=20,
                   help="promotion requires at least this many val samples")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--every", type=int, default=0,
                   help="loop every N seconds (0 = run once and exit)")
    p.add_argument("--force", action="store_true",
                   help="retrain even without enough new corpus lines")
    args = p.parse_args()

    if args.every <= 0:
        run_round(args)
        return 0
    while True:
        try:
            run_round(args)
        except Exception as e:  # noqa: BLE001 — the loop must outlive one bad round
            print(f"[autotrain] round failed: {type(e).__name__}: {e}")
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
