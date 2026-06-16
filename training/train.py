"""Train the behavior NN on the exported keypoint corpus → ONNX.

The one declared-Python piece of the runtime stack: PyTorch trains here,
the Rust service runs the exported ONNX (`service/src/behavior.rs`).

Usage
-----
::

    # staged/hand labels (the real path):
    python training/train.py --corpus corpus.jsonl --labels-csv labels.csv \
        --out models/behavior.onnx

    # weak-label bootstrap from the pipeline's task_category (walking/
    # loitering only — see WEAK_TASK_CATEGORY_MAP):
    python training/train.py --corpus corpus.jsonl --weak-from-task-category \
        --out models/behavior.onnx

    # smoke the full loop on synthetic data (no cameras needed):
    python training/gen_synth_corpus.py --out /tmp/synth
    python training/train.py --corpus /tmp/synth/corpus.jsonl \
        --labels-csv /tmp/synth/labels.csv --epochs 15 --out /tmp/synth/behavior.onnx

Outputs, next to ``--out``:
  * ``behavior.onnx``               — input ``x`` `[B,T,17,C]` (C=6), output ``logits`` `[B,n_classes]`
  * ``behavior.onnx.classes.json``  — ``{"classes", "seq_len", "min_joint_conf"}``
                                       (the Rust side refuses to guess these)
  * ``behavior.onnx.metrics.json``  — per-class P/R/F1, macro-F1, night slice

Labels
------
1. ``--labels-csv sample_id,label`` — authoritative (staged after-hours
   clips, hand review). This is the real path; behaviors like crouching
   and climbing basically never get clean weak labels.
2. ``--weak-from-task-category`` — bootstrap walking/loitering from the
   pipeline's existing task_category. Unmapped categories are SKIPPED,
   not bucketed into "other": silence beats noise labels.

Discipline (from training/README.md): deterministic split by sample_id
hash; class-weighted CE for the inevitable imbalance; night-slice metric
reported separately because the aggregate will lie about the regime that
matters; if this model can't beat features.py's heuristics, it isn't
trained yet.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import Sample, is_night, load_corpus  # noqa: E402

# Conservative weak-label map: pipeline task_category → behavior class.
# ONLY categories that genuinely imply the motion; everything else skips.
WEAK_TASK_CATEGORY_MAP: dict[str, str] = {
    "person_walking_passing": "walking",
    "person_check_in": "walking",
    "loitering_benign": "loitering",
    "loitering_suspicious": "loitering",
    "prolonged_loitering": "loitering",
}


def _is_val(sample_id: str) -> bool:
    """Deterministic ~20% validation split (stable across retrains so the
    val set never leaks into train between autotrain rounds)."""
    h = hashlib.sha256(sample_id.encode()).digest()
    return h[0] % 5 == 0


def _resolve_labels(
    samples: list[Sample], labels_csv: str | None, weak: bool,
) -> list[tuple[Sample, str]]:
    csv_map: dict[str, str] = {}
    if labels_csv:
        with open(labels_csv, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0].strip() and not row[0].startswith("#"):
                    csv_map[row[0].strip()] = row[1].strip()
    out: list[tuple[Sample, str]] = []
    for s in samples:
        label = csv_map.get(s.sample_id)
        if label is None and weak:
            cat = (s.labels or {}).get("task_category")
            label = WEAK_TASK_CATEGORY_MAP.get(cat or "")
        if label:
            out.append((s, label))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", action="append", required=True,
                   help="export JSONL (repeatable)")
    p.add_argument("--labels-csv", default=None,
                   help="CSV sample_id,label — authoritative labels")
    p.add_argument("--weak-from-task-category", action="store_true",
                   help="bootstrap walking/loitering from task_category")
    p.add_argument("--classes", default=None,
                   help="CSV class order (default: sorted labels found)")
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--min-joint-conf", type=float, default=0.0)
    p.add_argument("--min-visibility", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cpu/cuda/mps (default auto)")
    p.add_argument("--out", default="models/behavior.onnx")
    p.add_argument("--augment", action="store_true",
                   help="train-time skeleton augmentation: rotation (camera "
                        "angles) + per-joint confidence dropout (night/IR)")
    p.add_argument("--aug-rot-deg", type=float, default=15.0,
                   help="max random in-plane rotation, degrees")
    p.add_argument("--aug-conf-drop", type=float, default=0.1,
                   help="per-joint dropout prob (simulates low visibility)")
    p.add_argument("--aug-flip-prob", type=float, default=0.5,
                   help="train-time horizontal flip probability — mirror L/R "
                        "joints (the most natural, label-preserving skeleton "
                        "aug; 0 disables). Active only with --augment.")
    p.add_argument("--patience", type=int, default=0,
                   help="early-stop after N epochs with no val macro-F1 gain "
                        "(0 = run all epochs; the best checkpoint is exported "
                        "either way)")
    p.add_argument("--class-weight", default="sqrt_inv",
                   choices=["sqrt_inv", "inv", "effective", "none"],
                   help="class-imbalance weighting. sqrt_inv (default) tames "
                        "the blow-up raw inverse-freq ('inv') causes on rare "
                        "classes; 'effective' = Cui et al. 2019; 'none' = off")
    p.add_argument("--class-weight-cap", type=float, default=10.0,
                   help="clamp normalized class weights to [1/cap, cap]")
    p.add_argument("--label-smoothing", type=float, default=0.05,
                   help="CE label smoothing — calibrates confidence for the "
                        "fail-open threshold (0 = off)")
    args = p.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("error: PyTorch required for training — pip install torch", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available() else "cpu"
    )

    # ---- Load + label -----------------------------------------------------
    samples: list[Sample] = []
    for path in args.corpus:
        samples.extend(load_corpus(
            path, seq_len=args.seq_len,
            min_joint_conf=args.min_joint_conf,
            min_visibility=args.min_visibility,
        ))
    labeled = _resolve_labels(samples, args.labels_csv, args.weak_from_task_category)
    if not labeled:
        print("error: no labeled samples (provide --labels-csv or "
              "--weak-from-task-category over a corpus with task_category)",
              file=sys.stderr)
        return 2

    classes = ([c.strip() for c in args.classes.split(",") if c.strip()]
               if args.classes else sorted({lb for _, lb in labeled}))
    cls_idx = {c: i for i, c in enumerate(classes)}
    labeled = [(s, lb) for s, lb in labeled if lb in cls_idx]

    train_set = [(s, lb) for s, lb in labeled if not _is_val(s.sample_id)]
    val_set = [(s, lb) for s, lb in labeled if _is_val(s.sample_id)]
    counts = Counter(lb for _, lb in labeled)
    print(f"samples: {len(labeled)} labeled "
          f"({len(train_set)} train / {len(val_set)} val) · device {device}")
    for c in classes:
        print(f"  {c:<12} {counts.get(c, 0)}")
    if not train_set or not val_set:
        print("error: empty train or val split — need more labeled data",
              file=sys.stderr)
        return 2

    def to_xy(rows: list[tuple[Sample, str]]):
        x = torch.tensor([s.x for s, _ in rows], dtype=torch.float32)
        y = torch.tensor([cls_idx[lb] for _, lb in rows], dtype=torch.long)
        return x, y

    xtr, ytr = to_xy(train_set)
    xva, yva = to_xy(val_set)

    n_channels = int(xtr.shape[-1])  # 6 since the global-motion channels

    # ---- Model: compact temporal conv net over [B,T,17,C] ------------------
    # Small on purpose (~120k params): the corpus starts small, and the
    # inference contract (ONNX op coverage in ort) stays trivial.
    class TemporalConvNet(nn.Module):
        def __init__(self, n_classes: int):
            super().__init__()
            in_ch = 17 * n_channels
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, 128, 3, padding=1),
                nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Conv1d(128, 128, 3, padding=2, dilation=2),
                nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(128, n_classes)

        def forward(self, x):  # x: [B, T, 17, C]
            b, t, v, c = x.shape
            x = x.reshape(b, t, v * c).permute(0, 2, 1)  # [B, V*C, T]
            z = self.net(x).squeeze(-1)
            return self.head(z)

    model = TemporalConvNet(len(classes)).to(device)

    # Class-imbalance weights. Raw inverse-frequency (len/count) is unstable:
    # an intrusion-grade class with 1 of 6000 samples gets a ~6000x weight and
    # dominates the gradient. Default to sqrt-inverse-freq, normalized to mean 1
    # over *supported* classes and clamped to [1/cap, cap]. `inv` restores the
    # old behavior; `none` disables. Metric (F1) is unweighted regardless.
    import math

    def _class_weights(mode: str, cap: float) -> list[float]:
        n = [counts.get(c, 0) for c in classes]
        if mode == "none":
            ws = [1.0 for _ in n]
        elif mode == "inv":
            tot = sum(n)
            ws = [tot / max(1, ni) for ni in n]
        elif mode == "effective":  # Cui et al. 2019, beta=0.999
            beta = 0.999
            ws = [(1.0 - beta) / (1.0 - beta ** ni) if ni > 0 else 1.0 for ni in n]
        else:  # sqrt_inv (default)
            ws = [1.0 / math.sqrt(ni) if ni > 0 else 1.0 for ni in n]
        sup = [wi for wi, ni in zip(ws, n) if ni > 0]
        mean = sum(sup) / len(sup) if sup else 1.0
        ws = [wi / mean for wi in ws]
        return [min(cap, max(1.0 / cap, wi)) for wi in ws]

    cw = _class_weights(args.class_weight, args.class_weight_cap)
    print(f"class weights ({args.class_weight}, cap {args.class_weight_cap}): "
          + "  ".join(f"{c}={wi:.2f}" for c, wi in zip(classes, cw)))
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- Train --------------------------------------------------------------
    # ---- Train-time skeleton augmentation (different angles + lighting) -----
    # Channels per joint: [x, y, conf, vel, gdx, gdy]. Rotate (x,y) around the
    # skeleton centroid and the global-motion (gdx,gdy) by a per-sample random
    # angle → the model sees every track at varied camera angles. Randomly drop
    # joints' local channels → simulates night/IR low visibility. Velocity is a
    # magnitude (rotation-invariant), left as-is. Fresh per batch each epoch,
    # train split only — val stays clean. Free + scales to all data (no re-encode).
    # COCO-17 left/right joint swap (nose stays): eyes 1/2, ears 3/4,
    # shoulders 5/6, elbows 7/8, wrists 9/10, hips 11/12, knees 13/14,
    # ankles 15/16. Used by the horizontal flip below.
    FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    def augment_batch(xb):
        bsz = xb.shape[0]
        xb = xb.clone()

        # --- Horizontal flip: mirror the skeleton left/right. The most natural,
        # perfectly label-preserving skeleton aug (a free 2x). Mirror x around
        # the skeleton's OWN centroid (keeps it in the same screen location, so
        # zone/position meaning survives), reverse global horizontal travel
        # (gdx), and swap L/R joint indices so anatomy stays consistent. y and
        # the velocity magnitude are flip-invariant; gap joints (conf=0) keep 0.
        if args.aug_flip_prob > 0:
            fl1 = (torch.rand(bsz, device=xb.device) < args.aug_flip_prob).view(bsz, 1, 1)
            valid = xb[..., 2] > 0
            vf = valid.float()
            cnt = vf.flatten(1).sum(1).clamp(min=1.0)
            cx = ((xb[..., 0] * vf).flatten(1).sum(1) / cnt).view(bsz, 1, 1)
            xb[..., 0] = torch.where(fl1 & valid, 2 * cx - xb[..., 0], xb[..., 0])
            xb[..., 4] = torch.where(fl1, -xb[..., 4], xb[..., 4])
            xb = torch.where(fl1.unsqueeze(-1), xb[:, :, FLIP_IDX, :], xb)

        # --- Rotation: vary the camera angle. Recomputes geometry on the
        # (possibly flipped) skeleton above. ---
        rad = args.aug_rot_deg * 3.14159265 / 180.0
        ang = (torch.rand(bsz, device=xb.device) * 2 - 1) * rad
        cos = torch.cos(ang).view(bsz, 1, 1)
        sin = torch.sin(ang).view(bsz, 1, 1)
        x, y, conf = xb[..., 0], xb[..., 1], xb[..., 2]
        valid = conf > 0
        vf = valid.float()
        cnt = vf.flatten(1).sum(1).clamp(min=1.0)
        cx = ((x * vf).flatten(1).sum(1) / cnt).view(bsz, 1, 1)
        cy = ((y * vf).flatten(1).sum(1) / cnt).view(bsz, 1, 1)
        dx, dy = x - cx, y - cy
        xb[..., 0] = torch.where(valid, cx + dx * cos - dy * sin, x)
        xb[..., 1] = torch.where(valid, cy + dx * sin + dy * cos, y)
        gdx, gdy = xb[..., 4].clone(), xb[..., 5].clone()
        xb[..., 4] = gdx * cos - gdy * sin
        xb[..., 5] = gdx * sin + gdy * cos
        if args.aug_conf_drop > 0:
            drop = (torch.rand_like(conf) < args.aug_conf_drop) & valid
            for ch in (0, 1, 2, 3):
                xb[..., ch] = torch.where(drop, torch.zeros_like(xb[..., ch]), xb[..., ch])
        return xb

    # Per-class P/R/F1 on val — shared by best-checkpoint selection (below)
    # and the final metrics report.
    def prf(mask_pred, mask_true):
        tp = int((mask_pred & mask_true).sum())
        fp = int((mask_pred & ~mask_true).sum())
        fn = int((~mask_pred & mask_true).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return prec, rec, f1

    def val_pred():
        model.eval()
        with torch.no_grad():
            return model(xva.to(device)).argmax(1).cpu()

    def val_macro_f1(pred):
        return sum(prf(pred == i, yva == i)[2]
                   for i in range(len(classes))) / len(classes)

    n = len(xtr)
    best_f1, best_epoch, best_state, since = -1.0, -1, None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = xtr[idx].to(device)
            if args.augment:
                xb = augment_batch(xb)
            yb = ytr[idx].to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        # Best-checkpoint selection: keep the weights from the epoch with the
        # best val macro-F1, not whatever the last epoch lands on (Dropout +
        # small data make the final epoch a coin-flip). Selecting on val mildly
        # inflates the reported number — a frozen test split is the honest gate
        # (next change). Cheap: val set is small, model tiny.
        pred = val_pred()
        vf1 = val_macro_f1(pred)
        if vf1 > best_f1 + 1e-6:
            best_f1, best_epoch, since = vf1, epoch, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            acc = float((pred == yva).float().mean())
            print(f"  epoch {epoch:>3}  loss {total / n:.4f}  "
                  f"val acc {acc:.3f}  val macroF1 {vf1:.3f}")
        if args.patience and since >= args.patience:
            print(f"  early stop @ epoch {epoch} "
                  f"(no val macro-F1 gain in {args.patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"restored best checkpoint: val macro-F1 {best_f1:.3f} @ epoch {best_epoch}")

    # ---- Metrics (per-class + macro + night slice) on the best checkpoint ---
    pred = val_pred()

    per_class = {}
    for c, i in cls_idx.items():
        prec, rec, f1 = prf(pred == i, yva == i)
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": int((yva == i).sum())}
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)

    night_idx = [i for i, (s, _) in enumerate(val_set) if is_night(s)]
    night = None
    if night_idx:
        nm = torch.tensor(night_idx)
        nf1 = []
        for c, i in cls_idx.items():
            _, _, f1 = prf(pred[nm] == i, yva[nm] == i)
            nf1.append(f1)
        night = {"macro_f1": round(sum(nf1) / len(nf1), 4), "n": len(night_idx)}

    print(f"\nval macro-F1 {macro_f1:.3f}"
          + (f" · night macro-F1 {night['macro_f1']} (n={night['n']})" if night else
             " · night slice: no night samples in val"))
    for c, m in per_class.items():
        print(f"  {c:<12} P {m['precision']:.2f}  R {m['recall']:.2f}  "
              f"F1 {m['f1']:.2f}  (n={m['support']})")

    # ---- Export: ONNX + sidecars (the Rust-facing contract) -----------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, args.seq_len, 17, n_channels, device=device)
    export_kwargs = dict(
        input_names=["x"], output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    onnx_ok = True
    try:
        try:
            # Legacy TorchScript exporter: no onnxscript dependency, and this
            # tiny conv net is squarely inside what it handles. torch ≥2.6
            # defaults to the dynamo exporter, which pulls in onnxscript.
            torch.onnx.export(model, dummy, str(out), dynamo=False, **export_kwargs)
        except TypeError:  # older torch without the dynamo kwarg
            torch.onnx.export(model, dummy, str(out), **export_kwargs)
    except Exception as e:  # noqa: BLE001 — e.g. `onnx` not installed
        # NON-FATAL: the weights npz + sidecars below are the portable truth
        # (numpy/MLX backends run from them, and the ONNX can be rebuilt on
        # any box that has the `onnx` package). Losing a finished training
        # run to a missing export dep is the worse failure.
        onnx_ok = False
        print(f"WARNING: ONNX export failed ({e}); writing weight sidecars "
              "anyway — rebuild the .onnx on a box with `pip install onnx`",
              file=sys.stderr)
    Path(f"{out}.classes.json").write_text(json.dumps({
        "classes": classes,
        "seq_len": args.seq_len,
        "min_joint_conf": args.min_joint_conf,
        "channels": n_channels,
    }))
    # Raw weights sidecar — lets non-ONNX backends (numpy reference, MLX on
    # Apple silicon — see infer_mlx.py) run the exact same model without an
    # ONNX parser. Float tensors only; BN's int counter is training bookkeeping.
    import numpy as np
    np.savez(f"{out}.weights.npz", **{
        k: v.detach().cpu().numpy().astype("float32")
        for k, v in model.state_dict().items()
        if v.dtype.is_floating_point
    })
    Path(f"{out}.metrics.json").write_text(json.dumps({
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "night": night,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "classes": classes,
        "seq_len": args.seq_len,
        "min_joint_conf": args.min_joint_conf,
        "label_source": ("csv" if args.labels_csv else "weak_task_category"),
        "best_epoch": best_epoch,
        "class_weight": args.class_weight,
        "label_smoothing": args.label_smoothing,
    }, indent=2))
    print(f"\nwrote {out if onnx_ok else '(no onnx)'} (+ .classes.json, "
          ".weights.npz, .metrics.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
