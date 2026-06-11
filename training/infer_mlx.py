"""Run the behavior NN from its weight sidecars — numpy anywhere, MLX on Macs.

The Rust service runs the ONNX via ort (CPU EP everywhere, CoreML on
macOS) — that's the production path. This runner is the *portable* one:
the same trained model, no ONNX runtime, no torch.

Backends
--------
* ``numpy``  — pure-numpy forward pass. Runs anywhere Python + numpy run.
               Also the reference implementation the other backends are
               verified against.
* ``mlx``    — Apple-silicon GPU via MLX (``pip install mlx``). Same
               weights, same math, unified memory.

Both load ``<model>.onnx.weights.npz`` + ``<model>.onnx.classes.json``
written by ``train.py`` next to the ONNX. The network is TemporalConvNet
(train.py): Conv1d(17·C→128,k3,p1) → BN → ReLU → Conv1d(128→128,k3,p2,d2)
→ BN → ReLU → GAP(T) → Linear(128→n_classes). Dropout is inactive at
inference.

::

    # classify a corpus, report accuracy against a labels CSV:
    python training/infer_mlx.py --model models/behavior.onnx \
        --corpus data/corpus.jsonl --labels-csv data/labels.csv

    # on the Mac, on the GPU, cross-checked against numpy first:
    python training/infer_mlx.py --model models/behavior.onnx \
        --corpus data/corpus.jsonl --backend mlx --verify 32
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import Sample, load_corpus  # noqa: E402

BN_EPS = 1e-5


# ───────────────────────── weights / metadata ──────────────────────────────

def load_sidecars(model_path: str) -> tuple[dict, dict]:
    """(weights, meta) from ``<model>.weights.npz`` + ``<model>.classes.json``."""
    import numpy as np
    mp = Path(model_path)
    wp = Path(f"{mp}.weights.npz")
    cp = Path(f"{mp}.classes.json")
    if not wp.exists():
        raise FileNotFoundError(
            f"{wp} missing — retrain with the current train.py (it writes the "
            "raw-weights sidecar alongside the ONNX)")
    if not cp.exists():
        raise FileNotFoundError(f"{cp} missing — the model contract sidecar")
    weights = {k: np.asarray(v) for k, v in np.load(str(wp)).items()}
    meta = json.loads(cp.read_text())
    return weights, meta


# ───────────────────────── numpy reference backend ─────────────────────────

def _conv1d_np(x, w, b, *, pad: int, dil: int):
    """x [B,Cin,T], w [Cout,Cin,K] → [B,Cout,T] (stride 1, 'same' length)."""
    import numpy as np
    bsz, cin, t = x.shape
    cout, _, k = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    # Gather the K dilated taps: windows [B, Cin, K, T]
    taps = np.stack([xp[:, :, i * dil : i * dil + t] for i in range(k)], axis=2)
    y = np.einsum("bckt,ock->bot", taps, w, optimize=True)
    return y + b[None, :, None]


def _bn_np(x, g, b, mean, var):
    import numpy as np
    return (x - mean[None, :, None]) / np.sqrt(var[None, :, None] + BN_EPS) \
        * g[None, :, None] + b[None, :, None]


def forward_numpy(weights: dict, x):
    """x [B,T,17,C] float32 → logits [B,n_classes]. Mirrors train.py exactly."""
    import numpy as np
    b, t, v, c = x.shape
    z = x.reshape(b, t, v * c).transpose(0, 2, 1)          # [B, 17C, T]
    z = _conv1d_np(z, weights["net.0.weight"], weights["net.0.bias"], pad=1, dil=1)
    z = _bn_np(z, weights["net.1.weight"], weights["net.1.bias"],
               weights["net.1.running_mean"], weights["net.1.running_var"])
    z = np.maximum(z, 0.0)
    z = _conv1d_np(z, weights["net.4.weight"], weights["net.4.bias"], pad=2, dil=2)
    z = _bn_np(z, weights["net.5.weight"], weights["net.5.bias"],
               weights["net.5.running_mean"], weights["net.5.running_var"])
    z = np.maximum(z, 0.0)
    z = z.mean(axis=2)                                      # GAP over T
    return z @ weights["head.weight"].T + weights["head.bias"]


# ───────────────────────────── MLX backend ──────────────────────────────────

def forward_mlx_factory(weights: dict):
    """Build an MLX forward closure. Import deferred — only Macs have mlx.

    Layout notes (the part that bites): torch Conv1d is NCL with weight
    [Cout, Cin, K]; mlx.core.conv1d is **NLC** with weight [Cout, K, Cin].
    """
    import mlx.core as mx

    w = {k: mx.array(v) for k, v in weights.items()}

    def conv(z, wk, bk, *, pad, dil):
        y = mx.conv1d(z, mx.transpose(w[wk], (0, 2, 1)), stride=1,
                      padding=pad, dilation=dil)
        return y + w[bk]

    def bn(z, prefix):
        return (z - w[f"{prefix}.running_mean"]) \
            / mx.sqrt(w[f"{prefix}.running_var"] + BN_EPS) \
            * w[f"{prefix}.weight"] + w[f"{prefix}.bias"]

    def forward(x_np):
        b, t, v, c = x_np.shape
        z = mx.array(x_np).reshape(b, t, v * c)             # NLC: [B, T, 17C]
        z = conv(z, "net.0.weight", "net.0.bias", pad=1, dil=1)
        z = mx.maximum(bn(z, "net.1"), 0.0)
        z = conv(z, "net.4.weight", "net.4.bias", pad=2, dil=2)
        z = mx.maximum(bn(z, "net.5"), 0.0)
        z = z.mean(axis=1)                                   # GAP over T
        logits = z @ mx.transpose(w["head.weight"]) + w["head.bias"]
        mx.eval(logits)
        import numpy as np
        return np.asarray(logits)

    return forward


# ──────────────────────────────── CLI ───────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="path to the .onnx (sidecars resolved from it)")
    p.add_argument("--corpus", action="append", required=True)
    p.add_argument("--labels-csv", default=None, help="score accuracy against these")
    p.add_argument("--backend", default="numpy", choices=["numpy", "mlx"])
    p.add_argument("--verify", type=int, default=0, metavar="N",
                   help="cross-check this backend vs numpy on N samples first")
    p.add_argument("--min-visibility", type=float, default=0.25)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    import numpy as np
    weights, meta = load_sidecars(args.model)
    classes = meta["classes"]
    seq_len = int(meta.get("seq_len", 16))
    min_joint_conf = float(meta.get("min_joint_conf", 0.0))

    samples: list[Sample] = []
    for path in args.corpus:
        samples.extend(load_corpus(path, seq_len=seq_len,
                                   min_joint_conf=min_joint_conf,
                                   min_visibility=args.min_visibility))
        if args.limit and len(samples) >= args.limit:
            samples = samples[: args.limit]
            break
    if not samples:
        print("no samples", file=sys.stderr)
        return 1
    x = np.asarray([s.x for s in samples], dtype=np.float32)
    n_ch = x.shape[-1]
    if meta.get("channels") and int(meta["channels"]) != n_ch:
        print(f"error: model expects C={meta['channels']}, corpus has C={n_ch}",
              file=sys.stderr)
        return 2

    if args.backend == "mlx":
        fwd = forward_mlx_factory(weights)
    else:
        fwd = lambda a: forward_numpy(weights, a)  # noqa: E731

    if args.verify and args.backend != "numpy":
        k = min(args.verify, len(x))
        ref = forward_numpy(weights, x[:k])
        got = fwd(x[:k])
        max_dev = float(np.max(np.abs(ref - got)))
        same = int((ref.argmax(1) == got.argmax(1)).sum())
        print(f"verify vs numpy: max |Δlogit| {max_dev:.2e}, "
              f"argmax agreement {same}/{k}")
        if same != k:
            print("error: backend disagrees with the numpy reference", file=sys.stderr)
            return 3

    logits = fwd(x)
    pred = logits.argmax(1)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    conf = (e / e.sum(axis=1, keepdims=True)).max(axis=1)

    label_map: dict[str, str] = {}
    if args.labels_csv:
        with open(args.labels_csv, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[1].strip() and not row[0].startswith("#"):
                    label_map[row[0].strip()] = row[1].strip()

    hits = total = 0
    counts: dict[str, int] = {}
    for s, pi, ci in zip(samples, pred, conf):
        cls = classes[int(pi)]
        counts[cls] = counts.get(cls, 0) + 1
        truth = label_map.get(s.sample_id)
        mark = ""
        if truth:
            total += 1
            ok = truth == cls
            hits += ok
            mark = "  ok" if ok else f"  X (label: {truth})"
        if len(samples) <= 40 or truth:
            print(f"{s.sample_id:<44} {cls:<10} {ci:.2f}{mark}")
    print(f"\nbackend {args.backend} · {len(samples)} samples · "
          + " · ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    if total:
        print(f"accuracy vs labels: {hits}/{total} = {hits / total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
