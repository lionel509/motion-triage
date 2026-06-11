"""Generate the train/infer featurization parity fixture.

Builds a deterministic synthetic track (presence gaps, low-confidence
joints, a walking-ish trajectory), runs it through the CANONICAL
featurizer (``dataset.track_to_tensor``), and writes
``service/testdata/feature_parity.json``. The Rust unit test
(`behavior::tests::featurization_matches_python_dataset_py`) replays the
same frames through ``normalize_track`` and demands numeric agreement —
the mechanical guarantee that training features and inference features
can never silently drift.

Re-run whenever either featurizer changes::

    python training/gen_parity_fixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import track_to_tensor  # noqa: E402

SEQ_LEN = 16
MIN_JOINT_CONF = 0.30

OUT = Path(__file__).resolve().parent.parent / "service" / "testdata" / "feature_parity.json"


def _frame(t: int) -> dict:
    """Deterministic synthetic frame t: a person drifting right with a
    sinusoid-free (pure polynomial — reproducible in any float impl)
    limb wobble, one low-conf joint, varying box size."""
    x1 = 100.0 + 7.0 * t
    y1 = 50.0 + 1.5 * t
    h = 200.0 + 3.0 * t
    box = {"x1": x1, "y1": y1, "x2": x1 + 0.4 * h, "y2": y1 + h}
    kps = []
    for j in range(17):
        kx = x1 + 10.0 + 3.0 * j + 0.5 * t * (1 + (j % 3))
        ky = y1 + 8.0 * j + 0.25 * t * (j % 5)
        # joint 9 (l_wrist) is deliberately low-conf on odd frames —
        # exercises the min_joint_conf masking + velocity-reset path.
        c = 0.12 if (j == 9 and t % 2 == 1) else round(0.55 + 0.02 * ((j + t) % 10), 4)
        kps.append([round(kx, 4), round(ky, 4), c])
    return {"i": t, "box": box, "conf": 0.9, "kps": kps}


def main() -> int:
    # 21 frames (> SEQ_LEN → exercises subsampling) with two gaps.
    frames: list[dict | None] = []
    for t in range(21):
        frames.append(None if t in (5, 13) else _frame(t))

    x, vis = track_to_tensor(frames, seq_len=SEQ_LEN, min_joint_conf=MIN_JOINT_CONF)
    flat = [v for frame in x for joint in frame for v in joint]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "seq_len": SEQ_LEN,
        "min_joint_conf": MIN_JOINT_CONF,
        "frames": frames,
        "expected": flat,
        "visibility": vis,
    }))
    print(f"wrote {OUT} ({len(flat)} features, visibility {vis:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
