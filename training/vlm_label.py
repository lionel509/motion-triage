"""VLM teacher labeler — a local Ollama vision model labels a clip's behavior so
the fast on-device behavior-NN can be distilled from it.

The teacher in the auto-distillation flywheel: pull clips -> augment -> *label
here* -> pose-extract -> train -> gate. Run the VLM FREE (open-vocab describe +
multi-label + allowed to say normal/dismiss); we then map its behaviors onto the
NN taxonomy. Clip-level label (applies to every augmented variant of the clip).

Usage:
    python vlm_label.py --model qwen3-vl:8b --clip path/to/clip.mp4
    python vlm_label.py --model qwen3-vl:8b --frames a.jpg b.jpg c.jpg
    python vlm_label.py --model qwen3-vl:8b --clips-dir corpus/ --out labels.jsonl
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# NN behavior taxonomy the student can learn (door is the priority class).
TAXONOMY = [
    "walking", "standing", "pacing", "loitering", "carrying",
    "pickup", "door", "running", "crouching", "climbing",
]

# VLM free-vocabulary phrase -> NN class. Generous: the teacher talks freely,
# we normalize. Unknown phrases drop (logged in raw description, not a class).
BEHAVIOR_MAP = {
    "walking": "walking", "walk": "walking", "walking by": "walking",
    "standing": "standing", "stand": "standing", "waiting": "standing",
    "pacing": "pacing", "back and forth": "pacing",
    "loitering": "loitering", "lingering": "loitering",
    "carrying": "carrying", "carry": "carrying", "holding": "carrying",
    "pickup": "pickup", "picking up": "pickup", "picking_up": "pickup",
    "setting down": "pickup", "putting down": "pickup", "bending": "pickup",
    "door": "door", "entering": "door", "exiting": "door", "leaving": "door",
    "opening door": "door", "entering door": "door", "at the door": "door",
    "knocking": "door", "doorway": "door",
    "running": "running", "fleeing": "running",
    "crouching": "crouching", "kneeling": "crouching",
    "climbing": "climbing", "scaling": "climbing",
}

# Structured-output schema. Ollama (like vLLM guided decoding) fills every field
# when given a schema; a bare format="json" returns empty fields for this model.
SCHEMA = {
    "type": "object",
    "properties": {
        "primary": {"type": "string"},
        "behaviors": {"type": "array", "items": {"type": "string"}},
        "door_interaction": {"type": "boolean"},
        "suspicious": {"type": "boolean"},
        "reason": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["primary", "behaviors", "door_interaction", "suspicious",
                 "reason", "description"],
}

PROMPT = (
    "You are labeling a surveillance video clip to teach a small, fast on-device "
    "model. You are shown several frames sampled in time order from ONE clip. "
    "Decide what the main person is doing across the clip. You have full freedom "
    "to say it is normal / nothing suspicious. Pay special attention to DOOR "
    "interaction: approaching, opening, entering, or exiting a door. "
    'Return ONLY JSON with keys: "primary" (one behavior phrase), "behaviors" '
    '(list of behavior phrases present), "door_interaction" (true/false), '
    '"suspicious" (true/false), "reason" (short), "description" (one sentence). '
    "Behavior phrases should be plain words like: " + ", ".join(TAXONOMY) + ". "
    "Output ONLY the JSON object — no markdown, no code fences, no extra text."
)


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def _extract_json(txt: str) -> dict:
    """Pull the first JSON object from a free-text response (handles ``` fences)."""
    s = txt.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"primary": "", "behaviors": [], "description": s[:300]}


def sample_frames(clip: str, n: int) -> list[str]:
    """Extract n evenly-spaced JPEG frames from a clip via ffmpeg/ffprobe."""
    try:
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", clip],
            capture_output=True, text=True, timeout=30).stdout.strip())
    except Exception:
        dur = 0.0
    td = tempfile.mkdtemp(prefix="vlmlbl_")
    out: list[str] = []
    if dur > 0:
        for i in range(n):
            ts = dur * (i + 0.5) / n
            dest = os.path.join(td, f"f{i:02d}.jpg")
            subprocess.run(["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", clip,
                            "-frames:v", "1", "-q:v", "3", dest],
                           capture_output=True, timeout=60)
            if os.path.exists(dest):
                out.append(dest)
    if not out:  # fallback: first n decoded frames
        subprocess.run(["ffmpeg", "-y", "-i", clip, "-vf", "fps=1",
                        "-frames:v", str(n), os.path.join(td, "g%02d.jpg")],
                       capture_output=True, timeout=120)
        out = [str(p) for p in sorted(Path(td).glob("g*.jpg"))[:n]]
    return out


def vlm_label(model: str, frame_paths: list[str]) -> dict:
    """One Ollama vision call over the frames -> parsed + taxonomy-mapped label."""
    body = json.dumps({
        "model": model,
        "prompt": PROMPT,
        "images": [_b64(p) for p in frame_paths],
        # Do NOT set "format": qwen3-vl in Ollama returns EMPTY under the
        # structured-output grammar constraint (verified 2026-06-13, 1 or N
        # images). Free-text vision works great — ask for JSON, parse it out.
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = json.loads(resp.read().decode())
    parsed = _extract_json(raw.get("response", "") or "")

    def norm(phrase: str) -> str | None:
        p = str(phrase).strip().lower()
        if p in BEHAVIOR_MAP:
            return BEHAVIOR_MAP[p]
        for key, cls in BEHAVIOR_MAP.items():
            if key in p:
                return cls
        return None

    classes = []
    for b in ([parsed.get("primary")] + list(parsed.get("behaviors", []))):
        c = norm(b) if b else None
        if c and c not in classes:
            classes.append(c)
    if parsed.get("door_interaction") and "door" not in classes:
        classes.append("door")
    primary = norm(parsed.get("primary", "")) or (classes[0] if classes else "walking")
    return {
        "primary": primary,
        "classes": classes,
        "door_interaction": bool(parsed.get("door_interaction")),
        "suspicious": bool(parsed.get("suspicious")),
        "reason": parsed.get("reason", ""),
        "description": parsed.get("description", ""),
        "vlm_raw": parsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--clip")
    ap.add_argument("--frames", nargs="+")
    ap.add_argument("--clips-dir")
    ap.add_argument("--out")
    ap.add_argument("--n-frames", type=int, default=6)
    args = ap.parse_args()

    records = []
    if args.frames:
        rec = vlm_label(args.model, args.frames)
        rec["source"] = "frames"
        records.append(rec)
    elif args.clip:
        rec = vlm_label(args.model, sample_frames(args.clip, args.n_frames))
        rec["source"] = os.path.basename(args.clip)
        records.append(rec)
    elif args.clips_dir:
        for clip in sorted(Path(args.clips_dir).glob("*.mp4")):
            try:
                rec = vlm_label(args.model, sample_frames(str(clip), args.n_frames))
            except Exception as exc:  # noqa: BLE001
                rec = {"error": str(exc)}
            rec["source"] = clip.name
            records.append(rec)
            print(f"{clip.name}: {rec.get('primary')} {rec.get('classes')} "
                  f"sus={rec.get('suspicious')}", flush=True)
    else:
        ap.error("need --clip, --frames, or --clips-dir")

    if args.out:
        with open(args.out, "a", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
    else:
        print(json.dumps(records[0] if len(records) == 1 else records, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
