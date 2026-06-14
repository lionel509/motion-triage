"""VLM teacher labeler — a local Ollama vision model labels a clip across the
multi-head taxonomy (docs/TRAINING_SPEC.md) so the fast on-device behavior-NN can
be distilled from it.

The teacher in the auto-distillation flywheel: pull clips -> augment -> *label
here* -> pose-extract -> train -> gate. Run the VLM FREE (open-vocab) and FAIL-OPEN
(uncertain -> escalate, NEVER silently dismiss). We then map its free vocabulary
onto the NN taxonomy heads: verdict, behaviors (21), activities (12), scene (20).
Clip-level label (applies to every augmented variant of the clip).

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

# ---- Taxonomy heads (docs/TRAINING_SPEC.md Parts B / C / E) -----------------
VERDICTS = ["alert", "escalate", "dismiss"]

BEHAVIORS = [  # Part B (21) — the per-track behavior head
    "intrusion", "camera_approach", "running", "scaling", "crouch",
    "door_entry", "zone_intrusion", "fence_cross", "line_cross",
    "loitering", "prolonged_loitering", "pacing", "approach",
    "direction_change", "u_turn", "sudden_stop", "erratic",
    "entered_vehicle", "arrived_by_vehicle", "multi_person", "converging",
]
THREAT_BEHAVIORS = {  # never-miss set (Part B threat + Part C threat micro)
    "intrusion", "camera_approach", "running", "scaling", "fence_cross",
    "fall", "fighting", "weapon",
}

ACTIVITIES = [  # Part C (12) + pose micro-behaviors — per-person activity head
    "walking", "running", "standing", "sitting", "kneeling", "crouching",
    "bending", "reaching", "falling", "climbing", "throwing", "fighting",
    "cooking", "fall", "peering", "hands_up", "crawling", "concealing",
]

NEGATIVES = ["walk_by", "edge_exit", "transient", "blip", "no_person"]  # Part D

SCENES = [  # Part E (20) — scene-conditioning head/input
    "residential_entry", "residential_yard", "driveway", "garage",
    "perimeter_fence", "gate", "parking_lot", "loading_dock",
    "retail_interior", "retail_register", "lobby", "hallway",
    "warehouse_interior", "public_sidewalk", "public_street", "alley",
    "backyard", "rooftop", "stairwell", "unknown",
]

# Free-vocabulary phrase -> canonical class. Generous: the teacher talks freely,
# we normalize. Unknown phrases drop (kept in the raw description, not a class).
BEHAVIOR_MAP = {
    "intrusion": "intrusion", "vanished": "intrusion", "disappeared": "intrusion",
    "camera approach": "camera_approach", "approaching camera": "camera_approach",
    "tamper": "camera_approach", "fills the frame": "camera_approach",
    "running": "running", "fleeing": "running", "sprinting": "running",
    "scaling": "scaling", "climbing fence": "scaling", "climbing wall": "scaling",
    "crouch": "crouch", "crouching": "crouch", "ducking": "crouch",
    "door": "door_entry", "door entry": "door_entry", "entering": "door_entry",
    "exiting": "door_entry", "entering door": "door_entry", "at the door": "door_entry",
    "opening door": "door_entry", "knocking": "door_entry", "doorway": "door_entry",
    "zone intrusion": "zone_intrusion", "restricted area": "zone_intrusion",
    "fence cross": "fence_cross", "crossing fence": "fence_cross",
    "line cross": "line_cross", "tripwire": "line_cross",
    "loitering": "loitering", "lingering": "loitering", "hanging around": "loitering",
    "prolonged loitering": "prolonged_loitering",
    "pacing": "pacing", "back and forth": "pacing",
    "approach": "approach", "came closer": "approach", "approaching": "approach",
    "direction change": "direction_change", "changed direction": "direction_change",
    "u turn": "u_turn", "u-turn": "u_turn", "turned around": "u_turn",
    "sudden stop": "sudden_stop", "stopped": "sudden_stop",
    "erratic": "erratic", "wandering": "erratic",
    "entered vehicle": "entered_vehicle", "got in car": "entered_vehicle",
    "arrived by vehicle": "arrived_by_vehicle", "got out": "arrived_by_vehicle",
    "multi person": "multi_person", "multiple people": "multi_person", "group": "multi_person",
    "converging": "converging", "closing in": "converging",
}
ACTIVITY_MAP = {a: a for a in ACTIVITIES}
ACTIVITY_MAP.update({
    "walk": "walking", "stand": "standing", "sit": "sitting", "kneel": "kneeling",
    "bend": "bending", "reach": "reaching", "carry": "reaching", "carrying": "reaching",
    "fall": "fall", "fallen": "fall", "lying down": "fall", "collapsed": "fall",
    "fight": "fighting", "punching": "fighting", "physical altercation": "fighting",
    "throw": "throwing", "cook": "cooking", "peering": "peering", "peeking": "peering",
    "looking into": "peering", "hands up": "hands_up", "arms raised": "hands_up",
    "crawl": "crawling", "crawling": "crawling", "concealing": "concealing", "hiding": "concealing",
})
NEGATIVE_HINTS = {
    "walk_by": ["walking by", "passing", "passing through", "walk by", "pass-through"],
    "edge_exit": ["walked out", "left frame", "exited frame", "out of view"],
    "no_person": ["no person", "no people", "empty", "nobody", "no one"],
    "transient": ["flicker", "brief", "momentary"],
}

PROMPT = (
    "You are labeling a surveillance video clip to teach a small, fast on-device "
    "model. You are shown several frames in time order from ONE clip. Decide what "
    "the main person is doing across the whole clip. You have full freedom to say it "
    "is normal / nothing suspicious. \n"
    "FAIL-OPEN RULE: if you genuinely cannot tell whether a person is present or what "
    "is happening (too dark, occluded, ambiguous), set verdict='escalate' and say so "
    "in reason — NEVER set verdict='dismiss' when unsure. A missed real person is the "
    "worst outcome.\n"
    "Pay special attention to DOOR interaction (approaching/opening/entering/exiting a "
    "door) and to genuine threats (intrusion, someone approaching the camera, running/"
    "fleeing, climbing/scaling, fighting, a fall, a visible weapon).\n"
    'Return ONLY a JSON object (no markdown, no code fence) with keys: '
    '"verdict" (one of alert/escalate/dismiss), '
    '"behaviors" (list of behavior phrases present), '
    '"activities" (list of what the body is physically doing), '
    '"scene" (one location type), "door_interaction" (true/false), '
    '"suspicious" (true/false), "weapon_visible" (true/false), '
    '"reason" (short), "description" (one sentence).\n'
    "Behavior phrases like: " + ", ".join(BEHAVIORS) + ". "
    "Activity words like: " + ", ".join(ACTIVITIES) + ". "
    "Scene like: " + ", ".join(SCENES) + "."
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
    return {"verdict": "escalate", "behaviors": [], "activities": [],
            "description": s[:300], "reason": "unparseable VLM output"}


def _map(phrase, table) -> str | None:
    p = str(phrase).strip().lower()
    if p in table:
        return table[p]
    for key, cls in table.items():
        if key in p:
            return cls
    return None


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
    if not out:
        subprocess.run(["ffmpeg", "-y", "-i", clip, "-vf", "fps=1",
                        "-frames:v", str(n), os.path.join(td, "g%02d.jpg")],
                       capture_output=True, timeout=120)
        out = [str(p) for p in sorted(Path(td).glob("g*.jpg"))[:n]]
    return out


def vlm_label(model: str, frame_paths: list[str]) -> dict:
    """One Ollama vision call over the frames -> multi-head taxonomy label.

    No structured-output 'format': qwen3-vl in Ollama returns EMPTY under the
    grammar constraint (verified 2026-06-13). Free-text vision works -> parse JSON.
    """
    body = json.dumps({
        "model": model,
        "prompt": PROMPT,
        "images": [_b64(p) for p in frame_paths],
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = json.loads(resp.read().decode())
    parsed = _extract_json(raw.get("response", "") or "")

    behaviors: list[str] = []
    for b in parsed.get("behaviors", []) or []:
        c = _map(b, BEHAVIOR_MAP)
        if c and c not in behaviors:
            behaviors.append(c)
    if parsed.get("door_interaction") and "door_entry" not in behaviors:
        behaviors.append("door_entry")

    activities: list[str] = []
    for a in parsed.get("activities", []) or []:
        c = _map(a, ACTIVITY_MAP)
        if c and c not in activities:
            activities.append(c)

    scene = parsed.get("scene", "unknown")
    scene = scene if scene in SCENES else (_map(scene, {s: s for s in SCENES}) or "unknown")

    verdict = parsed.get("verdict", "escalate")
    verdict = verdict if verdict in VERDICTS else "escalate"  # fail-open default

    weapon = bool(parsed.get("weapon_visible"))
    if weapon and "weapon" not in behaviors:
        behaviors.append("weapon")

    # Negative inference when nothing fired and not suspicious.
    desc = (parsed.get("description", "") or "").lower()
    negatives: list[str] = []
    if not behaviors and not parsed.get("suspicious"):
        for neg, hints in NEGATIVE_HINTS.items():
            if any(h in desc for h in hints):
                negatives.append(neg)
        if not negatives and "walking" in activities:
            negatives.append("walk_by")

    threat = any(b in THREAT_BEHAVIORS for b in behaviors) or weapon
    primary = behaviors[0] if behaviors else (activities[0] if activities else
              (negatives[0] if negatives else "walk_by"))
    return {
        "verdict": verdict,
        "primary": primary,
        "behaviors": behaviors,
        "activities": activities,
        "negatives": negatives,
        "scene": scene,
        "door_interaction": bool(parsed.get("door_interaction")),
        "suspicious": bool(parsed.get("suspicious")),
        "weapon_visible": weapon,
        "threat": threat,
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
            print(f"{clip.name}: verdict={rec.get('verdict')} "
                  f"beh={rec.get('behaviors')} act={rec.get('activities')} "
                  f"scene={rec.get('scene')} threat={rec.get('threat')}", flush=True)
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
