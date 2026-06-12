"""MEVA KPF activity annotations -> behavior labels for our extracted tracks.

MEVA ships human ground-truth activities (``*-activities.yml``) with actor
IDs + source-frame spans, and per-frame actor boxes (``*-geom.yml``). We
extracted our own pose tracks from the same clips. This matches the two by
time-window + box overlap, so a MEVA ``person_picks_up_object`` annotation
becomes a ``pickup`` label on the corresponding track in our corpus.

These are AUTHORITATIVE labels (human-annotated), unlike the displacement
prelabels — so the merge step lets them win on conflict.

::

    python training/meva_labels.py \
        --corpus data/corpus_carry.jsonl data/corpus_meva.jsonl data/corpus_meva2.jsonl \
        --kpf-root data/meva-data-repo/annotation \
        --out data/labels_meva_activity.csv

Attribution: MEVA (Kitware/IARPA), CC BY 4.0.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

MEVA_SRC_FPS = 30.0

# MEVA activity -> our behavior flag. Only clear, single-person, useful ones;
# everything else is intentionally dropped (silence beats noise labels).
ACTIVITY_MAP = {
    "person_carries_heavy_object": "carrying",
    "person_picks_up_object": "pickup",
    "person_sets_down_object": "pickup",
    "person_sitting_down": "sitting",
    "person_standing_up": "sitting",
    "person_opens_facility_door": "door",
    "person_closes_facility_door": "door",
    "person_enters_through_structure": "door",
    "person_exits_through_structure": "door",
    "object_transfer": "handoff",
    "hand_interaction": "handoff",
    "specialized_texting_phone": "phone",
    "specialized_talking_phone": "phone",
    "person_reading_document": "phone",
}

CLIP_RE = re.compile(r"^meva(?:carry)?_(\w+?)_(G\d+)__(\d{8})T(\d{6})Z")
KPF_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})\.(\d{2})-(\d{2})-(\d{2})\.[\d-]+\.(\w+)\.(G\d+)[.-]activities\.yml$")


def index_kpf(root: Path) -> dict[tuple, Path]:
    """(date8, hhmmss, site, cam) -> activities.yml path."""
    idx = {}
    for p in root.rglob("*activities.yml"):
        m = KPF_RE.search(p.name)
        if m:
            key = (m.group(1).replace("-", ""),
                   m.group(2) + m.group(3) + m.group(4), m.group(5), m.group(6))
            idx[key] = p
    return idx


def parse_activities(path: Path) -> list[tuple[str, int, int, int]]:
    """-> [(behavior, actor_id, start_frame, end_frame)] for mapped activities.

    Parsed with a real YAML loader — KPF wraps long tokens across lines, so
    naive whitespace handling silently corrupts long activity names (e.g.
    ``person_carries_heavy_object``). ``act2`` loads as ``{name: conf}``.
    """
    doc = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or []
    out = []
    for item in doc:
        act = item.get("act") if isinstance(item, dict) else None
        if not act:
            continue
        a2 = act.get("act2") or {}
        beh = ACTIVITY_MAP.get(next(iter(a2), ""))
        if not beh:
            continue
        for actor in act.get("actors", []):
            aid = actor.get("id1")
            for ts in actor.get("timespan", []):
                tsr = ts.get("tsr0")
                if aid is not None and tsr:
                    out.append((beh, int(aid), int(tsr[0]), int(tsr[1])))
    return out


def parse_geom(path: Path) -> dict[int, dict[int, tuple]]:
    """actor_id -> {source_frame: (x1,y1,x2,y2)}. ``g0`` is a 'x1 y1 x2 y2' str."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or []
    geo: dict[int, dict[int, tuple]] = defaultdict(dict)
    for item in doc:
        g = item.get("geom") if isinstance(item, dict) else None
        if not g:
            continue
        aid, ts, box = g.get("id1"), g.get("ts0"), g.get("g0")
        if aid is None or ts is None or not box:
            continue
        parts = str(box).split()
        if len(parts) == 4:
            geo[int(aid)][int(ts)] = tuple(int(float(v)) for v in parts)
    return geo


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, max(1.0, b[3] - b[1]))


def match_track(frames, step, acts, geo) -> tuple[str, int] | None:
    """Best (behavior, overlap_count) for one track, or None."""
    best, best_n = None, 0
    for beh, aid, f0, f1 in acts:
        actor = geo.get(aid)
        if not actor:
            continue
        hits = 0
        for fr in frames:
            if not fr:
                continue
            src = fr["frame_index"] * step
            if not (f0 - 30 <= src <= f1 + 30):  # ±~1s: tracks fragment at edges
                continue
            near = min(actor, key=lambda t: abs(t - src), default=None)
            if near is None or abs(near - src) > step * 3:
                continue
            acx, acy, ah = center(actor[near])
            b = fr["box"]
            ocx, ocy = (b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0
            if ((ocx - acx) ** 2 + (ocy - acy) ** 2) ** 0.5 < 0.7 * ah:
                hits += 1
        if hits >= 2 and hits > best_n:
            best, best_n = beh, hits
    return (best, best_n) if best else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", nargs="+", required=True)
    p.add_argument("--kpf-root", default="data/meva-data-repo/annotation")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    idx = index_kpf(Path(args.kpf_root))
    print(f"indexed {len(idx)} KPF activity files")

    # Group tracks by clip so we parse each KPF once.
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for cpath in args.corpus:
        for line in open(cpath, encoding="utf-8"):
            rec = json.loads(line)
            stem = re.sub(r"-t\d+$", "", str(rec["sample_id"]))
            by_clip[stem].append(rec)

    rows, counts = [], defaultdict(int)
    clips_matched = 0
    for stem, recs in by_clip.items():
        m = CLIP_RE.match(stem)
        if not m:
            continue
        key = (m.group(3), m.group(4), m.group(1), m.group(2))
        apath = idx.get(key)
        if not apath:
            continue
        gpath = Path(str(apath).replace("activities.yml", "geom.yml"))
        if not gpath.exists():
            continue
        acts = parse_activities(apath)
        if not acts:
            continue
        geo = parse_geom(gpath)
        clips_matched += 1
        for rec in recs:
            step = round(MEVA_SRC_FPS / float(rec.get("fps_sampled") or 3.75))
            r = match_track(rec["frames"], step, acts, geo)
            if r:
                rows.append((rec["sample_id"], r[0]))
                counts[r[0]] += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# sample_id", "label  (MEVA KPF activity — authoritative)"])
        w.writerows(rows)
    print(f"clips matched to KPF: {clips_matched} | activity-labeled tracks: {len(rows)}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<10} {v}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
