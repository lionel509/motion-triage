"""Smoke helper: summarize a /triage JSON response (stdin or argv[1] path —
the path form sidesteps PowerShell's pipe re-encoding)."""
import json
import sys

if len(sys.argv) > 1:
    with open(sys.argv[1], encoding="utf-8-sig") as f:
        r = json.load(f)
else:
    r = json.load(sys.stdin)
ts = r.get("tracks", [])
print("decision:", r.get("decision"), "| reason:", r.get("reason"),
      "| tracks:", len(ts))
for t in ts[:6]:
    kps = t.get("keypoints") or []
    first = next((f for f in kps if f), None)
    print(f"  track id={t.get('id')} reason={t.get('reason')} "
          f"kp_frames={len(kps)} joints_in_first={len(first) if first else 0}")
