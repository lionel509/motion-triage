# motion-triage — service contract

**Pixels in, verdicts out.** Callers depend only on this document — never on the
model, language, or version behind it. The detector and behavior model can change
freely as long as this schema holds.

## Transport

Primary: **HTTP / multipart**, matching the shape most NVR pipelines already
speak. A gRPC mirror ([`proto/motion.proto`](../proto/motion.proto)) exists for
streaming / typed callers; both return the same `Verdict`.

---

## `POST /analyze`

Run detection → tracking → pose → behavior over a frame sequence.

### Request — `multipart/form-data`

| field     | type              | notes                                                                 |
|-----------|-------------------|-----------------------------------------------------------------------|
| `frame`   | jpeg (repeated)   | 1..N frames, time-ordered. Send **1** for the fast gate, **~16** for full triage. |
| `zones`   | json (optional)   | operator-drawn zones / lines, frame-fraction coords.                  |
| `context` | json (optional)   | `{"night": true}` — site-local context for suspicion scoring. The caller knows its timezone; the service never guesses it from UTC frames. |

`zones` payload:

```json
{
  "zones": [ { "name": "yard", "polygon": [[0.5,0.0],[1.0,0.0],[1.0,1.0],[0.5,1.0]] } ],
  "lines": [ { "name": "gate", "a": [0.5,0.0], "b": [0.5,1.0], "in_direction": "a_to_b" } ]
}
```

All coordinates are **fractions of the frame** (`[0,1]`) — resolution-independent.
The service multiplies back by the decoded frame size, so the caller never has to
know the camera's native resolution.

### Response — `application/json`

```json
{
  "decision": "alert",
  "reason": "loitering",
  "detect_ms": 42,
  "frames": 16,
  "active_window": { "start_frac": 0.10, "end_frac": 0.98 },
  "tracks": [
    {
      "id": 0,
      "cls": "person",
      "decision": "alert",
      "reason": "loitering",
      "n": 14,
      "straightness": 0.21,
      "span": 0.08,
      "dwell_frac": 0.88,
      "start_frac": 0.10,
      "end_frac": 0.98,
      "behavior": "pacing",
      "behavior_conf": 0.91,
      "sus_score": 1.0,
      "sus_alert": true,
      "keypoints": [
        [[612.0, 410.0, 0.94], "… 17 COCO joints as [x, y, conf] …"],
        "… one entry per frame the track was seen …"
      ]
    }
  ]
}
```

- `decision` — `"alert"` | `"dismiss"` (event-level: alert if **any** track alerts).
- `reason` — the highest-severity track reason (see taxonomy).
- `tracks[].behavior` / `behavior_conf` — the NN behavior flag + softmax
  confidence. Present only when a `BEHAVIOR_MODEL` is loaded.
- `tracks[].sus_score` / `sus_alert` — suspicion in `[0,1]` and whether it
  crossed the policy threshold (see below). Present only when a `SUS_POLICY`
  is loaded.
- `tracks[].keypoints` — per-frame 17-joint skeletons, COCO order, `[x, y, conf]`
  in source-frame pixels. Omitted/empty until the pose layer ships.
- `tracks[].identity` — present **only** on a track the v3 reversal-split audit
  carved out of a merged one: `{ "split_from": <orig track id>, "decided_by":
  "color" | "body" | "face" }`. Absent on ordinary tracks (see v3 below).
- `tracks[].start_frac` / `end_frac` — the **temporal window** this track occupies
  within the clip, as fractions of the clip duration: `start = first_frame/total`,
  `end = (last_frame+1)/total`, so `end_frac − start_frac == dwell_frac`. Where the
  dwell *is*, not just how long. Synthetic event-level rows (multi-person, zone/line)
  report `0.0/0.0`.
- `active_window` — top-level `{ start_frac, end_frac }` = the **union** of every
  real track's window (synthetic `id == usize::MAX` and zero-span rows excluded).
  Absent when no real track had a window. The dashboard converts this to seconds
  against the actual clip length and **trims the VLM's dense frames / clip to this
  span** instead of the whole padded clip — same frame budget, every frame lands on
  the subject. Additive and backward-compatible.

### Suspicion scoring

Every behavior flag (and rule reason) carries a weight; context multiplies it;
a threshold flags it:

```text
sus = max(flag_weight[behavior], reason_weight[reason])
        × (night ? night_mult : 1)
        × (in_zone ? zone_mult : 1)          # clamped to [0,1]
sus_alert = sus >= threshold
```

`in_zone` is true when the track's footpoint falls inside an operator zone.
The policy is a plain JSON file ([`service/sus_policy.example.json`](../service/sus_policy.example.json)) —
the same `pacing` is benign at noon and an alert at 3 a.m., and that's tuned per
site in seconds, not retrained. Walking weighs `0.0`: never suspicious on its own.

---

## The two-shot fast-exit pattern

The caller avoids fetching and decoding a full clip for the ~40–60% of events
that contain no person:

1. **Gate** — `POST /analyze` with the **single** keyframe already in hand.
   - `decision: dismiss`, `reason: no_person` → **stop. No clip fetch.**
   - any person detected → continue.
2. **Triage** — fetch the clip, extract ~16 frames, `POST /analyze` again.
   - returns the full behavior verdict.

One endpoint, one model, two calls — this is what lets motion-triage replace
*both* the old cheap prefilter and the expensive triage pass. `no_person` is a
first-class dismiss reason so the gate result is never ambiguous.

> Run the single-frame gate at a deliberately **low** confidence threshold: it's
> the one decision that kills an event before the clip is even fetched, so bias
> it toward letting borderline frames through to full triage.

---

## Reason taxonomy

The event reason is the highest-severity reason across all tracks.

| reason              | sev | meaning                                              |
|---------------------|-----|------------------------------------------------------|
| `camera_approach`   | 10  | box grows to fill the frame — approach / tamper      |
| `intrusion`         | 9   | vanished mid-frame while moving (not an edge exit)   |
| `door_entry`        | 8   | crossed an operator line inward                      |
| `zone_intrusion`    | 8   | footpoint entered an operator zone                   |
| `entered_vehicle`   | 7   | vanished at a vehicle, having moved toward it        |
| `line_cross`        | 6   | crossed an operator line                             |
| `pacing`            | 6   | ≥2 heading reversals, small net displacement         |
| `running`           | 6   | sustained fast straight gait                         |
| `u_turn`            | 5   | ~180° reversal in and back out                       |
| `scaling`           | 5   | rises in frame at constant apparent size — climbing  |
| `crouch`            | 5   | box collapses without receding                       |
| `converging`        | 5   | two tracks closing on each other                     |
| `direction_change`  | 4   | heading changed > ~60°                               |
| `sudden_stop`       | 4   | clearly moving, then frozen                          |
| `multi_person`      | 4   | ≥2 spatially distinct people present at once         |
| `loitering`         | 3   | stationary, sustained dwell                          |
| `approach`          | 3   | came closer and stayed                               |
| `erratic`           | 2   | wandering, low-straightness path                     |
| `arrived_by_vehicle`| 1   | appeared at a vehicle and walked away                |
| `no_person`         | 0   | **dismiss** — no person detected (fast-gate exit)    |
| `walk_by`           | 0   | **dismiss** — normal pass-through                    |
| `edge_exit`         | 0   | **dismiss** — walked out of frame                    |
| `transient`         | 0   | **dismiss** — short near-stationary dropout          |
| `blip`              | 0   | **dismiss** — too few points to judge a trajectory   |

---

## v3 — appearance identity (clothing + body + face)

Two **different** people passing a camera in opposite directions used to be
stitched by the geometry-only tracker into one track that reverses direction —
a false `pacing` / `u_turn` / `direction_change` / `loitering`. v3 gives the
tracker an identity sense so they stay distinct, fused with graceful
degradation (strongest available signal wins):

1. **Clothing color** — pure-Rust HSV histogram of the shirt + pants bands.
   Always on; vetoes a frame-to-frame association whose outfit is clearly unlike
   the track's own → the second person spawns their own track.
2. **Body Re-ID** — OSNet x1.0 512-d (env `REID_MODEL`), run only on the few
   ambiguous tracks.
3. **Face** — ArcFace `w600k_r50` 512-d (env `FACE_MODEL`), aligned from the
   pose facial keypoints. Opportunistic; abstains when the face is too small.

Effects on this contract (all additive):

- A merged reversal track may be returned as **two** tracks, each tagged with
  `identity` and typically re-classified to a benign `walk_by`.
- A single person fragmented by an occlusion is no longer miscounted as
  `multi_person`; and a `multi_person` of only clean walk-bys is recorded with
  `route: "dismiss"` instead of escalating.

**Config (env):** `IDENTITY_GATING` (default on; `0`/`false` reverts to exact
geometry-only behavior), `REID_MODEL`, `FACE_MODEL`, and `ID_*` threshold
overrides (`ID_VETO_COLOR_DIST`, `ID_COLOR_DIFF`, `ID_BODY_DIFF`, `ID_FACE_DIFF`,
`ID_REID_MIN_BOX_H`, …). Conservative defaults bias toward **not** splitting.

## Versioning

The contract is **additive**. New `reason` values and new `tracks[]` fields may
appear; callers must tolerate unknown reasons — treat any unknown reason on an
`"alert"` decision as an alert. Breaking changes bump a `/v2/analyze` path so old
callers keep working.
