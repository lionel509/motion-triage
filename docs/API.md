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

| field   | type              | notes                                                                 |
|---------|-------------------|-----------------------------------------------------------------------|
| `frame` | jpeg (repeated)   | 1..N frames, time-ordered. Send **1** for the fast gate, **~16** for full triage. |
| `zones` | json (optional)   | operator-drawn zones / lines, frame-fraction coords.                  |

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
- `tracks[].keypoints` — per-frame 17-joint skeletons, COCO order, `[x, y, conf]`
  in source-frame pixels. Omitted/empty until the pose layer ships.

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

## Versioning

The contract is **additive**. New `reason` values and new `tracks[]` fields may
appear; callers must tolerate unknown reasons — treat any unknown reason on an
`"alert"` decision as an alert. Breaking changes bump a `/v2/analyze` path so old
callers keep working.
