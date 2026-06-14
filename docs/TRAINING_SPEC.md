# Behavior NN — Training Spec (multi-head)

**Spec version:** v1.0 (2026-06-14)
**Goal:** the on-device YOLO + behavior-NN does ~99% of the VLM's *classification*
job (measured as NN↔VLM agreement); the DGX VLM (Nemotron-Omni :8000) becomes
fallback-only. **Fail-open is the cardinal rule: "couldn't determine" ≠ "dismiss."**

---

## PART A — The model is multi-head, not one classifier

It's a stack of heads, each its own training problem with its own labels.

| Head | Output | Labels | Status |
|---|---|---|---|
| Detection | person / vehicle / animal / bag boxes | COCO subset | ✅ YOLO |
| Tracking | per-track association | (regression) | ✅ greedy IoU |
| Pose | 17 COCO keypoints + per-joint conf | 17×3 | ✅ corpus logging (flag off) |
| Per-person activity | multi-label | 12 | ✅ enum exists (VLM only) |
| Per-track behavior | multi-label | 21 | ✅ geometry (muted) |
| Event-level behavior | multi_person / converging | 2 | ✅ |
| Verdict | alert / escalate / dismiss | 3 | ✅ scene-conditioned |
| Severity | P1–P5 | 5 | ✅ taxonomy |
| Confidence / calibration | scalar + abstain | — | ⚠️ needed for fail-open |
| Attributes (forensic) | clothing/bag/hat/gender/age | ~6 | ⚠️ VLM-only |
| Scene type | 20 | 20 | ✅ VLM scene profiler |

Student NN's primary job = **behavior + verdict + activity + pose** heads.
Attribute/scene heads can stay with the VLM longer.

> **Output contract — the NN emits CLASSES, not prose.** The model is not an LLM;
> it outputs the structured heads above (verdict, behaviors, activities, scene,
> severity, confidences). Human-readable alert text is built **deterministically
> from a template** filled by those classes (e.g. "Door entry · front door ·
> 9:18 PM"), never a generated sentence. The VLM teacher's free-text
> reason/description is **label-audit metadata only** (kept under `vlm_raw`), never
> a training target or a product output. *Future option:* distill a tiny caption
> head onto the NN if a one-line natural-language summary is ever wanted — out of
> scope for v1.

## PART B — 21 behavior labels (canonical, `workers/scene_classifier.py:28`)

**Threat — never-miss (keep ungated, fire even at low conf / bad light):**
- `intrusion` — vanishes mid-frame interior (not frame-exit). Hard: occlusion fakes it.
- `camera_approach` — box grows to fill frame (tamper). Hard: benign walk-up to near cam.
- `running` — sustained fast gait. Hard: jogger vs flee; distance.
- `scaling` — climbing fence/wall/drainpipe. Hard: box-aspect artifact → currently `_ARTIFACT`.
- `crouch` — kneeling/ducking. Hard: box-aspect artifact; **pose solves this**.

**Door / zone (the promoted alert):**
- `door_entry` (crossed door line inward) · `zone_intrusion` (restricted polygon) ·
  `fence_cross` (perimeter line either way) · `line_cross` (directional tripwire, non-door)

**Loiter family (FP epicenter — the main reason to train an NN):**
- `loitering` · `prolonged_loitering` (server time-upgrade) · `pacing` (≥2 reversals) ·
  `approach` (closer + stayed) · `direction_change` (>60°) · `u_turn` (~180°) ·
  `sudden_stop` · `erratic` (wandering, low straightness)

**Vehicle:** `entered_vehicle` · `arrived_by_vehicle`
**Event-level (needs all tracks):** `multi_person` (≥2) · `converging` (closing)

## PART C — Per-person activity (12, `llm/activities.py:37`) + pose micro-behaviors

Multi-label, descriptive (no alert by itself; feeds caption + bishul_akum):
`walking · running · standing · sitting · kneeling · crouching · bending · reaching ·
falling · climbing · throwing · fighting · cooking`

**Pose-derived micro-behaviors to ADD (pose >> box-geometry; anonymous → schema-OK):**
- `fall` (torso horizontal, not box-collapse) · `reaching/testing` (hand to handle/window/car door) ·
  `peering` (head into window/car) · `hands-up` (surrender/arms-raised) ·
  `crawling` (low locomotion) · `concealing` (hand-to-waistband, item-under-jacket)

## PART D — Negatives & hard negatives (train as hard as positives)

Ground truth lives in `.claude/wrong_calls.md` (operator-labeled FPs).

**Dismiss classes the NN must emit:** `walk_by` · `edge_exit` · `transient` · `blip` (<3 pts) · `no_person`

**Hard-negative scene catalog (the FP clips):**
- Static clutter mis-detected as person: camera housing, power brick, cables, furniture, foliage (Brooklyn ch14 desk, House porch)
- Distant tiny figures + passing traffic (Jefferson 47c64e81, street car 70b6aa50)
- Weather: rain, snow, wind foliage, headlight sweep
- Night-IR artifacts: insects/moths, spider webs, IR hotspots
- Animals/pets · OSD/timestamp text overlay motion
- The 8-in-a-row street-fleet cluster (crouch blind-fired on public streets → crouch/scaling became `_ARTIFACT`)

## PART E — Scene conditioning (20 scenes, `scene_classifier.py:37`) — verdict depends on location

`residential_entry, residential_yard, driveway, garage, perimeter_fence, gate,
parking_lot, loading_dock, retail_interior, retail_register, lobby, hallway,
warehouse_interior, public_sidewalk, public_street, alley, backyard, rooftop,
stairwell, unknown`. **Model INPUT (or per-scene head), not a global rule.**

- **Public** (street/sidewalk/alley/parking/dock): dismiss pacing/loitering/prolonged + all ambient; `intrusion` capped at **escalate, never dropped**.
- **Residential** (entry/yard/driveway/garage/gate/fence/backyard): float door_entry/approach/arrivals/loiter/pacing/zone/fence/line up to ≥escalate — never silently dismiss.
- **Interior** (retail/lobby/warehouse/stairwell/rooftop): alert only on camera_approach + zone_intrusion + fence_cross; escalate running/scaling/intrusion/line/door; dismiss rest.
- **Universal:** crouch/scaling never blind-alert anywhere (box artifacts).

## PART F — Forensic / attribute outputs (caption + watchlist + search)

Today VLM (`VLM_ATTRIBUTES_ENABLED`) + Re-ID. Target NN heads (OpenPAR/PromptPAR, MIT):
upper-color · lower-color · bag(yes/type) · hat/hood · gender · age band — each with
confidence (degrades night/IR → surface it, keep VLM fallback). Body Re-ID embedding
(SOLIDER, same-visit). Face embedding (ArcFace, watchlist) — **never in pose corpus (schema firewall)**.

## PART G — Object & vehicle sub-tasks (mostly proposed)

Vehicle type (car/truck/moto/bus), parked vs moving, approaching/departing (✅ entered/arrived).
`abandoned_object` (appeared, person left, object stays) · `removed_object` (porch-piracy core) ·
`package_present` (benign-notify) · `bag_co_motion` (theft trigger, `triggers.py`).

## PART H — Specialized verticals (per-customer)

Kashrut (cooking→bishul_akum, mashgiach presence, sabbath, meat/dairy, seal break — live).
Workplace/PPE (hard-hat missing, worker-alone, overnight). Fire/safety (fire/smoke/blocked exit — taxonomy, no detector). Livestock v0 (sheep present/count).

## PART I — Meta-outputs: calibration, abstention, severity

- **Calibrated confidence per head** → principled fail-open-to-VLM threshold.
- **Abstain / uncertain class** → escalate. #145 cardinal rule encoded in the model.
- **Severity P1–P5** → urgency → notify channel (WhatsApp vs Call Police).

## PART J — Data coverage (conditions, not classes — stratify the export)

Lighting day/dusk/**night-IR** (highest value) · Distance near/mid/far · Perspective
lateral/toward/away (center pinned while box grows — broke loitering) · Weather clear/rain/snow/wind ·
Occlusion + frame-edge clip · Vendor/stream Uniview/Hik, main vs sub · Geography street/residential/interior.

## PART K — Distillation mechanics + the blocker

- **Teacher:** VLM (Nemotron-Omni :8000 in prod; local `qwen3-vl:8b` for bulk labeling) → soft labels.
- **Student:** temporal NN on the 2080S; train on pose corpus + VLM labels.
- Use **soft labels (logits)**, not just argmax — preserve teacher uncertainty.
- **Active learning:** prioritize teacher↔student disagreements + operator-corrected events (`wrong_calls.md`, dispositions).
- **Hard-negative mining:** feed confirmed FPs back as negatives every cycle.
- 🚨 **BLOCKER (own notes):** the prod VLM can only confirm/alert — it cannot veto a YOLO FP → DISMISS. Until fixed, the teacher emits no "dismiss, NN was wrong" labels → student can't learn Part D negatives from prod. **Fix VLM-can-dismiss first** (the local `qwen3-vl` teacher in `vlm_label.py` already emits `suspicious:false`, partially mitigating for the flywheel).
- Augment per clip (re-angle, filters) for many instances/clip; **delete raw clips after pose extraction** (guardrail).

## PART L — Governance / eval guardrails

- Held-out eval set, **per-class precision/recall** (walk_by dominates — accuracy lies).
- **Per-class regression gate:** block promotion if any threat class (intrusion, running, fence_cross, fall) recall drops.
- Model backups + git tags + one-command revert every round.
- Claude periodic accuracy audit vs fresh operator dispositions.
- Track **false-dismiss rate** separately, weight it heavily (missed person = cardinal sin; redundant VLM call = cheap).

---

### Versioning
Behavior models: `v1.x` (`behavior_latest.onnx` symlink → `behavior_v1x*.onnx` + `.weights.npz` + `.metrics.json`), archived to `models/archive/v1.x_<date>/`, git-tagged each promotion. Spec doc versioned in its header.
