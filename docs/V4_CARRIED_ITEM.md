# motion-triage v4 — carried-item detection (grounding)

**Scope:** detect when a person **brings**, **leaves** (abandons), or **removes** an item — packages, duffels, weapons, and the unusual break-in/threat items COCO lacks (crowbar, bolt cutters, jerry can / accelerant, ladder, long-gun, pry bar). Then weight `item × scene × hour → sus` and feed the 3-way route (`alert | vlm | dismiss`).
**Posture:** closed commercial product. **MIT/Apache CODE never launders a WEIGHT trained on research-only data.** Every checkpoint is pinned to its training-data provenance; AGPL/GPL is forbidden to vendor.
**Synthesis basis:** open-vocab + closed-set + flywheel research, with adversarial verdicts overriding on conflict.

> ⚠️ **Read the "Codebase-grounded corrections" section at the bottom first** — a post-review pass against the actual repo flips two recommendations (weapon routing posture; the Phase-1 flywheel prerequisite) and corrects the sus-policy schema + tracker edit points. Where the corrections conflict with the body above, the corrections win.

---

## TL;DR (5 lines)

1. **Do NOT put an open-vocab grounding model on the hot path.** No model is simultaneously clean-weight, ONNX/CoreML-friendly, and real-time on the Apple GPU. All three already-wired grounding backends are unshippable: **YOLO-World (GPL-3.0) and YOLOE (AGPL-3.0) are license-forbidden; GroundingDINO is Apache-code but research-tainted weights AND ONNX-export-hostile.**
2. **On-device runtime = a closed-set Apache detector** (D-FINE COCO-only or **YOLOX**) emitting item boxes, sharing the detector pass that already runs — this *also* lands the AGPL Ultralytics YOLO26 swap that DATASETS.md already mandates.
3. **brought/left/removed is a TRACKING problem, not a detection problem** — derive it geometrically from the existing class-aware IoU tracker (`services/rust/yolo-triage/src/main.rs`); no new model, no dataset license.
4. **The VLM stays in-path** as the open-vocab describer for the unusual-item long tail, the second guard for ambiguous events, and the **label teacher** for the flywheel (Qwen3-VL Apache-2.0 / Nemotron Open Model License both permit output-as-training-data).
5. **Single cleanest deployable open-vocab option = Florence-2 (MIT code + MIT weights)** — but as a *local, offline* teacher/labeler at per-event latency, NOT a real-time runtime. Treat its weight as **conditional pending counsel sign-off on its LAION-sourced FLD-5B lineage**.

---

## Architecture decision (ranked)

The candidate architectures, ranked for THIS deployment (Apple-GPU on-device, closed commercial, ~410 ms/16-frame budget):

### #1 — Closed-set Apache detector + geometric person-object association + VLM-as-teacher  ✅ RECOMMENDED
- **License:** clean if the backbone is Apache **code + weights** and the released detector is trained from a clean init on VLM-teacher + CC-BY/CC0 data. No research-only weight enters the shipped artifact.
- **On-device:** one closed-set detector pass (replacing the AGPL YOLO26 pass already running) + a sub-millisecond geometric association pass over existing tracks. No open-vocab text encoder = no prompt drift, lower latency.
- **Why it wins:** (a) COCO **already** detects common carried items (backpack 24, handbag 26, suitcase 28, umbrella 25, bottle 39), so ordinary bags need no new model; (b) the *unusual* items are a small fixed vocabulary (~8–12 classes) — a closed-set head, not open-vocab; (c) brought/left/removed is geometry over the tracker you already built; (d) it reuses the proven behavior-NN flywheel pattern (VLM dual-JSON → autotrain → promote → reload).
- **This is the same shape as the weapon path DATASETS.md already chose** (RF-DETR/D-FINE/YOLOX finetuned on Roboflow Pistols CC0). Generalize that decision; do not introduce a grounding-family runtime.

### #2 — VLM-only (no new on-device model)
- **License:** clean (Qwen3-VL Apache / Nemotron permit commercial use + output-as-training-data).
- **On-device:** zero new on-device cost, but every carried-item event becomes a DGX round-trip — duplicating the in-path VLM you already pay for, and it inherits the macOS Local-Network reachability failure mode that already breaks the DGX link.
- **Verdict:** correct as the **Phase-1 fallback and the long-tail path**, wrong as the steady state. This is where v4 *starts* (ship the VLM `carried_items` field first), but the flywheel exists to move the common case on-device.

### #3 — On-device open-vocab grounding model (the operator's literal v4 idea)
- **Verdict: REJECTED for the runtime.** Adversarially confirmed: there is no clean-weight + ONNX-friendly + real-time open-vocab detector. Every permissive-code option ships a checkpoint trained on research-only grounding corpora (Cap4M, GoldG = GQA+Flickr30k, GRIT = LAION/COYO pseudo-labels, WebLI, SA-1B). The open-vocab models survive only **offline, as auto-labelers** (you ship the labels, not the weights) — and even there the cleanest teacher is Florence-2, not the grounding-DINO family.

**Net:** closed RF-DETR/D-FINE/**YOLOX** detector for the known + small-fixed-vocabulary item classes, **cheap geometric person-object association** for carry/abandon/remove, **Florence-2 (MIT)** as the one clean *offline* open-vocab teacher, and the **VLM in-path** as second guard + label teacher.

---

## Model shortlist (open-vocab / grounding)

Three columns per the strict posture: **(a) CODE license, (b) WEIGHT verdict, (c) on-device feasibility.** Weight verdict and on-device reflect the adversarial downgrades.

| Model | CODE license | WEIGHT verdict (commercial) | On-device (Apple GPU) | Role |
|---|---|---|---|---|
| **Florence-2** (base 0.23B / large 0.77B, MS) | **MIT** ([model card](https://huggingface.co/microsoft/Florence-2-large), [onnx port](https://huggingface.co/onnx-community/Florence-2-base-ft)) | **conditional** — MIT weights, but FLD-5B includes LAION-sourced images; Microsoft made no data-clearance warranty ([discussion 83](https://huggingface.co/microsoft/Florence-2-large/discussions/83)). Cleanest in a dirty field; needs counsel sign-off. | heavy_but_possible — autoregressive (token-by-token) detection → per-EVENT, not per-frame; community CoreML ~2–3 s/image, cpuOnly ([CoreML notes](https://github.com/john-rocky/CoreML-Models)) | **Offline teacher / labeler; DGX-unreachable fallback confirm.** The one clean-ish open-vocab option. |
| **OWLv2 / OWL-ViT** (Google) | **Apache-2.0** ([scenic](https://github.com/google-research/scenic/blob/main/scenic/projects/owl_vit/README.md)) | **finetune-only** — self-trained on **WebLI** (proprietary, research-only) + LVIS/O365 (O365 [academic-only](https://www.objects365.org/download.html)); card omits the self-train corpus ([paper](https://arxiv.org/html/2306.09683v3)). Apache code does not launder it. | **no (runtime)** — vanilla export FAILS ([optimum #1713](https://github.com/huggingface/optimum/issues/1713)); only an unlicensed auto-converted ONNX port exists; ~400 ms+/image even on Orin+TensorRT. | Secondary offline labeler at most; dominated by Florence-2. |
| **GroundingDINO** (IDEA) — **WIRED** | **Apache-2.0** ([repo](https://github.com/IDEA-Research/GroundingDINO)) | **reference-only** — grounding-dino-tiny trained on O365 + **GoldG** + **Cap4M** (Cap4M never released; GoldG research-licensed). | **no** — ONNX-export-hostile (custom MultiScaleDeformableAttention, fixed-batch only — [#223](https://github.com/IDEA-Research/GroundingDINO/issues/223)). | **Keep DEV/EVAL + offline-labeler only. Drop from any shipped binary.** |
| **YOLO-World** (Tencent) — **WIRED** | **GPL-3.0** ([LICENSE](https://github.com/AILab-CVC/YOLO-World/blob/master/LICENSE)) | **forbidden** — GPL conveys-source copyleft; "supports commercial usage" = GPL's right to *sell*, not a waiver. | n/a | **DROP from any shipped binary.** Tencent commercial license only. |
| **YOLOE** (THU-MIG / Ultralytics) — **WIRED** | **AGPL-3.0** ([Ultralytics license](https://www.ultralytics.com/license)) | **forbidden** — AGPL §13 network copyleft; Ultralytics states AGPL covers code AND produced models. | n/a | **DROP from any shipped binary.** Enterprise license only. Strictly worse than YOLO-World. |
| **OmDet-Turbo** (om-ai-lab) | **Apache-2.0** ([repo](https://github.com/om-ai-lab/OmDet)) | **finetune-only** — sole checkpoint trained on O365 + **GoldG** (Flickr30k non-commercial) ([HF card](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf)). | heavy_but_possible/unproven — backbone is **Swin-T + CLIP + deformable-attn** (NOT RT-DETR ResNet as some sources claim); ONNX path is fixed-size, post-proc excluded; all speed numbers are NVIDIA+TensorRT, **zero CoreML/Apple validation**. | Best *architecture* for a cacheable fixed-vocab head IF you fund a from-scratch retrain + a real CoreML export spike. The one open-vocab on-device experiment worth prototyping — but high-risk. |
| **MM-Grounding-DINO** (OpenMMLab) | **Apache-2.0** ([config](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md)) | **forbidden-to-vendor** — Cap4M dropped, but trained on **GoldG (Flickr30k non-commercial)** + **GRIT (LAION/COYO scrape)**; same risk class as wired GDINO. | **no** — same DETR+BERT export wall; ~18.5 FPS even on A100. | Even as an offline teacher its tainted lineage caps it at internal-eval-only. Dominated. |
| **T-Rex2 / DINO-X / GLIP** (IDEA / MS) | non-commercial / API-only / MIT | **forbidden / reference-only** — T-Rex2 IDEA non-commercial; DINO-X weights API-only; GLIP Cap4M-tainted. | n/a (no local weights) | Exclude. |

**WIRED-backend verdict:** keep **GroundingDINO** as a dev/eval buffet member + offline labeler; **drop YOLO-World and YOLOE** from every shipped path (GPL/AGPL). None of the three is shippable on-device.

---

## Person-object association (the cheap layer, no new model)

brought_in / left_behind / removed is **derived from track geometry**, not learned. The detector only answers *"is there an item box, and what class."* The tracker computes the state.

**The machinery already exists** in `services/rust/yolo-triage/src/main.rs`:
- `fn iou()` (L120), `struct Track { id, cls, first_frame, last_frame, first, last, centers, heights }` (L67), class-aware NMS, and IoU tracking that **never merges a person into a vehicle track** (`if used[di] || d.cls != t.cls { continue }`).
- `near_vehicle()/approached_vehicle()/departed_vehicle()/entered_vehicle/arrived_by_vehicle` **already implement the exact person-vs-other-class spatial-temporal association pattern.** Carried-item is the same shape with item class ids substituted for `VEHICLE_CLASSES`.

**The three predicates (temporal logic over track histories):**
- **CARRYING / brought_in** — item box near/overlapping a person box AND co-moving with that person's track (footpoint distance stays small across the clip); brought_in if it *entered* with the person.
- **LEFT_BEHIND / abandoned** — item track APPEARS, then PERSISTS after the owning person's track exits/separates for N frames (item stationary, owner gone).
- **REMOVED** — item present in early frames, associated to a person, then GONE simultaneously with a person's track leaving.

**Two hard requirements (adversarial):**
1. **Widen the discard filter.** Today L163/L187 hard-drop everything that is not person(0) or vehicle(2,3,5,7): `if cls != PERSON_CLASS && !is_vehicle(cls) { continue; }`. The item boxes the detector emits are thrown away **before** the tracker sees them. Add the item class ids to the kept set (mirror the `custom_backend.py` `_CUSTOM_ID_BASE=1000` offset so they don't collide with COCO ids).
2. **The association layer is INERT alone.** It inherits the license + provenance of whatever detector feeds it. It is **not** independently shippable — it ships only once a commercially-clean item detector lands AND the discard filter is widened. (Correct verdict: the algorithm is clean; its inputs are not.)

**State is FP-prone under occlusion** — owner-association breaks when the tracker loses the owner through occlusion, and the spike's lightweight IoU tracker is weaker than shape/heat-model trackers in the literature. So **left_behind/removed route to the VLM (escalate), never direct-fire** — consistent with the existing "geometric reasons demoted alert→escalate" rule.

**Eval-only benchmarks (NEVER training weights):** ABODA (no LICENSE file → all-rights-reserved), PETS2006 (research-only), AVSS2007 / i-LIDS (Home-Office-gated, paid + research-only EULA). They are sanity checks; the abandoned-object logic is algorithmic so no training corpus is needed.

---

## Routing taxonomy (item × context → sus)

New per-track reasons slot into the existing `lanes_triage.py` structures (`_TRIAGE_MESSAGE` / `_TRIAGE_CATEGORY` / `_REASON_SEVERITY` / `GEOMETRIC_REASONS` / `_route_triage` cascade) and `sus_policy.json`:

| Reason | Severity | Default route | Notes |
|---|---|---|---|
| `weapon_shape` | 12 | **ALERT (direct-fire)** | The one carried-item reason that direct-fires WhatsApp with no VLM (mirrors `armed_person`). **Operator sign-off required** — CLAUDE.md demoted geometric/behavioral reasons to escalate; confirm a misread "long object" FP cost is acceptable. |
| `item_abandoned` | 10 | ALERT in transit/perimeter zone; **escalate(VLM)** elsewhere | Abandoned-package detection. FP-prone under occlusion → prefer escalate by default. |
| `burglary_tool` (crowbar/bolt-cutters/jerry-can/ladder carried) | 9 | **escalate(VLM)** | Policy *weight* (not route) makes it ALERT at night near a fence. |
| `item_removed` | 6 | escalate(VLM) | Possible theft. |
| `item_brought_in` | 4 | escalate only if scene+hour weight crosses | Benign by default. |

**`sus_policy.json` gets an `item_weights` block** mirroring the per-behavior weights with the SAME `night_mult` / `zone_mult` multipliers — `weight = base(item) × scene_mult(zone_type) × hour_mult`, fully per-site tunable with no retraining:

| Item × scene × hour | sus | Outcome |
|---|---|---|
| ladder / bolt_cutters / crowbar × fence_zone × night (3am) | 0.95 | **alert** |
| jerry_can / accelerant × any × any | >0.85 | **alert** (accelerant has no benign reading) |
| abandoned package × transit_zone | 0.90 | **alert** |
| backpack × storefront × noon | 0.0 | **dismiss** (the "street walker weighs 0.0" rule) |
| weapon_shape | bypasses weighting | **direct alert** |

**The 3-way route falls out of the existing thresholds:** sus ≥ `alert_at` (0.85) AND confident → **alert**; `review_at` (0.40) ≤ sus < 0.85, OR a class below `conf_floor` (0.50) → **vlm**; sus < 0.40 → **dismiss**. A flagged-but-unweighted item **floors at vlm** (never silently dropped — CLAUDE.md fail-open).

---

## Flywheel (VLM `carried_items` field → on-device detector)

The proven behavior-NN flywheel pattern applied to items. **Two phases.**

**PHASE 1 — ships immediately, no new model:**
- **VLM output** (`prompts_schemas.py` `RESPONSE_SCHEMA.properties`): add `carried_items: [{ item: string (open-vocab, free text), state: enum[brought_in, left_behind, removed, present_unchanged], confidence: number, person_index: int|null }]`.
- **Self-label half** (box `llm/prompts.py`): add `training.carried_items: [{ item_class: <closed enum>, state, bbox_frac, person_index }]`. The operator half stays **free-text open-vocab**; a small server-side string→class dictionary snaps it to a CLOSED label for training (exactly how `task_category` enums constrain the training label today). The VLM self-labels the TRUE item even when its operator verdict is a false positive.
- **Persist:** zero change — `lanes_event_verdict.py:76` already writes the whole verdict to `events.llm_raw_json` (no migration).
- **State machine** (Option C): add the tracker association + carry/abandon/remove predicates in the Rust spike.

This alone gives v4 carried/brought/left detection end-to-end via the VLM-in-path and **starts filling the corpus** for the long-tail items that have no clean public set.

**PHASE 2 — flywheel matures, on-device head:**
- **Export:** `export_pose_dataset.py += --carried-items` → `carried_items.csv` keyed by the same pseudonymized `sample_id = sha256("{salt}:{event_id}")[:16]`.
- **Train:** extend the Rust `continuous-train` daemon (same `NOTIFY training_label_added` trigger, debounce 15 s, same promote + checkpoint + reload + rollback) with a carried-item training branch alongside `autotrain.py`.
- **Promote** an Apache-backbone closed-set item head once the corpus is large enough; the VLM demotes to second-guard for the open-vocab tail only. **Weight in-domain (VLM-labeled) samples over MEVA** in the promote-gate — MEVA is overhead/facility footage, a cold-start bootstrap, not the customers' storefront/perimeter domain.

**Bootstrap datasets — USE vs eval-only:**

| Dataset | License | Use |
|---|---|---|
| **VLM-teacher labels** (Qwen3-VL / Nemotron) | Apache / Nemotron Open Model License — [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct), [Nemotron](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/) | **USE** — output-as-training-data permitted; this is the acquisition strategy for unusual items. |
| **MEVA** (`person_abandons_package`, `person_carries_heavy_object`, `person_picks_up_object`, `person_sets_down_object`) | CC-BY-4.0 ([mevadata.org](https://mevadata.org/)) | **USE** — cold-start for brought/left/removed. Bootstrap-only; let in-domain dominate. |
| **Roboflow Pistols** (weapon) | **CC0** ([page](https://public.roboflow.com/object-detection/pistols)) | **USE** — already approved in DATASETS.md. |
| **Roboflow Universe** bag/tool/luggage sets | varies (CC0/CC-BY/MIT/**undeclared**) | **USE only after per-dataset license check** — undeclared sets are unusable until confirmed. |
| **COCO** common bags | CC-BY annotations ([terms](https://cocodataset.org/#termsofuse)) | **USE** (annotations) — already covered by the existing detector. |
| **PETS2006** | research-only (UK ICO) | **EVAL-ONLY — never weights.** |
| **i-LIDS / AVSS2007** | paid + research-only EULA | **EVAL-ONLY — never weights.** |

---

## DO-NOT-SHIP callout

Every forbidden / finetune-only asset, with the **weight-vs-data** reason:

**Forbidden to vendor (CODE copyleft — hard stop before any weight question):**
- **YOLO-World** — **GPL-3.0** code. Conveying inside a closed binary = GPL derivative = must open your source. ([LICENSE](https://github.com/AILab-CVC/YOLO-World/blob/master/LICENSE))
- **YOLOE** — **AGPL-3.0** code; §13 network copyleft, covers produced models. ([Ultralytics license](https://www.ultralytics.com/license))
- **Ultralytics YOLO26 / YOLO11-pose** (current detector + pose feeder) — **AGPL-3.0**; the substrate the association layer rides on today is itself non-shippable. Swap to RF-DETR/D-FINE/YOLOX (detect) per DATASETS.md Phase-0.

**Forbidden to ship the WEIGHT (research-only training data — Apache/MIT code does NOT launder it):**
- **GroundingDINO** — O365 + GoldG + **Cap4M** (Cap4M never released). ([repo](https://github.com/IDEA-Research/GroundingDINO))
- **MM-Grounding-DINO** — **GoldG (Flickr30k non-commercial)** + **GRIT (LAION/COYO scrape)**. ([dataset prepare](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/dataset_prepare.md), [Flickr30k terms](https://github.com/BryanPlummer/flickr30k_entities))
- **OWLv2** — **WebLI** (proprietary research-only) + O365 ([academic-only](https://www.objects365.org/download.html)). ([paper](https://arxiv.org/html/2306.09683v3))
- **OmDet-Turbo** (released checkpoint) — O365 + **GoldG**. ([HF card](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf))
- **GLIP** — Cap4M + GoldG. **APE** — **SA-1B** (research-license) + GoldG + VG.
- **RF-DETR / D-FINE Objects365-pretrained checkpoints** — the strongest checkpoints inherit **Objects365's academic-only image license** ([paper Table 5](https://arxiv.org/abs/2511.09554), [O365 terms](https://www.objects365.org/download.html)). **Ship the COCO-only / clean-init checkpoint only**; D-FINE labels its COCO-only family explicitly, RF-DETR does not — prefer D-FINE/YOLOX for provenance clarity.
- **RF-DETR XL/2XL ("Plus")** — **PML-1.0** weights, not Apache. Use Nano/Small/Medium/Large only.

**Forbidden as a checkpoint, API-only, or non-commercial:**
- **T-Rex2** — IDEA non-commercial license. **DINO-X / GroundingDINO-1.5/1.6 Pro/Edge** — weights API-only (no local export).

**EVAL-ONLY datasets — bytes never enter a shipped weight:** PETS2006, i-LIDS/AVSS2007, ABODA (unlicensed), NTU RGB+D, and the full DATASETS.md research blocklist.

---

## What to prototype first

1. **Land the AGPL→Apache detector swap with v4** — finetune **D-FINE COCO-only** (or **YOLOX** if CoreML export reliability is the binding constraint) on COCO carried classes + Roboflow Pistols CC0. **Benchmark a *finetuned* export under the ort CoreML EP on this Mac BEFORE committing** — RF-DETR's finetuned variant silently falls back to CPU on Apple Silicon ([rf-detr #427](https://github.com/roboflow/rf-detr/issues/427)), and DETR deformable-attn / GridSample ops can partition the graph to CPU and blow the 410 ms budget. YOLOX (pure CNN) is the lower-risk Apple-GPU fallback; it needs a new decode branch for its raw-grid output shape.
2. **Widen the tracker discard filter** (`main.rs` L163/L187) to keep item class ids, and add the carry/abandon/remove geometric predicates next to `_refine_track_reason` / the `event_level`/`zone_line_reason` extension points (L685-700).
3. **Ship the VLM `carried_items` field + `training.carried_items` half** (Phase-1 flywheel) — gives end-to-end carried-item detection via the VLM-in-path immediately and starts the corpus.
4. **Wire the routing taxonomy** — new reasons into `lanes_triage.py` + `item_weights` in `sus_policy.json`; route `left_behind`/`removed`/`burglary_tool` to **escalate**, confirm `weapon_shape` direct-fire with the operator.
5. **Stand up Florence-2 (MIT) as a local offline labeler** — pending counsel sign-off on the LAION-sourced FLD-5B lineage — to distill carried-item labels and as a DGX-unreachable fallback confirm.
6. **(Optional, high-risk) one open-vocab on-device experiment = OmDet-Turbo** — only if funded for a full from-scratch retrain off the GoldG checkpoint AND a real CoreML/ort export spike for its Swin+CLIP+deformable-attn graph. Default to the closed-set path if the spike fails.

---

## Codebase-grounded corrections (post-review — these override the body above on conflict)

A review pass against the actual repo + CLAUDE.md found the synthesis made a few wrong assumptions. The **license verdicts and the overall architecture (closed detector + geometric association + VLM teacher) stand** — these correct the integration details and one routing posture:

1. **🔴 `weapon_shape` must default to `escalate(VLM)`, NOT direct-fire ALERT.** The table above has this backwards. Per CLAUDE.md (2026-06-17), **all** FP-prone behavioral/model-inferred reasons were deliberately demoted **alert → escalate**, and **only operator-drawn boundaries** (`zone_intrusion` / `line_cross` / `fence_cross`) may direct-fire WhatsApp. A model-inferred "weapon shape" is exactly the FP-prone non-boundary class the policy forbids from blind-firing (a misread umbrella/tool → a police call is the cardinal FP). **Default `weapon_shape` to escalate; the VLM confirms-and-fires.** Direct-fire is an explicit per-camera operator opt-in only, like the existing `cameras.scene_profile` override.

2. **The `training:` dual-JSON self-label half does NOT exist in this checkout.** Phase-1 says "add `training.carried_items` half" as if extending an existing field — but the dual-JSON flywheel (`training:{behavior,route,confidence}`) is **box-only** ("LIVE on the box" per CLAUDE.md); this checkout's `llm/prompts.py` has only `task_category`/`recommended_action`. **Prerequisite:** port the dual-JSON self-label mechanism first (or do this work on the box tree), then add `carried_items` to it.

3. **The tracker discard filter is in TWO places, not one** (the line numbers in step 2 above are also off). `detect()` filters non-person/non-vehicle classes in **both** decode branches — the YOLO26 NMS-free e2e head (`if cls != PERSON_CLASS && !is_vehicle(cls) { continue; }`, ~`main.rs:189`) **and** the YOLO11 raw-grid path (same line, ~`main.rs:214`). **Both** must widen to keep item class ids, or items are dropped on whichever model is loaded. (Mirror the v3 `VEHICLE_CLASSES` constant with an `ITEM_CLASSES` set.)

4. **`item_weights = base × scene_mult(zone_type) × hour_mult` is NOT config-only — it needs `sus.rs` work.** The real `SusPolicy` (verified in `service/src/sus.rs`) has only scalar `night_mult` / `zone_mult` and string-keyed `flag_weight` / `reason_weight` maps — there is **no** per-zone-type `scene_mult` and **no** `hour_mult`. So: adding `item_weights` as a string-keyed `reason_weight`-style map **is** config-only; but the **multiplier formula** (per-zone-type scene mult, hour mult) requires extending `SusPolicy` + `score()` in `sus.rs`. Phase 1 can ship items as plain `reason_weight` entries reusing the existing `night_mult`/`zone_mult`; defer the richer formula to a deliberate `sus.rs` change.

5. **D-FINE is also DETR-family (deformable attention) — same Apple-GPU export risk it's being recommended to avoid.** Treat "benchmark a *finetuned* CoreML/ort export on this Mac" as a **hard gate before D-FINE is ranked #1**, not a footnote. **YOLOX (pure CNN) is the only low-export-risk on-device runtime** — but it still needs a new raw-grid decode branch written into `detect()` (no one has written it). If the D-FINE export spike partitions to CPU, fall to YOLOX.

6. **Phase-2 needs a defined storage column for the per-track item label before `export_pose_dataset.py --carried-items` can work.** The exporter keys on `task_category`/`llm_verdict`/`sample_id` over `event_pose_frames`; there is **no** `carried_items` column in the corpus, the DB, or `event_pose_frames` today. Decide where the per-track item annotation persists (simplest: read it back out of `events.llm_raw_json->'training'->'carried_items'` joined by `event_id`, no migration) and name it in the export join.

7. **Bootstrap dataset licenses to re-confirm at acquisition** (asserted, not all re-fetched this pass): MEVA CC-BY + the `person_abandons_package`/`carries_heavy_object` class availability, Roboflow Pistols CC0 (already approved in DATASETS.md ✓), COCO annotations CC-BY. Any Roboflow Universe set stays unusable until its per-dataset license is confirmed.

**Net:** the v4 thesis is sound — **closed Apache detector (lands the AGPL-YOLO swap too) + free geometric carry/abandon/remove association over the existing tracker + VLM as open-vocab teacher/second-guard; no open-vocab grounder on the hot path; Florence-2 (MIT) as an offline labeler pending counsel.** Fix the weapon-routing posture and the dual-JSON prerequisite before building.
