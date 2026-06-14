# NN Bootstrap — Pretrained Model & Dataset Acquisition + Integration Plan

**Source:** license-verified research workflow (15 agents, adversarial license check), 2026-06-14.
**Scope:** license-clean external assets to bootstrap our skeleton behavior NN
(TemporalConvNet over `[T=16, V=17 COCO, C=6]`).
**Default posture:** the **shipped NN stays skeleton-based and trained only on
commercially-clean data**; RGB models are teachers/auto-labelers, *not* bundled artifacts.

> **🚨 Phase-0 gate — Ultralytics is AGPL-3.0.** Our current `yolo26n-pose`, `yolo26m`
> (triage detector), and `yolo26n` (prefilter) are all Ultralytics = **AGPL-3.0**, which
> network-copyleft-taints a closed commercial product. Clean Apache/permissive
> replacements exist (pose: **ViTPose++ / SOLIDER-HumanPose**; detect: **RF-DETR / D-FINE /
> YOLOX**). Decide whether AGPL is acceptable before scaling the corpus — all downstream
> skeleton provenance depends on the chosen pose feeder.

> **Three-way license split applies to everything:** (a) *code* license, (b) *weight*
> license (= derived from *training-data* license), (c) *dataset* license. An Apache/MIT
> codebase with **NTU/Kinetics-trained checkpoints** is safe as code, but an NTU-trained
> `.pth` is **not shippable**. Pin every checkpoint to its training-data provenance.

---

## 1. Recommended stack by head / class-family

| Head / class-family | Best license-clean asset | License | Integration |
|---|---|---|---|
| **Pose feeder** (upstream of all) | **SOLIDER-HumanPose** (offline GT) / **ViTPose++** (live drop-in) — COCO-17 | Apache-2.0 | **use_as_is** — same 17-kp output, no tensor change. Clean replacement for AGPL `yolo26n-pose`. Benchmark Mac-GPU latency (both heavier). |
| **Skeleton-action backbone/teacher** (12 activities) | **PYSKL** (ST-GCN++ / PoseConv3D), HRNet COCO-17 | Apache-2.0 (**code only**) | **distill** — teacher to pseudo-label; GCN/3D-heatmap won't load into the TCN. **Don't ship NTU checkpoints.** |
| (architecture reference) | **MS-G3D** (MIT), **ST-GCN** (BSD-2) | MIT / BSD-2 | **distill/reference** — MIT lets us vendor & relicense the graph/temporal-conv layers. MS-G3D graph must remap OpenPose-18→COCO-17. |
| **12 activities — clean DATA** | **Kinetics-400/700** (re-extract skeletons) | CC-BY-4.0 | **data_only** — running/climbing/throwing. Key labels to retained clips (YouTube link-rot). |
| **RGB teacher** (auto-label + soft labels) | **InternVideo2** (1B + S/B/L) | Apache-2.0 | **distill** — frozen soft-label teacher over per-event RGB clips → distill into skeleton TCN. |
| **Streaming RGB teacher** (edge-shaped) | **MoViNet** (A0–A5 stream) | Apache code + CC-BY (K600) | **distill** — causal streaming, closest to on-device. |
| **Lightweight RGB backbones** | **PyTorchVideo / PySlowFast** (X3D, SlowFast, MViT) | Apache + CC-BY (K400) | **distill** / optional finetune — pin to Kinetics/AVA checkpoints; reject NTU/SSv2. |
| **Fall** | **GMDCSA-24** (MIT) + **CAUCAFall** (CC-BY) + **UP-Fall** (verify CC-BY) | MIT / CC-BY | **data_only** — pose → `[16,17,6]`, `fall` positives + ADL negatives. GMDCSA night/low-res = ideal hard positives. |
| **Fight / violence** | **Surveillance Camera Fight Dataset** (seymanurakti) | MIT | **data_only** — only cleanly-MIT surveillance-POV fight set (~300 clips). **Add pair-interaction head** (inter-skeleton distance + closing velocity). |
| **Fight/throw/fall/climb — localized** | **AVA / AVA-Kinetics** | CC-BY-4.0 | **data_only** — person-localized labels K400 lacks; crop per-person tubes → pose. |
| **21-behavior trajectory head** | **MEVA** (anchor) + **HR-ShanghaiTech** (BSD-2) + **Street Scene/MERL** (CC-BY-SA) | CC-BY / BSD-2 / CC-BY-SA | **data_only** — MEVA 37 ActEV types → intrusion/loiter/enters-vehicle/door/abandon; HR-ShanghaiTech = best clean **skeleton** anomaly corpus. Re-extract skeletons (don't reuse AlphaPose order); label-map config onto our 21/12. |
| **Unsupervised abnormal-trajectory** | **STG-NF idea** | CC-BY-**NC** | **reference only** — clean-room reimplement the flow-over-pose scorer; do NOT import code/weights. |
| **Weapon (appearance detector)** | **RF-DETR core N/S/M/L** / **D-FINE** / **YOLOX** | Apache-2.0 | **finetune** — no permissive pretrained weapon model; train ourselves. **Avoid RF-DETR XL/2XL (PML-1.0).** Separate 2D detector beside pose, feeds `weapon_present`. |
| **Weapon — clean DATA** | **Roboflow Pistols** (Granada) | **CC0** | **data_only** — cleanest, zero attribution. Finetune RF-DETR-S/D-FINE-S/YOLOX-s → ONNX/CoreML on `:8091`. |
| **Weapon-operating pose micro-behavior** | **MBE-2023** (blueprint, no weights) | paper CC-BY | **blueprint** — label `weapon-operating` in our corpus, train as a pose-micro head. |
| **Forensic attributes** | **SOLIDER-PAR** / **OpenPAR/PromptPAR** | Apache / MIT (code) | **finetune** — train attr head **only on PA-100K** (no NC data). Late-fused RGB branch. |
| **Attributes — clean DATA** | **PA-100K** (100K crops, 26 attrs) | CC-BY-4.0 | **data_only** — the **one** clean commercial PAR dataset. Attribution mandatory. |
| **Re-ID** | **SOLIDER-REID** + SOLIDER backbone | MIT + Apache | **finetune** on our own IDs (not Market/MSMT — research-only). Plugs into `reid_upgrades/cross_camera.py`. |
| **Scene (20 types, conditioning input)** | **Places365-CNNs** (ResNet50) | MIT code + CC-BY weights | **finetune** — 365→20 map → scene-conditioning vector. CC-BY attribution binding. |
| **Negatives** | **MEVA** + **Street Scene** + ADL from fall sets | CC-BY / CC-BY-SA / MIT | **data_only** — dog/animal, vehicle_passing, U-turn, walk_by, sitting. |
| **Vehicle type / abandoned-removed** | **MEVA** (+ Street Scene) | CC-BY-4.0 | **data_only** — enters/exits-vehicle + sparse abandonment; keep on our rules + corpus. |

**Heads with NO clean external source — keep fully in-house (our rules + labeled corpus):**
the trajectory/relational behaviors — `loitering`, `pacing`, `u_turn`, `direction_change`,
`intrusion`, `line_cross`, `fence_cross`, `converging`, `multi_person`-as-relation,
`entered_vehicle`, `arrived_by_vehicle`. No clip-action corpus covers these; MEVA/ShanghaiTech/
Street Scene only seed a subset.

---

## 2. DO NOT USE — license blocklist

**Copyleft / AGPL taint (forbidden in a closed product):**
- **Ultralytics YOLO / YOLOv7/8/11-pose / YOLOE / BoxMOT** — **AGPL-3.0**. Includes our current `yolo26n-pose`, `yolo26m`, `yolo26n`. Swap to ViTPose++/SOLIDER (pose), RF-DETR/D-FINE/YOLOX (detect).
- **Fight-Detection-Yolov8-Pose** + all YOLOv8-pose fight repos — AGPL backbone.
- **SPHAR** — GPL-v3 scripts + mixed per-source video licenses; only the CC-BY (MEVA)/MIT (UT-Interaction)/VIRAT subset, never whole.

**Non-commercial CODE *and* weights (repo itself blocks commercial use):**
- **CTR-GCN**, **2s-AGCN** — CC-BY-NC repos (reimplement layers from paper / use PYSKL).
- **STG-NF** — CC-BY-NC code+weights (idea only).
- **VideoMAE V2** — code MIT but **weights CC-BY-NC** → NC taints distillation; **not even as a teacher.**

**Non-commercial / research-only DATASETS (eval/reference only, never shipped weights):**
- **NTU RGB+D 60/120** — academic-only + forbids derived datasets. **The central trap:** any NTU-trained PoseC3D/ST-GCN++ checkpoint and any NTU-derived skeleton are disqualified.
- **Toyota Smarthome**, **UCF-Crime/HR-Crime**, **CUHK Avenue/HR-Avenue**, **NWPU Campus**, **UBnormal** (CC-BY-NC-ND), **OmniFall** (NC-SA), **URFD/UR-Fall** (NC-SA), **Le2i/IMVIA** (unverified — the canonical fall trap), **GajuuzZ/Human-Falling** (no license + Le2i-trained), **RWF-2000**, **RLVS** (+ "Skeleton-RLVS"), **AIRTLab**, **DVD** (NC-ND), **XD-Violence/Hockey/Movies** (copyrighted source), **UPAR** (NC-SA), **Rethinking_of_PAR** (no license), **Mendeley firearm** (NC), **USRT** (NC), **Monash Guns** (dataset no commercial license).

**Ethical-source / surveillance-forbidding:** any Re-ID under **KPR / Hippocratic** licenses. SOLIDER (MIT/Apache) is the safe Re-ID path.

**Conditional / verify-before-use:**
- **VIRAT Ground** — gated agreement, commercial unstated → unclear (eval only; annotations CC-BY).
- **Sohas weapon** — CC-BY-SA (ShareAlike reach over model output unsettled) → legal sign-off; prefer CC0 Pistols.
- **CCTV-Gun** — Apache code, weights inherit NC; retrain on CC0/CC-BY.
- **RF-DETR XL/2XL** — PML-1.0 (use core N/S/M/L only).
- **MMAction2/PYSKL checkpoints** — Apache code, pin to Kinetics/AVA, reject NTU/SSv2.
- **MotionBERT** — Apache code, action checkpoint NTU-finetuned; only the SSL encoder is clean (remap H36M→COCO-17, retrain head).

---

## 3. Phased acquisition order

**Phase 0 — Pose-feeder decision (gate for everything).** Decide whether AGPL `yolo26n-pose`
is acceptable. If not → stand up **ViTPose++** (live) / **SOLIDER-HumanPose** (offline GT), both
Apache, same COCO-17, no tensor change. All downstream extraction uses the chosen feeder.

**Phase 1 — Never-miss threats that map cleanly to the pose NN:**
1. **Fall** — GMDCSA-24 (MIT) → CAUCAFall (CC-BY) → UP-Fall. Immediate `fall` head + ADL negatives.
2. **Fight** — Surveillance Camera Fight (MIT) + AVA/AVA-Kinetics (CC-BY); add pair-interaction head.
3. **Skeleton-action backbone + teacher** — PYSKL (Apache code) + Kinetics-400 (CC-BY) re-extracted → 12-activity head.

**Phase 2 — Surveillance trajectory head + negatives:**
4. **MEVA** (CC-BY) — anchor 21-behavior corpus + negatives + vehicle/abandon seed.
5. **HR-ShanghaiTech** (BSD-2, from svip-lab) — clean skeleton anomaly corpus.
6. **Street Scene/MERL** (CC-BY-SA) — street geometry, animal/vehicle negatives.

**Phase 3 — RGB distillation flywheel:**
7. **InternVideo2** (Apache) teacher; **MoViNet** (Apache+CC-BY) streaming; **PyTorchVideo X3D/SlowFast** (Apache+CC-BY) lightweight — over the per-event RGB clips we already retain.

**Phase 4 — Weapon, attributes, scene (separate non-skeleton heads):**
8. **Weapon** — Roboflow Pistols (CC0) + RF-DETR/D-FINE/YOLOX (Apache) → `:8091`. + skeleton-native `weapon-operating` head.
9. **Attributes** — SOLIDER-PAR trained **only on PA-100K** (CC-BY).
10. **Re-ID** — SOLIDER-REID finetuned on our own IDs → `cross_camera.py`.
11. **Scene** — Places365 → 20-scene conditioning vector.

**Always in-house:** the relational/trajectory behaviors (loitering, pacing, u_turn, intrusion, line_cross, fence_cross, converging, vehicle arrival/entry).

---

## 4. Integration notes

**Skeleton-action models → our `[T=16,V=17,C=6]` TCN.** PYSKL/MS-G3D/ST-GCN tops are GCN/3D-heatmap
— their `.pth` **won't load** into the TCN; they're pseudo-label **teachers** + architecture refs.
PYSKL's HRNet/SOLIDER pipeline is already COCO-17 (no remap); MS-G3D (OpenPose-18) and MotionBERT
(H36M) must remap to COCO-17. Uniform pipeline per dataset: **our pose feeder → window `[16,17,6]` →
project native taxonomy onto our 21/12 heads via a label-map config → merge into corpus.** Never
reuse a dataset's bundled keypoint ordering — re-extract for parity + clean provenance.

**RGB models → teacher/distillation (shipped NN stays skeleton-based).** InternVideo2/MoViNet/X3D run
as frozen soft-label teachers over the per-event RGB clip, keyed to the same event as the pose
trajectory; soft labels distill into the skeleton heads. Only our own TCN weights ship.
**VideoMAE V2 excluded** (NC weights taint output).

**Datasets → VLM-teacher labeling + supervised fine-tuning.** CC-BY/CC0/MIT/BSD sets feed (1) direct
supervised fine-tuning after pose extraction and (2) the existing VLM-distillation flywheel.
**NC-derived pseudo-labeling is a gray area** — before using any model as an auto-labeler, retrain
its backbone on the CC-BY dataset (e.g., SOLIDER-PAR on PA-100K); the NTU-distillation path must end
with the TCN **re-fit on clean data** so shipped weights aren't NTU-derived.

**Non-skeleton heads = parallel branches, not into the TCN.** Weapon detector (RF-DETR/D-FINE/YOLOX)
= separate 2D detector on `:8091` → `weapon_present` flag. Attributes (SOLIDER-PAR) + Re-ID
(SOLIDER-REID) = late-fused RGB-crop branches (shared Swin backbone). Scene (Places365) = the 20-d
scene-conditioning vector. SOLIDER-HumanPose is the one RGB asset that *produces* the skeleton.

**STG-NF complement:** clean-room reimplement the normalizing-flow-over-pose scorer for an
unsupervised "how abnormal" signal alongside the supervised TCN (loitering/intrusion without labels).

**Attribution ledger (carry in shipped credits):** Kinetics (Google), AVA (Google), MEVA, Street
Scene (Jones & Ramachandra / MERL), CAUCAFall, UP-Fall, **PA-100K**, **Places365** (link the paper).
CC0 (Pistols) + BSD-2 (ShanghaiTech — retain notice) = no attribution burden. Street Scene CC-BY-SA:
do **not** redistribute a modified dataset copy; the trained model is unaffected.
