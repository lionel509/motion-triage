# motion-triage v3 identity — bootstrap weights + VLM-reinforcement plan

> **Source:** license-verified research workflow (24 agents, adversarial license + weight-provenance check), 2026-06-17. Companion to [`DATASETS.md`](DATASETS.md) (which covers the *behavior* NN). This doc covers the **v3 identity/appearance signals** (clothing / body Re-ID / face) and how to bootstrap them on open assets, then reinforce with the existing VLM flywheel. A repo-grounded **verification addendum** at the bottom resolves the feasibility questions the research couldn't (it had no repo access).

## TL;DR (per-signal head-start pick + shippability)

1. **CLOTHING** — Keep the hand-rolled HSV histogram as the always-on free baseline; train an optional learned PA-100K attribute head on the **PA-100K dataset (CC-BY-4.0, ship-as-is data)** using the SOLIDER-PAR / OpenPAR *architecture only*. **The downloadable SOLIDER-PAR and OpenPAR checkpoints are NOT shippable** (LUPerson / CLIP-WIT backbone taint) → `finetune_only`.
2. **BODY RE-ID** — Bootstrap a fresh 512-d head on a **DINOv2 ViT-S/14 backbone (Apache-2.0, weights AND code) → `ship_as_is`**, trained from scratch on customer footage. The current OSNet path is only clean if it is the **ImageNet-pretrained** `osnet_x1_0` backbone (`conditional` — backbone + own-data finetune); any `*_market/*_msmt/*_duke` OSNet or any SOLIDER-REID checkpoint is `reference_only`.
3. **FACE** — **NO-GO by default.** The deployed `buffalo_l` / `antelopev2` weights are `forbidden` (WebFace600K / Glint360K research-only). The only legal ship path is buying the **InsightFace commercial license** AND clearing a BIPA/CUBI consent gate. Keep ArcFace architecture; use confident opportunistic face matches only as a high-purity *labeler for the body embedder*, do not retrain ArcFace.
4. **FLYWHEEL** — Cold-start is **tracklet self-supervision** (free, first-party, `ship_as_is`); reinforce with **confidence- + agreement-gated VLM same/different pair mining** (`finetune_only_on_own_data`, labels noisy — gate hard); promotion-gate every embedder exactly like the behavior NN.
5. **GOLDEN RULE** — MIT/Apache *code* never launders a *weight* trained on research-only data. Ship weights derived only from (a) a permissive self-supervised backbone on commercially-clean data (DINOv2-LVD142M ✓; LUPerson ✗) plus (b) the customer's own footage labeled by tracklets / gated-VLM / face.

---

## CLOTHING signal

**Recommendation: histogram stays; add one optional learned attribute head, trained yourself on PA-100K + VLM color labels.**

### Download (DATA, ship-as-is)
- **PA-100K dataset** — `https://github.com/xh-liu/HydraPlus-Net` (Google Drive linked from repo).
  - **Provenance / verdict:** `CC-BY-4.0` (HydraPlus-Net README: "The PA-100k dataset is under CC-BY 4.0 license"). Real outdoor-surveillance pedestrian crops, 26 attributes incl. upper/lower clothing color, bag, hat, glasses. **`conditional ship_as_is`**: commercially usable **but** CC-BY-4.0 attribution is **mandatory** (cite PA-100K / HydraPlus-Net, Liu et al., in shipped credits), and CC-BY clears *copyright* only — not the per-subject privacy/biometric rights (use-context caveat).
  - Source: `https://github.com/xh-liu/HydraPlus-Net` · `https://creativecommons.org/licenses/by/4.0/`

### Architecture (CODE clean; WEIGHTS NOT)
- **SOLIDER-PAR** (`tinyvision/SOLIDER` + `SOLIDER-PersonAttributeRecognition`, Apache-2.0 code) — **`finetune_only`, NOT `ship_as_is`.** The only downloadable PA-100K checkpoint is a *fused* `.pth` = Swin attribute head **+ a backbone self-supervised-pretrained on LUPerson** (explicitly "commercial usage is forbidden", YouTube-sourced). Self-supervised/unlabeled does **not** exempt the derived weight. Use the Apache **architecture**, put it on a clean backbone.
  - Sources: `https://github.com/tinyvision/SOLIDER` · `https://github.com/tinyvision/SOLIDER-PersonAttributeRecognition` · `https://github.com/DengpanFu/LUPerson`
- **OpenPAR / PromptPAR** (`Event-AHU/OpenPAR`, MIT code) — `finetune_only`. PA-100K head is data-clean, but (a) the backbone is CLIP ViT-L/14 trained on proprietary unreleased WIT-400M = unvetted provenance, and (b) the repo co-locates RAP/PETA checkpoints behind the *same* download links (shared BaiduYun code / Drive folder) — high risk of pulling a research-only RAP/PETA weight. Heavy + ONNX-hostile on the hot path. Architecture/teacher reference only.
  - Sources: `https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/README.md` · `https://arxiv.org/abs/2312.10692`

### Shippable build
Apache SOLIDER-PAR **Swin attribute architecture** → **commercially-clean backbone** (DINOv2-Apache or ImageNet/Apache) → train the 26-attribute head **only on PA-100K (CC-BY) + your own VLM-labeled crops**. A plain Swin classifier exports to clean ONNX (no prompt/VLM machinery), loads in the existing Rust `ort`/CoreML session.

### I/O + fusion
- Input: person crop (256×192 typical for PA-100K Swin). Output: 26 attribute logits (upper/lower color, bag, hat, glasses, age/gender).
- **Plug-in:** late-fuse as a per-keyframe RGB-crop branch **beside** the HSV histogram in `identity.rs`; structured color attributes upgrade the color signal AND power the "red-shirt-near-entrance-after-10pm" hybrid pgvector query. Run only on the **gated subset** of frames, not all 16.
- **Night/IR policy:** color collapses on IR sub-streams for BOTH histogram and learned head → auto-down-weight color (tie to the existing `context.night` flag); fall back to body Re-ID + face.
- **Forbidden CLIP variant:** `MetaCLIP` weights are CC-BY-NC = **forbidden** even though its code is MIT. For a holistic appearance embedding prefer an **Apache-2.0 OpenCLIP** card on a commercially-licensed open dataset; OpenAI CLIP is `conditional` (model card forbids surveillance + facial recognition; HF weights carry **no** license tag).

---

## BODY RE-ID signal

**Recommendation: replace the cold-start backbone with DINOv2-Apache; train the 512-d Re-ID head from scratch on customer data. Verify the currently-shipped OSNet is the ImageNet variant, not Market.**

### Download (head-start backbone, ship-as-is)
- **DINOv2 ViT-S/14 (Apache-2.0, code AND weights)** — `https://github.com/facebookresearch/dinov2`
  - Direct: `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth` (verified live, ~88 MB).
  - **Provenance / verdict:** Self-supervised on **LVD-142M** (142M *unlabeled* curated web images, no identity labels, faces blurred). Meta relicensed code+weights to **Apache-2.0 on 2023-08-31** for commercial use ("DINOv2 code and model weights are released under the Apache License 2.0"). → **`ship_as_is` backbone.** No Market/MSMT/Duke/MS1M ID-data taint — the structural reason it's clean where SOLIDER-REID/OSNet-Market are not.
  - **Carve-out:** the **biology variants (Cell-DINO / Channel-Adaptive-DINO / XRay-DINO) are FAIR-Noncommercial → do NOT pull those.** Pin the exact `.pth` hash.
  - Sources: `https://github.com/facebookresearch/dinov2/blob/main/LICENSE` · `https://huggingface.co/facebook/dinov2-base`
  - ⚠️ **Accuracy unproven for person Re-ID** — license is confirmed, but there is no person-Re-ID parity number in evidence (only a *wildlife* zero-shot figure). **Benchmark before adopting** (see addendum).

### The current OSNet slot — verify provenance now
- **`osnet_x1_0` ImageNet-pretrained backbone** (`KaiyangZhou/deep-person-reid`, MIT code) — separate "ImageNet pretrained models" zoo section (`<model>_imagenet.pth`, Drive `1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY`; HF `kaiyangzhou/osnet`). **`conditional`** (backbone + own-data finetune): plain-ImageNet-classification-pretrained, NOT Re-ID-tainted — but ImageNet is itself NC-research at the dataset level, so ship as "backbone + own-data finetune," not the raw binary.
  - Sources: `https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html` · `https://image-net.org/accessagreement`
- **`osnet_x1_0_market1501 / _msmt17 / _duke` → `reference_only`. DO NOT SHIP** (weights inherit Market/MSMT research-only; DukeMTMC research-only **and RETRACTED**).
  - **🔴 ACTION ITEM:** confirm v3's deployed OSNet ONNX was baked from the **ImageNet** weight, not a Market checkpoint. If it came from `*_market/*_msmt/*_duke`, that is **a live shipped-weight licensing exposure** — re-bake from the ImageNet weight (same arch, same 512-d, drop-in).

### SOLIDER-REID / TransReID-SSL / FastReID / CLIP-ReID — all `finetune_only` / `reference_only`
**Every** downloadable Re-ID checkpoint in the OSS zoo is tainted (LUPerson SSL pretrain and/or Market/MSMT/Duke heads). Use only as **frameworks/architectures** + internal eval baselines.

### Shippable build + fusion
- Attach a **512-d Re-ID projection head + BNNeck** to DINOv2 (or the ImageNet-OSNet backbone) → train the head (optionally LoRA/last-blocks) on tracklet + gated-VLM pairs from the customer's own cameras → export to ONNX/CoreML for the **`REID_MODEL`** env slot.
- I/O matches `identity.rs ReidCtx`: **1×3×256×128, ImageNet-norm (0.485/0.456/0.406, 0.229/0.224/0.225), L2-normalized output** (confirmed — see addendum).
- **Latency gate:** a plain ViT is heavier than OSNet on the Apple GPU. **Benchmark ViT-S/14 (frozen + small head) vs OSNet on your own association task AND CoreML latency before replacing OSNet.** Likely: ViT-S/14 (or a distilled CNN student) on-device, full ViT as the DGX-side teacher.
- **Do not re-taint on finetune:** never mix Market/MSMT/Duke/LTCC into the finetune set.

---

## FACE signal

**Recommendation: NO-GO by default. Architecture stays; weights do not ship for free; lean on body + clothing.**

### Currently-deployed weights — `forbidden`
- **`buffalo_l` (RetinaFace + `w600k_r50`, 512-d)** — recognition head trained on **WebFace600K** (subset of WebFace260M = academic-research-only, signed agreement, "not allowed to use ... for any commercial purposes"). InsightFace model-zoo README: "ALL models are available for non-commercial research purposes only." → `forbidden`.
- **`antelopev2` (`glintr100`, R100, 512-d)** — same verdict; head trained on **Glint360K** (non-commercial research, overlaps MS-Celeb identities). r50→r100 changes nothing for licensing.
  - Sources: `https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md` · `https://www.face-benchmark.org/download.html`

### Permissive-code alternatives — also NOT shippable
- **AdaFace (CVLFace, MIT)** and **MagFace (Apache-2.0)** — `finetune_only`. Every checkpoint is MS1MV2/MS1MV3 (RETRACTED MS-Celeb-1M), WebFace4M/12M, CASIA-WebFace, or VGGFace2 — all research-only. **Zero** clean-data face checkpoint exists in either zoo. AdaFace's quality-adaptive margin + MagFace's magnitude=quality abstain are useful **architecture references** for tiny surveillance faces — retrain on own/consented data only.
  - Sources: `https://github.com/IrvingMeng/MagFace/blob/main/LICENSE` · `https://exposing.ai/msceleb/`

### The ONE legal ship path for face weights
- **InsightFace commercial license** for `buffalo_l` / `antelopev2` / `buffalo_s` / `buffalo_m` — `recognition-oss-pack@insightface.ai`, `https://www.insightface.ai/solutions/face-recognition-licensing`. Same ONNX artifacts (zero code change to v3); the license is the only thing that changes. **Get in writing that the EULA clears the upstream WebFace600K / Glint360K data license**, not just InsightFace's packaging.

### FACE GO/NO-GO (legal overlay, distinct from OSS license)
**Default = NO-GO.** A clean weight still doesn't clear the **biometric-privacy product gate**:
- **Illinois BIPA** — private right of action, $1k/$5k per violation; Clearview settled $51.75M. A 512-d faceprint embedding *is* a biometric identifier under BIPA.
- **Texas CUBI** — drove >$2.77B in Meta + Google settlements. **Washington** — consent before commercial enrollment.

**Enable face ONLY if the customer (a) buys the InsightFace commercial license AND (b) accepts a per-site written opt-in/consent + retention/destruction schedule.** Given tiny surveillance faces abstain most of the time, **quantify the lift of face over body+clothing on real footage before paying or taking on the exposure.**

### Face's role in the flywheel (no retraining)
Keep `FaceCtx` (112×112 ArcFace from pose eye-keypoints) frozen. Use **confident** face matches as the **highest-purity labeler for the BODY embedder's hardest case** — same person, different clothes across visits (exactly the pairs tracklets can't give and the VLM is unreliable on): face=SAME → gold cross-outfit body positive; face=DIFFERENT → gold hard negative. **Do NOT finetune a face head on customer faces.**

---

## DATASETS — USE vs EVAL-ONLY

### USE (commercial-clean — can bootstrap / train shippable weights)

| Dataset | License | URL | What it's for |
|---|---|---|---|
| **MEVA** | CC-BY-4.0 (consented actors, IRB) — *re-confirm primary source* | `https://mevadata.org/` · `s3://mevadata-public-01` | **Cornerstone trainable set.** 29 cameras, 100+ consented actors, person-person crossing. Crop crossing tubes to bootstrap the association head. Attribution mandatory. *Asterisk:* `drop-4-hadcv22` subset may not train NIST ActEV submissions (benchmark clause, not a commercial limit). |
| **PA-100K** | CC-BY-4.0 (attribution mandatory) | `https://github.com/xh-liu/HydraPlus-Net` | The one clean commercial attribute set. Train your clothing-attribute head on it + own VLM labels. Never ship someone else's PA-100K checkpoint without re-checking *its* backbone. |
| **Street Scene** | CC-BY-SA-4.0 (Zenodo) | `https://zenodo.org/records/10870472` | Crossing/occlusion + negatives. **`conditional`:** never redistribute a modified copy; keep attribution; keep trained weights closed. |
| **DINOv2 weights** | Apache-2.0 (code + standard ViT weights) | `https://github.com/facebookresearch/dinov2` | Clean self-supervised body-Re-ID backbone. Avoid the FAIR-NC biology variants. |
| **Tracklet self-supervision** (recipe) | MIT (`pytorch-metric-learning`) / BSD-2 (`SupContrast`) | `https://github.com/KevinMusgrave/pytorch-metric-learning` | First-party-only contrastive signal — the cleanest cold-start; loss code ships no weights. |

> **HR-ShanghaiTech — REMOVED from USE / correct in [`DATASETS.md`](DATASETS.md).** The behavior-NN doc lists it BSD-2 `ship_as_is`; the adversarial check **refuted** this: the `svip-lab` LICENSE 404s, the underlying ShanghaiTech Campus footage is research-only no-consent real surveillance video of identifiable people, and the HR split is `RomeroBarata/skeleton_based_anomaly_detection` (not svip-lab). Re-extracting skeletons does not launder it. → **`reference_only`.**

### EVAL-ONLY (validate v3 keeps two crossers as two tracks — report HOTA/IDF1/IDSW — but never ship derived weights)

| Dataset | License | URL | Validates |
|---|---|---|---|
| **PersonPath22** | CC-BY-NC-4.0 | `https://github.com/amazon-science/tracking-dataset` | **Primary held-out eval** — static cameras, most surveillance-realistic crossing/occlusion. |
| **MOT20** (+ MOT17) | CC-BY-NC-SA-3.0 | `https://motchallenge.net/data/MOT20/` | Dense-crowd occlusion; report IDSW before/after the v3 layer. |
| **DanceTrack** | Repo MIT, video NC | `https://dancetrack.github.io/` | Uniform-appearance crossing — where color + body-ReID collapse and v3 must lean on motion. |
| **SportsMOT** | CC-BY-NC-4.0 | `https://github.com/MCG-NJU/SportsMOT` | Uniform-appearance crossing. |
| **JRDB / JRDB-Pose** | CC-BY-NC-SA-3.0 | `https://jrdb.stanford.edu/` | Occlusion-aware association + pose-through-occlusion. |
| **LTCC / PRCC / DeepChange / Celeb-reID** | academic/signed/web-scraped | (per-set) | **Reference numbers only** to justify the "same-visit only" premise (body-ReID ~20% mAP on clothing change). |

> NC permits **internal** benchmarking, forbids commercial redistribution / shipped weights. Standard held-out harness = **PersonPath22 + MOT20 + DanceTrack**.
> **Forbidden even for eval:** Occluded-Duke / P-DukeMTMC (RETRACTED DukeMTMC), TAO (no unified license), MOTSynth (GTA-V EULA), WILDTRACK (consent flag).

---

## VLM-REINFORCEMENT RECIPE — cold-start sequence wired to the existing flywheel

The flywheel already exists (`training/vlm_label.py`, `autotrain.py`, `continuous-train` daemon, `export_pose_dataset.py`, Postgres `NOTIFY`, `sample_id = sha256("{salt}:{event_id}")[:16]`). **Hook in, don't redesign.**

### DAY 0 — Tier-1 tracklet self-supervision (ship immediately, fully clean)
1. **Backbone:** DINOv2 ViT-S/14 (Apache) or ImageNet-OSNet backbone — no identity supervision baked in.
2. **Free labels from your own footage:** within one clip, **same track = positive**; **two spatially-disjoint simultaneous tracks (boxes don't overlap) = perfectly-correct hard negative**. Mine negatives only when the existing `strong_iou` (0.50) gate says boxes are disjoint, so partial occlusion never creates a false negative.
3. **New export — `export_reid_pairs.py`** (beside `export_pose_dataset.py`): emit `pairs.jsonl` keyed by the **same** `sample_id`, rows `{sample_id, crop_a, crop_b, label: same|diff, label_source: tracklet, confidence}`. Crops come from the `event_pose_frames` track boxes you already store (✓ confirmed — addendum).
4. **New train target — `train_reid.py`:** a SupCon/triplet contrastive head (MIT `pytorch-metric-learning` / BSD-2 `SupContrast`) → **512-d ONNX matching `identity.rs ReidCtx`**.

### DAYS 1–N — Tier-2 gated VLM pair mining
5. **Extend `vlm_label.py` with `add_pair_judgment(crop_a, crop_b)`** reusing the DGX path + `VLM_ATTRIBUTES_ENABLED` color output. Returns `{same: true|false|unsure, confidence, matched_on, reason}`. Send the two crops as **separate images** (Qwen3-VL native multi-image).
6. **MANDATORY adversarial gating** (LVLMs give "catastrophic answers" at fine-grained matching → noisy weak labeler, never a sole oracle):
   - Keep a VLM pair only if **confidence ≥ a rising floor** AND it **agrees with the cheap color-histogram distance** from `identity.rs` AND a **temporal/geometric feasibility check** (reuse `cross_camera.py` topology windows).
   - **Prefer VLM negatives over positives.** **Tracklet positives always override** a conflicting VLM label. On disagreement → `unsure`, never train (fail-toward-not-merging, same bias as `identity_match` → `Unknown`).
7. **Face as gold labeler:** confident `FaceCtx` matches inject the hardest cross-outfit pairs with `label_source: face`.
8. Append to `pairs.jsonl`; fire `NOTIFY` (new `training_reid_pair_added`) on commit so `continuous-train` debounces (15 s) and runs the reid round.

### Ongoing — Tier-3 pseudo-label-cleaned reinforcement
9. Nightly: camera-aware DBSCAN cluster on-camera embeddings → **soft** pseudo-IDs (MMT-style soft labels beat hard cluster labels) → contrastive refine. Reference impl **OpenUnReID (Apache)** — never vendor AGPL BoxMOT. Split clusters by camera for purity.
10. **Promotion gate — extend `autotrain.py` with `--task reid`:** metric = **same-visit rank-1 / mAP on a held-out tracklet-verified split**. On promote, **re-tune `identity.rs` `body_same`/`body_diff` thresholds**, archive a checkpoint, reload the spike (`pkill -f "spike serve"`). Push every promote to GitHub like the behavior NN. *(Cold-start has no champion → set an absolute acceptance bar, not "beats champion.")*

> **Privacy:** cross-event person linkage is more sensitive than pose skeletons. The pseudonymized `sample_id` keying helps; decide a retention/consent stance for stored crop pairs.

---

## What's NEW vs already in DATASETS.md / reid_upgrades/README.md

| Already chosen (keep) | What this report ADDS / CORRECTS |
|---|---|
| SOLIDER-REID as the MIT OSNet replacement | **CORRECTION:** released SOLIDER-REID **and** raw SOLIDER backbone **weights are NOT shippable** (LUPerson research-only + Market/MSMT heads). Use the architecture only; **swap the LUPerson backbone for Apache DINOv2.** The *code* is safe, the *weights* are not. |
| OpenPAR / PA-100K attribute path | PA-100K **stays** (CC-BY, attribution). NEW: OpenPAR/SOLIDER-PAR **checkpoints** are `finetune_only`; RAP/PETA checkpoints co-located in the same download = trap. |
| HR-ShanghaiTech as "best clean BSD skeleton corpus" | **CORRECTION — remove BSD/ship_as_is + the svip-lab URL.** Research-only no-consent footage → `reference_only`. |
| (new) | **DINOv2-Apache** as the shippable self-supervised body backbone. |
| (new) | **MEVA** as the cornerstone *trainable* crossing/occlusion substrate (CC-BY-4.0). |
| (new) | The **tracklet + gated-VLM-pair** flywheel recipe, the `export_reid_pairs.py` / `train_reid.py` / `autotrain.py --task reid` hooks, and the **FACE BIPA/CUBI legal gate**. |

---

## PHASED ORDER (fastest correct head start)

1. **Audit the deployed OSNet weight TODAY** — confirm ImageNet `osnet_x1_0`, not Market/MSMT/Duke. If tainted, re-bake from the ImageNet weight. *(Closes a possible live exposure.)*
2. **Ship Tier-1 tracklet self-supervision** on the current backbone — free, clean, works from event 1. Add `export_reid_pairs.py` + `train_reid.py`.
3. **Stand up DINOv2 ViT-S/14**, benchmark vs OSNet on your own association + CoreML latency. Adopt only if it wins.
4. **Turn on gated VLM pair mining** (`add_pair_judgment` + the 3-way gate). Negatives first.
5. **Wire `autotrain.py --task reid`**; let `continuous-train` reinforce nightly.
6. **Clothing learned head** (SOLIDER-PAR arch on DINOv2 backbone, PA-100K + VLM color labels) — only if same-visit failures justify it; histogram stays the free baseline.
7. **Face stays OFF** until a customer buys the InsightFace commercial license AND clears BIPA/CUBI; quantify face's lift over body+clothing first.

---

## LICENSE TRAPS / DO-NOT-SHIP (weight-vs-data reason)

**The trap:** MIT/Apache **code** never cures a **weight** whose training data is research-only. Self-supervised/unlabeled pretraining does **not** exempt the weight (LUPerson, ImageNet, WIT-400M all still bind).

- **SOLIDER-REID / SOLIDER backbone** → `finetune_only`/`reference_only`. Backbone SSL on **LUPerson** (commercial forbidden); heads on **Market/MSMT**.
- **TransReID-SSL** → `finetune_only`. **LUPerson** SSL pretrain.
- **OSNet `*_market/*_msmt/*_duke`** → `reference_only`. Market/MSMT research-only; **DukeMTMC RETRACTED**.
- **OSNet `*_imagenet`** → `conditional`. ImageNet NC-research at source; ship as backbone + own-data finetune.
- **FastReID / Bag-of-Tricks / LightMBN / CLIP-ReID** → `reference_only`/`finetune_only`. All weights Market/MSMT/Duke/VeRi.
- **SOLIDER-PAR / OpenPAR (RAP, PETA)** → `finetune_only`/`forbidden`. LUPerson or CLIP-WIT backbone; RAP signed-agreement, PETA aggregate.
- **MetaCLIP weights** → `forbidden`. CC-BY-NC (code MIT). **OpenAI CLIP weights** → `conditional`/avoid (no license tag; model card bars surveillance + face recognition).
- **buffalo_l / antelopev2** → `forbidden`. WebFace600K / Glint360K research-only. Legal path = paid InsightFace license + BIPA gate.
- **AdaFace / MagFace** → `finetune_only`. MS1MV2/3 (RETRACTED), WebFace4M/12M, CASIA-WebFace, VGGFace2.
- **DigiFace-1M** → `forbidden` (Microsoft NC-research) — promising synthetic direction if a commercial license appears.
- **HR-ShanghaiTech**, **Occluded-Duke / P-DukeMTMC**, **MOTSynth**, **TAO**, **WILDTRACK** → `reference_only`/`forbidden` (see datasets).
- **LTCC / PRCC / DeepChange / Celeb-reID** → `forbidden` for training; reference numbers only.

---

## Local verification addendum (repo-grounded — resolves the research's open feasibility items)

The research agents had no repo access. Checked against this codebase 2026-06-17:

- ✅ **Tier-1 self-supervision is feasible as written.** `event_pose_frames` stores a per-frame `box` per detection, and `scripts/export_pose_dataset.py` already joins `p.box` + `p.keypoints` + `e.llm_verdict` + `task_category` and groups them into tracks via `dashboard.workers.pose_logger.greedy_iou_tracks`. `export_reid_pairs.py` is a thin sibling reusing the same boxes + tracker (same track = positive; disjoint simultaneous tracks = negative). No schema migration needed.
- ✅ **`identity.rs` tensor contracts confirmed** (this is the v3 code we wrote): `ReidCtx` = **1×3×256×128, ImageNet mean/std, L2-normalized** output; `FaceCtx` = **112×112**, RGB `(x−127.5)/127.5`, aligned from pose eye-keypoints. ONNX exports must match these exactly to drop into `REID_MODEL`/`FACE_MODEL`.
- ⚠️ **Deployed labeler VLM (output-license check before relying on it).** This checkout pins `LLM_BACKEND_PRIMARY=qwen` (Qwen3-VL-30B); the box runs Nemotron-Omni. Qwen3 dense text models are Apache-2.0, but **verify the exact Qwen3-VL LICENSE / Nemotron Open Model License permit using outputs to train a commercial model** before scaling Tier-2. Low risk (own deployment, own footage, swappable labeler) but confirm. `https://github.com/QwenLM/Qwen3-VL/blob/main/LICENSE`
- ⚠️ **`add_pair_judgment` needs a new non-event call shape.** Existing `vlm_label.py` is event-keyed (one event + its keyframes). A cross-event two-crop same/different judgment is a new request shape + response schema — small adapter work, not free; spec it before wiring Tier-2.
- ⚠️ **Open verification items inherited from the research** (license verdicts are primary-source-confirmed; these are accuracy/liveness items): DINOv2 person-Re-ID accuracy is **unproven** — benchmark before adopting; MEVA + PA-100K license lines + download liveness should be re-fetched at acquisition time; the "LVLMs give catastrophic fine-grained answers" claim (justifying the hard gate) is from arXiv 2501.18698 — directionally safe regardless.
