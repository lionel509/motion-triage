# v3 identity — replacing the shelved face signal

> Status: synthesis for motion-triage v3. Face is shelved (BIPA/CUBI exposure + every OSS face weight is research-only). This document decides what license-clean, privacy-lighter signal(s) replace or augment it, and — more importantly — separates the signals that actually fix the v3 merge bug from the harder, different need of restoring face's cross-outfit role.
>
> Not legal advice. Every "biometric" determination below is fact- and jurisdiction-specific; route any deployed identity signal through counsel exactly as face was.

---

## TL;DR — ranked, by impact-per-effort

The single most important framing: **face was the CROSS-OUTFIT / long-horizon anchor.** The v3 bug is **SAME-VISIT, SAME-OUTFIT** (two people in one 60 s clip). So face was *opportunistic, never load-bearing* for this bug. The merge fix is **motion + body + color**, not a face replacement. Two separate problems, ranked separately below.

### A. Fixes the actual v3 merge bug (A-exits-right stitched to B-enters-left)

| # | Action | Effort | Cost / model | License + legal | Impact on the bug |
|---|--------|--------|--------------|-----------------|-------------------|
| **1** | **Add a motion model to `track()`** — constant-velocity Kalman predict/update + Mahalanobis gating + a velocity-incompatibility veto | Medium (pure-Rust from first principles) | **FREE** — no model, no weight | **Clean by construction**, zero biometric exposure | **Highest** — directly severs the geometry stitch that *causes* the bug |
| **2** | **Pose soft-biometrics** — viewpoint-robust anthropometric ratio vector (clip-mean) as a new `IdSig.shape`, fused below color, used as a *veto only* | Low-Medium (own math over keypoints we already decode) | **FREE** — pure Rust, model size 0 | **License-clean by construction**; lightest legal posture but **NOT a safe harbor** — keep ephemeral, treat as a *potential* biometric | **Weak / situational** — only catches gross-build differences and night/IR cases; does **not** separate same-build, same-outfit people |
| **3** | **Keep clothing-color HSV histogram** (already shipped) as the primary within-clip re-association signal | None (exists) | FREE — pure Rust, no weight | Clean | Strong same-visit when lighting is good; dies in IR |

### B. Replaces face's cross-outfit / long-horizon role (a different, harder need the bug does **not** require)

| # | Action | Effort | Cost / model | License + legal | Verdict |
|---|--------|--------|--------------|-----------------|---------|
| **4** | **DINOv2-Apache body Re-ID** (already the verified-clean body backbone) | Already in plan | Needs a model (clean weight) | **Clean weight** (Apache backbone); body Re-ID is closer to "appearance" than "biometric template" — still review before deploy | Best available cross-outfit-*adjacent* signal; degrades under outfit change but carries no forbidden weight |
| **5** | **Gait** (OpenGait / FastPoseGait / GaitGraph / synthetic) | High → infeasible | Needs a model + a clean corpus that **does not exist** | **OFF THE TABLE** — forbidden weights + re-imports a biometric legal gate (EU AI Act names gait explicitly) | Does not fix the bug; not privacy-lighter than face once you ship into EU; park as a speculative *later* spike only |

**One-line plan:** Ship #1 first (it is the real fix), add #2 behind a conservative legal+threshold gate as a free night/IR augment, keep #3, lean on #4 for the cross-outfit gap, and **do not adopt #5**.

---

## The bug vs the need — do not conflate them

- **The bug (what broke v3):** the tracker MERGES two different people within one 60 s clip because `track()` is pure geometry — IoU ≥ 0.30, then a raw center-distance fallback within 1.2 box-heights, bridging gaps up to 8 frames, with **no Kalman filter, no velocity model, no Mahalanobis gating.** That is exactly the path that stitches "person A exits right" to "person B enters left," producing a false reversing track → false *pacing / loitering*. The two people are in the **same visit, same outfit**, so color and body Re-ID already discriminate them when present; the gap is purely **association geometry**.
- **The need face was filling (a different, harder problem):** cross-outfit, long-horizon identity — recognizing the same person across visits or after a clothing change. Face was the anchor for *that*, and it was opportunistic here. Replacing that role is a separate project (Section B / DINOv2), and **nothing in the v3 bug requires solving it.**

Conclusion the evidence forces: **stop trying to replace face to fix the merge.** Fix the tracker (Section 1), then add a free weak shape veto (Section 2). Treat cross-outfit as a distinct, lower-priority track.

---

## 1. Motion model — the real fix (ranked #1)

The current association in `track()`:

1. IoU ≥ 0.30 between detection and last track box → match.
2. else raw center-distance fallback within 1.2 box-heights → match.
3. bridge gaps up to 8 frames.

This has **no state estimate and no motion consistency check**, so a detection that lands near where a *departed* track last was will be claimed by it regardless of velocity direction. That is the failure.

### What to add

**(a) Per-track constant-velocity Kalman filter.**
State `[cx, cy, w, h, vx, vy, vw, vh]` (center, size, and their velocities — the SORT/ByteTrack-style state). On each frame:
- **predict:** advance the state by the motion model to get the expected box for this frame;
- **update:** correct with the matched detection.

The predicted box (not the *last observed* box) is what you match against — so a track that has been moving right *predicts* further right, and a detection appearing on the left no longer falls inside its gate.

**(b) Mahalanobis (or predicted-IoU) gating.**
Gate association on the Mahalanobis distance between the detection and the track's predicted distribution, not raw pixels. This is the standard DeepSORT gate (`chi2inv95` threshold on the 4-D measurement). A detection that is improbable under the track's predicted motion is rejected even if it is pixel-close.

**(c) Velocity-incompatibility veto (cheap, do this even before full Kalman).**
If a track's recent velocity is strongly rightward and a candidate detection requires an instantaneous leftward jump (sign-flip of `vx` beyond a threshold, or a position delta inconsistent with `|v|·Δt`), **veto** the association. This single rule kills the "exits right → enters left" stitch directly and is a few lines of Rust over the displacement history you can already derive from the track's box centers.

**(d) Association method.**
Replace greedy nearest-box with a **Hungarian / linear-assignment** over a cost matrix = gated (predicted-IoU or Mahalanobis) distances, so two simultaneous people are assigned jointly rather than first-come-first-served. ByteTrack's two-stage match (high-confidence dets first, then recover low-confidence) is a clean, well-documented recipe to mirror.

### License — all clean to reimplement from first principles

The Kalman filter is classical math (1960), not anyone's IP. The algorithms you'd mirror are all permissively licensed *as code* and trivially reimplementable in Rust:

- **SORT** — GPL-3.0 as a repo, but the *algorithm* (Kalman + Hungarian + IoU) is from the paper and is public; reimplement from the paper, do not vendor the repo. https://github.com/abewley/sort/blob/master/LICENSE
- **DeepSORT** — GPL-3.0 repo; again reimplement the *gating math* (Mahalanobis + `chi2inv95`) from the paper, not the code. https://github.com/nwojke/deep_sort/blob/master/LICENSE
- **ByteTrack** — **MIT** code, fully reusable as a reference and portable to Rust. https://github.com/FoundationVision/ByteTrack/blob/main/LICENSE
- **BoxMOT** — **AGPL-3.0**, forbidden to vendor in a closed product; use only as documentation. https://github.com/mikel-brostrom/boxmot/blob/master/LICENSE

> Note: these trackers add *no biometric signal at all* — they only constrain geometry/motion. Zero license-weight taint, zero BIPA/CUBI exposure. This is the highest-leverage, lowest-risk change on the table.

### Expected impact

Directly removes the structural cause of the v3 merge. The "A exits right, B enters left" stitch is velocity-incompatible by construction, so a predicted-state gate + velocity veto refuses it without needing to know *who* either person is. This is the change to ship first and measure on the real failing 60 s clips before investing in any appearance/biometric signal.

---

## 2. Pose soft-biometrics — free augment, honest about its weakness (ranked #2)

We already run YOLO-pose every frame (17 COCO keypoints), so **skeleton-derived anthropometric features are FREE** — no extra model, no weight license, pure Rust — and they survive in IR where the clothing-color histogram collapses to gray/black/white value-bins.

### What to fuse into `IdSig`

A new `IdSig.shape: Option<[f32; K]>`: a small fixed-length vector of **self-normalized body-proportion ratios** computed from the keypoints we already decode, each divided by a body-internal scale (shoulder width or torso length) so it is pixel-scale and distance invariant:

- torso-length : leg-length
- shoulder-width : hip-width
- upper-leg : lower-leg
- (head-to-shoulder : torso)

Computed per frame, then **averaged across the track's frames into a clip-mean** (the pose analogue of the existing `color_mean`), compared with a normalized L1/cosine distance. Fuse it **below** color (color is stronger when present), and use it as the body-Re-ID tier's free fallback when OSNet/DINOv2 isn't run.

**Start with the 2–3 most viewpoint-stable, most-often-visible ratios** (torso:total, shoulder:hip, leg:torso) behind a single conservative threshold, then widen to the full set once measured on real clips.

### Honest weak-discriminator caveat (the verdicts were emphatic here)

- It is **WEAK alone** and only additive as a fusion signal. The headline "~72–77% alone / +~20% fusion" numbers come from **RGB-D / Kinect 3-D-skeleton** benchmarks (Pala et al.) — *easier* than our 2-D projected keypoints. Hand-crafted skeleton descriptors score far lower in real 2-D settings (e.g. D16 ≈ 28%/14% rank-1 on BIWI). Do not market these numbers as ours. https://www.sciencedirect.com/science/article/abs/pii/S0921889016305073 · https://arxiv.org/html/2401.15296v2
- **"Viewpoint-robust" is aspirational, not proven.** A 2-D skeleton is a projection of a 3-D one; foreshortening changes apparent limb ratios as a person turns toward/away from camera. The v3 bug is *A frontal / B profile* — precisely the regime where raw 2-D ratios are least reliable. Restrict to self-normalized proportions and keep thresholds wide. https://arxiv.org/abs/2307.12917
- **Jitter:** 2-D keypoints oscillate frame-to-frame even on a static subject. A per-frame ratio is noisy → **must** clip-mean average before trusting it, or you risk *falsely splitting a real pacer* (the cardinal sin). Gate on a minimum keypoint-confidence (analogue of `ColorSig.weak`) so a profile/occluded skeleton **abstains** instead of producing garbage.
- **It does not solve the actual bug.** For two same-build, same-outfit people it abstains and adds nothing. Its real, narrow wins are: (a) **night/IR**, where the color histogram is blind, and (b) two similarly-dressed people of **clearly different build** (tall-lanky vs short-stocky).

### Usage rule (matches `identity.rs` semantics)

`IdSig.shape` may only **REFUSE a merge** (return `Different` → veto the association) on a **wide, hysteresis-gated gross-build margin**, and otherwise return `Unknown` (fails toward NOT-splitting). It must **never force a merge**. Pair it with the Section-1 motion gate, which addresses the stitch more directly.

### Lighter legal posture — but NOT a clean safe harbor

- BIPA's definition of *biometric identifier* is a **closed enumerated list** — retina/iris scan, fingerprint, voiceprint, scan of hand or face geometry — and **expressly excludes** "physical descriptions such as height, weight, hair color, or eye color." A static build/proportion descriptor is closer to an excluded physical description than to the enumerated templates. 740 ILCS 14/10. https://www.ilga.gov/Legislation/ILCS/Articles?ActID=3004&ChapterID=57 · https://law.justia.com/codes/illinois/chapter-740/act-740-ilcs-14/
- Texas CUBI mirrors BIPA's closed list and commentators confirm it "does not cover … gait analysis, or behavioral biometrics." https://statutes.capitol.texas.gov/Docs/BC/htm/BC.503.htm
- **Adversarial caveats that downgrade this from "safe by construction" to "legal_gate":**
  - The exclusion lists are *scalar* descriptors. A **multi-dimensional ratio vector engineered to discriminate identity** is arguably "biometric information used to identify an individual" under BIPA's second prong — its very purpose is to tell people apart. https://www.ilga.gov/Legislation/ILCS/Articles?ActID=3004&ChapterID=57
  - The "must actually identify" defense rests on a **single non-binding district decision** (*Martell v. X Corp.*, 2024 WL 3011353, N.D. Ill.) that favored a defendant because the data was *not designed to identify*; our vector *is* designed to discriminate persons — the opposite posture. https://www.insideprivacy.com/privacy-and-data-security/illinois-federal-court-dismisses-bipa-suit-against-x-holding-biometric-identifiers-must-identify-individuals/
  - **Ephemerality is necessary but not sufficient:** *Cothron v. White Castle* (Ill. 2023) holds BIPA liability attaches at each **collection/capture** event, not at storage — "discarded at clip-end" does not by itself avoid liability if the vector qualifies as a biometric.
  - **CA CCPA and Washington MHMDA define biometric/behavioral data broadly** and enumerate gait — so the safety is jurisdiction-specific to IL/TX enumerated-list statutes and hinges on this staying a **static build descriptor**, not movement-rhythm. https://statescoop.com/future-state-biometric-privacy-body-data-2025/

**Net legal rule:** keep `shape` strictly ephemeral, **never persist it as a cross-visit re-id template** (do not write it to `pose_log` as a stored identity vector), and route the keep/kill decision through counsel as a *potential* biometric — not as a green-lit safe harbor. As an engineering signal it is a free, weak, IR-capable fusion veto worth adding behind that gate.

---

## 3. Gait — plain verdict: OFF THE TABLE now, speculative later

**Verdict: do not adopt gait for v3.** It fails the same three gates that shelved face *and* adds two of its own, and it does not even fix the bug.

### License — same trap as Re-ID/face, not an escape

- **OpenGait** (GaitBase, DeepGaitV2, GaitGL, GaitSet, SwinGait): **no SPDX license** + explicit README clause — code is "only used for academic purposes, people cannot use this code for anything that might be considered commercial use." Forbidden to vendor. https://github.com/ShiqiYu/OpenGait/blob/master/README.md
- **FastPoseGait** (GPGait, GaitTR, GaitGraph2): no SPDX + "strictly intended for academic purposes and can not be utilized for any form of commercial use," and it is *built on* OpenGait — doubly blocked. https://github.com/BNU-IVC/FastPoseGait
- **GaitGraph v1** (tteepe): the one genuinely **MIT** code repo — but its released checkpoint is trained on **CASIA-B** (research-only), so the **weight is forbidden** even though the code is MIT. Textbook "MIT code never launders a research-only weight." https://raw.githubusercontent.com/tteepe/GaitGraph/main/LICENSE · https://github.com/tteepe/GaitGraph/blob/main/README.md
- **Datasets are the fatal wall:** CASIA-B, OU-MVLP/OU-ISIR, GREW, Gait3D, SUSTech1K, CCPG are **all research-only with signed agreements banning commercial use and modification**, so **every released gait checkpoint is dataset-tainted** and there is **no clean corpus to retrain on**. GREW: "not allowed to use this dataset … for any commercial purposes." SUSTech1K: "shall not be modified or used commercially." https://www.grew-benchmark.org/doc/license_agreement_for_grew_dataset.pdf · https://lidargait.github.io/static/resources/SUSTech1KAgreement.pdf
- Synthetic escape hatches (VersatileGait, BarbieGait) have **unverified licenses** and arguable derivative-of-real-biometric provenance; not relied upon. https://arxiv.org/pdf/2105.14421

### Legal — gait re-imports the biometric gate that shelved face (this kills the point)

This is the decisive correction to any "gait is privacy-lighter" premise:

- **Under US enumerated-list statutes, gait is genuinely lighter than face.** BIPA's closed six-item physiological list omits gait; Texas CUBI's closed list explicitly does not reach "gait analysis … or behavioral biometrics." So in **IL/TX**, gait carries a *lighter* gate than face. https://www.ilga.gov/Legislation/ILCS/Articles?ActID=3004&ChapterID=57 · https://statutes.capitol.texas.gov/Docs/BC/htm/BC.503.htm · https://www.texasattorneygeneral.gov/consumer-protection/file-consumer-complaint/consumer-privacy-rights/biometric-identifier-act
- **But that is jurisdiction-specific, not universal.** The **EU AI Act's** biometric-data concept **names "behavioral characteristics, such as gait or voice print,"** GDPR Art. 9 treats it as special-category data when used to uniquely identify, and the EDPB/EDPS jointly called for a ban on automated recognition "including faces, gait, fingerprints, DNA, voice." **Colorado HB24-1130** and **CCPA** also explicitly name gait. https://iapp.org/news/a/biometrics-in-the-eu-navigating-the-gdpr-ai-act · https://www.twobirds.com/en/insights/2023/global/biometrics-under-the-eu-ai-act · https://www.dwt.com/blogs/privacy--security-law-blog/2025/05/colorado-privacy-act-biometrics-updates-july-2025
- **For a closed commercial product with any EU / CA / CO / WA exposure, gait carries the SAME legal gate as face.** Adopting it would **re-create the BIPA/CUBI-style exposure you just removed by shelving face** — under a different statute. That defeats the entire reason to look past face.

### Compute / scope — gait does not even fix the bug

Gait is a **cross-outfit / long-horizon anchor** needing **multiple consistent-viewpoint gait cycles** and (for silhouette models) per-frame person segmentation we don't run. A single fixed-camera 60 s pass-by rarely yields them, and the reported 78–88% accuracies are on controlled multi-cycle, multi-view benchmarks, not surveillance pass-bys. CASIA-B accuracy already drops to ~66% under clothing change. The v3 bug is same-visit, same-outfit — gait's only unique value (cross-outfit) is exactly the role face played and that this bug never needed.

### Disposition

- **Now:** off the table (forbidden weights + biometric legal gate in EU/CA/CO + wrong tool for a same-outfit bug).
- **Later (speculative only):** *if* a future, verified-permissive synthetic gait corpus lets you train a from-scratch weight on MIT-code GaitGraph **and** counsel clears the deployed signal under EU/CO/CA biometric law — re-run the analysis then. Do not build on it today.

---

## 4. What to build first — ordered, tied to `track()` / `identity.rs`

1. **`track()` motion model (Section 1).** Add a constant-velocity Kalman state per track, gate associations on predicted-IoU/Mahalanobis, and add a velocity-incompatibility veto so "exits right" can't claim "enters left." Reimplement from the SORT/DeepSORT papers and the MIT ByteTrack code; vendor no GPL/AGPL repo. **Measure on the real failing 60 s clips before adding any appearance signal** — this alone may close the bug.
2. **`IdSig.shape` (Section 2), conservative subset first.** Compute 2–3 self-normalized clip-mean proportion ratios from the pose keypoints already decoded; abstain below a keypoint-confidence floor; allow it to **veto-only** on a wide gross-build margin (fails toward `Unknown`). Keep it strictly ephemeral — **never write it to `pose_log` as an identity template** — and route the keep/kill through counsel as a *potential* biometric. Widen the ratio set only after measuring on real clips. Its real win is **night/IR** and clearly-different-build pairs.
3. **Keep the clothing-color HSV histogram (Section 3, existing)** as the primary same-visit re-association signal; it remains the strongest cheap, clean signal when lighting is good.
4. **Lean on DINOv2-Apache body Re-ID for the cross-outfit gap** (the role face vacated) — the one verified-clean body backbone — as a separate, lower-priority track. Quantify the current color + body cross-outfit miss rate on real two-visit footage before spending any biometric-risk budget on more.
5. **Do not adopt gait.**

---

## Open questions to resolve with measurement / counsel

1. Does the Kalman + Mahalanobis (or velocity-veto) tracker gate **alone** resolve the same-visit A-exits/B-enters merge on the real failing 60 s clips? If yes, appearance/biometric signals are augmentation, not necessity.
2. Within one 60 s clip, is the clip-mean `shape` ratio vector stable for a *single* person across viewpoint change as they turn — i.e., does foreshortening move proportions more than the gap between two different builds? Only a measurement on captured clips answers whether `shape` can veto at all.
3. What per-component threshold and minimum-keypoint-confidence gate make `shape` **abstain** (→ `Unknown`) on profile/occluded skeletons instead of triggering a false split?
4. Should `shape` veto **only when color is weak/unavailable** (night/IR), or always as an OR'd veto? Always-on risks fragmenting a real single pacer on jitter; night-only is safer but narrower.
5. Legal: write the no-enrollment / no-gallery / clip-ephemeral design into the privacy posture doc so the "non-identifying, not persisted" claim is true in code, and confirm with counsel whether a same-clip, never-persisted ratio vector is defensibly outside "biometric identifier" in every shipped jurisdiction (BIPA, CUBI, CCPA, WA MHMDA, CO CPA) given *Cothron*'s collection-not-storage rule.
