# Camera-Only Race Car Localization with a Neural Visual EKF
**Format:** 6 slides, about 45 s each, roughly 4:30 total (30 s buffer under the 5:00 limit).
**Script length:** about 100-110 words per slide, read at about 145 words per minute.
**Source:** `README.md` and `brainstorming_and_diagnostics.md`. Every number comes from those files.

---

## Slide 1: Problem & Motivation
**Title:** *Where Is the Car? Cameras Only.*

**Bullets**
- Camera-only pose, no GPS
- Train: night. Test: daylight, faster
- No starting pose given
- Causal, 10 Hz output
- Score: position RMSE (lower wins)

**Dominant visual**
- Full-bleed night frame from `camera_fl`, with motion blur and floodlight glare visible.
- Overlay the Yas Marina track map (`track_left` / `track_right` borders) as a minimap.
- On the minimap, draw the GT trajectory as a thin line. Add a pulsing dot labeled "x, y, yaw = ?".
- Optionally show a 2x2 grid of the four cameras (fl, fr, cl, cr) with a "?" over the car.

**Speaker notes (about 45 s)**
> Picture a race car at forty meters per second, at night, with no GPS. Just cameras. Our task is to recover its position and heading ten times a second, and to do it causally, using only past images. The data is a night session at Yas Marina, almost twenty minutes long, including a pit-lane pass. Why is this hard? Floodlights, motion blur, and long straights that look identical frame after frame. A tenth of a second of timing error is already four meters of position error. The ground truth is off limits at runtime, and the test gives no starting pose at all, so the system has to find where it is from the images alone. Our score is position RMSE, and lower is better.

---

## Slide 2: Previous Limitations
**Title:** *Drift or Ambiguity. Pick One.*

**Bullets**
- VO drifts; retrieval aliases
- 98% of error: along-track
- Gate locks out correct place
- Baseline: 24.5 m RMSE

**Dominant visual**
- Left, about 60% of the screen: a stacked bar or donut of squared position error. Longitudinal is 98.3% (24.302^2 / 24.508^2). Lateral is about 1.7% (3.172^2 / 24.508^2).
- Annotate the chart with "yaw only 2.2 deg".
- Right: a small top-down schematic of the failure. Draw the true car position and the drifted predicted position 30+ m down a straight.
- Draw the 35 m / 35 deg gate circle around the *predicted* pose, with the true location outside it. Label it "correct place excluded".

**Speaker notes (about 45 s)**
> Camera-only systems usually fall into two camps. Visual odometry is smooth, but it drifts without bound. Place retrieval is drift-free, but ambiguous on repetitive straights. Our baseline fused both, and scored twenty-four point five meters RMSE. Before changing anything, we built a diagnostic tool. It showed that ninety-eight percent of the squared error was along the track. Heading was fine, around two degrees. Lateral error was about three meters. The car knew where it was across the track, just not how far along it. Worse, once drift exceeded the retrieval gate, the filter could never recover.

---

## Slide 3: Proposed Method - Architecture
**Title:** *RacingLocNet: One Backbone, Two Heads*

**Bullets**
- Shared ResNet-18, four cameras
- Attention gate suppresses blur
- Head 1: place embedding
- Head 2: ego-motion + uncertainty

**Dominant visual** (full-width architecture schematic, left to right)
- Four grayscale crops (256x128; fl, fr, cl, cr) feed one shared ResNet-18 box.
- Next to the backbone, a small "per-camera attention gate" block with four weights. The side-camera weights are visibly smaller.
- The network splits into two colored branches:
  - **Branch A (blue):** 256-D L2-normalized place embedding -> arrow into a "place database (training laps)" cylinder.
  - **Branch B (orange):** frame pair (t-1, t) -> (dx, dy, dyaw) + log sigma^2.
- Label the output of each branch with its failure mode: "drift-free but ambiguous" and "smooth but drifting".

**Speaker notes (about 45 s)**
> This is RacingLocNet. Four grayscale cameras, cropped to the track region, share a single ResNet-18 backbone. A learned attention gate down-weights motion-blurred side views at speed. Two heads sit on top. The place head produces a 256-dimensional embedding, matched against a database built from the training laps. The ego-motion head takes consecutive frames and predicts body-frame motion, plus a learned uncertainty, trained with a heteroscedastic loss. So the network tells us when it is unsure. One network, two complementary signals. One is drift-free but ambiguous. The other is smooth but drifting. Everything stays causal, at ten hertz.

---

## Slide 4: Proposed Method - Fusion and the Fix
**Title:** *Fix the Filter, Not the Network*

**Bullets**
- VO predicts; retrieval corrects
- Learned variance sets process noise
- Sweep-tuned trust, no retraining
- Q floor 1.0 m, R x 0.03

**Dominant visual**
- A clean EKF loop diagram on SE(2) [x, y, yaw]. **Predict** (VO delta-pose, with log sigma^2 becoming Q) feeds **Update** (gated top-K retrieval -> softmax-weighted pose, with anisotropic R). A loop arrow returns to Predict.
- Beside it, a **before/after "trust bar"**: two horizontal sliders per sensor.
  - Before: VO trust very high, retrieval trust low.
  - After: balanced.
  - Label the changes "Q_long floor = 1.0 m/step" and "R scale = 0.03".
- Footer chip: "Diagnostics use GT offline only. Never in the submission path."

**Speaker notes (about 45 s)**
> The two heads meet in a causal Extended Kalman Filter. Odometry predicts, and its learned variance becomes the process noise. Retrieval updates, gated to candidates within thirty-five meters and thirty-five degrees, and weighted by similarity. Our diagnostics found the real problem. The filter over-trusted odometry along the track, and under-trusted retrieval. A parameter sweep fixed it with two numbers: a longitudinal noise floor of one meter per step, and a measurement covariance scaled by point-oh-three. No retraining, no new architecture. Just calibrated trust between sensors. The diagnostic tool uses ground truth, so it never touches the submission path.

---

## Slide 5: Key Results
**Title:** *24.5 m -> 13.7 m, Same Network*

**Bullets**
- 24.5 -> 13.7 m RMSE
- Median 9.0 -> 1.5 m
- Lateral, yaw unchanged
- p95 still 20 m

**Dominant visual**
- Main chart, about 70% of the screen: a grouped bar chart, baseline (`rmse-24.508`) vs tuned (`rmse-13.688`), with these bars:
  - position RMSE: 24.5 -> 13.7 (-44%)
  - median: 9.0 -> 1.5 (-83%)
  - p95: 56.9 -> 20.0 (-65%)
  - longitudinal RMSE: 24.3 -> 13.3
  - Use a bold "-44%" callout on the RMSE pair.
- Small inset: lateral (3.17 -> 3.20) and yaw (2.16 deg -> 2.37 deg) bars side by side, nearly identical. Caption: "only along-track trust changed".
- Small inset: "within 1 m" goes from 12.8% -> 32.8%.
- *Stretch visual, needs generating from `inference_telemetry.csv`:* a top-down trajectory overlay of GT vs baseline vs tuned, with the baseline drifting off by tens of meters on a straight.

**Speaker notes (about 45 s)**
> Same network, same data. Only the filter's trust changed. Position RMSE falls from twenty-four point five to thirteen point seven meters, a forty-four percent reduction. The median error drops sixfold, from nine meters to one and a half. Frames within one meter rise from thirteen to thirty-three percent. Lateral and yaw stay flat, exactly as the diagnostics predicted, since we only touched along-track trust. To be honest about the limits: the ninety-fifth percentile is still twenty meters, and the remaining error comes from long along-track excursions. Results are on held-out laps.

---

## Slide 6: Conclusion
**Title:** *Diagnose First. Then Fix.*

**Bullets**
- Diagnose first, then fix
- Calibrated trust beat retraining
- Next: sequence matching, recovery
- Next: geometric refinement

**Dominant visual**
- A three-step roadmap strip in the center.
  1. **Now:** 13.7 m RMSE (solid, filled).
  2. **Next:** "Sequence matching + same-place/different-lap contrastive mining + global recovery" (dashed).
  3. **Goal:** "Local geometric refinement -> sub-meter" (dashed, faded).
- Beneath the strip, a faint thumbnail of the slide-5 bar chart as a recap.
- Include a QR code or repo link in the corner.

**Speaker notes (about 45 s)**
> Three takeaways. First, diagnose before you train. One offline tool localized the failure to the fusion, not the network. Second, calibrated uncertainty beat retraining: trust tuning alone cut the error by forty-four percent. Third, the remaining error is along-track, and that tells us where to go next. Sequence matching over several frames. Retrieval trained with same-place, different-lap positives. Global recovery when the filter locks onto the wrong place. And local geometric refinement, which is the path to sub-meter accuracy. Thank you. Happy to take questions.

---

## Appendix A: State of the Art and Roadmap (for later use)

Reference material for the roadmap on slide 6, a backup slide, or questions from the jury. The summaries below are
paraphrased from the linked sources. Content was rephrased for compliance with licensing restrictions. None of these
methods has been benchmarked on our data, so present them as **directions**, not as results.

### Where each part of our pipeline stands

| Part | What we use | Newer methods | What the sources report | Fit for us |
|---|---|---|---|---|
| Place descriptor (which part of the track is this?) | ResNet-18 + MLP, contrastive loss, trained on one night session | MixVPR, SALAD, BoQ, DINOv2-based (DINO-Mix, AnyLoc, MegaLoc) | [BoQ](https://arxiv.org/html/2405.07364v1) reports beating NetVLAD, MixVPR and EigenPlaces. [MixVPR](https://arxiv.org/html/2303.02190v1) reports a large margin with under half the parameters of NetVLAD/CosPlace. [SALAD](https://arxiv.org/html/2311.15937v2) and [DINO-Mix](https://arxiv.org/html/2311.00230v2) build on foundation-model features. A [2026 comparison](https://arxiv.org/html/2603.13917) evaluates these families on driving sequences. | Highest value for the night→day gap: foundation-model features are trained on very varied imagery. Fits on one inference GPU. **Next step.** |
| Lighting change (night training, daylight test) | Photometric augmentation (gamma, exposure, haze, sky glow, shadows, blur) | ToDayGAN; GAN-generated examples as metric-learning augmentation | [ToDayGAN](https://arxiv.org/abs/1809.09767) translates night driving images to daytime to improve retrieval-based localization. [Generating Diverse Night Examples](https://arxiv.org/html/2309.16351v1) uses a GAN generator as augmentation for retrieval training and enforces edge consistency. | Considered and not used: it needs real daylight images, and the organisers' daylight example may not be used for training. Would need a public daylight dataset. Roadmap. |
| Visual odometry (how far did we move?) | Early-fusion CNN regression on two frames | DROID-SLAM, DPVO, DPV-SLAM | [DPVO](https://arxiv.org/html/2208.04726v2) reports beating DROID while using a third of the memory and running 3× faster. [DPV-SLAM](https://arxiv.org/html/2408.01654v1) reports DROID-level accuracy at 2.5× the speed. [DROID-SLAM](https://arxiv.org/html/2108.10869v1) reports far fewer catastrophic failures than prior work. | Strong, but monocular scale has to come from somewhere (camera height, multi-camera rig). Roadmap. |
| Pose from images (exact position once the place is known) | Snap to the stored reference pose | Scene coordinate regression (ACE), relative pose regression (Reloc3r), map-relative regression | [PoseNet](https://arxiv.org/html/1505.07427v3) (the classic regression baseline) reports about 2 m outdoors. [ACE](https://arxiv.org/html/2305.14059v1) reports mapping up to 300× faster than earlier scene-coordinate methods at similar accuracy. [Reloc3r](https://arxiv.org/html/2412.08376v1) is trained on about 8 M image pairs and generalises to new scenes. | The path from about 1.5–2 m to sub-metre. Roadmap. |
| Using time (a sequence of views, not one) | EKF + gate + similarity-weighted trust + global relocalization | Learned sequence matching | [Sequential Place Learning](https://arxiv.org/html/2103.02074v1) fuses visual embeddings with motion in an LSTM. [Sequential descriptors](https://arxiv.org/html/2207.03868v1) learn a descriptor for a whole sequence. [Filtering under appearance change](https://arxiv.org/pdf/1811.08063v4) shows that temporal continuity improves localization. | Natural extension of our filter on long, similar-looking straights. Roadmap. |

### Methods we looked at and rejected for the lighting gap

- **[Restormer](https://arxiv.org/html/2111.09881v2):** an image-restoration transformer for deraining, deblurring and
  denoising, trained on paired clean/degraded images. It does not turn night into day, and we have no paired
  night/day images.
- **[TransWeather](https://arxiv.org/html/2111.14813v2):** removes rain, fog and snow. Same limits.
- **MobileViT-style hybrid:** a backbone for recognition, not an image generator. It could replace ResNet-18 as an
  encoder, but it doesn't produce daylight images.

### Added after the organisers' final clarification (use on slide 3 or 4)

- **Telemetry odometry.** Speed and yaw rate from `/constructor0/state_estimation` (explicitly allowed) replace the learned
  visual odometry as the motion model. It does not depend on lighting or speed, so the daylight, higher-speed test
  does not hurt it. The learned VO stays as an automatic fallback.
- **No initial pose.** Candidate start positions are tracked in parallel for a few seconds and scored by image
  similarity and by whether their dead-reckoned path stays on the mapped track.
- **Declared resources.** ImageNet-pretrained ResNet-18 only; no external datasets. The organisers' daylight example is
  not used for training, tuning or validation (one frame appears on a slide as an illustration of the test domain).

### Suggested use on the slides

- Slide 6 roadmap strip: **foundation-model place descriptor → geometric refinement (scene coordinates / relative pose) → learned sequence matching**.
- Speaker note: "Our filter and data pipeline are modular. Each of these swaps one block without changing the rest."
- Keep the presenter note below: no head-to-head comparison against these methods was run, so avoid "state-of-the-art" claims.

---

## Timing Plan

| Slide | Section | Target |
|---|---|---|
| 1 | Problem & Motivation | 0:45 |
| 2 | Previous Limitations | 0:45 |
| 3 | Architecture | 0:45 |
| 4 | Fusion and Fix | 0:45 |
| 5 | Key Results | 0:45 |
| 6 | Conclusion | 0:45 |
| | **Total** | **4:30** |

## Notes for the Presenter
- **Slide count:** 6 slides, not 7. Seven slides at 45 s each would be 5:15, over the limit.
- **Results scope:** The only comparison in the repo is baseline vs. tuned (`rmse-24.508` vs `rmse-13.688`). There are no external-method baselines, so avoid "state-of-the-art" claims.
- **Evaluation split:** `brainstorming_and_diagnostics.md` states that scores are on held-out validation laps. `README.md` doesn't say which split its milestone tables use. Confirm before presenting.
- **"Previous limitations":** Slide 2 uses your own baseline's failure analysis plus the general VO-drift vs retrieval-aliasing trade-off. It is not a literature comparison. Add cited prior work if the venue expects it.
- **Charts to generate:** The slide 2 donut and the slide 5 bar chart can be built from the table values. The trajectory overlay needs the telemetry CSV, which isn't in the repo snapshot.
