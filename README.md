# Camera-Only Race Car Localization — Neural Visual EKF

Estimate the pose of a race car (`constructor0/base_link`) in the track frame `map` — position
`x`, `y` and heading `yaw` — using **only the onboard cameras**, at camera rate (10 Hz). The dataset
is a night session at Yas Marina North (constructor0, 2026-10-13), laps up to ~40 m/s including a
pit-lane pass.

The pipeline is causal: a pose at time `t` only uses images with stamps `<= t`. The single ground-truth
pose at the start of the run is allowed as the initial pose; all other GT is used for training/scoring
only, never as a runtime input.

## Current milestone

| | |
|---|---|
| **position RMSE [m]** | **24.508**  ← score (lower is better) |
| coverage (1 s bins) | 100.0 % (ranked; threshold ≥ 90 %) |
| position median / p95 [m] | 9.003 / 56.861 |
| lateral RMSE [m] | 3.172 |
| longitudinal RMSE [m] | 24.302 |
| yaw RMSE [deg] | 2.159 |
| within 0.25 / 0.50 / 1.00 m | 3.0 % / 6.3 % / 12.8 % |

This version is tagged in git (see [Versioning & milestones](#versioning--milestones)). The error is
dominated by the **longitudinal** component (24.3 m of the 24.5 m total), i.e. drift *along* the track;
lateral error (3.2 m) and heading (2.2°) are far tighter. Future improvements should drive the position
RMSE down from here — this number is the baseline to beat.

## Approach

Three stages, run in order.

### 1. Build dataset — `Scripts/1_build_dataset.py`

Reads the MCAP bag and produces a synchronized training cache (`data/training_cache.h5`).

- Decodes the four front/front-side cameras (`camera_fl`, `camera_fr`, `camera_cl`, `camera_cr`) to
  grayscale, crops the vertical track ROI (18 %–74 % of height, dropping floodlights and the nosecone)
  and resizes to 256×128.
- Anchors a causal multi-camera sample on every `camera_fl` frame, recording the per-camera timestamp
  offsets so the model can reason about asynchronous exposures.
- Interpolates ground-truth pose (`/tf`: `map` → `base_link`) at each `camera_fl` timestamp, unwrapping
  yaw across ±π before interpolation.

Output datasets: `timestamps_ns`, `images` (N×4×128×256 uint8), `dt_offsets` (N×4), `poses`
(N×[x, y, z, yaw]), and `track_borders` when present.

### 2. Train model — `Scripts/2_train_model.py`

`RacingLocNet`: a shared ResNet-18 backbone (first conv adapted to 1-channel input) applied to all four
cameras, with a learned per-camera attention gate (down-weights motion-blurred side views at speed).
Two heads:

- **Place descriptor head** — a 256-D L2-normalized embedding for fast GPU place retrieval, plus an
  auxiliary direct-pose head to structure the embedding space.
- **Visual ego-motion head** — predicts body-frame motion `(dx, dy, dyaw)` and a heteroscedastic
  `log_var` uncertainty from a consecutive frame pair `(t-1, t)`.

Losses: heteroscedastic VO loss (yaw weighted 5×), a supervised metric-contrastive place loss (pulls
embeddings from different laps at the same physical location together), and an auxiliary pose loss.

The checkpoint bundles the model weights, coordinate normalization stats, a **neural place database**
(embeddings + poses for all training frames), track borders, and the validation-split start timestamp.

Split: first 80 % of frames for training, last 20 % (~2 laps) held out for validation. `--full-train`
uses 100 % for a final submission.

### 3. Causal inference — `Scripts/3_causal_infer.py`

A causal Extended Kalman Filter on SE(2) `[x, y, yaw]` that fuses the two heads online at 10 Hz:

- **Predict** — the visual ego-motion `(dx, dy, dyaw)` propagates the state; the learned `log_var`
  becomes the process noise `Q` (rotated from body to world frame).
- **Update** — spatially-gated anisotropic neural place retrieval. The filter gates the place database
  to candidates near the current estimate (≤ 35 m, ≤ 35° yaw), takes the top-K by embedding similarity,
  forms a softmax-weighted mean pose measurement, and builds an anisotropic covariance from the sample
  spread (naturally wide along straights, tight laterally). The update only fires when top-1 similarity
  exceeds 0.45, otherwise the step is visual-odometry only.

The EKF is initialized from the first GT pose of the run (allowed), falling back to global neural
relocalization if `/tf` is absent. Writes `submission.csv` plus a per-frame `inference_telemetry.csv`.

The filter's trust is tunable from the command line. The current defaults — a longitudinal
process-noise floor `q_x_floor = 1.0` m/step and a position measurement-covariance scale
`r_scale = 0.03` — were selected by the diagnostic sweep and are what took the score from 24.508 m to
13.688 m (see `brainstorming_and_diagnostics.md`).

### Diagnostics — `Scripts/4_diagnostics.py`

An offline analysis tool that **uses ground truth on purpose**, so it is never part of the submission
path. On the held-out validation frames it compares four variants on identical timestamps
(`retrieval_top1`, `aux_head`, `vo_only`, `fused`), decomposes every error into longitudinal/lateral,
and reports reference-map coverage, gate-exclusion rate, VO step statistics, filter consistency
(NIS/NEES), plus a parameter sweep over `q_x_floor`, `r_scale`, and gate size ranked by validation RMSE.
This is how the fusion weaknesses were isolated instead of guessing. Requires a checkpoint trained
*without* `--full-train` (so a validation split exists).

## Repository layout

> The layout below is the `shubi` branch pipeline (milestones `rmse-10.426-shubi`, `rmse-8.684-shubi`).
> The sections above describe the original `main` pipeline.

| path | |
|---|---|
| `Scripts/1_build_dataset.py` | bag → HDF5 cache (images, GT poses, per-marker track borders); `--export-gt` |
| `Scripts/2_train.py` | train PlaceNet + OdomNet → checkpoint (DB, calibration, track map, preprocessing) |
| `Scripts/3_calibrate.py` | odometry bias curve + track map into an existing checkpoint (no retraining) |
| `Scripts/4_infer.py` | causal inference → submission + telemetry (+ stress hooks) |
| `Scripts/5_evaluate.py` | score (same rules as `tools/evaluate.py`) + error slices + DB coverage → JSON |
| `Scripts/6_stress_test.py` | robustness suite (dusk, lighting, image quality, sensor faults, kidnap) |
| `Scripts/lib/` | shared library: models, data, track map, filters, memory, perturbations, metrics, provenance |
| `tools/extract_gt.py` | extract ground-truth poses from `/tf` to CSV |
| `tools/evaluate.py` | score a submission against ground truth |
| `brainstorming_and_diagnostics.md` | milestone-by-milestone problem → diagnosis → fix → benefit log |
| `chats/` | debugging walkthroughs that drove each improvement |
| `checkpoints/` | trained checkpoint lives here (produced on the server — see note) |
| `data/` | dataset + HDF5 cache (not tracked in this repo) |

> **Note:** training runs on a server, so the trained checkpoint (`checkpoints/*.pth` / `*.pt`) is
> **not** included here. Place the checkpoint in `checkpoints/` before running inference. `data/` is
> also excluded from the repo (large MCAP bag and HDF5 cache).

## Usage

```bash
pip install -r requirements.txt
BAG=Data/c0_2026-10-13_22-53-42_camloc_0.mcap

# 1. bag -> cache (+ full GT CSV)
python Scripts/1_build_dataset.py $BAG --output data/training_cache.h5 --export-gt data/gt.csv
# 2. train a condition-robust model (GPU)
python Scripts/2_train.py --data-h5 data/training_cache.h5 --checkpoint checkpoints/robust.pt \
    --clahe --norm robust --aug structured --seed 0
# 3. odometry bias curve + track map
python Scripts/3_calibrate.py --checkpoint checkpoints/robust.pt --output checkpoints/robust_cal.pt \
    --data-h5 data/training_cache.h5 --plot checkpoints/robust_track.png
# 4. causal inference on the held-out laps (or without --val-only for a submission)
python Scripts/4_infer.py $BAG --checkpoint checkpoints/robust_cal.pt --val-only --output runs/val.csv
# 5. score + where the error comes from
python Scripts/5_evaluate.py --gt data/gt.csv --sub runs/val.csv --checkpoint checkpoints/robust_cal.pt --val-only
# 6. robustness suite
python Scripts/6_stress_test.py --mcap $BAG --checkpoint checkpoints/robust_cal.pt --gt data/gt.csv --out stress/robust
```

Every output gets a `<file>.meta.json` sidecar (git commit, arguments, input fingerprints, and the
checkpoint's provenance chain `build → train → calibrate`).

### Fairness / data use

- The only training input is the provided night session bag. The place database is built solely from its
  training-split frames. No evaluation data is read, and no parameter is tuned against it.
- Condition robustness is a stated design goal, handled generically: broad, symmetric augmentation over
  weather / time-of-day / illumination (brightening and darkening equally likely), robust normalisation,
  CLAHE, and causal self-adaptation during a run (experience memory and optional AdaBN) that use only the
  run's own past frames and the tracker's own estimate — the same information the filter already uses.
- The stress suite perturbs inputs to measure robustness; it is a proxy and never part of the submission
  path, and its scenarios are generic conditions, not a reconstruction of any specific run.
- The only runtime use of ground truth is the single allowed initial pose.

## Metric

`position RMSE [m] = sqrt(mean(dx^2 + dy^2))` is the score — lower is better. Errors are split into the
GT car frame (longitudinal along heading, lateral across it); lateral matters most for racing. A
submission needs ≥ 90 % coverage of the 1 s bins of the run to be ranked. See `data/README.md` for the
full dataset, topic, and metric reference.

## Versioning & milestones

Scored milestones are recorded as annotated git tags, with the full metric block in the tag message.
The baseline is `rmse-24.508`; the current best is **`rmse-13.688`** (position RMSE 13.688 m). Each
later tag should show a lower score; compare against these to track progress. The full
problem → diagnosis → fix → benefit story behind each milestone lives in
`brainstorming_and_diagnostics.md`, with the source debugging walkthroughs under `chats/`.
