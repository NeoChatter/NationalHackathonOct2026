# Camera-Only Race Car Localization — combined release

> **To run it:** [`HOWTORUN_README.md`](HOWTORUN_README.md). **Why each part was chosen:** [`DECISIONS.md`](DECISIONS.md).
> **Jury:** [`JURY_README.md`](JURY_README.md).

Estimate the pose of a race car (`constructor0/base_link`) in the track frame `map` — position `x`, `y` and heading
`yaw` — from the onboard cameras plus the car's speed / yaw-rate / acceleration signals, at camera rate (10 Hz). The
training run is a night session at Yas Marina North (constructor0, 2026-10-13), laps up to ~40 m/s including a
pit-lane pass. The test run is one daylight lap at higher speed.

This folder merges the two parallel branches `rmse-2.740-vidya-2026-10-18-1712` (V) and `rmse-4.363-shubi` (S)
following `analysis/Answers.md`: V's causal camera sync, network, training recipe and validation protocol, with S's
calibrated odometry, 4-state EKF, track map, along-track start finding, modular `lib/`, provenance and stress suite.

## Results

**The combined pipeline has not been scored yet** (it has only passed the synthetic end-to-end smoke test). The
reference to beat, on the same protocol (2 held-out middle laps, frames 4777–6975, no start pose, seed 0):

| | position RMSE [m] | median / p95 [m] | lateral / longitudinal [m] | yaw [deg] | < 1 m |
|---|---|---|---|---|---|
| V `rmse-2.740-vidya-2026-10-18-1712` (reference, `results/reference_v2.740/`) | **2.740** | 0.944 / 7.399 | 2.040 / 1.829 | 0.872 | 53.3 % |
| combined (this folder) | *run `python run_submission.py validate`* | | | | |

S's `rmse-4.363` used a different window (last 20 % of the run incl. the unmapped pit lane; ~0.95 m on its other
2000 frames by the analysis' arithmetic), so it is not comparable with the row above.

## Rules this solution follows

| Rule (organisers) | How it is met |
|---|---|
| Causal: a pose at time `t` uses only images stamped `<= t` | `lib/common.select_causal_image` (used by stage 1 and stage 4) takes, per camera, the newest image with `stamp <= t`; every pose is stamped with its camera_fl stamp. Speed / yaw rate are integrated only from samples that have already arrived (`lib/state_est.py`). |
| **No initial pose is given** | None is used. The along-track grid filter (`lib/sfilter.py`) starts from a uniform belief over the lap and seeds the EKF once a peak is confident, distinctive and consistent with the odometry. |
| Ground truth is for development and scoring only | `/tf` is read only by `Scripts/1_build_dataset.py` (training labels) and the scoring tools. `Scripts/4_infer.py` subscribes to the four camera topics and the state estimation only. |
| Speed, accelerations and yaw rate may be used, **pose fields may not** | `lib/state_est.SE_FIELDS = ('v_mps', 'wz_radps', 'ax_mps2', 'ay_mps2')` is the only whitelist; pose fields (`x_m`, `y_m`, `yaw_rad`, `sn_map_state`, ...) are never read. |
| The shared daylight example must not be used to train, pre-train, tune or validate | Not used for any of these. Validation laps come from the training run only. The only reader is `presentation/prepare_assets.py` (one frame as a slide illustration). |
| Public external datasets and pre-trained models are allowed but must be declared | See below. |

## Declared external resources

| Resource | Where | Used for |
|---|---|---|
| ResNet-18, ImageNet-1k weights (`torchvision.models.ResNet18_Weights.DEFAULT`) | torchvision | Initialisation of the place encoder and the motion network (first conv averaged to grayscale), then fine-tuned on the training run. |

**External datasets:** none.

## Approach

1. **Build** (`Scripts/1_build_dataset.py`): bag → HDF5 cache. Four cameras cropped to the track band (18–74 % of the
   height), 256×128 grayscale; one sample per camera_fl frame with the causal companion images; GT pose interpolated
   at the anchor (labels only); per-marker track borders; whitelisted state estimation; full-run GT CSV.
2. **Train** (`Scripts/2_train.py`): `RacingLocNet` — shared ResNet-18 over the 4 cameras with a learned camera gate →
   256-D place descriptor, an auxiliary pose head, and an early-fusion network that predicts frame-to-frame motion
   with its own uncertainty (fallback odometry). Same-place / different-lap paired batches, photometric + structured
   lighting augmentation (no geometric change), CLAHE + robust normalisation. Validation: two whole laps from the
   middle of the run with a 5 s guard band kept out of training and the map. The checkpoint holds the weights, the
   place database, the retrieval calibration, the track map and the provenance chain.
3. **Calibrate** (`Scripts/3_calibrate.py`): speed scale, yaw-rate scale and bias, odometry noise, VO bias curve and
   the track map (centre line, real borders, pit corridor, heading prior) — all fitted on the training frames only.
4. **Infer** (`Scripts/4_infer.py`): causal 10 Hz loop. EKF on `[x, y, yaw, odometry scale]`; prediction from the
   calibrated speed / yaw rate (learned VO only if they are missing or implausible, and as a cross-check); update
   from gated place retrieval with track-aligned noise; along-track grid filter for start finding and relocalisation;
   heading prior and border clamp. One pose per camera_fl frame, never skipped.
5. **Evaluate** (`Scripts/5_evaluate.py`): official metric plus error slices by filter mode and frame block, and the
   database coverage of every evaluated pose.
6. **Stress** (`Scripts/6_stress_test.py`): 26 perturbation / kidnap scenarios (10 in the quick set) on the validation block.

## Repository layout

| path | |
|---|---|
| `config.toml` | every path, the seed and every hyperparameter |
| `run_submission.py` | one command per goal: `smoke`, `validate`, `submit`, `all` |
| `Scripts/1_build_dataset.py` … `6_stress_test.py` | the pipeline stages, in run order |
| `Scripts/lib/` | shared library (see `Scripts/lib/__init__.py`) |
| `tools/evaluate.py`, `tools/extract_gt.py` | the organisers' scorer and GT export (unchanged) |
| `tools/gt_window.py` | cut the GT to a pose file's span so `evaluate.py` can score the validation block |
| `tools/visualize.py`, `tools/plot_training.py` | trajectory / error figures + metrics, training curves |
| `tools/make_synthetic_bag.py` | synthetic bag for the smoke test (never for scoring) |
| `results/reference_v2.740/` | the V branch's validation outputs, sweep and training history (reference) |
| `presentation/` | slide deck sources (`build_deck.py`, assets) |
| `brainstorming_and_diagnostics.md` | problem → diagnosis → fix → benefit log of the milestones |
| `docs/history/` | superseded planning documents of both branches |
| `chats/` | debugging walkthroughs behind the early milestones |

Large files (`data/`, `checkpoints/*.pt`, `runs/`, `stress/`) are not in Git.

## Quick run

```bash
pip install -r requirements.txt
python run_submission.py smoke                          # end-to-end check on a synthetic bag
python run_submission.py validate                       # validation model + official score of the 2-lap block
python run_submission.py submit --test-bag <TEST.mcap>  # submission model + results/poses/test_poses.csv
```

## Metric

`position RMSE [m] = sqrt(mean(dx^2 + dy^2))` is the score — lower is better. Errors are split into the GT car frame
(longitudinal along the heading, lateral across it; lateral RMSE is the tie-breaker). A submission needs ≥ 90 %
coverage of the 1 s bins of the run to be ranked.

## Milestones (history)

| tag | position RMSE [m] | protocol | what changed |
|---|---|---|---|
| `rmse-24.508` | 24.508 | GT start pose, last 20 % | baseline neural visual EKF |
| `rmse-13.688` | 13.688 | GT start pose, last 20 % | EKF trust re-tuning (last shared commit `e9c31ef`) |
| `rmse-10.426-shubi` | 10.426 | GT start pose, last 20 % | S: VO bias curve, anisotropic tracker, PlaceNet / OdomNet |
| `rmse-8.684-shubi` | 8.684 | GT start pose, last 20 % | S: track map |
| `rmse-7.395-shubi` | 7.395 | GT start pose, last 20 % | S: grid-filter re-seeding, numbered stages + `lib/` |
| `rmse-4.363-shubi` | 4.363 | **no start pose**, last 20 % (incl. pit lane) | S: speed / yaw-rate odometry, global initialisation |
| `rmse-2.740-vidya-2026-10-18-1712` | 2.740 | **no start pose**, 2 held-out middle laps | V: causal sync, telemetry odometry, multi-candidate start, new recipe |
| combined (this folder) | not scored yet | same as the row above | this merge |

The protocols differ (the organisers' rules changed), so rows with different protocols are not comparable.
