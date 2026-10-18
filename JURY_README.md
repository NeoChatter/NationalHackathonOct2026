# Jury Guide: Camera-Only Race Car Localization

> **Team:** `<TEAM NAME>` · **Challenge:** Constructor Labs, Camera-based Localization on a Race Track
<<<<<<< HEAD
> **Repository:** https://github.com/NeoChatter/NationalHackathonOct2026 (this snapshot: combined release of the
> branches `vidya` (tag `rmse-2.740-vidya-2026-10-18-1712`) and `shubi` (tag `rmse-4.363-shubi`))

This page tells you where every submission item lives and how to reproduce our pose files. The technical write-up
(rules checklist, declared external resources) is in [`README.md`](README.md); the full run guide is
[`HOWTORUN_README.md`](HOWTORUN_README.md).
=======
> **Repository:** https://github.com/NeoChatter/NationalHackathonOct2026 (this snapshot: branch `vidya`,
> tag `rmse-2.740-vidya-2026-10-18-1712`)

This page tells you where every submission item lives and how to reproduce our pose files. The technical write-up
(including the rules checklist and the declaration of external resources) is in [`README.md`](README.md).
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510

---

## 1. Submission items

| Item | Location |
|---|---|
<<<<<<< HEAD
| Source code | [`Scripts/`](Scripts) (stages 1-6 + `lib/`), [`tools/`](tools) (scoring, visualisation), [`run_submission.py`](run_submission.py) |
| Pose files | `results/poses/val_block_poses.csv` (held-out laps, written by `run_submission.py validate`), `results/poses/test_poses.csv` `<ADD AFTER TEST>` |
| Validation results | `results/validation/official_score.txt`, `metrics.txt` / `metrics.json`, `results/poses/val_block_poses.csv.eval.json` `<ADD AFTER RUN>` |
| Trajectory visualisation | `results/validation/`: `trajectory.png`, `error_timeline.png`, `error_distribution.png` `<ADD AFTER RUN>` |
| Training progress | `results/figures/training_progress.png` (data: `checkpoints/racing_model_val_history.csv`) `<ADD AFTER RUN>` |
| Robustness suite | `stress/run/summary.md` `<ADD AFTER RUN>` |
| Reference result of the V branch | [`results/reference_v2.740/`](results/reference_v2.740) (poses, metrics, figures, filter sweep) |
| Slides | [`presentation/project-overview.pdf`](presentation) (source: `presentation/build_deck.py`; result chips `<FILL AFTER RUN>`) |
=======
| Source code | [`Scripts/`](Scripts) (pipeline), [`tools/`](tools) (scoring and visualisation) |
| Pose files | [`results/poses/val_block_poses.csv`](results/poses) (held-out laps, 2199 poses), `test_poses.csv` `<ADD AFTER TEST>` |
| Validation results | [`results/validation/metrics.txt`](results/validation) (+ `metrics.json`), section 8 below |
| Trajectory visualisation | [`results/validation/`](results/validation): `trajectory.png`, `trajectory.gif`, `error_timeline.png`, `error_distribution.png` |
| Training progress | [`results/figures/training_progress.png`](results/figures) (data: `checkpoints/racing_model_val_history.csv`) |
| Filter tuning | [`diagnostics/sweep_results.csv`](diagnostics), `diagnostics/recommended_args.txt` |
| Slides | [`presentation/project-overview.pdf`](presentation) (source: `project-overview.pptx`) |
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510
| Video (≤ 5 min) | `<LINK PLACEHOLDER>` |

### Team members

| Name | Role | Contact |
|---|---|---|
| `<NAME>` | `<ROLE>` | `<EMAIL>` |
| `<NAME>` | `<ROLE>` | `<EMAIL>` |

---

## 2. Downloads (large files are not in Git)

| File | Put it at | Download |
|---|---|---|
<<<<<<< HEAD
| Submission model (all frames, calibrated) | `checkpoints/racing_model_cal.pt` | `<LINK PLACEHOLDER>` |
| Validation model (never saw the held-out laps, calibrated) | `checkpoints/racing_model_val_cal.pt` | `<LINK PLACEHOLDER>` |
| Training recording | `data/c0_2026-10-13_22-53-42_camloc_0.mcap` | `<provided by the organisers>` |
| Training cache (optional, skips stage 1) | `data/training_cache.h5` | `<LINK PLACEHOLDER>` |

`<SHA-256 CHECKSUMS PLACEHOLDER>` (also recorded in every checkpoint's provenance chain and `<output>.meta.json`)
=======
| Trained localization model (submission) | `checkpoints/racing_model.pt` | `<LINK PLACEHOLDER>` |
| Validation model (never saw the held-out laps) | `checkpoints/racing_model_val.pt` | `<LINK PLACEHOLDER>` |
| Training recording | `data/c0_2026-10-13_22-53-42_camloc_0.mcap` | `<LINK PLACEHOLDER / provided by organisers>` |
| Training cache (optional, skips step 1) | `data/training_cache.h5` | `<LINK PLACEHOLDER>` |

`<SHA-256 CHECKSUMS PLACEHOLDER>`
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510

---

## 3. Setup

<<<<<<< HEAD
- **Python:** 3.11 or newer.
- **GPU:** one CUDA 13 capable NVIDIA GPU. Training was done on `<SERVER GPU PLACEHOLDER>`.
- **Install:** `pip install -r requirements.txt` (exact versions; CUDA wheels on Linux, CPU-only elsewhere).
- **Check:** `python run_submission.py smoke` runs every stage on a small synthetic bag (minutes).

All commands run from the repository root. Paths come from `config.toml` (`[paths]`).
=======
- **Python:** 3.10 or newer (the inference script uses parenthesised `with` blocks).
- **GPU:** inference runs on a single CUDA GPU. Training was done on `<SERVER GPU PLACEHOLDER>`.
- **Install:**

```bash
pip install -r requirements.txt   # exact versions from our server (Linux, CUDA 13 wheels: torch 2.14.1, torchvision 0.29.1)
# Other OS / CUDA versions: install the matching torch build first, then
pip install numpy opencv-python h5py mcap mcap-ros2-support matplotlib
```

All commands below run from the repository root.
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510

---

## 4. Reproduce a pose file and score it with the organisers' scorer

<<<<<<< HEAD
Inference is causal and runs at the 10 Hz camera rate. **No initial pose is used**: the start position is found
from the images; motion comes from the speed and yaw rate. `/tf` and the pose fields of
`/constructor0/state_estimation` are never read by inference.
=======
Inference is causal and runs at the 10 Hz camera rate. **No initial pose is needed or accepted**: the start position
is found from the images, and the motion comes from the telemetry speed and yaw rate. `/tf` and the pose fields of
`/constructor0/state_estimation` are never read. No extra flags are needed: the defaults are the validated settings.
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510

### 4a. Test recording

```bash
<<<<<<< HEAD
python Scripts/4_infer.py <TEST_BAG.mcap> --checkpoint checkpoints/racing_model_cal.pt \
    --output results/poses/test_poses.csv
python tools/evaluate.py <TEST_GT.csv> results/poses/test_poses.csv      # official score (organisers' GT)
```

One pose per `camera_fl` frame for the whole run, header `timestamp_ns,x,y,yaw` (`timestamp_ns` = `header.stamp`
of that camera_fl image; every companion image used is stamped at or before it).

### 4b. Held-out validation laps (with ground truth)

```bash
python Scripts/1_build_dataset.py                      # cache + data/gt.csv (if not done yet)
python Scripts/4_infer.py --val-only --checkpoint checkpoints/racing_model_val_cal.pt \
    --output results/poses/val_block_poses.csv
python tools/gt_window.py data/gt.csv results/poses/val_block_poses.csv results/gt_val_block.csv
python tools/evaluate.py results/gt_val_block.csv results/poses/val_block_poses.csv
```

Why the cut: the held-out laps are 220 s of a 1194 s recording; against the full ground truth `tools/evaluate.py`
reports ~18 % coverage and "NOT RANKED". Against the cut ground truth every error number is the same and the coverage
is measured over the predicted laps. Expected output: `poses submitted 2199 evaluated 2199`, `coverage 100.0 %`,
then the scores `<ADD AFTER RUN>`.
=======
python Scripts/3_causal_infer.py <TEST_BAG.mcap> --checkpoint checkpoints/racing_model.pt \
    --output results/poses/test_poses.csv --log results/poses/test_telemetry.csv

# official score, with the organisers' ground truth of the test run
python tools/evaluate.py <TEST_GT.csv> results/poses/test_poses.csv
```

The pose file has the header `timestamp_ns,x,y,yaw` and one pose per `camera_fl` frame for the whole run
(`timestamp_ns` = `header.stamp` of that image), so `tools/evaluate.py` scores it directly. The console reports which
odometry was used (`telemetry on N/M frames`), the frame at which the start position was committed, and the number of
retrieval updates.

### 4b. Held-out validation laps (with ground truth)

Use the **validation** checkpoint, which never saw these laps.

```bash
python tools/extract_gt.py data/c0_2026-10-13_22-53-42_camloc_0.mcap results/gt.csv
python Scripts/3_causal_infer.py data/c0_2026-10-13_22-53-42_camloc_0.mcap --val-only \
    --checkpoint checkpoints/racing_model_val.pt \
    --output results/poses/val_block_poses.csv --log results/poses/val_block_telemetry.csv

# organisers' scorer: cut the ground truth to the 220 s that were predicted, then score
python tools/gt_window.py results/gt.csv results/poses/val_block_poses.csv results/gt_val_block.csv
python tools/evaluate.py results/gt_val_block.csv results/poses/val_block_poses.csv

# extra figures and metrics.txt / metrics.json (same numbers)
python tools/visualize.py --pred results/poses/val_block_poses.csv --gt results/gt.csv \
    --checkpoint checkpoints/racing_model_val.pt --telemetry results/poses/val_block_telemetry.csv \
    --out-dir results/validation --animate
```

Why the cut: the held-out laps are 220 s of a 1194 s recording. Against the full ground truth, `tools/evaluate.py`
reports 18.5 % coverage and "NOT RANKED". Against the cut ground truth it reports 100.0 % coverage; every error number
is the same either way. Expected output:

```
poses submitted        2199   evaluated 2199
coverage (1 s bins)    100.0 %   (>= 90 % to be ranked)
position RMSE  [m]     2.740   <- SCORE (lower is better)
position median/p95    0.944 / 7.399
lateral RMSE   [m]     2.040
longitudinal RMSE [m]  1.829
yaw RMSE     [deg]     0.872
within 0.25 m          5.1 %
within 0.50 m          15.5 %
within 1.00 m          53.3 %
```

This is the same protocol as the test: no initial pose, start found from the images, telemetry odometry.
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510

---

## 5. Full reproduction (training server)

<<<<<<< HEAD
```bash
python run_submission.py validate                       # stages 1-5 on the 2-lap validation protocol
python run_submission.py submit --test-bag <TEST.mcap>  # stages 2-4 with all frames -> results/poses/test_poses.csv
```

| Step | Script | Produces |
|---|---|---|
| 1 | `Scripts/1_build_dataset.py` | `data/training_cache.h5`, `data/gt.csv` |
| 2 | `Scripts/2_train.py` (`--full-train` for the submission) | checkpoint + `*_history.csv` |
| 3 | `Scripts/3_calibrate.py` (`--full` for the submission) | calibrated checkpoint (odometry, track map) |
| 4 | `Scripts/4_infer.py` | pose file + per-frame telemetry + provenance |
| 5 | `tools/gt_window.py` + `tools/evaluate.py`, `Scripts/5_evaluate.py`, `tools/visualize.py` | official score, error slices, figures |
| 6 | `Scripts/6_stress_test.py` | robustness table |

Our server: `<TIMINGS AND GPU PLACEHOLDER>` (stage 4 prints `[TIMING] per frame`).

## 6. How the method works

- **Reference map.** Every training frame is stored as a 256-D image descriptor with its known pose.
- **Network (`RacingLocNet`).** One ResNet-18 encodes the 4 front and side cameras into the descriptor; a second,
  early-fusion network sees two consecutive frames and predicts the motion (fallback odometry).
- **Motion.** Speed and yaw rate are integrated between two camera stamps and calibrated (scale, bias) on the training
  laps; the filter also estimates the remaining scale error online.
- **Finding the start and getting un-lost.** A 1-D belief over the whole lap is updated with the similarity to every
  map entry, judged relative to the lap's own median/spread; once one peak is confident and moves with the odometry,
  the filter starts there (also used to recover when lost).
- **Filter.** EKF on `[x, y, yaw, odometry scale]`, corrected by the most similar map entries near the estimate with a
  track-aligned uncertainty, then a heading prior and the real track borders.
- **Lighting robustness.** Photometric and structured lighting augmentation in training, CLAHE and robust per-camera
  normalisation. No geometric augmentation.
=======
| Step | Script | Produces |
|---|---|---|
| 1 | `Scripts/1_build_dataset.py` | `data/training_cache.h5` (4 cameras, 128×256 grayscale, causal sync, interpolated GT, telemetry odometry) |
| 2 | `Scripts/2_train_model.py` | checkpoint (network + reference map) and per-epoch `*_history.csv` |
| 3 | `Scripts/4_diagnostics.py --sweep` | filter settings chosen on the held-out laps (offline, uses GT) |
| 4 | `Scripts/3_causal_infer.py` | pose file + per-frame telemetry |
| 5 | `tools/gt_window.py`, `tools/evaluate.py` | official score of the validation laps |
| 6 | `tools/plot_training.py`, `tools/visualize.py` | figures and metrics |

```bash
NIGHT=data/c0_2026-10-13_22-53-42_camloc_0.mcap

python Scripts/1_build_dataset.py $NIGHT --output data/training_cache.h5

# validation model: 2 full laps held out, used for every number we report
python Scripts/2_train_model.py --checkpoint checkpoints/racing_model_val.pt
python tools/plot_training.py checkpoints/racing_model_val_history.csv --out results/figures/training_progress.png
python Scripts/4_diagnostics.py --sweep --checkpoint checkpoints/racing_model_val.pt   # -> diagnostics/recommended_args.txt

# validation poses, official score, figures: see section 4b

# submission model: same recipe on all laps
python Scripts/2_train_model.py --full-train --checkpoint checkpoints/racing_model.pt
```

The sweep's chosen settings are already the defaults of `Scripts/3_causal_infer.py`. After changing the training
recipe, re-run the sweep and pass its flags (`$(cat diagnostics/recommended_args.txt)`) or update `NEW_PRESET`.

Our server: build 172 s; training 20 epochs × 12.6 s; inference 7.2 ms per frame (excluding bag decoding).
`<GPU MODEL PLACEHOLDER>`

---

## 6. How the method works (one paragraph per part)

- **Reference map.** Every training frame is stored as a 256-D image descriptor together with its known pose.
- **Network (`RacingLocNet`).** One ResNet-18 encodes the 4 front and side cameras into the place descriptor. A second,
  early-fusion network sees two consecutive frames stacked together and predicts the body-frame motion with its own
  uncertainty. It is the fallback odometry.
- **Motion.** Speed and yaw rate from the telemetry are integrated between two camera stamps (about 100 Hz inside a 10 Hz
  step). The learned odometry is used only when the telemetry is missing, stale or out of range.
- **Finding the start.** Up to 96 candidate start positions are tracked in parallel for the first seconds. Each is scored by
  how well the images keep matching the map around it and by how well its path stays on the mapped track. The best is committed to.
- **Filter.** An EKF on `[x, y, yaw]` predicts with the motion estimate and corrects with the poses of the most similar map
  entries near the current estimate, trusting weak matches less. After the commit, a confident match far from the current
  estimate relocalizes it.
- **Lighting robustness.** Training uses photometric augmentation (gamma, exposure, haze, sky glow, shadows, blur). It does
  not change image geometry.
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510

## 7. Rules compliance

- **Causal:** a pose at time `t` uses only images with `header.stamp <= t` and telemetry that has already arrived.
- **No initial pose:** none is used. `--init-pose` exists for experiments and is not part of any command in this guide.
<<<<<<< HEAD
- **Ground truth:** used for training labels, calibration (training laps only) and scoring. `Scripts/4_infer.py` does
  not subscribe to `/tf`.
- **Telemetry:** only `v_mps`, `wz_radps`, `ax_mps2`, `ay_mps2` of `/constructor0/state_estimation` (`lib/state_est.py`).
- **Organisers' daylight example:** not used for training, pre-training, tuning or validation; only
  `presentation/prepare_assets.py` reads one frame as a slide illustration.
- **External resources:** ImageNet-pretrained ResNet-18 (torchvision); no external datasets.
=======
- **Ground truth:** used for training, offline diagnostics and scoring only. `Scripts/3_causal_infer.py` does not subscribe to `/tf`.
- **Telemetry:** only `v_mps`, `wz_radps`, `ax_mps2`, `ay_mps2` of `/constructor0/state_estimation` are read (see `TelemetryTrack` in `Scripts/1_build_dataset.py`).
- **Organisers' daylight example:** not used for training, pre-training, tuning or validation. The only file that reads it is `presentation/prepare_assets.py`, which extracts one frame as a slide illustration of the test domain.
- **External resources:** ImageNet-pretrained ResNet-18 (torchvision); no external datasets. Declared in [`README.md`](README.md#declared-external-resources).
- **`Scripts/4_diagnostics.py`:** uses ground truth by design and is never part of the submission path.
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510

## 8. Results

Held-out laps of the training run (frames 4777–6975, 220 s), validation model, no initial pose:

<<<<<<< HEAD
| | Position RMSE [m] | Median / p95 [m] | Lateral [m] | Longitudinal [m] | Yaw [deg] | < 1 m |
|---|---|---|---|---|---|---|
| combined release | `<ADD AFTER RUN>` | | | | | |
| reference: V branch `rmse-2.740` | 2.740 | 0.944 / 7.399 | 2.040 | 1.829 | 0.872 | 53.3 % |

Night conditions only: the test lap is in daylight and faster, which these numbers do not probe. Earlier milestones
used other protocols (see `README.md`).

## 9. Troubleshooting

See [`HOWTORUN_README.md`](HOWTORUN_README.md#7-troubleshooting).
=======
| Position RMSE [m] | Median / p95 [m] | Lateral [m] | Longitudinal [m] | Yaw [deg] | < 1 m | < 2 m | < 5 m |
|---|---|---|---|---|---|---|---|
| **2.740** | 0.944 / 7.399 | 2.040 | 1.829 | 0.872 | 53.3 % | 84.2 % | 90.6 % |

- Start-up: the first 60 frames (6 s, before the start position is committed) have RMSE 5.24 m; error at the commit
  2.23 m; RMSE after the commit 2.64 m.
- Odometry: telemetry on 2198 of 2199 frames; a retrieval update on every frame; no relocalizations needed.
- Night conditions only: the test lap is in daylight and faster, which this number does not probe.

History of the milestones: `git tag -n3`. The earlier tags (`rmse-24.508`, `rmse-13.688`) used the older protocol with
a given initial pose and a different validation window, so they are not directly comparable.

## 9. Troubleshooting

- **`[WARN] telemetry values out of range`:** the learned odometry is used for those frames. Check `[ODOM]` in the cache build log for the telemetry scale.
- **`[INFO] the learned VO disagrees with the telemetry`:** informational; the telemetry is kept.
- **`ModuleNotFoundError: mcap_ros2`:** run `pip install mcap-ros2-support`.
- **CPU only:** inference works but is slow.
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510
