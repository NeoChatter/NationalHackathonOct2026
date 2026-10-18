# Jury Guide: Camera-Only Race Car Localization

**Team:** Vidya Sagar Vepa and Shubhangee Singhal · **Challenge:** Constructor Labs, Camera-based Localization on a Race Track  
**Repository:** https://github.com/NeoChatter/NationalHackathonOct2026

Quick links: [Submission items](#1-submission-items) · [Setup & downloads](#2-setup--downloads) · [Run inference](#3-run-inference) · [Full reproduction](#4-full-reproduction) · [Method](#5-how-the-method-works) · [Results](#6-results)

---

## 1. Submission items

| Item | Location |
|---|---|
| Source code | [`Scripts/`](Scripts) (stages 1-6 + `lib/`), [`tools/`](tools) (scoring, visualisation), [`run_submission.py`](run_submission.py) |
| Pose files | `results/poses/val_block_poses.csv` (held-out laps), `results/poses/test_poses.csv` |
| Validation results | `results/validation/official_score.txt`, `metrics.txt`, `results/poses/val_block_poses.csv.eval.json` |
| Trajectory visualisation | `results/validation/`: `trajectory.png`, `error_timeline.png`, `error_distribution.png` |
| Training progress | `results/figures/training_progress.png` |
| Robustness suite | `stress/run/summary.md` |
| Reference result (V branch, 2.740 m) | [`results/reference_v2.740/`](results/reference_v2.740) |
| Slides | `presentation/project-overview.pdf` |
| Video (≤ 5 min) | `<LINK PLACEHOLDER>` |

---

## 2. Setup & downloads

**Requirements:**
- Python 3.11 or newer
- CUDA 13 capable NVIDIA GPU
- `pip install -r requirements.txt`

**Large files (not in Git):**

| File | Location | Download |
|---|---|---|
| Submission model (all frames, calibrated) | `checkpoints/racing_model_cal.pt` | `<LINK PLACEHOLDER>` |
| Validation model (held-out laps, calibrated) | `checkpoints/racing_model_val_cal.pt` | `<LINK PLACEHOLDER>` |
| Training recording | `data/c0_2026-10-13_22-53-42_camloc_0.mcap` | `<provided by organisers>` |
| Training cache (optional, skips stage 1) | `data/training_cache.h5` | `<LINK PLACEHOLDER>` |

**Verify setup:** `python run_submission.py smoke` (runs all stages on synthetic data, ~minutes)

---

## 3. Run inference

**3a. Test recording**

```bash
python Scripts/4_infer.py <TEST_BAG.mcap> --checkpoint checkpoints/racing_model_cal.pt \
    --output results/poses/test_poses.csv
python tools/evaluate.py <TEST_GT.csv> results/poses/test_poses.csv
```

Output: one pose per camera frame, CSV with `timestamp_ns,x,y,yaw`

**3b. Held-out validation laps (with ground truth)**

```bash
python Scripts/1_build_dataset.py
python Scripts/4_infer.py --val-only --checkpoint checkpoints/racing_model_val_cal.pt \
    --output results/poses/val_block_poses.csv
python tools/gt_window.py data/gt.csv results/poses/val_block_poses.csv results/gt_val_block.csv
python tools/evaluate.py results/gt_val_block.csv results/poses/val_block_poses.csv
```

Expected output: `coverage 100.0 %`, then the scores.

---

## 4. Full reproduction

```bash
python run_submission.py validate                       # stages 1-5, 2-lap validation protocol
python run_submission.py submit --test-bag <TEST.mcap>  # stages 2-4, all frames
```

| Step | Script | Produces |
|---|---|---|
| 1 | `Scripts/1_build_dataset.py` | `data/training_cache.h5`, `data/gt.csv` |
| 2 | `Scripts/2_train.py --full-train` | checkpoint + training history CSV |
| 3 | `Scripts/3_calibrate.py --full` | calibrated checkpoint |
| 4 | `Scripts/4_infer.py` | pose CSV + provenance |
| 5 | `tools/evaluate.py`, `Scripts/5_evaluate.py` | official score + figures |
| 6 | `Scripts/6_stress_test.py` | robustness summary |

---

## 5. How the method works

- **Reference map.** Every training frame is stored as a 256-D image descriptor with its known pose.
- **Network (RacingLocNet).** ResNet-18 encodes 4 front/side cameras into a descriptor; a second network predicts motion (fallback odometry).
- **Motion.** Speed and yaw rate are integrated and calibrated (scale, bias) on training laps; the filter estimates remaining scale error online.
- **Start and recovery.** A 1-D belief over the lap is updated with image similarities. Once a peak is confident and matches odometry, the filter starts there (also used to recover when lost).
- **Filter.** EKF on `[x, y, yaw, odometry scale]`, corrected by the most similar map entries with track-aligned uncertainty.
- **Robustness.** Photometric and structured lighting augmentation in training, CLAHE and robust normalization. No geometric augmentation.

---

## 6. Results

Held-out laps of the training run (frames 4777–6975, 220 s), validation model, no initial pose:

| | Position RMSE [m] | Median / p95 [m] | Lateral [m] | Longitudinal [m] | Yaw [deg] | < 1 m |
|---|---|---|---|---|---|---|
| combined release | `<ADD AFTER RUN>` | | | | | |
| reference: V branch (2.740 m) | 2.740 | 0.944 / 7.399 | 2.040 | 1.829 | 0.872 | 53.3 % |

---

## 7. Rules compliance

- **Causal:** poses use only images with `header.stamp <= t` and telemetry already arrived.
- **No initial pose:** start position found from images; motion from speed and yaw rate.
- **Ground truth:** used for training, calibration (training laps only), and scoring.
- **Telemetry:** only `v_mps`, `wz_radps`, `ax_mps2`, `ay_mps2`.
- **Organisers' daylight example:** not used for training or tuning.
- **External resources:** ImageNet-pretrained ResNet-18 (torchvision); no external datasets.
