# Jury Guide: Camera-Only Race Car Localization

> **Team:** `<TEAM NAME>` · **Challenge:** Constructor Labs, Camera-based Localization on a Race Track
> **Repository:** https://github.com/NeoChatter/NationalHackathonOct2026 (this snapshot: combined release of the
> branches `vidya` (tag `rmse-2.740-vidya-2026-10-18-1712`) and `shubi` (tag `rmse-4.363-shubi`))

This page tells you where every submission item lives and how to reproduce our pose files. The technical write-up
(rules checklist, declared external resources) is in [`README.md`](README.md); the full run guide is
[`HOWTORUN_README.md`](HOWTORUN_README.md).

---

## 1. Submission items

| Item | Location |
|---|---|
| Source code | [`Scripts/`](Scripts) (stages 1-6 + `lib/`), [`tools/`](tools) (scoring, visualisation), [`run_submission.py`](run_submission.py) |
| Pose files | `results/poses/val_block_poses.csv` (held-out laps, written by `run_submission.py validate`), `results/poses/test_poses.csv` `<ADD AFTER TEST>` |
| Validation results | `results/validation/official_score.txt`, `metrics.txt` / `metrics.json`, `results/poses/val_block_poses.csv.eval.json` `<ADD AFTER RUN>` |
| Trajectory visualisation | `results/validation/`: `trajectory.png`, `error_timeline.png`, `error_distribution.png` `<ADD AFTER RUN>` |
| Training progress | `results/figures/training_progress.png` (data: `checkpoints/racing_model_val_history.csv`) `<ADD AFTER RUN>` |
| Robustness suite | `stress/run/summary.md` `<ADD AFTER RUN>` |
| Reference result of the V branch | [`results/reference_v2.740/`](results/reference_v2.740) (poses, metrics, figures, filter sweep) |
| Slides | [`presentation/project-overview.pdf`](presentation) (source: `presentation/build_deck.py`; result chips `<FILL AFTER RUN>`) |
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
| Submission model (all frames, calibrated) | `checkpoints/racing_model_cal.pt` | `<LINK PLACEHOLDER>` |
| Validation model (never saw the held-out laps, calibrated) | `checkpoints/racing_model_val_cal.pt` | `<LINK PLACEHOLDER>` |
| Training recording | `data/c0_2026-10-13_22-53-42_camloc_0.mcap` | `<provided by the organisers>` |
| Training cache (optional, skips stage 1) | `data/training_cache.h5` | `<LINK PLACEHOLDER>` |

`<SHA-256 CHECKSUMS PLACEHOLDER>` (also recorded in every checkpoint's provenance chain and `<output>.meta.json`)

---

## 3. Setup

- **Python:** 3.11 or newer.
- **GPU:** one CUDA 13 capable NVIDIA GPU. Training was done on `<SERVER GPU PLACEHOLDER>`.
- **Install:** `pip install -r requirements.txt` (exact versions; CUDA wheels on Linux, CPU-only elsewhere).
- **Check:** `python run_submission.py smoke` runs every stage on a small synthetic bag (minutes).

All commands run from the repository root. Paths come from `config.toml` (`[paths]`).

---

## 4. Reproduce a pose file and score it with the organisers' scorer

Inference is causal and runs at the 10 Hz camera rate. **No initial pose is used**: the start position is found
from the images; motion comes from the speed and yaw rate. `/tf` and the pose fields of
`/constructor0/state_estimation` are never read by inference.

### 4a. Test recording

```bash
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

---

## 5. Full reproduction (training server)

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

## 7. Rules compliance

- **Causal:** a pose at time `t` uses only images with `header.stamp <= t` and telemetry that has already arrived.
- **No initial pose:** none is used. `--init-pose` exists for experiments and is not part of any command in this guide.
- **Ground truth:** used for training labels, calibration (training laps only) and scoring. `Scripts/4_infer.py` does
  not subscribe to `/tf`.
- **Telemetry:** only `v_mps`, `wz_radps`, `ax_mps2`, `ay_mps2` of `/constructor0/state_estimation` (`lib/state_est.py`).
- **Organisers' daylight example:** not used for training, pre-training, tuning or validation; only
  `presentation/prepare_assets.py` reads one frame as a slide illustration.
- **External resources:** ImageNet-pretrained ResNet-18 (torchvision); no external datasets.

## 8. Results

Held-out laps of the training run (frames 4777–6975, 220 s), validation model, no initial pose:

| | Position RMSE [m] | Median / p95 [m] | Lateral [m] | Longitudinal [m] | Yaw [deg] | < 1 m |
|---|---|---|---|---|---|---|
| combined release | `<ADD AFTER RUN>` | | | | | |
| reference: V branch `rmse-2.740` | 2.740 | 0.944 / 7.399 | 2.040 | 1.829 | 0.872 | 53.3 % |

Night conditions only: the test lap is in daylight and faster, which these numbers do not probe. Earlier milestones
used other protocols (see `README.md`).

## 9. Troubleshooting

See [`HOWTORUN_README.md`](HOWTORUN_README.md#7-troubleshooting).
