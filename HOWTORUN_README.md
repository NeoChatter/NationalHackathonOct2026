# How to run the combined release on the server

Follow this top to bottom. Every command runs from the repository root (the folder with `config.toml`), and every
stage reads `config.toml`, so paths, the seed and all hyperparameters are set in one place.

What the pipeline is, in one line: V's causal camera sync, RacingLocNet training recipe and 2-lap validation
protocol, combined with S's calibrated speed/yaw-rate odometry, 4-state EKF, track map and along-track start
finding (details and reasons per choice: [`DECISIONS.md`](DECISIONS.md)).

> **Not scored yet.** No number in this repository comes from this combined code. `2.740 m` is the reference result
> of the V branch, kept in `results/reference_v2.740/`. Run section 5 to get the first combined number.

---

## 1. Prerequisites and environment

| Item | Required | Source |
|---|---|---|
| OS | Linux x86_64 | `analysis/Answers.md` Q3/Q13 (training workstation) |
| GPU | NVIDIA GPU, driver that supports CUDA 13 (the torch 2.14.1 wheel bundles the CUDA 13 runtime) | same |
| Python | 3.11 or newer (3.11-3.14) | required by the pins (numpy 2.4.6, scipy 1.17.1, pandas 3.0.6) and `tomllib` |
| Disk | ~12 GB free: bag 7.0 GiB, cache ~1.6 GB, 4 checkpoints ~110 MB each, outputs | sizes from the analysis |
| RAM | 16 GB or more (stages 2-3 load all 11,943 × 4 images, ~1.6 GB, plus DataLoader workers) | estimate |
| Internet | once, for the ImageNet ResNet-18 weights (stage 2, into `$TORCH_HOME`) | `[train].pretrained = true` |

```bash
cd combinedrelease
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt                     # exact pins; ~3-4 GB of CUDA wheels on Linux
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.14.1 True <your GPU>`. If `False`, fix the driver before going on (stages 2-4 also run on CPU, but
training would take hours). Only for the slides: `pip install -r presentation/requirements.txt`.

## 2. Data and checkpoint locations

All paths are in `[paths]` of `config.toml`, relative to the repository root unless absolute. Defaults:

| Key | Default | What | Needed by |
|---|---|---|---|
| `bag` | `data/c0_2026-10-13_22-53-42_camloc_0.mcap` | organisers' night training recording (7.0 GiB) | 1, 4 `--val-only`, 6 |
| `test_bag` | *(empty)* | hidden test recording; or pass `--test-bag` | 4 (submission) |
| `cache` | `data/training_cache.h5` | written by stage 1 | 2, 3 |
| `gt_csv` | `data/gt.csv` | full-run ground truth, written by stage 1 | 5, 6, scoring |
| `val_checkpoint` → `val_checkpoint_cal` | `checkpoints/racing_model_val.pt` → `..._val_cal.pt` | validation model (2 laps held out), before / after stage 3 | 3-6 |
| `checkpoint` → `checkpoint_cal` | `checkpoints/racing_model.pt` → `..._cal.pt` | submission model (all frames) | 3, 4 |
| `results_dir`, `stress_dir`, `runs_dir` | `results/`, `stress/`, `runs/` | outputs | – |
| `day_sample_bag` | `data/rosbag_2026-10-13_16-08-46_cameras_lap/..._0.mcap` | organisers' daylight sample, **slides only** | `presentation/prepare_assets.py` |

To use other locations on the server, do not edit the defaults; create `server.toml` next to `config.toml`:

```toml
extends = "config.toml"
[paths]
bag = "/data/hackathon/c0_2026-10-13_22-53-42_camloc_0.mcap"     # example: put the real server paths here
cache = "/scratch/racing/training_cache.h5"
gt_csv = "/scratch/racing/gt.csv"
```

and run everything with `export RACING_CONFIG=$PWD/server.toml` (or `--config server.toml` on each command).
`$VAR` and `~` are expanded in paths. Checkpoints are written by stages 2-3; none are needed to start.

## 3. Smoke test first (minutes on a GPU, no real data needed)

```bash
python run_submission.py smoke
```

It writes a synthetic 5-lap bag (`tools/make_synthetic_bag.py`) and runs **every** stage and tool on it with a tiny
configuration (2 epochs × 3 batches, no ImageNet download): build → train → plot → calibrate → infer → official
scorer → slices → figures → 2 stress scenarios → full-train → calibrate → infer on a "test" bag → figures.
Expected last lines:

```
[SMOKE TEST PASSED] every stage ran end to end on the synthetic bag (... s). ...
[SMOKE] removed .../runs/smoke (use --keep to inspect it)
```

The RMSE it prints is meaningless (synthetic images, an untrained network). It took 13-18 min on a laptop CPU
(Windows, CPU-only torch); on the server GPU it should take a few minutes, and it also checks that CUDA works for
every stage. `--keep` keeps `runs/smoke/` for inspection.

## 4. Commands in run order

`python run_submission.py validate` runs 4.1-4.6 in this order; `python run_submission.py submit --test-bag <TEST>`
runs 4.7. `--dry-run` prints the exact commands, `--from-stage N` resumes. The individual commands are:

| # | Command | Writes | Expected output | Rough time (GPU) |
|---|---|---|---|---|
| 4.1 | `python Scripts/1_build_dataset.py` | `data/training_cache.h5`, `data/gt.csv` | `[DONE] 11943 synchronized 4-camera samples` (± a few), `[SYNC] ... future-stamped ... ~50 %`, `[ODOM] raw distance scale ~0.997` | ~3 min |
| 4.2 | `python Scripts/2_train.py` | `checkpoints/racing_model_val.pt`, `..._val_history.csv` | `[SPLIT] block: frames 4777-6975 (t = 478-698 s ...)`, 20 epoch lines with `[VAL-RET]` / `[VAL-VO]`, `[CALIB] ...`, `[TRACK] L=... borders=yes` | ~15-40 min |
| 4.2b | `python tools/plot_training.py checkpoints/racing_model_val_history.csv --out results/figures/training_progress.png` | PNG | `saved ...` | seconds |
| 4.3 | `python Scripts/3_calibrate.py --plot results/figures/track_map.png` | `checkpoints/racing_model_val_cal.pt` | `[SE] fitted on ... TRAINING frame pairs: v_scale ~1.00, wz_scale ~1.0 ... (scale+bias)`, `[SE] held-out: ... (report only)`, `[TRACK] ...` | 1-3 min |
| 4.4 | `python Scripts/4_infer.py --val-only` | `results/poses/val_block_poses.csv` (+ `_telemetry.csv`, `.meta.json`) | `2199 poses written`, `Start: global initialisation after N frames`, `Odometry: ~2198 frames from state estimation` | 1-3 min |
| 4.5 | `python tools/gt_window.py data/gt.csv results/poses/val_block_poses.csv results/gt_val_block.csv` then `python tools/evaluate.py results/gt_val_block.csv results/poses/val_block_poses.csv` | (score on stdout) | `poses submitted 2199 evaluated 2199`, `coverage 100.0 %`, `position RMSE [m] ...` | seconds |
| 4.6 | `python Scripts/5_evaluate.py --val-only` and `python tools/visualize.py --pred results/poses/val_block_poses.csv --gt data/gt.csv --checkpoint checkpoints/racing_model_val_cal.pt --telemetry results/poses/val_block_poses_telemetry.csv --out-dir results/validation` | `.eval.json`, figures, `metrics.txt/json` | same RMSE as 4.5, error slices by mode / 200-frame block | < 1 min |
| (6) | `python Scripts/6_stress_test.py` (or `validate --stress`) | `stress/run/summary.{csv,md}` | 10 scenarios, `RMSE ... x clean` | 15-30 min |
| 4.7 | `python Scripts/2_train.py --full-train` → `python Scripts/3_calibrate.py --full` → `python Scripts/4_infer.py <TEST.mcap> --output results/poses/test_poses.csv` | `checkpoints/racing_model{,_cal}.pt`, **`results/poses/test_poses.csv`** | one pose per camera_fl frame of the test bag, header `timestamp_ns,x,y,yaw` | ~15-30 min |

Times are scaled from the reference run's logs (build 172 s, 20 epochs × 12.6 s, 7.2 ms per frame) and are not
measured for this code: CLAHE, robust normalisation (two medians per camera image) and the structured augmentation
add DataLoader CPU work, and the per-epoch held-out evaluation adds time. If the GPU waits for data, raise
`[train].workers` (one per free CPU core).

## 5. Reproduce the reported RMSE (protocol)

The protocol is the one of the `rmse-2.740` reference (analysis §6, Answers ruling 2), so the numbers are comparable:

- **Split:** 2 whole laps from 40 % of the run, frames **4777-6975** (t = 478-698 s), ±5 s guard band out of training
  and the map (`[split]` in `config.toml`). Check that the `[SPLIT]` line of stage 2 prints these frames.
- **Seed:** 0 (`[run].seed`), no initial pose, speed/yaw-rate odometry.
- **Metric:** the organisers' `tools/evaluate.py` (position RMSE; lateral, longitudinal, yaw, within-x m), on the
  ground truth cut to the predicted span by `tools/gt_window.py`, i.e. 100 % coverage of the block.

```bash
python run_submission.py validate          # = 4.1 ... 4.6; the reported number is in results/validation/official_score.txt
```

Compare with `results/reference_v2.740/validation/metrics.txt` (2.740 / median 0.944 / lateral 2.040 /
longitudinal 1.829 / yaw 0.872 deg). The `rmse-4.363` result used another window (last 20 %, incl. the pit lane) and
is **not** comparable; `[split].val_mode = "tail"` reproduces that split if needed.

**Quick A/B without retraining (optional, ~5 min):** the filter half of the merge can be checked with the existing
V model `racing_model_val.pt` (108 MB, on the training workstation; copy it to `checkpoints/v2740/`) before the long
retrain:

```bash
python Scripts/3_calibrate.py --checkpoint checkpoints/v2740/racing_model_val.pt --output checkpoints/v2740/racing_model_val_cal.pt
python Scripts/4_infer.py --val-only --checkpoint checkpoints/v2740/racing_model_val_cal.pt --output runs/v2740_newfilter.csv
python tools/gt_window.py data/gt.csv runs/v2740_newfilter.csv runs/gt_cut.csv && python tools/evaluate.py runs/gt_cut.csv runs/v2740_newfilter.csv
```

That checkpoint has no `preproc` entry, so it runs with V's mean/std normalisation, as it was trained. Stage 3 also
measures its missing retrieval calibration. It needs the cache from 4.1 (same causal sync, so the frames match).

## 6. Known issues

- **Not trained or scored end to end on real data.** Only the synthetic smoke test ran. The split frames, the
  calibration values and the RMSE are unverified until section 5 has run.
- **CLAHE + robust normalisation + structured augmentation were never used in a scored run** (rulings 7 / Q11). If the
  validation number is worse than 2.740, first retry with `[preproc] clahe = false, norm = "std"` and
  `[augment] structured_p = 0.0` (= V's recipe) to separate the training change from the filter change.
- **Similarity thresholds are model-specific.** `[tracker].sim_floor = 0.80` (absolute) comes from S's model;
  `sim_thresh` is measured per checkpoint. Daylight may push similarities below the floor; the along-track filter
  (relative z-scores) still works, the EKF updates then stop.
- **The VO bias curve is fitted on training pairs** (no leakage), so it is close to identity; the VO is only a fallback.
- **Start-up:** with no confident peak the tracker seeds after `init_timeout_frames = 300` frames (30 s) anyway; the
  synthetic validation run (barely trained network) hit this timeout. Check `Start: global initialisation after N
  frames` on the real block (S reported 8 frames).
- **No ground-truth replay / sweep tool** (V's `4_diagnostics.py` drives the old filter and was not ported).
- **Single seed, night only**: no variance estimate; the test lap is daylight and faster.
- `results/reference_v2.740/racing_model_val_history.csv` has 5 empty `*_day` columns from an older code revision.
- The slide deck was rebuilt without the CycleGAN text, but its result chips still say "fill in" (section 9).

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `[CONFIG] config file not found` | run from the repository root, or set `RACING_CONFIG` / `--config` |
| `[ERROR] bag not found` | set `[paths].bag` (or `test_bag`) or pass the bag as the first argument |
| `[ERROR] this cache was not built with the causal camera sync` | rebuild with stage 1 (old caches from either branch are not reused) |
| `torch.cuda.is_available()` is False | driver too old for CUDA 13, or a CPU-only torch was installed: reinstall from `requirements.txt` |
| stage 2 hangs at the first epoch or `DataLoader worker ... killed` | lower `[train].workers` (RAM), or set it to 0 to see the real error |
| `Weights only load failed` | the checkpoint is from another code base; only checkpoints from these scripts or V's `racing_model_val.pt` load |
| `[SE][WARN] ... implausible` or `scale fit rejected` | check the state-estimation fields; stage 4 then falls back to the learned VO |
| `[INFO] the learned VO disagrees with the speed signal` | informational: the speed signal is kept |
| `Start: NEVER initialised` | the track map is missing (`[TRACK][WARN]`) or the descriptors do not match: re-run stage 3, check `sim_thresh` |
| `NOT RANKED: coverage below threshold` on the validation block | score against the cut GT (4.5), not the full GT |
| `ModuleNotFoundError: mcap_ros2` | `pip install -r requirements.txt` in the active venv |
| ImageNet download fails (no internet) | pre-fill `$TORCH_HOME/hub/checkpoints/resnet18-*.pth`, or `--no-pretrained` (worse results) |

## 8. Before production checklist

- [ ] `pip install -r requirements.txt` succeeded on the server and `torch.cuda.is_available()` is True.
- [ ] `python run_submission.py smoke` passes on the server (checks CUDA and every stage).
- [ ] Server paths set in a `server.toml` (`extends = "config.toml"`), not by editing defaults; `RACING_CONFIG` exported.
- [ ] Stage 1 log: ~11,943 samples, `[SYNC]` future-stamped share ~50 %, `[ODOM]` scale ~0.997.
- [ ] Stage 2 log: `[SPLIT] block: frames 4777-6975`; otherwise the score is not comparable with 2.740.
- [ ] Stage 3 log: `v_scale` and `wz_scale` near 1 with `(scale+bias)`, held-out drift small, track map `borders=yes`.
- [ ] Validation RMSE from 4.5 compared with 2.740 (and lateral with 2.040, the tie-breaker); decide on the
      preprocessing / augmentation defaults (section 6) before training the submission model.
- [ ] Optional quick A/B of section 5 done, so the filter change and the training change are known separately.
- [ ] Stress suite run; no scenario fails; `Start:` within a few seconds on `clean`.
- [ ] Submission: `--full-train` model, `test_poses.csv` has one pose per camera_fl frame, header
      `timestamp_ns,x,y,yaw`, strictly increasing stamps, coverage ≥ 90 % of 1 s bins.
- [ ] No `--init-pose` in any submission command; `/tf` is never an input of stage 4 (it is not in its topic list).
- [ ] Record the GPU model, ms per frame (`[TIMING]` line of stage 4) and the SHA-256 of the final checkpoints (in
      `<output>.meta.json` / checkpoint provenance) in `JURY_README.md` and `presentation/results_to_fill.json`.
- [ ] Rebuild the slides (`prepare_assets.py` → `build_deck.py` → `render_deck.ps1` on Windows) and fill the placeholders.

## 9. Slides

```bash
pip install -r presentation/requirements.txt
python presentation/prepare_assets.py    # reads [paths].bag, cache, day_sample_bag (one frame each)
python presentation/build_deck.py        # -> presentation/project-overview.pptx
powershell -File presentation/render_deck.ps1   # Windows + PowerPoint only: -> project-overview.pdf
```

## 10. Assumptions (not defined in `analysis/`)

- **GPU model and count:** unknown (`<SERVER GPU PLACEHOLDER>` in the analysis). Assumed one CUDA 13 capable GPU;
  the code uses one device (`[run].device`).
- **Absolute server paths:** not defined; the analysis only gives repository-relative `data/` and `checkpoints/`.
  Defaults are those relative paths; set real ones in `server.toml`.
- **Python version on the server:** not stated; the pinned packages need 3.11+.
- **CPU cores / RAM:** not stated; `[train].workers = 4` (V's default) and ≥ 16 GB RAM assumed.
- **Internet access** for the one-time ImageNet weight download: assumed.
- **Test bag:** assumed to have the same camera and state-estimation topics as the training bag, and no `/tf`.
- **Existing checkpoints** (`racing_model_val.pt`, `racing_model.pt`, `se_cal.pt`) are on the training workstation
  only; they are not needed (stages 2-3 recreate everything), except for the optional A/B in section 5.
