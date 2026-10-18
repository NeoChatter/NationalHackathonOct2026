# Reproduce the model and the validation score

This is for reproducing our training. To only localise a recording with our trained model, use
[JURY_README.md](JURY_README.md).

Run every command from the repository root (the folder with `config.toml`). All stages read `config.toml`, so paths,
the seed and the hyperparameters are set in one place. Paths can be moved without editing the defaults: create
`server.toml` with `extends = "config.toml"` and a `[paths]` section that lists only the changed keys, then set
`RACING_CONFIG=server.toml` (or pass `--config server.toml` to a stage). `$VAR` and `~` are expanded in paths.

## 1. Requirements

| Item | Needed |
|---|---|
| OS | Linux x86_64 (CUDA wheels); other systems get CPU-only torch |
| GPU | NVIDIA GPU with a CUDA 13 capable driver (training on CPU takes hours) |
| Python | 3.11 or newer |
| Disk / RAM | about 12 GB free disk; 16 GB RAM or more (stages 2-3 load all 11,943 × 4 images, ~1.6 GB) |
| Internet | once, for the ImageNet ResNet-18 weights (stage 2, into `$TORCH_HOME`) |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"     # expect True on the GPU machine
```

## 2. Data

Put the organisers' training recording at `data/c0_2026-10-13_22-53-42_camloc_0.mcap` (`[paths].bag`, 7.0 GiB). Stage 1
writes the cache `data/training_cache.h5` and the ground truth `data/gt.csv`; stages 2-3 write the checkpoints into
`checkpoints/`.

## 3. Validation model: train and score the held-out laps

The validation model never sees two whole laps from the middle of the run (frames 4777-6975, t = 478-698 s, with a 5 s
guard band on each side). Seed 0, no initial pose.

```bash
python Scripts/1_build_dataset.py                                  # cache + data/gt.csv (~3 min)
python Scripts/2_train.py                                          # -> checkpoints/racing_model_val.pt
python tools/plot_training.py checkpoints/racing_model_val_history.csv --out results/figures/training_progress.png
python Scripts/3_calibrate.py --plot results/figures/track_map.png # -> checkpoints/racing_model_val_cal.pt
python Scripts/4_infer.py --val-only                               # -> results/poses/val_block_poses.csv
python tools/gt_window.py data/gt.csv results/poses/val_block_poses.csv results/gt_val_block.csv
python tools/evaluate.py results/gt_val_block.csv results/poses/val_block_poses.csv      # the validation score
python Scripts/5_evaluate.py --val-only                            # error slices, database coverage
python tools/visualize.py --pred results/poses/val_block_poses.csv --gt data/gt.csv \
    --checkpoint checkpoints/racing_model_val_cal.pt --telemetry results/poses/val_block_poses_telemetry.csv \
    --out-dir results/validation
```

`tools/gt_window.py` cuts the ground truth to the span of the predicted laps. Against the full 1194 s run,
`tools/evaluate.py` reports about 18 % coverage and "NOT RANKED"; against the cut ground truth every error value is
the same and the coverage is measured over the predicted laps.

Optional robustness suite: `python Scripts/6_stress_test.py` (writes `stress/run/summary.csv`).

## 4. Submission model: train on all frames

```bash
python Scripts/2_train.py --full-train                             # -> checkpoints/racing_model.pt
python Scripts/3_calibrate.py --full                               # -> checkpoints/racing_model_cal.pt
```

`checkpoints/racing_model_cal.pt` is the model that `run_submission.py` uses (the download in `JURY_README.md`). With it
in place:

```bash
python run_submission.py path/to/recording.mcap
```

## 5. Method in short

- **Reference map:** every training frame is stored as a 256-D image descriptor with its known pose.
- **Network (`RacingLocNet`):** one ResNet-18 encodes the four cameras into the descriptor; an early-fusion network sees
  two consecutive frames and predicts the motion (fallback odometry). Training uses same-place / different-lap paired
  batches, photometric and structured lighting augmentation (no geometric change), CLAHE and robust per-camera
  normalisation.
- **Motion:** speed and yaw rate are integrated between camera stamps, with scale and bias calibrated on the training
  laps; the filter also estimates the remaining scale error online.
- **Start and recovery:** a 1-D belief over the whole lap is updated with the similarity to every map entry; once one
  peak is confident and moves with the odometry, the filter starts there (also used to recover when lost).
- **Filter:** EKF on `[x, y, yaw, odometry scale]`, corrected by the most similar map entries near the estimate with a
  track-aligned uncertainty, then a heading prior and the track borders.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `[CONFIG] config file not found` | run from the repository root, or set `RACING_CONFIG` / `--config` |
| `[ERROR] bag not found` | set `[paths].bag` or pass the bag as the first argument |
| `torch.cuda.is_available()` is False | driver too old for CUDA 13, or a CPU-only torch was installed: reinstall from `requirements.txt` |
| stage 2 stalls at the first epoch or `DataLoader worker ... killed` | lower `[train].workers` (RAM), or set it to 0 to see the real error |
| `Weights only load failed` | the checkpoint was not written by these scripts |
| `NOT RANKED: coverage below threshold` on the validation block | score against the cut ground truth (`tools/gt_window.py`), not the full one |
| `ModuleNotFoundError: mcap_ros2` | `pip install -r requirements.txt` in the active venv |
| ImageNet download fails (no internet) | pre-fill `$TORCH_HOME/hub/checkpoints/resnet18-*.pth` |
