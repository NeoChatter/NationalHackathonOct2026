# Decisions: what came from which branch, and why

V = `rmse-2.740-vidya-2026-10-18-1712/`, S = `rmse-4.363-shubi/`. "Ruling N" / "Qn" = `analysis/Answers.md`,
"report §n" = `analysis/ANALYSIS_REPORT.md`. Items marked **(own)** are choices the analysis did not make.

## Structure and packaging

| # | Decision | From | Why |
|---|---|---|---|
| 1 | Numbered stages `1_build_dataset` → `2_train` → `3_calibrate` → `4_infer` → `5_evaluate` → `6_stress_test` + `Scripts/lib/` | S | Ruling 8 and the blueprint (stage 3 = calibration, stage 4 = inference); scripts now run in numeric order (V's ran 4 before 3). |
| 2 | V's `4_diagnostics.py` (GT replay + sweep) not ported | – | It drives V's `FusionLocalizer`/`MultiStartLocalizer`, which rulings 4-6 replace, and its sweep tuned on the scored laps (report §6.3); `5_evaluate` slices + `6_stress_test` cover part of it. |
| 3 | One config file `config.toml` (paths, seed, every hyperparameter; `extends` overlays; CLI flags still override) | **(own)** | Task requirement: no hardcoded local paths. |
| 4 | Single run script `run_submission.py` (`smoke`, `validate`, `submit`, `all`) | **(own)** | Blueprint step 4 asks for one script from raw bag to `submission.csv`. |
| 5 | `requirements.txt` = V's 48 pins + `scipy==1.17.1`; Linux markers on the CUDA/triton wheels | V | Blueprint step 4; scipy 1.17.1 supports Python 3.11-3.14 and numpy 2.4.6 (1.18 needs ≥ 3.12). |
| 6 | `presentation/requirements.txt` numpy 2.5.4 → 2.4.6 | V | Blueprint step 4 (pin conflict, report §2.1). |
| 7 | Provenance records on every output and checkpoint, plus full SHA-256 (cached per path/size/mtime) and the resolved config | S | Ruling 8, blueprint step 1. |
| 8 | Checkpoints saved weights-only safe, always loaded with `torch.load(weights_only=True)` | **(own)** | No pickle code execution on load; V's tools already loaded checkpoints that way. |
| 9 | Python ≥ 3.11 (V's docs said ≥ 3.10) | – | Required by the pins (numpy 2.4.6, pandas 3.0.6, scipy 1.17.1) and by `tomllib`. |

## Data and protocol

| # | Decision | From | Why |
|---|---|---|---|
| 10 | Camera sync: newest image with stamp ≤ camera_fl (`select_causal_image`, buffer 64), same function in stages 1 and 4; poses stamped at camera_fl; S's newest-logged pairing and `pose_at` extrapolation removed | V | Ruling 1, Q9 (non-causal pairing = disqualification risk). |
| 11 | Cache = V's samples + S's `state_est/*`, `track_bounds/marker_*`, `--export-gt`, `--aux-only`, provenance; training refuses a cache without `causal_sync=1` | V + S | Q2 ruling and blueprint step 1. |
| 12 | V's `TelemetryTrack` / integrated `odom` dataset dropped; S's `lib/state_est` is the only telemetry reader | S | Ruling 4; report §9 ("logic implemented twice"); only V's dropped `4_diagnostics` read `odom`. |
| 13 | Flat `track_borders` built with S's `markers_from_msg` (marker pose applied) | S | Report §3.3: identical to V if marker poses are identity, more correct otherwise. |
| 14 | Validation split: 2 whole laps from 40 % of the run, ±5 s guard kept out of training and the map (frames 4777-6975 on the real bag); `val_mode = "tail"` kept for S-comparable runs | V | Ruling 2; report §6 (scores only comparable on the same window). |
| 15 | Seed 0, no cuDNN determinism flag by default (`[run].deterministic` switch added) | V | Same as the 2.740 reference run (report §6.7). |
| 16 | Submission model trained and mapped on 100 % of the frames (`--full-train`) | V | Ruling 3. |

## Model and training

| # | Decision | From | Why |
|---|---|---|---|
| 17 | `RacingLocNet` (ResNet-18 place descriptor + aux pose head + early-fusion `PairMotionNet` VO), soft spatial contrastive loss, beta-NLL VO loss, `LocationPairBatchSampler`; S's PlaceNet/OdomNet/InfoNCE dropped | V | Ruling 7. |
| 18 | V's 2.740 hyperparameters (20 epochs, batch 64, lr 3e-4, still 0.10, noise 0.04, photo-aug 1.0 / p 0.85) | V | Ruling 7 names V's recipe; S's 40 + 15 epochs belong to the dropped networks. |
| 19 | CLAHE on + robust (median/MAD) normalisation, stored in `ck['preproc']` | S | Ruling 7 ("S's CLAHE and robust normalization") and Q11. Unverified in any scored run. |
| 20 | Augmentation = V's PhotometricAugment + S's structured effects on top with `structured_p = 0.5`; S's own gamma/gain/blur/noise draws not repeated | V + S | Q11 enables both; **(own)** p = 0.5 so two full-strength pipelines are not stacked on every sample. |
| 21 | V's image-standstill threshold (`still_thresh`) dropped | V | Q7: zero effect with telemetry; S's tracker has its own stationary handling. |
| 22 | Retrieval calibration (noise stds, similarity threshold) measured on training frames against the map minus ±20 s (`train_crosspass`); S's measurement on the validation laps kept as `source = "val"` | **(own)** | Report §5b/§6.3 flag S's version as leakage; also gives the full-train model its own calibration (replaces S's `--calib-ckpt`). |

## Calibration and inference

| # | Decision | From | Why |
|---|---|---|---|
| 23 | Speed scale, yaw-rate scale + bias, per-frame noise, fitted on the training (= database) frames, not `ts < val_start` | S | Ruling 4 + blueprint step 2 (`~val_mask`). |
| 24 | VO bias curve + VO noise fitted on training pairs (S fitted them on the scored laps) | S | Blueprint "strictly on the non-validation split"; near-identity expected, VO is only a fallback. |
| 24b | Bug fix in S's `fit_calibration`: a degenerate / implausible yaw-rate scale fit now falls back to scale 1 + bias-only fit (S kept the other solution's bias, doubling the heading change) | S, fixed | Found by the smoke test (constant yaw rate); no effect when the scale fit is plausible (0.5-2). |
| 25 | Stage 3 fills a missing retrieval calibration, so V's existing `racing_model_val.pt` can run through the new filter without retraining | **(own)** | Cheap A/B of rulings 4-6 on the server before the long retrain. |
| 26 | Track map (centre line, real borders, pit corridor, heading prior, border clamp); reference-lap search skips "laps" that jump across the hole left by the validation block | S | Ruling 5; the gap check is needed because V's split cuts the block out of the middle of the database. |
| 27 | Start finding and relocalisation: S's `AlongTrackFilter` (median/MAD relative similarity); V's `MultiStartLocalizer` dropped | S | Ruling 6. |
| 28 | Filter: S's 4-state EKF `[x, y, yaw, k]`, chi² gating, track-aligned covariance, constants in `[tracker]`; V's `FusionLocalizer` and its sweep-tuned `NEW_PRESET` dropped | S | Rulings 4-6; V's tuned values do not transfer to another filter. |
| 29 | Odometry per frame: speed/yaw rate if plausible (V's range check), else learned VO, else coast; V's `OdomHealth` VO cross-check, monitoring only | V + S | Blueprint step 3. |
| 30 | Every camera_fl anchor gets a pose; a missing or > 300 ms old companion image only skips the visual update (S skipped the whole set) | V | Blueprint step 3 (100 % emission, ≥ 90 % coverage rule). |
| 31 | Start pose: global initialisation only; `--init-pose X Y YAW` (handed in by the caller) replaces S's `--init first-pose`, which read `/tf` inside the inference script | **(own)** | Q14 allows optional seeding "if provided by the harness"; keeps "inference never reads /tf" true. |
| 32 | Experience memory, adaptive similarity, AdaBN kept but off | S | Q11. |
| 33 | `--val-only` starts reading the bag 10 s before the block (V decoded from the start) | **(own)** | Same companion images and telemetry for the first validation frame, much faster; stops at `val_end` like V. |

## Evaluation and reporting

| # | Decision | From | Why |
|---|---|---|---|
| 34 | Reported number = organisers' `tools/evaluate.py` on GT cut by `tools/gt_window.py`; `5_evaluate --val-only` cuts GT the same way (S cut at `val_start` only) | V | Report §6.6; S's crop would score the block against GT up to the end of the run. |
| 35 | Telemetry CSV = S's columns + camera age, visual-update flag, VO dx, VO/speed ratio; `visualize.py` reads both formats | S + V | Needed by `5_evaluate` slices and V's figures. |
| 36 | V's 2.740 artefacts (poses, metrics, figures, sweep, history CSV) moved to `results/reference_v2.740/` and labelled as reference | V | Q12: the deck shows 2.740 as the verified benchmark; they are not outputs of this code. |
| 37 | Deck: CycleGAN / daylight-training statements and the old start / re-lock text replaced in `build_deck.py`; `.pptx` and `.pdf` rebuilt from it; result chips stay "fill in" | V | Q6 ruling ("purge all references"), report §7; no combined result exists yet. |
| 38 | `presentation_outline.md` and `pipeline_comparison_shubi.md` moved to `docs/history/` with a "superseded" banner | V, S | Report §7 (stale). |
| 39 | Synthetic MCAP generator + `run_submission.py smoke` | **(own)** | Neither branch had a dry run; the task asked for a quick end-to-end check. |
