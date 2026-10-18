"""
Library code shared by the numbered pipeline scripts in Scripts/ (no command-line entry points here).

  config      config.toml loader: paths, seeds, hyperparameters for every stage                       (new)
  common      camera topics, crop, causal camera sync (select_causal_image), stamps, angles          (V + S)
  provenance  git + argument + config + input fingerprints (full SHA-256) for every run / checkpoint   (S)
  models      RacingLocNet (place descriptor + VO), preprocessing (CLAHE, robust norm), checkpoint I/O (V + S)
  augment     PhotometricAugment + structured lighting effects for training                         (V + S)
  data        training dataset, paired batch sampler, held-out metrics, calibration measurements    (V + S)
  split       guarded 2-lap validation block / tail / full-train split                               (V)
  state_est   whitelisted speed / yaw-rate inputs, causal integration, calibration, plausibility     (S + V)
  track_map   track borders -> centre line, Frenet (s, n), corridors, heading prior                  (S)
  sfilter     along-track grid filter (start finding without a start pose, relocalisation)          (S)
  tracker     4-state EKF [x, y, yaw, odometry scale] + lost mode + track constraints                (S)
  memory      causal experience memory (off by default)                                             (S)
  perturb     stress perturbations for the robustness suite                                         (S)
  metrics     scoring (same rules as tools/evaluate.py / gt_window.py), slices, coverage, recovery   (S)

V = rmse-2.740-vidya-2026-10-18-1712, S = rmse-4.363-shubi (see DECISIONS.md).
"""
