"""
Library code shared by the numbered pipeline scripts in Scripts/ (no command-line entry points here).

  common      camera topics, crop, stamps, angles, MCAP readers (GT / first pose)
  provenance  git + argument + input fingerprints for every run and every checkpoint
  models      PlaceNet / OdomNet, input preprocessing (CLAHE, normalisation), training augmentation,
              test-time BatchNorm adaptation
  data        training datasets, embedding, retrieval / odometry evaluation used for calibration
  track_map   track borders -> centerline, Frenet (s, n), corridors, heading prior
  sfilter     along-track grid filter (condition-independent relocalisation)
  memory      causal experience memory (adds today's descriptors to the place database)
  tracker     the localisation filter (EKF + lost mode)
  perturb     stress perturbations for the robustness suite
  metrics     scoring (same rules as tools/evaluate.py), slices, coverage, recovery
"""
