# Pipeline comparison — `shubi` branch vs. `rmse-13.688` baseline

Branch: `shubi`  ·  Baseline tag: `rmse-13.688`  ·  Milestones: `rmse-10.426-shubi` → `rmse-8.684-shubi` (latest)
Evaluation: `--val-only` run on the held-out laps, scored with `Tools/evaluate.py gt_val.csv val_submission.csv`.

> This doc has two milestones. Milestone 1 (below) is the odometry-calibration work. Milestone 2 (at the
> bottom) is the track-frame work that took the val RMSE from 10.426 to 8.684 m.

> Note on comparability: the `10.426` number here is a **validation-split** score (last ~20% of frames,
> 2387 poses). The baseline `13.688` recorded in the repo tag is the sweep-tuned milestone for the same
> pipeline family. Treat the headline as "my `shubi` pipeline scores 10.426 on the val split"; a clean
> apples-to-apples claim needs the baseline re-run on the *same* `gt_val.csv` split. See "Caveats" below.

## Headline

| metric | baseline run (uncalibrated) | `shubi` (calibrated) | change |
|---|---:|---:|---:|
| **position RMSE [m]**  | 29.680 | **10.426** | **−19.25 (−65%)** |
| position median [m]    | 1.811  | 1.062  | −0.75 |
| position p95 [m]       | 93.733 | 32.427 | −61.3 |
| **longitudinal RMSE [m]** | 29.315 | 8.109 | **−21.2 (−72%)** |
| **lateral RMSE [m]**   | 4.641  | 6.552  | **+1.91 (+41%) — worse** |
| yaw RMSE [deg]         | 4.190  | 1.710  | −2.48 |
| within 0.25 m          | 2.6 %  | 6.1 %  | +3.5 pp |
| within 0.50 m          | 9.9 %  | 20.6 % | +10.7 pp |
| within 1.00 m          | 26.6 % | 47.7 % | +21.1 pp |

The two runs above are the `[53]/[54]` run (model `racing_model.pt`, no odometry calibration) and the
`[56]/[57]` run (model `racing_model_cal.pt`, after `calibrate_odom.py`). The headline improvement comes
almost entirely from fixing the **longitudinal** (along-track) drift.

## What changed in the `shubi` pipeline

The code moved from one monolithic `3_causal_infer.py` (with `CausalSE2EKF` inline) to a modular layout:

- **`Scripts/locnet.py`** (new) — shared model + preprocessing, replacing the `importlib` load hack.
  Splits the old single `RacingLocNet` into two explicit heads:
  - `PlaceNet`: 4 cameras → one L2-normalised place descriptor (per-camera attention gate, no BatchNorm in the head).
  - `OdomNet`: **early-fusion** ego-motion. The first conv sees `[prev, curr, curr−prev]` per camera, so
    it can actually measure pixel displacement. Output is body-frame `(dx, dy, dyaw)`.
- **`Scripts/tracker.py`** (new) — numpy-only SE(2) filter (`Tracker`) replacing `CausalSE2EKF` plus the
  mode logic. Key physical priors encoded here:
  1. Car moves along its heading → only `(dx_body, dyaw)` from vision are used; `dy_body` is ignored.
  2. VO has a **scale error** → process noise grows with `dx`, and an online scale `k` is estimated.
  3. Place retrieval is accurate **across** the track, poor **along** it → anisotropic, heading-rotated
     measurement covariance (`retr_along_std` ≫ `retr_lat_std`).
  4. Losing the track never jumps: search radius grows while lost; a reset needs 5 consecutive consistent matches.
  5. Corridor clamp: lateral offset from the nearest training path is clamped to `lat_clamp` (7 m).
- **`Scripts/calibrate_odom.py`** (new) — the key accuracy change. Measures the odometry network's
  **speed-dependent forward-distance bias** on the held-out laps and stores a correction curve
  (`dx_map_pred` → `dx_map_true`) in the checkpoint. No retraining.
- **`Scripts/2_train_model.py`**, **`Scripts/3_causal_infer.py`** (modified) — rewritten to use the modules above.
- `Scripts/4_diagnostics.py` is **unchanged** from `main` (not adapted to this pipeline — see Caveats).

## Why longitudinal error collapsed (the win)

The calibration log is explicit about the bias:

```
[CAL] forward-step error: raw median |err| 0.398 m, bias -107.7% -> calibrated median |err| 0.338 m
```

The raw `true/pred` ratios show the OdomNet **under-reads distance at low speed and over-reads at high
speed** (classic regression-to-the-mean):

| ~speed | true/pred |
|---:|---:|
| 12.0 m/s | 0.69 (over-reads) |
| 18.3 m/s | 1.13 |
| 25.4 m/s | 1.12 |
| 33.2 m/s | 1.06 (under-reads) |

Because the error is in the **forward** step and it accumulates between place-retrieval updates, it shows
up as along-track (longitudinal) drift — exactly the component that dropped from 29.3 m to 8.1 m. The
per-frame `odom scale k` in the two logs confirms it: pre-calibration `k` swings wildly (0.94–1.37);
post-calibration it sits much closer to 1.0 because the curve already removed most of the bias, leaving
the online `k` only a small residual to track.

Secondary effect: stable odometry keeps the filter in `TRACKING` far more. In the first run the filter
fell into long `RELOCALIZING` stretches (`lost=112`, `lost=192`); after calibration most of the run stays
`TRACKING`, which is why p95 error dropped from 93.7 m to 32.4 m and yaw RMSE more than halved.

## Why lateral RMSE got worse (4.64 → 6.55 m)

This is the regression to investigate. A few code-grounded hypotheses, most likely first:

1. **The filter now trusts odometry more, so it leans less on across-track retrieval corrections.**
   Retrieval is the component that pins down lateral position (prior #3). With the longitudinal problem
   fixed, fewer/weaker updates fire, and small lateral biases in the VO integration are no longer being
   corrected as aggressively. The anisotropic R intentionally keeps `retr_lat_std` small (tight laterally),
   but that only helps when an update actually fires.

2. **`lat_clamp = 7.0 m` is a blunt lateral prior.** The corridor clamp snaps the estimate toward the
   nearest training path. On a wide racing line or where laps differ laterally, clamping to the nearest
   DB path can *introduce* lateral error rather than remove it. Worth sweeping `lat_clamp` (e.g. 3–10 m)
   and logging how often `excess != 0` fires.

3. **Relocalization tail.** The end of the calibrated run still has a `RELOCALIZING` tail
   (`lost=92`, `lost=192`, sim ~0.81–0.86 around frames 2200–2300). During reloc the lateral estimate
   comes from a 5-match chain mean, which can sit off the true line and inflate lateral RMSE even while
   longitudinal looks fine.

4. **Retrieval top-K blend.** The `update()` blends neighbours within 6 m / 20° of the best match with a
   softmax over similarity. If the gated neighbourhood spans two nearby racing lines, the blended `z` sits
   *between* them — a lateral averaging error.

### Suggested experiments to recover lateral accuracy

- Sweep `lat_clamp` and `retr_lat_std`; log clamp-activation rate and per-frame lateral innovation.
- Try firing retrieval updates more often (lower `sim_thresh` / soften the `RELOCALIZING` entry at
  `lost >= 10`) and watch whether lateral RMSE drops without longitudinal regressing.
- Decompose error by `mode` (TRACKING vs RELOCALIZING) — confirm whether the lateral inflation is
  concentrated in the reloc tail (hypothesis 3) or spread across tracking (hypothesis 1).
- Weight the top-K blend by inverse lateral spread, or take the single best match laterally while keeping
  the blend longitudinally.

## How the two pipelines inform each other

- **Teammate's `main` pipeline** (sweep-tuned EKF trust, `q_x_floor`/`r_scale`) and **this `shubi`
  pipeline** (bias-calibrated odometry + anisotropic tracker) attack the same dominant failure —
  along-track drift — from different ends. `main` tuned *how much to trust* each source; `shubi` fixed the
  *source bias* itself. These are complementary: calibrated odometry should make `main`'s trust sweep
  easier, and `main`'s lateral trust settings may be exactly what cures `shubi`'s lateral regression.
- Next joint step: run both pipelines on the **identical** `gt_val.csv` split and compare the
  longitudinal/lateral decomposition frame-by-frame, not just the headline RMSE.

## Caveats (read before quoting the 65% number)

- **Different split basis.** `13.688` is the repo milestone tag; the numbers compared head-to-head above
  are the two `--val-only` runs from the `shubi` session logs (29.680 → 10.426). Re-run the baseline
  checkpoint with `--val-only` against the same `gt_val.csv` to make the comparison exact.
- **`4_diagnostics.py` is still the `main` version** and was not adapted to the `shubi` tracker, so its
  variant comparison / NIS-NEES output may not match this pipeline. Don't rely on it for `shubi` until ported.
- `2 incomplete image sets skipped` in the calibrated run (2387 vs 2386 poses) — a minor pairing
  difference, not expected to move RMSE materially.
- Scores are on one night session (Yas Marina North). Generalisation to other conditions is untested.

---
*Generated on the `shubi` branch for joint analysis. Evaluation numbers transcribed from the session's
`3_causal_infer.py` / `evaluate.py` logs.*


---

# Milestone 2 — track-frame constraints (`rmse-8.684-shubi`)

Tag: `rmse-8.684-shubi`  ·  Builds on `rmse-10.426-shubi`.
No retraining: the track map was added to the existing `racing_model_cal.pt` with `Scripts/build_track_map.py`
(checkpoint used: `racing_model_track.pt`). Same `--val-only` split, 2387 poses.

## Headline (vs. the previous `shubi` milestone)

| metric | `rmse-10.426-shubi` | `rmse-8.684-shubi` | change |
|---|---:|---:|---:|
| **position RMSE [m]**  | 10.426 | **8.684** | **−1.74 (−17%)** |
| position median [m]    | 1.062  | 1.067  | ~flat |
| position p95 [m]       | 32.427 | 28.695 | −3.73 |
| longitudinal RMSE [m]  | 8.109  | 6.800  | −1.31 (−16%) |
| **lateral RMSE [m]**   | 6.552  | **5.401** | **−1.15 (−18%) — regression recovered** |
| yaw RMSE [deg]         | 1.710  | 1.410  | −0.30 |
| within 1.00 m          | 47.7 % | 47.7 % | flat |

The lateral regression flagged in milestone 1 (4.64 → 6.55 m) is now partly recovered (→ 5.40 m), while
longitudinal kept improving. Both the along-track and across-track components moved in the right direction.

## What changed

The track frame replaces the blunt `lat_clamp` (distance to the nearest training path) with the real track
geometry, built offline and stored in the checkpoint. New `Scripts/track_map.py` + `Scripts/build_track_map.py`;
`1_build_dataset.py` now stores border markers per line; `2_train_model.py` bakes a `track_map` into the
checkpoint; `tracker.py` / `3_causal_infer.py` use it when present (fully backward-compatible without one).

- **Per-marker borders.** `track_bounds` is kept per line instead of flattened. The run found 4 markers
  (left=1, right=1, other=2) and both walls at 100% of stations.
- **Centerline + (s, n).** A 3006 m closed centerline (3006 stations) re-centred between the real borders,
  median width L/R 6.8/6.8 m. DB poses are expressed in Frenet (s, n).
- **Real-border clamp** replaces `lat_clamp` — fired on only 15 frames (the estimate was already inside the
  borders almost everywhere).
- **Heading prior** (soft pull toward the racing-line heading at s) — applied on 2374/2387 frames; measured
  heading spread median 0.9°, so the prior is tight and trustworthy. This is the main driver of the yaw and
  lateral improvement.
- **Pit / run-off corridor** — 760 stations carry an extra allowed lateral band from where training laps
  left the main track.

## Reading the run

- The whole lap stays `TRACKING` (`lost=0`) until the pit-lane pass near the end, where it enters
  `RELOCALIZING` (frames ~2100–2300, sim 0.81–0.93) — same structural weak spot as before, now the dominant
  remaining error (p95 still 28.7 m). That region is the next thing to attack.
- Runtime unchanged in practice: median 14.0 ms / p95 19.3 ms per image set (track-frame ops are numpy-cheap).

## Still open

- **Pit-lane relocalization tail** is now the largest remaining error source. Worth checking whether the
  pit corridor (760 stations) is being used during that stretch, or whether retrieval simply has no nearby
  database coverage there.
- Settings (`border_margin`, heading-prior strength/gate) are first guesses; a small sweep via the A/B flags
  (`--no-heading-prior`, `--border-margin`) could squeeze more.
- Same split caveat as milestone 1: scores are on the `--val-only` held-out laps.
