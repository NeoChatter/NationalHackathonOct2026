# Brainstorming & Diagnostics Log

A running record of how this camera-only localization system is being diagnosed and improved,
milestone by milestone. It is written to be read on its own (and to be handed to an LLM as source
material for a project presentation): each entry states the **problem**, the **diagnosis**, the
**fix**, and the **measured benefit**.

The score throughout is **position RMSE [m]** on the held-out validation laps — lower is better. The
error is split into **longitudinal** (along the car's heading / along the track) and **lateral**
(across it). For this project the error is overwhelmingly longitudinal, which drives every decision
below.

---

## Milestone timeline

| tag | position RMSE [m] | median / p95 [m] | longitudinal | lateral | yaw [deg] | < 1.0 m | what changed |
|---|---|---|---|---|---|---|---|
| `rmse-24.508` | 24.508 | 9.003 / 56.861 | 24.302 | 3.172 | 2.159 | 12.8 % | baseline neural visual EKF |
| `rmse-13.688` | **13.688** | 1.547 / 20.004 | 13.308 | 3.201 | 2.368 | 32.8 % | EKF trust re-tuning from the diagnostic sweep |

**Headline:** a 44 % cut in position RMSE (24.508 → 13.688 m) and a 6× cut in **median** error
(9.003 → 1.547 m) came from *re-tuning how much the filter trusts each sensor* — not from retraining
the network or changing the architecture. The typical frame is now localized to ~1.5 m; the remaining
RMSE is dominated by a smaller number of large along-track excursions.

---

## The core insight: the error is longitudinal, not heading

The very first diagnostic observation reframed the whole problem. At the 24.5 m baseline:

```
longitudinal² / total²  =  24.302² / 24.508²  ≈  0.983
```

So **~98 % of the squared position error is along-track**. The car usually knows its heading (yaw RMSE
~2°) and its lateral position (3.2 m) reasonably well, but can be tens of metres ahead of or behind its
true location — exactly the failure mode you expect on long, visually repetitive straights at night.

Consequence: the highest-value work is anything that fixes **along-track** position — place retrieval,
temporal consistency, and recovery — not heading or more training epochs. A note that was important to
internalize early: the 0.304 m figure in the dataset README is *synthetic ground truth plus noise*, a
sanity check for the scorer, not an achievable baseline for a camera method.

---

## How the pipeline can fail (the suspect list)

Before touching anything, the pipeline was audited end to end. The components most likely to cause
along-track error:

| Component | Weakness identified | Why it matters for along-track error |
|---|---|---|
| Place descriptor | Contrastive loss uses a broad similarity kernel over random batches; it does not explicitly mine same-location-different-lap positives. | The embedding can tell track *sections* apart without resolving positions a few metres apart along a straight. |
| Visual odometry (VO) | The VO head sees globally pooled ResNet features and their difference, not spatial correspondences or a correlation volume. | Can learn approximate motion but struggles with precise translation → steady along-track drift. |
| Retrieval gate | Candidates must be within 35 m / 35° of the *predicted* state, with no global recovery after init. | Once drift exceeds the gate, the correct location is excluded indefinitely — the filter locks onto the wrong place. |
| Retrieval measurement | Top-8 references are averaged; their spread sets the covariance. | Visually similar but physically distant places get blended into a confident-looking wrong measurement. |
| Camera sync | Buffers keep the most recently *seen* image rather than the latest with `stamp ≤ anchor`. | Log order ≠ exposure-time causality; at 40 m/s, 0.1 s of skew ≈ 4 m of along-track error. |
| Checkpoint selection | Training saves the final epoch; no validation-RMSE-based selection. | Can't tell whether more epochs help or hurt. |

The leading hypothesis: **weak along-track retrieval + VO drift + a gate that blocks recovery**, acting
together.

---

## Diagnose before changing anything: `Scripts/4_diagnostics.py`

Rather than guess, an offline diagnostic tool was built. It deliberately uses ground truth, so it is an
**analysis tool only and never part of the submission path**. It runs the network once over the
held-out validation frames and compares four variants on identical timestamps:

- `retrieval_top1` — pose of the single most-similar reference (global, no gate, no filter)
- `aux_head` — the network's direct global-pose output
- `vo_only` — integrate VO deltas from the permitted initial pose
- `fused` — a faithful replica of the EKF loop in `3_causal_infer.py`, with extra telemetry

It also produces the diagnostics that pointed straight at the fix:

- **Error decomposition** into longitudinal / lateral for every variant.
- **Reference-map coverage** (GT-assisted): is a correct reference even present in the map near each
  true pose? This separates "the map lacks this place" from "the filter mishandled a good match."
- **Gate exclusion**: how often a reference close to the *true* pose existed but was outside the gate
  around the *predicted* state — the signature of a filter locked onto the wrong place.
- **VO step statistics**: per-step RMSE, bias, and a forward-distance scale factor (systematic
  along-track over/under-shoot).
- **Filter consistency**: NIS / NEES, i.e. whether the filter's own covariance is honest or
  over/under-confident.
- A **parameter sweep** over process-noise floor, measurement-covariance scale, and gate size, ranked
  by validation RMSE.

This is the key move: **a few targeted ablations replaced another blind, expensive training run.**

---

## What the diagnostics revealed

The sweep and variant comparison localized the problem to **fusion, not the network**:

- Retrieval, on its own, had far more useful along-track information than the fused output was
  exploiting. The filter was effectively *ignoring good position measurements* in favor of a drifting
  VO prediction.
- The EKF was **over-trusting visual odometry along the track** (its learned process variance was too
  small longitudinally) and **under-trusting the retrieval measurement** (its position covariance was
  too large). The net effect: along-track drift accumulated and corrections were too weak to pull it
  back.

A subtle, related bug surfaced in the audit: the VO loss weights squared errors by `[1, 2, 5]`, but
inference used `exp(log_var)` directly as physical variance — so the training-time weighting and the
inference-time variance were inconsistent. The principled correction is `q_var /= [1, 2, 5]`.

---

## The fix that produced 13.688 m: re-tune the filter's trust

Two sweep-selected changes in `Scripts/3_causal_infer.py`, both about *how much the EKF trusts each
source*:

1. **Longitudinal process-noise floor — `q_x_floor = 1.0 m/step`.**
   Forces the filter to admit that its along-track VO prediction is uncertain. With a healthy floor on
   the forward process noise, the Kalman gain lets retrieval corrections actually move the estimate
   instead of being overwhelmed by an over-confident prediction.

2. **Position measurement-covariance scale — `r_scale = 0.03`.**
   Shrinks the retrieval measurement covariance (~33× tighter), telling the filter to believe a
   coherent retrieved position. Combined with (1), the correct place pulls the estimate back onto the
   track.

The variance-scaling correction (`--fix-q-scaling`, `q_var /= [1, 2, 5]`) is implemented and available,
but left **off by default** because the sweep achieved the best overall RMSE without it — a reminder to
keep what the measurements justify, not what merely sounds correct.

### Measured benefit

| metric | before (`rmse-24.508`) | after (`rmse-13.688`) | change |
|---|---|---|---|
| position RMSE [m] | 24.508 | **13.688** | −44 % |
| median [m] | 9.003 | 1.547 | −83 % |
| p95 [m] | 56.861 | 20.004 | −65 % |
| longitudinal RMSE [m] | 24.302 | 13.308 | −45 % |
| lateral RMSE [m] | 3.172 | 3.201 | ~flat |
| yaw RMSE [deg] | 2.159 | 2.368 | ~flat |
| within 1.0 m | 12.8 % | 32.8 % | +20 pts |

Lateral and yaw barely moved — expected, since the change only re-weighted the longitudinal/position
trust. The win is concentrated exactly where the diagnostics said the error lived.

---

## Why this is the right kind of progress

- **Cheap, reversible, measured.** No retraining; a tuning change validated against held-out data,
  guided by a sweep rather than intuition.
- **Targeted.** The diagnostics proved the error was along-track and lived in fusion, so the fix
  addressed fusion — and the gains landed where predicted.
- **Honest.** The improvement is real but partial: the error is *still* dominated by longitudinal
  excursions (13.3 of 13.7 m), and p95 is 20 m. The typical frame is good (~1.5 m median); the tail is
  the remaining work.

---

## What's next (prioritized by the diagnostics)

1. **Along-track place retrieval.** Train the descriptor with explicit same-location-different-lap
   positives and nearby along-track hard negatives, and track retrieval recall at 2 m / 5 m. This
   attacks the root cause of the remaining longitudinal error.
2. **Recovery + hypothesis separation.** Add a covariance-aware global recovery search and cluster
   retrieved candidates by location/branch instead of averaging them, so the filter can leave a wrong
   lock and never blends distinct places (straight vs. a look-alike straight vs. pit lane).
3. **Temporal sequence matching.** A causal beam search / HMM / particle filter over candidate
   track positions — a short sequence of views carries far more along-track information than one frame.
4. **Exposure-time-causal sync + invalid-interval handling.** Select the latest image with
   `stamp ≤ anchor`, log camera age, and reset the VO pair on bad intervals rather than feeding
   out-of-distribution pairs into the VO head.
5. **Local geometric refinement.** Once the correct place is found, estimate continuous displacement
   from a matched reference (feature matching + robust 2D–3D pose) instead of snapping to stored
   reference coordinates — the path to sub-metre accuracy.
6. **Validation-driven checkpoint selection.** Score full causal validation RMSE during training and
   keep the best checkpoint, not the last epoch.

---

## Glossary for the presentation

- **Position RMSE** — the competition score; root-mean-square 2D position error, lower is better.
- **Longitudinal vs lateral error** — along the car's heading vs across it; this project's error is
  almost entirely longitudinal.
- **VO (visual odometry)** — frame-to-frame motion estimate; accurate short-term, drifts long-term.
- **Place retrieval** — matching the current view against a database of known-location reference views.
- **EKF trust (Q / R)** — process noise `Q` says how much to trust the motion prediction; measurement
  noise `R` says how much to trust the retrieval measurement. The 13.688 m milestone is a Q/R
  re-tuning.
- **Gate** — the region around the predicted pose in which retrieval candidates are considered; too
  tight and it blocks recovery.
- **NIS / NEES** — consistency checks for whether the filter's stated uncertainty matches reality.
