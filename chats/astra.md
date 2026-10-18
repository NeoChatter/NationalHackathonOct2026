**Your main problem is longitudinal localization—knowing where the car is along the track—not primarily yaw.** About **98.3% of your squared position error is longitudinal**:

\[
\frac{24.302^2}{24.508^2}\approx 0.983.
\]

A car can have approximately the correct heading while being tens of metres ahead of or behind its true location, especially on a straight.

Also, the **0.304 m example is synthetic ground truth plus noise**, not a demonstrated camera-localization result or an expected baseline.

From reviewing your code, I would **not start by increasing epochs**: your current pipeline has weaknesses in local place discrimination, motion estimation, retrieval recovery, and uncertainty handling that should be investigated first. [allcodes.txt]

I have not run your model, so the distinction below is important: some issues are directly visible in the code; their contribution to your particular score needs measurement.

## 1. Where your current pipeline is vulnerable

| Component | What is happening | Why it matters |
|---|---|---|
| Place-descriptor training | The loss uses a broad spatial similarity kernel and randomly shuffled batches; despite the docstring, it does not explicitly select matching locations from different laps. [allcodes.txt] | The descriptor may distinguish track sections without distinguishing positions a few metres apart. |
| Visual odometry | The VO head receives globally pooled ResNet features, their difference, and timing features—not spatial feature correspondences or a correlation volume. [allcodes.txt] | This architecture can learn approximate motion, but precise translation from small image displacements is difficult. |
| Retrieval gate | References must be within **35 m and 35°** of the predicted state; there is no global recovery search after initialization. [allcodes.txt] | Once longitudinal drift becomes sufficiently large, the correct location can be excluded indefinitely. |
| Retrieval measurement | The top eight reference poses are averaged, and their spread determines positional measurement covariance. [allcodes.txt] | Similar-looking but different locations can produce a misleading average; a tightly clustered wrong match can appear confident. |
| Synchronization | The cache keeps the most recently encountered image, without checking that its exposure timestamp is no later than the anchor timestamp. [allcodes.txt] | Arrival/log order does not guarantee exposure-time causality or good synchronization. |
| Validation | Training reserves validation indices but does not evaluate validation loss or trajectory RMSE during training. [allcodes.txt] | You cannot select the best checkpoint or tell whether extra epochs help. |

**The combination I would investigate first is weak along-track retrieval plus VO drift plus a gate that prevents recovery.** That is a plausible explanation for your numbers, not yet a confirmed diagnosis.

---

## 2. Before retraining: measure which component is failing

You can learn much more from a few ablations than from another long training run.

Evaluate these variants on **the same validation timestamps**, without removing difficult frames:

| Variant | Prediction |
|---|---|
| **Global retrieval only** | Pose of the highest-similarity reference, without the spatial gate or EKF |
| **VO only** | Integrate predicted body-frame motion from the permitted initial pose |
| **Auxiliary pose head only** | Decode the network’s direct global-pose prediction |
| **Current fused pipeline** | Your existing output |

Your checkpoint already contains reference embeddings, reference poses, coordinate normalization statistics, and the auxiliary pose head needed for these comparisons. [allcodes.txt]

For the auxiliary head, decode:

\[
\hat{\mathbf p}_{xy}
=
\hat{\mathbf p}_{xy,\mathrm{norm}}\odot \mathbf{xy}_{\mathrm{std}}
+
\mathbf{xy}_{\mathrm{mean}},
\qquad
\hat{\psi}
=
\operatorname{atan2}(\widehat{\sin\psi},\widehat{\cos\psi}).
\]

These correspond to the outputs and normalization used in your training script. [allcodes.txt]

### How to interpret the results

- **Global retrieval is much better than fused:** focus on VO, gating, and covariance calibration.
- **VO drifts steadily, but retrieval is locally accurate:** improve motion estimation and give reliable retrieval more influence.
- **Global retrieval also has large longitudinal errors:** improve descriptors and add image-based pose refinement.
- **Failures concentrate in the pit lane:** check whether the training map adequately covers that route.
- **The auxiliary head is unexpectedly best:** use it as a baseline and potentially a coarse recovery proposal—not automatically as a trusted EKF measurement.

### Add two ground-truth-assisted diagnostics

These are **offline diagnostics only**, never inference inputs.

**A. Reference-map coverage**

For each validation pose, find the nearest training reference position, optionally requiring compatible heading:

\[
d_{\mathrm{nearest}}(t)
=
\min_{j\in\mathcal D}
\left\|\mathbf p_{\mathrm{GT}}(t)-\mathbf p_j\right\|_2.
\]

Report median, p95, and a plot over time.

This tells you whether relevant reference locations exist at all. It is a baseline for selecting a stored pose, **not a lower bound for interpolation or geometric refinement**.

**B. Gate exclusion**

At each frame, ask:

> Was a reference close to the true pose available in the database, but excluded by the predicted-state gate?

Your current telemetry logs similarity and the final estimate, but not gate size, predicted pose, VO increments, retrieval measurement, or covariance; add these fields. [allcodes.txt]

Useful fields are:

```text
timestamp_ns
dx_body_pred, dy_body_pred, dyaw_pred
pred_x, pred_y, pred_yaw
gate_size
retrieved_x, retrieved_y, retrieved_yaw
innovation_x, innovation_y, innovation_yaw
P_xx, P_yy, P_yawyaw
R_xx, R_yy, R_yawyaw
```

Then plot longitudinal error against time, speed, and gate exclusions. A smooth drift and a sudden wrong-place jump require different fixes.

---

## 3. Fix timestamp handling before rebuilding the model

This is both an accuracy issue and a causality requirement.

### A. Synchronize by exposure timestamp

In both dataset construction and inference, your secondary-camera buffer stores the latest encountered image rather than explicitly selecting the latest image with `stamp <= camera_fl_stamp`. [allcodes.txt]

Use a short timestamped buffer per camera and, at anchor time \(t\), select:

\[
I_c(t)=\operatorname*{latest}_{I:\,t_I\le t} I.
\]

Conceptually:

```python
eligible = [(stamp, image) for stamp, image in camera_buffer
            if stamp <= anchor_stamp]

selected = max(eligible, key=lambda item: item[0]) if eligible else None
```

Keep older entries long enough that a newly encountered future-stamped image does not overwrite the last eligible one.

Also log:

- Negative offsets: must not occur for selected images.
- Age of every selected camera image.
- Missing cameras.
- Abnormally large anchor intervals.

At \(40\,\mathrm{m/s}\), an **uncompensated** temporal offset of \(0.1\,\mathrm{s}\) corresponds to \(4\,\mathrm m\) of travel; \(0.3\,\mathrm m\) corresponds to only \(7.5\,\mathrm{ms}\).

Your network receives current camera offsets, but not the previous image set’s offsets, although both image sets are used for VO. [allcodes.txt]

For asynchronous paired-image motion estimation, supply timing information for **both** sets, or explicitly handle their different exposure times.

### B. Handle invalid intervals explicitly

Training accepts intervals between \(0.01\) and \(0.5\,\mathrm{s}\), whereas inference lower-clamps the interval and has no corresponding upper-bound handling. [allcodes.txt]

For invalid intervals, do not silently feed an out-of-distribution pair into VO. Instead:

- Reset the VO pair state.
- Attempt visual relocalization.
- Retain appropriately increased uncertainty.

Also exclude training labels spanning GT gaps larger than the evaluator’s allowed gap.

### C. Correct initialization and validation semantics

The inference script repeatedly overwrites `first_gt_pose` until the EKF is initialized, so it does not strictly preserve the first eligible GT pose. [allcodes.txt]

Capture the permitted initial pose once, store its timestamp, and account for the interval to the first camera output without reading subsequent GT as localization input.

Your `--val-only` mode also permits initialization from a GT pose at the validation start, rather than only from the beginning of the original recording. [allcodes.txt]

That is useful for an explicitly reset validation experiment, but it is not equivalent to a full-run evaluation under the “very first pose of the run” exception.

---

## 4. Make descriptor training discriminate nearby places

### Why the current loss is not enough

Your current target distribution is constructed as:

```python
spatial_sim = exp(-distance / 8.0) * exp(-yaw_difference / 0.35)
target_probs = softmax(spatial_sim * 5.0)
```

This is the implemented target construction. [allcodes.txt]

A distant pair has `spatial_sim ≈ 0`, but after softmax its unnormalized target weight is still \(\exp(0)=1\); this is a soft distribution over all other batch samples, not explicit positive/negative matching. [allcodes.txt]

That construction is not inherently invalid, but it can provide weak fine-scale supervision—particularly when random batches contain few close cross-lap pairs.

### Change the batch sampler first

For a batch of 64, a useful starting point is:

1. Select 32 anchors.
2. For each anchor, select a training image from another lap:
   - Within approximately **2 m**.
   - Heading difference below approximately **10°**.
3. Include hard negatives from nearby but distinct positions, especially **5–30 m ahead or behind** on similar-looking straights.

These thresholds are starting points, not universal constants.

Use lap identifiers derived offline from the training trajectory, or a sufficiently separated-time condition when lap segmentation is unavailable. Do not draw positives from held-out validation frames.

### Then use an explicit positive/negative loss

For example, the following is a reasonable starting replacement for the descriptor loss:

```python
def metric_contrastive_loss(emb, poses, temp=0.07):
    xy = poses[:, :2]
    yaw = poses[:, 4]

    distance = torch.cdist(xy, xy)
    angle = torch.abs(torch.atan2(
        torch.sin(yaw[:, None] - yaw[None, :]),
        torch.cos(yaw[:, None] - yaw[None, :]),
    ))

    off_diag = ~torch.eye(
        len(emb), dtype=torch.bool, device=emb.device
    )

    positive = (
        off_diag
        & (distance < 2.0)
        & (angle < math.radians(10.0))
    )

    negative = (
        off_diag
        & (
            (distance > 5.0)
            | (angle > math.radians(30.0))
        )
    )

    # Ignore ambiguous pairs between the positive/negative thresholds.
    allowed = positive | negative
    valid = positive.any(dim=1) & negative.any(dim=1)

    if not valid.any():
        return emb.sum() * 0.0

    logits = (emb @ emb.T) / temp
    logits = logits[valid].masked_fill(
        ~allowed[valid], float("-inf")
    )
    log_probs = logits - torch.logsumexp(
        logits, dim=1, keepdim=True
    )

    pos = positive[valid]
    positive_log_probs = log_probs.masked_fill(~pos, 0.0)

    return -(
        positive_log_probs.sum(dim=1) / pos.sum(dim=1)
    ).mean()
```

**Do not paste this in while leaving batching unchanged and assume the problem is solved.** Without close positive pairs, some or many batches can provide no contrastive training signal.

Track:

- Fraction of anchors with positives.
- Retrieval recall within 2 m and 5 m.
- Median and p95 retrieved-position error.
- Confusions between distinct along-track locations.

This improves **candidate retrieval**. A descriptor trained to group locations within 2 m is not, by itself, a sub-metre pose estimator.

---

## 5. Fix fusion and recovery

### A. Correct the learned process-variance interpretation

Your VO loss multiplies squared errors by `[1, 2, 5]`, but inference uses `exp(log_var)` directly as physical process variance. [allcodes.txt]

For the implemented loss,

\[
\mathcal L_i
=
\frac12 e^{-s_i}w_i e_i^2+\frac12s_i,
\]

the corresponding physical residual variance is:

\[
\sigma_i^2=\frac{e^{s_i}}{w_i}.
\]

Therefore, replace the inference conversion with:

```python
q_var = (
    np.exp(log_var[0].cpu().numpy())
    / np.array([1.0, 2.0, 5.0], dtype=np.float64)
)
```

This makes inference consistent with the weighting used during training, though the resulting variances still need empirical calibration. [allcodes.txt]

Also, the current `log_var >= -8` clamp, followed by direct exponentiation, imposes a minimum yaw standard deviation of about **1.05° per prediction step**, despite the smaller floor specified inside the EKF. [allcodes.txt]

That is worth correcting, but **do not expect this patch alone to remove 24 m of longitudinal error**.

### B. Add a real recovery mechanism

Your inference performs a global search only when initializing without GT; subsequent searches are always spatially gated. [allcodes.txt]

Instead:

1. Use local retrieval during normal tracking.
2. Monitor ambiguity and measurement consistency.
3. If local tracking becomes unreliable, generate global retrieval candidates.
4. Require consistency across several **past/current** frames before switching location.
5. Increase uncertainty when losing track rather than staying confidently locked onto the wrong section.

Use a covariance-aware search region with reasonable limits rather than relying exclusively on a fixed 35 m radius.

### C. Do not average different location hypotheses

Your top-eight weighted average does not first check whether the references form one coherent location hypothesis. [allcodes.txt]

Cluster retrieved candidates by position, heading, and route branch before forming a measurement.

For example:

- Candidate group A: current straight.
- Candidate group B: a visually similar section farther along.
- Candidate group C: pit lane.

These should remain separate hypotheses—not be averaged into a location between them.

Compare confidence between **spatially distinct clusters**, not merely between the top two individual reference images; two adjacent correct references should naturally have similar scores.

### D. Use temporal sequence matching

For your dominant longitudinal error, a causal sequence matcher is particularly worth testing:

- Maintain several candidate track positions.
- Score their visual matches.
- Penalize implausible motion between consecutive candidates.
- Keep route branches distinct.
- Use only observations up to the current time.

A beam search, hidden Markov model, or particle filter can do this.

A short sequence of views often contains more along-track information than one image set, even when each individual frame is ambiguous.

---

## 6. For genuinely small errors, add local geometric refinement

Your present global measurement ultimately comes from stored reference poses, not from estimating the query camera’s geometric displacement relative to a matched reference. [allcodes.txt]

That matters because the validation car may be:

- Between stored reference positions.
- On a different racing line.
- At a slightly different heading.
- Entering or leaving the pit lane.

A stronger architecture is:

```text
Current camera images
        ↓
Coarse place retrieval
        ↓
Match local image features to a training-built visual map
        ↓
Geometric pose estimation / refinement
        ↓
Robust causal temporal fusion
```

### A practical implementation direction

1. Build a sparse 3D visual map using **training images**, calibration, and permitted training poses.
2. Store descriptors and 3D landmarks around each reference keyframe.
3. Retrieve a small candidate region at inference.
4. Match current-image features to that region’s landmarks.
5. Estimate camera pose with robust 2D–3D pose estimation.
6. Convert camera pose to `base_link` using extrinsics.
7. Fuse only geometrically verified measurements.

This does not guarantee 0.3 m: calibration, blur, lighting, landmark geometry, and map quality all matter. But it provides a mechanism for estimating **continuous displacement from the reference**, rather than relying only on descriptor similarity and stored coordinates.

### Improve VO in the same spirit

Your current VO discards spatial layout through global pooling before predicting motion. [allcodes.txt]

For more accurate motion, test:

- Spatial feature correlation between consecutive frames.
- Tracked image features with calibrated geometric estimation.
- A hybrid learned/geometric VO front end.

Also ablate resolution, crop, and camera selection rather than assuming more cameras automatically improve accuracy.

Your preprocessing reduces every view to grayscale \(256\times128\) and removes the top and bottom portions of the image, which may discard useful location-specific details as well as glare. [allcodes.txt]

Test a less aggressive crop and higher-resolution **local matching**, even if the coarse retrieval network remains low-resolution.

---

## 7. The order I recommend

### First: diagnose the existing checkpoint

Run:

1. Global retrieval only.
2. VO only.
3. Auxiliary pose only.
4. Current fusion.
5. Reference-map coverage and gate-exclusion diagnostics.

**This determines whether to spend your next day on descriptor training, VO, or filtering.**

### Second: fix correctness and obvious mismatches

- Exposure-time-causal buffering.
- Camera-age checks.
- Invalid-interval handling.
- Initial-pose semantics.
- Process-variance scaling.
- Global recovery.

Rebuild and retrain if you change the input synchronization or timing features.

### Third: retrain with meaningful validation

- Cross-lap positive sampling.
- Nearby along-track hard negatives.
- Validation retrieval metrics.
- Full causal validation inference at checkpoints.
- Checkpoint selection by **position RMSE**, not just training loss.

Your current training script saves the final model after all epochs rather than selecting a checkpoint using validation localization accuracy. [allcodes.txt]

### Fourth: add temporal and geometric refinement

Use sequence consistency to reduce wrong-place matches, and geometric refinement to reduce the residual error once the correct place is identified.

Finally, your reported 100% coverage is for the filtered validation interval, because your workflow filters GT to the held-out suffix; it is not evidence of full-recording coverage. [allcodes.txt]

Do not judge generalization by enabling `--full-train` and evaluating against the same images in the reference database; clarify the organizers’ final training/evaluation protocol first.

**Bottom line:** the first question is not “How many more epochs?” It is **“Are the retrieved locations wrong, is VO drifting, or is fusion rejecting/corrupting otherwise good matches?”** Your ablation table will answer that—and your overwhelmingly longitudinal error makes along-track retrieval, temporal consistency, and recovery the highest-priority suspects.

I have written a py file called 4_diagnostics.py

```
#!/usr/bin/env python3
"""
4_diagnose.py -- OFFLINE diagnosis of the camera-localization pipeline.

Uses ground truth ON PURPOSE, so this is an analysis tool only. It must never be
part of the submitted inference path.

Put it next to 1_build_dataset.py / 2_train_model.py / 3_causal_infer.py and run:

    python 4_diagnose.py --data-h5 data/training_cache.h5 --checkpoint checkpoints/racing_model.pt

The checkpoint must come from a run WITHOUT --full-train (otherwise there are no
held-out validation frames to diagnose on).

What it does (on the held-out validation frames, identical timestamps for all variants):
  1. One network pass over the validation frames (embeddings, aux pose, VO deltas, VO log-variances).
  2. Four variants:
       retrieval_top1 : most similar DB reference, global search, no gate, no filter
       aux_head       : the network's direct global-pose output
       vo_only        : integrate VO deltas starting from the first validation GT pose
       fused          : replica of the EKF loop in 3_causal_infer.py, with extra logging
  3. Every error is split into along-track (longitudinal) and across-track (lateral) parts.
  4. Reference-map coverage and gate-exclusion diagnostics.
  5. VO step-error / scale-bias statistics and filter-consistency (NIS / NEES) statistics.
  6. How often the contrastive loss sees positive pairs in a random training batch.

Outputs (in --out-dir): variants_per_frame.csv, fused_telemetry.csv, diagnostics.png
"""
import argparse
import csv
import importlib.util
import math
import os
import sys

import numpy as np


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def decompose(pred_xy, pred_yaw, gt_xy, gt_yaw):
    """Position error in the GT heading frame: longitudinal (along track) / lateral, plus yaw error."""
    d = pred_xy - gt_xy
    c, s = np.cos(gt_yaw), np.sin(gt_yaw)
    e_long = c * d[:, 0] + s * d[:, 1]
    e_lat = -s * d[:, 0] + c * d[:, 1]
    e_yaw = wrap(pred_yaw - gt_yaw)
    return e_long, e_lat, e_yaw


def summarize(name, e_long, e_lat, e_yaw):
    ok = np.isfinite(e_long) & np.isfinite(e_lat) & np.isfinite(e_yaw)
    el, ea, ey = e_long[ok], e_lat[ok], e_yaw[ok]
    xy = np.hypot(el, ea)
    tot = float((xy ** 2).sum())
    return dict(
        name=name,
        n=int(ok.sum()),
        rmse_xy=float(np.sqrt((xy ** 2).mean())),
        rmse_long=float(np.sqrt((el ** 2).mean())),
        rmse_lat=float(np.sqrt((ea ** 2).mean())),
        rmse_yaw_deg=float(np.degrees(np.sqrt((ey ** 2).mean()))),
        med_xy=float(np.median(xy)),
        p95_xy=float(np.percentile(xy, 95)),
        max_xy=float(xy.max()),
        long_share=float((el ** 2).sum() / tot) if tot > 0 else float('nan'),
    )


def print_table(rows):
    head = f"{'variant':<15}{'RMSE xy':>9}{'long':>9}{'lat':>8}{'yaw[deg]':>10}{'median':>9}{'p95':>9}{'max':>9}{'long share':>12}"
    print(head)
    print('-' * len(head))
    for r in rows:
        print(f"{r['name']:<15}{r['rmse_xy']:>9.2f}{r['rmse_long']:>9.2f}{r['rmse_lat']:>8.2f}"
              f"{r['rmse_yaw_deg']:>10.2f}{r['med_xy']:>9.2f}{r['p95_xy']:>9.2f}{r['max_xy']:>9.2f}"
              f"{r['long_share']:>11.1%} ")


def spans(mask):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m) and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


# ----------------------------------------------------------------------------
# EKF: prefer the real class from 3_causal_infer.py so edits there are diagnosed too
# ----------------------------------------------------------------------------
class _FallbackEKF:
    """Verbatim copy of CausalSE2EKF from 3_causal_infer.py (used only if that file cannot be imported)."""

    def __init__(self, x0, y0, yaw0):
        self.state = np.array([x0, y0, yaw0], dtype=np.float64)
        self.P = np.diag([0.15 ** 2, 0.15 ** 2, math.radians(1.0) ** 2])

    def predict(self, dx_body, dy_body, dyaw, q_var):
        x, y, yaw = self.state
        c, s = math.cos(yaw), math.sin(yaw)
        self.state[0] = x + c * dx_body - s * dy_body
        self.state[1] = y + s * dx_body + c * dy_body
        self.state[2] = float(wrap(yaw + dyaw))
        F_jac = np.array([
            [1.0, 0.0, -s * dx_body - c * dy_body],
            [0.0, 1.0, c * dx_body - s * dy_body],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        R_rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        Q_body = np.diag([
            max(float(q_var[0]), 0.05 ** 2),
            max(float(q_var[1]), 0.02 ** 2),
            max(float(q_var[2]), math.radians(0.3) ** 2),
        ])
        Q_world = R_rot @ Q_body @ R_rot.T
        self.P = F_jac @ self.P @ F_jac.T + Q_world

    def update(self, z_meas, R_cov):
        y_res = np.array([
            z_meas[0] - self.state[0],
            z_meas[1] - self.state[1],
            float(wrap(z_meas[2] - self.state[2])),
        ], dtype=np.float64)
        S = self.P + R_cov
        K = self.P @ np.linalg.inv(S)
        self.state = self.state + K @ y_res
        self.state[2] = float(wrap(self.state[2]))
        self.P = (np.eye(3) - K) @ self.P


def get_ekf_class():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, '3_causal_infer.py')
    try:
        mod = load_module_from_path('infer_mod', path)
        return mod.CausalSE2EKF, 'CausalSE2EKF imported from 3_causal_infer.py'
    except Exception as exc:  # noqa: BLE001
        return _FallbackEKF, f'inline copy of CausalSE2EKF (could not import 3_causal_infer.py: {exc!r})'


# ----------------------------------------------------------------------------
# Step A: load checkpoint + cache, run the network once (the only torch-dependent part)
# ----------------------------------------------------------------------------
def load_inputs(args):
    import h5py
    import torch

    here = os.path.dirname(os.path.abspath(__file__))
    train_mod = load_module_from_path('train_mod', os.path.join(here, '2_train_model.py'))
    # The weights come from the checkpoint, so skip the ImageNet download in RacingLocNet.__init__.
    _orig_resnet18 = train_mod.models.resnet18
    train_mod.models.resnet18 = lambda *a, **k: _orig_resnet18(weights=None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = train_mod.RacingLocNet(embed_dim=256).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    val_start_ns = int(ckpt.get('val_start_timestamp_ns', 0))
    if val_start_ns <= 0:
        sys.exit('This checkpoint has no held-out validation split (trained with --full-train?). '
                 'Retrain without --full-train to diagnose generalization.')

    with h5py.File(args.data_h5, 'r') as h5f:
        ts_all = h5f['timestamps_ns'][:]
        poses_all = h5f['poses'][:]          # [x, y, z, yaw]
        dtoff_all = h5f['dt_offsets'][:]
        i0 = int(np.searchsorted(ts_all, val_start_ns))
        i1 = len(ts_all) if args.limit is None else min(len(ts_all), i0 + args.limit)
        imgs = h5f['images'][i0:i1]

    db_emb = ckpt['db_embeddings'].cpu().numpy().astype(np.float32)
    db_poses = ckpt['db_poses'].cpu().numpy().astype(np.float64)   # [x, y, yaw]
    if len(db_emb) != i0:
        print(f'[WARN] DB has {len(db_emb)} references but the cache has {i0} frames before the validation start. '
              f'Was the cache rebuilt after training? Results may be meaningless.')
    if ts_all[i0] != val_start_ns:
        print('[WARN] validation start timestamp not found exactly in the cache.')

    ts = ts_all[i0:i1]
    gt = poses_all[i0:i1]
    dtoff = dtoff_all[i0:i1]
    N = len(ts)

    def prep(u8):
        x = u8.astype(np.float32) / 255.0
        x = (x - x.mean(axis=(2, 3), keepdims=True)) / (x.std(axis=(2, 3), keepdims=True) + 1e-5)
        return torch.from_numpy(x).to(device)

    emb = np.zeros((N, 256), np.float32)
    aux = np.zeros((N, 4), np.float32)
    delta = np.full((N, 3), np.nan)
    logvar = np.full((N, 3), np.nan)
    dt_raw = np.full(N, np.nan)

    print(f'[NET] Running the network on {N} validation frames on {device} ...')
    with torch.no_grad():
        _, e0, a0 = model.forward_single(prep(imgs[0:1]))
        emb[0] = e0.cpu().numpy()[0]
        aux[0] = a0.cpu().numpy()[0]
        for s in range(1, N, args.batch):
            e = min(s + args.batch, N)
            dt_b = (ts[s:e] - ts[s - 1:e - 1]) * 1e-9
            dt_raw[s:e] = dt_b
            tf = np.concatenate([np.maximum(dt_b, 0.01)[:, None], dtoff[s:e]], axis=1).astype(np.float32)  # as in inference
            emb_b, aux_b, d_b, lv_b = model(prep(imgs[s - 1:e - 1]), prep(imgs[s:e]),
                                            torch.from_numpy(tf).to(device))
            emb[s:e] = emb_b.cpu().numpy()
            aux[s:e] = aux_b.cpu().numpy()
            delta[s:e] = d_b.cpu().numpy()
            logvar[s:e] = lv_b.cpu().numpy()

    return dict(
        ts=ts, gt=gt, emb=emb, aux=aux, delta=delta, logvar=logvar, dt_raw=dt_raw,
        db_emb=db_emb, db_poses=db_poses,
        xy_mean=ckpt['xy_mean'].cpu().numpy().astype(np.float64),
        xy_std=ckpt['xy_std'].cpu().numpy().astype(np.float64),
        train_poses=poses_all[:i0], train_ts=ts_all[:i0],
    )


# ----------------------------------------------------------------------------
# Step B: pure-numpy analysis
# ----------------------------------------------------------------------------
def integrate_vo(gt0, delta):
    out = np.zeros((len(delta), 3))
    out[0] = gt0
    for i in range(1, len(delta)):
        x, y, yaw = out[i - 1]
        dx, dy, dyaw = delta[i]
        c, s = math.cos(yaw), math.sin(yaw)
        out[i] = [x + c * dx - s * dy, y + s * dx + c * dy, float(wrap(yaw + dyaw))]
    return out


def coverage(gt, db_poses, compat_deg):
    """Nearest stored reference to each GT pose: position only, and heading-compatible."""
    N = len(gt)
    near_any = np.zeros(N)
    near_head = np.full(N, np.inf)
    compat = math.radians(compat_deg)
    for i in range(N):
        d = np.hypot(db_poses[:, 0] - gt[i, 0], db_poses[:, 1] - gt[i, 1])
        near_any[i] = d.min()
        m = np.abs(wrap(db_poses[:, 2] - gt[i, 3])) <= compat
        if m.any():
            near_head[i] = d[m].min()
    return near_any, near_head


def run_fused(d, args, EKF):
    """Replica of the EKF loop in 3_causal_infer.py (same gate, top-8 average, covariance), with telemetry."""
    gt, emb, delta, logvar = d['gt'], d['emb'], d['delta'], d['logvar']
    db, db_emb = d['db_poses'], d['db_emb']
    N = len(gt)
    ekf = EKF(gt[0, 0], gt[0, 1], gt[0, 3])   # permitted initial pose
    gate_r2 = args.gate_m ** 2
    gate_yaw = math.radians(args.gate_deg)
    close_yaw = math.radians(args.close_deg)
    w_loss = np.array([1.0, 2.0, 5.0])

    cols = ['frame', 'timestamp_ns', 'mode', 'gt_x', 'gt_y', 'gt_yaw',
            'dx_body_pred', 'dy_body_pred', 'dyaw_pred', 'q_x', 'q_y', 'q_yaw',
            'pred_x', 'pred_y', 'pred_yaw', 'gate_size_m', 'gate_n', 'top1_sim',
            'retrieved_x', 'retrieved_y', 'retrieved_yaw',
            'innovation_x', 'innovation_y', 'innovation_yaw', 'nis',
            'P_xx', 'P_yy', 'P_yawyaw', 'R_xx', 'R_yy', 'R_yawyaw',
            'est_x', 'est_y', 'est_yaw', 'err_long', 'err_lat', 'err_yaw',
            'meas_err_long', 'meas_err_lat', 'nees_xy',
            'ref_close_to_gt', 'ref_close_in_gate', 'gate_excluded_close_ref']
    rows = []
    est = np.zeros((N, 3))

    for i in range(N):
        r = {c: float('nan') for c in cols}
        r.update(frame=i, timestamp_ns=int(d['ts'][i]), mode='VO_ONLY',
                 gt_x=gt[i, 0], gt_y=gt[i, 1], gt_yaw=gt[i, 3],
                 gate_size_m=args.gate_m, top1_sim=0.0)

        if i > 0:
            q_var = np.exp(logvar[i])
            if args.fix_q_scaling:
                q_var = q_var / w_loss
            q_floor = getattr(args, 'q_x_floor', 0.0)
            if q_floor > 0:
                q_var[0] = max(q_var[0], q_floor ** 2)
            ekf.predict(float(delta[i, 0]), float(delta[i, 1]), float(delta[i, 2]), q_var)
            r.update(dx_body_pred=delta[i, 0], dy_body_pred=delta[i, 1], dyaw_pred=delta[i, 2],
                     q_x=q_var[0], q_y=q_var[1], q_yaw=q_var[2])
        r.update(pred_x=ekf.state[0], pred_y=ekf.state[1], pred_yaw=ekf.state[2])

        dxd = db[:, 0] - ekf.state[0]
        dyd = db[:, 1] - ekf.state[1]
        dyawd = np.abs(wrap(db[:, 2] - ekf.state[2]))
        gate_mask = ((dxd ** 2 + dyd ** 2) <= gate_r2) & (dyawd <= gate_yaw)
        gidx = np.nonzero(gate_mask)[0]
        r['gate_n'] = len(gidx)

        # Ground-truth-assisted: was a reference close to the TRUE pose available, and was it inside the gate?
        close = (np.hypot(db[:, 0] - gt[i, 0], db[:, 1] - gt[i, 1]) <= args.close_m) & \
                (np.abs(wrap(db[:, 2] - gt[i, 3])) <= close_yaw)
        has_close = bool(close.any())
        in_gate = bool((close & gate_mask).any())
        r.update(ref_close_to_gt=int(has_close), ref_close_in_gate=int(in_gate),
                 gate_excluded_close_ref=int(has_close and not in_gate))

        if len(gidx) >= 5:
            sims = db_emb[gidx] @ emb[i]
            k = min(8, len(gidx))
            top_local = np.argsort(-sims, kind='stable')[:k]
            top_vals = sims[top_local].astype(np.float64)
            top1_sim = float(top_vals[0])
            r['top1_sim'] = top1_sim
            if top1_sim > args.min_sim:
                e = np.exp(top_vals * 15.0 - (top_vals * 15.0).max())
                w = e / e.sum()
                matched = db[gidx[top_local]]
                z_x = float(np.sum(w * matched[:, 0]))
                z_y = float(np.sum(w * matched[:, 1]))
                z_yaw = math.atan2(float(np.sum(w * np.sin(matched[:, 2]))),
                                   float(np.sum(w * np.cos(matched[:, 2]))))
                diff_xy = matched[:, :2] - np.array([[z_x, z_y]])
                cov_xy = (diff_xy.T * w) @ diff_xy + np.eye(2) * (0.25 / max(top1_sim, 0.2)) ** 2
                R_cov = np.eye(3) * (math.radians(2.0) ** 2)
                R_cov[:2, :2] = cov_xy * getattr(args, 'r_scale', 1.0)

                innov = np.array([z_x - ekf.state[0], z_y - ekf.state[1],
                                  float(wrap(z_yaw - ekf.state[2]))])
                nis = float(innov @ np.linalg.solve(ekf.P + R_cov, innov))
                ml, ma, _ = decompose(np.array([[z_x, z_y]]), np.array([z_yaw]),
                                      gt[i:i + 1, :2], gt[i:i + 1, 3])
                r.update(mode='EKF_FUSED', retrieved_x=z_x, retrieved_y=z_y, retrieved_yaw=z_yaw,
                         innovation_x=innov[0], innovation_y=innov[1], innovation_yaw=innov[2], nis=nis,
                         R_xx=R_cov[0, 0], R_yy=R_cov[1, 1], R_yawyaw=R_cov[2, 2],
                         meas_err_long=float(ml[0]), meas_err_lat=float(ma[0]))
                ekf.update(np.array([z_x, z_y, z_yaw]), R_cov)

        est[i] = ekf.state
        el, ea, ey = decompose(est[i:i + 1, :2], est[i:i + 1, 2], gt[i:i + 1, :2], gt[i:i + 1, 3])
        e_xy = np.array([est[i, 0] - gt[i, 0], est[i, 1] - gt[i, 1]])
        try:
            nees = float(e_xy @ np.linalg.solve(ekf.P[:2, :2], e_xy))
        except np.linalg.LinAlgError:
            nees = float('nan')
        r.update(est_x=est[i, 0], est_y=est[i, 1], est_yaw=est[i, 2],
                 err_long=float(el[0]), err_lat=float(ea[0]), err_yaw=float(ey[0]),
                 P_xx=ekf.P[0, 0], P_yy=ekf.P[1, 1], P_yawyaw=ekf.P[2, 2], nees_xy=nees)
        rows.append(r)

    return est, rows, cols


def batch_positive_stats(poses, ts, batch=64, n_batches=500, seed=0):
    """How much positive signal does the contrastive loss see in a random training batch?"""
    dt = np.diff(ts) * 1e-9
    valid = np.nonzero((dt >= 0.01) & (dt <= 0.5))[0] + 1     # same rule as RacingDataset
    rng = np.random.default_rng(seed)
    xy, yaw = poses[:, :2], poses[:, 3]
    B = min(batch, len(valid))
    off = ~np.eye(B, dtype=bool)
    f_doc, f_tight, mass, ent = [], [], [], []
    for _ in range(n_batches):
        idx = rng.choice(valid, size=B, replace=False)
        p, a = xy[idx], yaw[idx]
        dist = np.linalg.norm(p[:, None] - p[None], axis=-1)
        ang = np.abs(wrap(a[:, None] - a[None]))
        spatial = np.exp(-dist / 8.0) * np.exp(-ang / 0.35)
        logits = np.where(off, spatial * 5.0, -np.inf)
        logits = logits - logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob = prob / prob.sum(axis=1, keepdims=True)
        pos_doc = off & (dist < 6.0) & (ang < math.radians(20))
        pos_tight = off & (dist < 2.0) & (ang < math.radians(10))
        f_doc.append(pos_doc.any(axis=1).mean())
        f_tight.append(pos_tight.any(axis=1).mean())
        mass.append((prob * pos_doc).sum(axis=1).mean())
        with np.errstate(divide='ignore', invalid='ignore'):
            h = -np.nansum(np.where(prob > 0, prob * np.log(prob), 0.0), axis=1) / math.log(B - 1)
        ent.append(h.mean())
    return dict(batch=B, frac_pos_6m_20deg=float(np.mean(f_doc)), frac_pos_2m_10deg=float(np.mean(f_tight)),
                mean_target_mass_on_pos=float(np.mean(mass)), norm_target_entropy=float(np.mean(ent)))


def make_plot(path, t, errs_long, fused_rows, near_any, near_head, speed, title):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f'[PLOT] skipped ({exc!r})')
        return
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for name, e in errs_long.items():
        ax[0].plot(t, e, lw=1, label=name)
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].set_ylabel('longitudinal error [m]')
    ax[0].legend(loc='upper left', ncol=4, fontsize=8)
    ax[0].set_title(title, fontsize=10)

    fe = np.array([r['err_long'] for r in fused_rows])
    me = np.array([r['meas_err_long'] for r in fused_rows])
    ex = np.array([r['gate_excluded_close_ref'] == 1 for r in fused_rows])
    nc = np.array([r['ref_close_to_gt'] == 0 for r in fused_rows])
    ax[1].plot(t, fe, lw=1.2, color='C3', label='fused estimate error')
    ax[1].plot(t, me, lw=0.8, color='C0', alpha=0.6, label='retrieval measurement error (fused loop)')
    for a, b in spans(ex):
        ax[1].axvspan(t[a], t[b], color='red', alpha=0.2, lw=0)
    for a, b in spans(nc):
        ax[1].axvspan(t[a], t[b], color='gray', alpha=0.25, lw=0)
    ax[1].set_ylabel('fused: longitudinal [m]')
    ax[1].set_title('red = close reference existed but was outside the gate;  gray = no close reference in the map',
                    fontsize=9)
    ax[1].legend(loc='upper left', fontsize=8)

    ax[2].plot(t, near_any, lw=1, label='nearest reference (position only)')
    ax[2].plot(t, np.where(np.isfinite(near_head), near_head, np.nan), lw=1, label='nearest heading-compatible reference')
    ax[2].set_ylabel('distance to map [m]')
    ax[2].set_xlabel('time since validation start [s]')
    ax[2].legend(loc='upper left', fontsize=8)
    ax2 = ax[2].twinx()
    ax2.plot(t, speed, color='k', lw=0.6, alpha=0.5)
    ax2.set_ylabel('GT speed [m/s]')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def sweep(d, args, EKF):
    """Try a grid of filter settings on the same network outputs; rank by error on frames the map covers."""
    import copy
    import itertools
    gt = d['gt']
    grid = list(itertools.product([0.0, 0.3, 0.5, 1.0],            # q_x_floor [m per frame]
                                  [1.0, 0.3, 0.1, 0.03],           # r_scale
                                  [(35.0, 35.0), (80.0, 60.0)]))   # gate (m, deg)
    print(f'[SWEEP] {len(grid)} filter settings on {len(gt)} validation frames (one dot = one setting done)')
    res, covered = [], None
    for qf, rs, (gm, gd) in grid:
        a = copy.copy(args)
        a.q_x_floor, a.r_scale, a.gate_m, a.gate_deg = qf, rs, gm, gd
        est, rows, _ = run_fused(d, a, EKF)
        if covered is None:
            covered = np.array([r['ref_close_to_gt'] == 1 for r in rows])
        el, ea, ey = decompose(est[:, :2], est[:, 2], gt[:, :2], gt[:, 3])
        e = np.hypot(el, ea)
        res.append(dict(q_x_floor=qf, r_scale=rs, gate_m=gm, gate_deg=gd,
                        cov_rmse=float(np.sqrt((e[covered] ** 2).mean())),
                        cov_median=float(np.median(e[covered])),
                        cov_p95=float(np.percentile(e[covered], 95)),
                        cov_yaw_deg=float(np.degrees(np.sqrt((ey[covered] ** 2).mean()))),
                        unc_rmse=float(np.sqrt((e[~covered] ** 2).mean())) if (~covered).any() else float('nan'),
                        all_rmse=float(np.sqrt((e ** 2).mean()))))
        print('.', end='', flush=True)
    print()
    res.sort(key=lambda r: r['cov_rmse'])
    base = next(r for r in res if r['q_x_floor'] == 0.0 and r['r_scale'] == 1.0 and r['gate_m'] == 35.0)
    head = (f"{'rank':>4}  {'q_x_floor':>9} {'r_scale':>8} {'gate':>9}  {'covered RMSE':>12} {'median':>7} {'p95':>6} "
            f"{'yaw[deg]':>9}  {'uncovered RMSE':>14} {'all RMSE':>9}")
    print(f'\nRanked by RMSE on frames the map covers ({covered.mean():.1%} of frames). '
          'The uncovered / all columns are shown for information only.')
    print(head)
    print('-' * len(head))

    def line(rank, r, tag=''):
        print(f"{rank:>4}  {r['q_x_floor']:>9.2f} {r['r_scale']:>8.2f} {r['gate_m']:>4.0f}/{r['gate_deg']:<4.0f}  "
              f"{r['cov_rmse']:>12.2f} {r['cov_median']:>7.2f} {r['cov_p95']:>6.2f} {r['cov_yaw_deg']:>9.2f}  "
              f"{r['unc_rmse']:>14.2f} {r['all_rmse']:>9.2f} {tag}")
    for i, r in enumerate(res[:10], 1):
        line(i, r, '<- current pipeline' if r is base else '')
    if base not in res[:10]:
        line(res.index(base) + 1, base, '<- current pipeline')
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'sweep_results.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    print(f'\n[FILES] {out}')


def analyze(d, args, EKF, ekf_note):
    os.makedirs(args.out_dir, exist_ok=True)
    ts, gt = d['ts'], d['gt']
    N = len(ts)
    t = (ts - ts[0]) * 1e-9
    gt_xy, gt_yaw = gt[:, :2], gt[:, 3]
    dt = np.maximum(np.diff(ts) * 1e-9, 1e-3)
    speed = np.r_[np.nan, np.hypot(*np.diff(gt_xy, axis=0).T) / dt]
    speed[0] = speed[1]

    print(f'\n[INFO] {ekf_note}')
    print(f'[INFO] fused settings: gate {args.gate_m:g} m / {args.gate_deg:g} deg, min_sim {args.min_sim:g}, '
          f'fix_q_scaling={args.fix_q_scaling}, q_x_floor={getattr(args, "q_x_floor", 0.0):g}, r_scale={getattr(args, "r_scale", 1.0):g}')
    print(f'[INFO] {N} validation frames, {t[-1]:.1f} s, {len(d["db_poses"])} DB references, '
          f'mean GT speed {np.nanmean(speed):.1f} m/s')

    # ---- variants -----------------------------------------------------------------------------
    db_emb, db_poses = d['db_emb'], d['db_poses']
    top1_idx = np.zeros(N, dtype=int)
    top1_sim = np.zeros(N)
    for s in range(0, N, 512):
        sims = d['emb'][s:s + 512] @ db_emb.T
        top1_idx[s:s + 512] = sims.argmax(axis=1)
        top1_sim[s:s + 512] = sims.max(axis=1)
    ret = db_poses[top1_idx]                                     # x, y, yaw

    aux_xy = d['aux'][:, :2].astype(np.float64) * d['xy_std'] + d['xy_mean']
    aux_yaw = np.arctan2(d['aux'][:, 2], d['aux'][:, 3])

    vo = integrate_vo([gt[0, 0], gt[0, 1], gt[0, 3]], d['delta'])
    est, fused_rows, fused_cols = run_fused(d, args, EKF)

    variants = {
        'retrieval_top1': (ret[:, :2], ret[:, 2]),
        'aux_head': (aux_xy, aux_yaw),
        'vo_only': (vo[:, :2], vo[:, 2]),
        'fused': (est[:, :2], est[:, 2]),
    }
    errs, summ = {}, []
    for name, (pxy, pyaw) in variants.items():
        el, ea, ey = decompose(pxy, pyaw, gt_xy, gt_yaw)
        errs[name] = (el, ea, ey)
        summ.append(summarize(name, el, ea, ey))

    print('\n=== 1. Variants on identical validation timestamps (metres; error axes follow the GT heading) ===')
    print_table(summ)
    sm = {r['name']: r for r in summ}

    covered = np.array([r['ref_close_to_gt'] == 1 for r in fused_rows])
    if covered.any() and (~covered).any():
        print(f'\n=== 1b. Same variants, split by map coverage '
              f'(covered = a reference within {args.close_m:.0f} m / {args.close_deg:.0f} deg of the true pose) ===')
        print(f"{'variant':<15}{'RMSE covered':>14}{'median covered':>16}{'RMSE uncovered':>16}{'uncovered share of total sq. error':>38}")
        for name, (el, ea, ey) in errs.items():
            sq = el ** 2 + ea ** 2
            share = np.nansum(sq[~covered]) / np.nansum(sq)
            print(f"{name:<15}{np.sqrt(np.nanmean(sq[covered])):>14.2f}{np.nanmedian(np.sqrt(sq[covered])):>16.2f}"
                  f"{np.sqrt(np.nanmean(sq[~covered])):>16.2f}{share:>37.1%}")
        print(f'({covered.mean():.1%} of frames are covered, {(~covered).mean():.1%} are not)')

    # global retrieval quality (independent of gate and filter)
    ret_dist = np.hypot(ret[:, 0] - gt_xy[:, 0], ret[:, 1] - gt_xy[:, 1])
    recall = {m: float((ret_dist <= m).mean()) for m in (2, 5, 10, 35)}
    print('\nGlobal top-1 retrieval: share of frames whose retrieved pose lies within '
          + ', '.join(f'{m} m: {recall[m]:.1%}' for m in recall)
          + f' | median top-1 similarity {np.median(top1_sim):.2f}')

    if covered.any():
        rc = {m: float((ret_dist[covered] <= m).mean()) for m in (2, 5, 10)}
        print('On covered frames only, global top-1 retrieval within '
              + ', '.join(f'{m} m: {rc[m]:.1%}' for m in rc))

    # ---- coverage -----------------------------------------------------------------------------
    near_any, near_head = coverage(gt, db_poses, args.close_deg)
    fin = near_head[np.isfinite(near_head)]
    print('\n=== 2. Reference-map coverage (GT-assisted; a baseline for stored-pose retrieval, not a bound for refinement) ===')
    print(f'nearest reference, position only      : median {np.median(near_any):.2f} m | p95 {np.percentile(near_any, 95):.2f} m | max {near_any.max():.2f} m')
    if len(fin):
        print(f'nearest heading-compatible (<= {args.close_deg:.0f} deg): median {np.median(fin):.2f} m | p95 {np.percentile(fin, 95):.2f} m | max {fin.max():.2f} m'
              f' | frames with none: {int((~np.isfinite(near_head)).sum())}')
    no_close = np.array([r['ref_close_to_gt'] == 0 for r in fused_rows])
    print(f'frames with no reference within {args.close_m:.0f} m and {args.close_deg:.0f} deg of the true pose: {no_close.mean():.1%}')

    # ---- gate exclusion -----------------------------------------------------------------------
    excl = np.array([r['gate_excluded_close_ref'] == 1 for r in fused_rows])
    used = np.array([r['mode'] == 'EKF_FUSED' for r in fused_rows])
    gate_n = np.array([r['gate_n'] for r in fused_rows])
    print(f'\n=== 3. Gate exclusion (gate {args.gate_m:.0f} m / {args.gate_deg:.0f} deg around the PREDICTED state) ===')
    print(f'frames where a close reference existed but the gate excluded it : {excl.mean():.1%}')
    if excl.any():
        first = int(np.argmax(excl))
        print(f'first such frame: #{first} (t = {t[first]:.1f} s); longest run: '
              f'{max(b - a + 1 for a, b in spans(excl))} frames')
    print(f'frames where a retrieval measurement was fused                  : {used.mean():.1%}')
    print(f'frames with fewer than 5 references in the gate (no update)     : {(gate_n < 5).mean():.1%}')

    # ---- VO -----------------------------------------------------------------------------------
    c, s_ = np.cos(gt_yaw[:-1]), np.sin(gt_yaw[:-1])
    gx, gy = np.diff(gt_xy[:, 0]), np.diff(gt_xy[:, 1])
    gt_step = np.stack([c * gx + s_ * gy, -s_ * gx + c * gy, wrap(np.diff(gt_yaw))], axis=1)
    pr = d['delta'][1:]
    err = pr - gt_step
    ok = np.isfinite(err).all(axis=1)
    err, pr, gt_step_ok, spd = err[ok], pr[ok], gt_step[ok], speed[1:][ok]
    print('\n=== 4. Visual odometry, per step (frame t-1 -> t) ===')
    print(f'step RMSE   : dx {np.sqrt((err[:, 0] ** 2).mean()):.3f} m | dy {np.sqrt((err[:, 1] ** 2).mean()):.3f} m | '
          f'dyaw {math.degrees(np.sqrt((err[:, 2] ** 2).mean())):.3f} deg')
    print(f'step bias   : dx {err[:, 0].mean():+.3f} m | dy {err[:, 1].mean():+.3f} m | dyaw {math.degrees(err[:, 2].mean()):+.3f} deg')
    print(f'forward-distance scale (sum predicted / sum true): {pr[:, 0].sum() / gt_step_ok[:, 0].sum():.3f}  '
          f'(1.000 = no systematic scale error)')
    edges = np.percentile(spd, [0, 33.3, 66.7, 100])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (spd >= lo) & (spd <= hi)
        if m.any() and gt_step_ok[m, 0].sum() != 0:
            print(f'   speed {lo:5.1f}-{hi:5.1f} m/s: scale {pr[m, 0].sum() / gt_step_ok[m, 0].sum():.3f}, '
                  f'step RMSE dx {np.sqrt((err[m, 0] ** 2).mean()):.3f} m')
    dt_raw = d['dt_raw'][1:]
    bad_dt = (dt_raw < 0.01) | (dt_raw > 0.5)
    print(f'frame intervals outside the training range [0.01, 0.5] s: {int(bad_dt.sum())} of {len(dt_raw)}'
          f' (max interval {np.nanmax(dt_raw):.3f} s)')
    lv = d['logvar'][1:][ok]
    pred_sd_now = np.sqrt(np.exp(lv).mean(axis=0))
    pred_sd_w = np.sqrt((np.exp(lv) / np.array([1.0, 2.0, 5.0])).mean(axis=0))
    act = np.sqrt((err ** 2).mean(axis=0))
    print('VO uncertainty calibration (actual RMS error / predicted std; about 1.0 is calibrated):')
    print(f'   as used now exp(log_var)        : dx {act[0] / pred_sd_now[0]:.2f} | dy {act[1] / pred_sd_now[1]:.2f} | dyaw {act[2] / pred_sd_now[2]:.2f}')
    print(f'   with exp(log_var)/[1,2,5] (fix) : dx {act[0] / pred_sd_w[0]:.2f} | dy {act[1] / pred_sd_w[1]:.2f} | dyaw {act[2] / pred_sd_w[2]:.2f}')

    # ---- filter consistency -------------------------------------------------------------------
    nis = np.array([r['nis'] for r in fused_rows])
    nees = np.array([r['nees_xy'] for r in fused_rows])
    print('\n=== 5. Filter consistency (fused loop) ===')
    print(f'NIS  (innovation vs. predicted S, 3 dof, ideal mean 3): mean {np.nanmean(nis):.1f} | median {np.nanmedian(nis):.1f}')
    print(f'NEES (position error vs. P, 2 dof, ideal mean 2)      : mean {np.nanmean(nees):.1f} | median {np.nanmedian(nees):.1f}')

    # ---- descriptor-training batches ----------------------------------------------------------
    bs = batch_positive_stats(d['train_poses'], d['train_ts'])
    print(f'\n=== 6. What the contrastive loss sees in a random training batch of {bs["batch"]} ===')
    print(f'anchors with >= 1 other sample within 6 m and 20 deg (the docstring thresholds): {bs["frac_pos_6m_20deg"]:.1%}')
    print(f'anchors with >= 1 other sample within 2 m and 10 deg                           : {bs["frac_pos_2m_10deg"]:.1%}')
    print(f'mean target probability mass on those 6 m / 20 deg pairs                       : {bs["mean_target_mass_on_pos"]:.3f}')
    print(f'normalized entropy of the target distribution (1.0 = uniform over the batch)   : {bs["norm_target_entropy"]:.3f}')

    # ---- interpretation hints -----------------------------------------------------------------
    print('\n=== Reading the numbers (rules of thumb, not proofs) ===')
    f_, r_, v_, a_ = (sm[k]['rmse_xy'] for k in ('fused', 'retrieval_top1', 'vo_only', 'aux_head'))

    def sm_cov(name):
        el, ea, _ = errs[name]
        return float(np.sqrt(np.nanmean((el ** 2 + ea ** 2)[covered])))
    if covered.any():
        f_c = sm_cov('fused')
        r_c = sm_cov('retrieval_top1')
        if r_c < 0.5 * f_c:
            print(f'- On covered frames raw retrieval ({r_c:.1f} m) beats the fused output ({f_c:.1f} m): the retrieval is good, '
                  'the filter is not using it enough (Q, R, gate, VO bias).')
        elif f_c < 0.5 * r_c:
            print(f'- On covered frames fused ({f_c:.1f} m) is much better than raw retrieval ({r_c:.1f} m): retrieval itself is the weak link.')
        else:
            print(f'- On covered frames fused ({f_c:.1f} m) and raw retrieval ({r_c:.1f} m) are similar.')
        if (~covered).any():
            fsq = errs['fused'][0] ** 2 + errs['fused'][1] ** 2
            unc = np.nansum(fsq[~covered]) / np.nansum(fsq)
            if unc > 0.5:
                print(f'- {unc:.0%} of the fused squared error comes from frames the map does not cover: check training coverage / the '
                      'evaluation protocol before tuning anything else.')
        if rc[5] < 0.5:
            print(f'- Only {rc[5]:.0%} of covered frames are retrieved within 5 m: the descriptor confuses places (improve training / sampling).')
    if excl.mean() > 0.05:
        print(f'- The gate hid a close reference in {excl.mean():.0%} of frames: the prediction drifted out of reach (add recovery / widen the gate).')
    if np.percentile(near_any, 95) > 5.0 or no_close.mean() > 0.05:
        print('- The map has coverage gaps on the validation route (check the plot, e.g. pit lane).')
    scale = pr[:, 0].sum() / gt_step_ok[:, 0].sum()
    if abs(scale - 1.0) > 0.02:
        print(f'- VO has a systematic forward-scale error ({scale:.3f}): this drifts along-track by itself.')
    if np.nanmean(nis) > 6 or np.nanmean(nees) > 4:
        print('- The filter is over-confident (NIS / NEES above the ideal): measurement or process covariances are too small.')
    if a_ < min(f_, r_, v_):
        print('- The auxiliary pose head beats everything else: keep it as a baseline / coarse recovery proposal.')
    if bs['norm_target_entropy'] > 0.9 or bs['frac_pos_6m_20deg'] < 0.3:
        print('- Random batches contain few close pairs, so the contrastive loss gets little positive signal: change the batch sampler first.')
    if sm['fused']['long_share'] > 0.9:
        print(f'- {sm["fused"]["long_share"]:.0%} of the fused squared error is along-track: lateral and yaw are not the main problem.')

    # ---- files --------------------------------------------------------------------------------
    tel = os.path.join(args.out_dir, 'fused_telemetry.csv')
    with open(tel, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fused_cols)
        w.writeheader()
        for r in fused_rows:
            w.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v) for k, v in r.items()})

    var_csv = os.path.join(args.out_dir, 'variants_per_frame.csv')
    with open(var_csv, 'w', newline='') as f:
        w = csv.writer(f)
        head = ['timestamp_ns', 't_s', 'gt_x', 'gt_y', 'gt_yaw', 'gt_speed', 'nearest_ref_m']
        for name in variants:
            head += [f'{name}_x', f'{name}_y', f'{name}_yaw', f'{name}_err_long', f'{name}_err_lat', f'{name}_err_yaw']
        w.writerow(head)
        for i in range(N):
            row = [int(ts[i]), f'{t[i]:.3f}', f'{gt_xy[i, 0]:.4f}', f'{gt_xy[i, 1]:.4f}', f'{gt_yaw[i]:.5f}',
                   f'{speed[i]:.3f}', f'{near_any[i]:.3f}']
            for name, (pxy, pyaw) in variants.items():
                row += [f'{pxy[i, 0]:.4f}', f'{pxy[i, 1]:.4f}', f'{pyaw[i]:.5f}',
                        f'{errs[name][0][i]:.4f}', f'{errs[name][1][i]:.4f}', f'{errs[name][2][i]:.5f}']
            w.writerow(row)

    png = os.path.join(args.out_dir, 'diagnostics.png')
    if not args.no_plot:
        make_plot(png, t, {k: v[0] for k, v in errs.items()}, fused_rows, near_any, near_head, speed,
                  f'fused RMSE {f_:.2f} m | retrieval_top1 {r_:.2f} m | vo_only {v_:.2f} m | aux_head {a_:.2f} m')
    print(f'\n[FILES] {tel}\n        {var_csv}' + (f'\n        {png}' if not args.no_plot else ''))
    return summ


def main():
    p = argparse.ArgumentParser(description='Offline, ground-truth-assisted diagnosis of the localization pipeline.')
    p.add_argument('--data-h5', default='data/training_cache.h5')
    p.add_argument('--checkpoint', default='checkpoints/racing_model.pt')
    p.add_argument('--out-dir', default='diagnostics')
    p.add_argument('--limit', type=int, default=None, help='only use the first N validation frames (quick test)')
    p.add_argument('--batch', type=int, default=32)
    # what-if knobs for the fused replica (defaults = current inference settings)
    p.add_argument('--gate-m', type=float, default=35.0)
    p.add_argument('--gate-deg', type=float, default=35.0)
    p.add_argument('--min-sim', type=float, default=0.45)
    p.add_argument('--fix-q-scaling', action='store_true', help='divide exp(log_var) by the VO loss weights [1,2,5]')
    p.add_argument('--q-x-floor', type=float, default=0.0,
                   help='what-if: minimum along-track VO process noise, as a std in metres per frame (0 = off)')
    p.add_argument('--r-scale', type=float, default=1.0,
                   help='what-if: multiply the retrieval measurement covariance (<1 trusts retrieval more)')
    # definition of "a reference close to the true pose"
    p.add_argument('--close-m', type=float, default=5.0)
    p.add_argument('--close-deg', type=float, default=20.0)
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--sweep', action='store_true',
                   help='instead of the full report, try a grid of filter settings and rank them')
    args = p.parse_args()

    EKF, note = get_ekf_class()
    data = load_inputs(args)
    if args.sweep:
        sweep(data, args, EKF)
        return
    analyze(data, args, EKF, note)


if __name__ == '__main__':
    main()
```

when i ran the test with python Scripts/4_diagnostics.py --sweep
i got the following table. apparently this is supposed to get some understanding. 
table:
	q_x_floor	r_scale	gate_m	gate_deg	cov_rmse	cov_median	cov_p95	cov_yaw_deg	unc_rmse	all_rmse
1	0.5	0.03	35.0	35.0	2.0741385717101277	1.1322267676597613	3.7297897771336523	1.3789942351343651	38.925752394621895	13.326368616991136
2	1.0	0.1	35.0	35.0	2.107597242077607	1.1163838292475692	3.7475984299148464	1.2955739956442616	39.855972155596426	13.642621390789266
3	0.3	0.03	35.0	35.0	2.1390796645668084	1.2048525784713484	3.777710264688839	1.397863099807711	34.279663189411195	11.782414818908835
4	1.0	0.03	35.0	35.0	2.1532942072285035	1.16147445406543	3.9004872881662758	1.364138645533744	27.025563998581358	9.374113461219713
5	1.0	0.3	35.0	35.0	2.1766831177779125	1.1793308495648063	3.7766259412087924	1.281007467583144	39.5288334992511	13.54269593168403
6	0.5	0.03	80.0	60.0	2.1964229303682217	1.1322267676597613	3.7297897771336523	1.331673399234185	62.45886376452611	21.25319650887516
7	0.5	0.1	35.0	35.0	2.211231138619502	1.2632015699858834	3.8363249525545315	1.301683157944042	38.831532477091166	13.314351024873622
8	1.0	0.1	80.0	60.0	2.2290949847703074	1.1163838292475692	3.7475984299148464	1.2580128475503702	62.75215821627852	21.355053172782718
9	0.3	0.03	80.0	60.0	2.230469156937089	1.2048525784713484	3.761307655998801	1.3529738485289655	61.616933759022416	20.9726172163112
10	1.0	0.3	80.0	60.0	2.2625205450327597	1.1793308495648063	3.76482226691244	1.2421621663479965	62.15701254657376	21.157624634446154
11	1.0	0.03	80.0	60.0	2.287499741450124	1.16147445406543	3.891740699085635	1.324765824731953	62.44140813447257	21.255819233414563
12	0.5	0.1	80.0	60.0	2.2905520664858585	1.2632015699858834	3.850477547498352	1.2666759399738328	62.236340717473404	21.18702064498005
13	1.0	1.0	35.0	35.0	2.5598338251779853	1.5931420207459546	4.5780703846818644	1.2688001651594412	38.089589569012695	13.122453142013551
14	0.3	0.1	35.0	35.0	2.601435809152871	1.651614622649128	4.728758039976265	1.3177616212582688	32.05607798900424	11.128712325964601
15	1.0	1.0	80.0	60.0	2.6157496556357915	1.5931420207459546	4.582220565664475	1.2434268316812682	58.291257561643256	19.893897221462296
16	0.3	0.1	80.0	60.0	2.657848814586779	1.651614622649128	4.730342713599957	1.2859455243761717	57.13736705181908	19.51121718145507
17	0.5	0.3	35.0	35.0	2.6606804017029315	1.6833226823862915	4.819724762232643	1.2810275964157083	40.00832023062671	13.778656718772377
18	0.5	0.3	80.0	60.0	2.713424956080849	1.6833226823862915	4.822594460439508	1.2532510021254577	56.98717322669557	19.46756159934996
19	0.0	0.03	80.0	60.0	3.109232881312499	2.033659970794321	5.848106932161757	1.4794475472531181	55.51826263730312	19.028194281319344
20	0.0	0.03	35.0	35.0	3.1175202444812853	2.033659970794321	5.848106932161757	1.5332456406075945	30.410397312553144	10.70844685846245
21	0.3	0.3	80.0	60.0	3.6340756075951104	2.3627439559275603	6.836905786370974	1.2780653158500441	56.45558058160701	19.42274395502099
22	0.3	0.3	35.0	35.0	3.6437256728838703	2.3627439559275603	6.836905786370974	1.2909325097964535	35.91043309344038	12.635527666733324
23	0.5	1.0	80.0	60.0	3.854243495063244	2.5216114484915084	7.259827763774712	1.2587878004671933	56.67717522088723	19.534018296533546
24	0.5	1.0	35.0	35.0	3.8586756616071356	2.5216114484915084	7.259827763774712	1.2606363838885046	36.49434965965713	12.881507910629697
25	0.0	0.1	80.0	60.0	4.746548345654148	3.027469388817394	9.276388267905556	1.415558861506488	56.03381739808676	19.495010813766484
26	0.0	0.1	35.0	35.0	4.748738226654522	3.027469388817394	9.276388267905556	1.4226748561079763	28.62937267065322	10.675705375041181
27	0.3	1.0	35.0	35.0	5.644172590481826	3.484532342663329	10.90081141373438	1.2915766056851459	52.2793931405205	18.484388247963157
28	0.3	1.0	80.0	60.0	5.6445241127933805	3.484532342663329	10.90081141373438	1.2915995461151788	56.59773023111337	19.889716032402387
29	0.0	0.3	35.0	35.0	7.253866190386562	4.330594270510437	14.40678195315053	1.456188451671695	51.513933619815425	18.733419607235888
30	0.0	0.3	80.0	60.0	7.281263542627039	4.330594270510437	14.40678195315053	1.456737349886301	56.5961705545277	20.354612592308968
31	0.0	1.0	80.0	60.0	11.539335903463492	6.644280902563275	21.130271339213586	1.505386406138905	76.0134721889386	27.93889955659389
32	0.0	1.0	35.0	35.0	11.776168353246481	6.70703065953404	21.130270146546472	1.5199588217533424	64.55692771027842	24.510492906929215


for reference refer the following chats:
```
No need to apologise, this output is informative. Before I explain it, I need to fix two things in my own script:

- **Coverage split:** it doesn't separate frames the map covers from frames it doesn't, and that split looks like the key to your numbers.
- **First hint:** the "retrieval is the weak link" line compares RMSEs, which outliers dominate, so I'm correcting it.

Yes, please send `fused_telemetry.csv`, which is about 1 MB. You can skip `variants_per_frame.csv`. The telemetry file has the per-frame filter covariances and retrieval similarity, which I need to confirm the main suspect below. I've also updated the script (`4_diagnose.py`): it now splits errors by map coverage and has two new what-if flags. Replace your `4_diagnostics.py` with it.

## How to read the plot

- **Top panel:** each variant's along-track error over time.
- **Middle panel:** the fused error (red) against the error of the retrieval measurement fed into the filter (blue). Red shading means a nearby reference existed but the gate hid it. Grey means the map has nothing near the true pose.
- **Bottom panel:** distance to the nearest reference, with the car's speed in grey.

## What your numbers say

There are two separate problems stacked on top of each other.

**1. The last ~27 s are outside your map.**
- 11.5% of validation frames have no reference within 5 m. In the grey zone the nearest reference is 9–13 m away, and the car slows to about 9 m/s and then stops. My guess is the pit lane or pit box, but check that.
- Raw retrieval jumps hundreds of metres there. The fused estimate follows the wrong retrievals up to about 100 m of error.
- By eye, that segment is roughly 80% of your squared error. The new table 1b prints the exact figure.

**2. On the covered track, retrieval is good but the filter lags behind it by 10–25 m.**
- Retrieval top-1 has a median error of 1.6 m, and 84% of frames land within 5 m. That is about 95% of the frames the map covers.
- The blue measurement error stays within about ±3 m, while the red fused error sits 10–25 m below it. The filter has a persistent negative offset, meaning the estimate sits behind the true position, even though its own measurements are accurate.
- VO shrinks speeds towards the average. Its forward scale is 1.28 below 20 m/s, 0.88 in the middle range and 0.80 above 30 m/s. At speed it under-reports motion, so the estimate falls behind.
- NEES has a median of 325 against an ideal of 2. The filter thinks it is accurate to well under a metre while it is about 10 m off.

My hypothesis is that the steady lag is roughly the VO shortfall per frame divided by the Kalman gain along the track, and that gain is tiny. VO's own along-track uncertainty is about 5× too small (calibration 5.26). The retrieval covariance is also wide along the track, because it comes from the spread of the top-8 reference positions. The `P_xx` and `R_xx` columns in the CSV will confirm or refute this.

## What to stop worrying about

- **`aux_head` and `vo_only`:** at 166 m and 391 m they are unusable alone, so don't tune them.
- **The gate:** it hid a nearby reference in only 0.8% of frames, one 20-frame episode at 110.8 s. That coincides with your worst on-track error (−47 m, beyond the 35 m gate radius). It is not the main problem, but it is the edge of a cliff, so widening it is cheap insurance.
- **Yaw and lateral error:** 2.2° and 3.2 m are fine.
- **Descriptor training:** the batch-sampler rework was the review's top suggestion. Only 18% of anchors have a positive in a batch, yet retrieval works well on covered track. Put it last.
- **The review's `[1,2,5]` fix:** it brings `dy` and `dyaw` calibration to 1.00 and 0.50, but leaves `dx` at 5.26, so it won't touch your main issue.

One correction: the old script printed "retrieval itself is the weak link". That was misleading, because the RMSE comparison was dominated by off-map outliers. The new version compares on covered frames only.

## What to do, in order

1. **Re-run the updated script with no flags.** Table 1b gives the exact covered/uncovered split.
2. **Probe the filter, with no retraining.** The goal is to bring the covered-frame error towards retrieval's own ±3 m:
   ```bash
   python Scripts/4_diagnostics.py --q-x-floor 0.5
   python Scripts/4_diagnostics.py --r-scale 0.1
   python Scripts/4_diagnostics.py --q-x-floor 0.5 --r-scale 0.1 --gate-m 80 --gate-deg 60
   ```
   Watch "RMSE covered", "median covered" and NEES in 1b. These values are probes, not recommendations, because I haven't seen your real `R` and `P`. Judge them on covered frames only. This is a single validation stretch, so keep any settings you adopt round and sensible.
3. **Port the winning settings into `3_causal_infer.py`, then fix the VO scale.** The cheapest way is to estimate the scale online from retrieval, either as an extra filter state or as a sliding-window ratio. Retraining the VO head to predict speed with a relative-error loss is the heavier alternative.
4. **Decide how much the off-map segment matters.**
   - Check whether the real evaluation includes that kind of area, since it depends on the organisers' protocol.
   - Plot `gt_x` and `gt_y` of the tail against your training positions.
   - Check in the CSV whether `top1_sim` drops there, because a similarity drop could serve as a rejection signal.
5. **Only then retrain the descriptor or VO.**


Both tools are done on my side: I've analysed your CSV and added a one-command parameter sweep to the script. I'm handing that over now with a plain-language read of your three runs.

Understood, I'll drive from here and you only need to run what I give you. Your three runs show clear progress, and I've found something important about the last seconds of your validation data.

## Where you stand

- **On the part of the track your map covers, the model is now accurate.** The combined run (`--q-x-floor 0.5 --r-scale 0.1 --gate-m 80 --gate-deg 60`) averages 2.3 m of error there. That beats raw retrieval at 2.95 m.
- **My explanation for the lag was partly wrong.** I guessed the retrieval's uncertainty was too wide, but your CSV shows it was small. The cure still worked: trusting the retrieval more.
- **The combined run's total of 21 m looks worse, but it isn't.** 88% of its squared error comes from the final 9 seconds, when the car is parked somewhere the map has never seen. Ignoring just those seconds, the total is 7.4 m. In that stretch the answer is close to luck: the three runs gave 36, 29 and 62 m. So don't pick settings by total error.

## Why the end can't be fixed by filter settings

- **Similarity is a clean "I'm lost" signal.** On-map matches score at least 0.946 for 90% of frames. Off-map matches score 0.89 or less for 99% of frames. A cutoff around 0.92 keeps 99% of the good matches and none of the bad ones.
- **Ignoring the bad matches makes the end worse, though.** In a replay on your logged data, I estimate the end error goes to about 100 m, because the visual odometry (VO) is broken there too. The car moves at a steady 8.3 m/s, but VO reports 8–24 m/s, and when parked it still reports 9 m/s. VO has never seen that place.
- **So only data can fix the end, not tuning.** The final model trained with `--full-train` will have this stretch in its map and in VO training. That only helps if the real test passes through the same place.

## What I'd like you to run now

Replace your `Scripts/4_diagnostics.py` with the updated `4_diagnose.py`, then run:

```bash
python Scripts/4_diagnostics.py --sweep
```

This takes about a minute. It tries 32 filter settings and ranks them by error on the covered part of the track. Your current pipeline is in the table for comparison. Paste the table back to me. I'll then write the improved `3_causal_infer.py` with the best settings, ready to drop in.

## One question

Do you know whether the hidden test is a separate recording of the same track, and whether it includes driving into the pit and parking? "I don't know" is a fine answer. If you can find out from the organisers, it decides how much the last seconds matter for your final score.
``` 

with this, i am expecting you to optimise the scripts. i am not sure if i understand all the technical terms and i will get to them at a later point in time. for now i want to improve the RMSE. if the info is good enough, let me know the updated files and i will test them. what i am looking for now is solution not how to


**Yes—your sweep gives enough information for a useful update without retraining.**

Use **row 4**, not row 1: the competition scores **all evaluated poses**, not just the map-covered portion.

| Configuration | Covered RMSE | Uncovered RMSE | **Overall RMSE** |
|---|---:|---:|---:|
| Original settings | 11.78 m | 64.56 m | **24.51 m** |
| Row 1: best covered-only result | 2.07 m | 38.93 m | **13.33 m** |
| **Row 4: recommended** | **2.15 m** | **27.03 m** | **9.37 m** |

That is approximately a **62% overall RMSE reduction in your diagnostic replay**. The actual inference run must confirm it.

The previous advice to ignore overall error was not appropriate for selecting the final scoring configuration. Covered-only error is useful for diagnosis, but **overall RMSE remains the objective**.

## Files to change

- **Keep `1_build_dataset.py`, `2_train_model.py`, your cache, and your checkpoint unchanged for this test**; the update below uses the existing model architecture and checkpoint contents. [allcodes.txt]
- **Replace `Scripts/3_causal_infer.py`** with the following.
- No retraining is needed.

The recommended defaults reproduce row 4’s settings:

```text
q_x_floor = 1.0
r_scale   = 0.03
gate_m    = 35
gate_deg  = 35
```

I have **not** enabled the variance-weight correction or a higher similarity cutoff: neither was part of the winning sweep configuration.

### Replacement: `Scripts/3_causal_infer.py`

This preserves the original model and retrieval approach, changes the tested filter settings, and adds timestamp-safe image buffering and more telemetry. [allcodes.txt]

```python
#!/usr/bin/env python3
"""
Causal camera localization with sweep-selected filter settings.

Default settings:
    q_x_floor = 1.0 m per step
    r_scale   = 0.03
    gate_m    = 35.0
    gate_deg  = 35.0
    min_sim   = 0.45

Compatible with the existing RacingLocNet checkpoint.

--val-only is a RESET validation experiment:
it may initialize from the first eligible validation GT pose.

Without --val-only, only the first eligible GT pose of the recording
is captured for initialization. No subsequent GT poses enter the estimator.
"""

import argparse
import csv
import importlib.util
import math
import os
import time
from collections import deque

import numpy as np
import torch

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class CausalSE2EKF:
    """
    State: [x, y, yaw].

    The sweep-selected Q/R adjustments are applied by the inference loop,
    not hidden inside this class. This keeps the diagnostics compatible.
    """

    def __init__(self, x0, y0, yaw0):
        self.state = np.array([x0, y0, yaw0], dtype=np.float64)
        self.P = np.diag([
            0.15 ** 2,
            0.15 ** 2,
            math.radians(1.0) ** 2,
        ])

    def predict(self, dx_body, dy_body, dyaw, q_var):
        x, y, yaw = self.state
        c, s = math.cos(yaw), math.sin(yaw)

        self.state[0] = x + c * dx_body - s * dy_body
        self.state[1] = y + s * dx_body + c * dy_body
        self.state[2] = wrap_angle(yaw + dyaw)

        F = np.array([
            [1.0, 0.0, -s * dx_body - c * dy_body],
            [0.0, 1.0,  c * dx_body - s * dy_body],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        rotation = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        Q_body = np.diag([
            max(float(q_var[0]), 0.05 ** 2),
            max(float(q_var[1]), 0.02 ** 2),
            max(float(q_var[2]), math.radians(0.3) ** 2),
        ])

        self.P = (
            F @ self.P @ F.T
            + rotation @ Q_body @ rotation.T
        )
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, measurement, R):
        residual = np.array([
            measurement[0] - self.state[0],
            measurement[1] - self.state[1],
            wrap_angle(measurement[2] - self.state[2]),
        ], dtype=np.float64)

        S = self.P + R

        # Equivalent to P @ inv(S), without explicitly taking an inverse.
        K = np.linalg.solve(S.T, self.P.T).T

        self.state += K @ residual
        self.state[2] = wrap_angle(self.state[2])

        # Joseph covariance update for numerical stability.
        I_K = np.eye(3) - K
        self.P = I_K @ self.P @ I_K.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)


def load_model(train_module, checkpoint, device):
    """
    Construct the original architecture without downloading ImageNet weights.
    All model parameters are subsequently loaded from the checkpoint.
    """
    original_resnet18 = train_module.models.resnet18

    def resnet18_without_download(*args, **kwargs):
        kwargs["weights"] = None
        return original_resnet18(*args, **kwargs)

    train_module.models.resnet18 = resnet18_without_download
    try:
        model = train_module.RacingLocNet(embed_dim=256)
    finally:
        train_module.models.resnet18 = original_resnet18

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def select_causal_image(buffer, anchor_ns):
    """Newest encountered image whose exposure stamp is <= anchor_ns."""
    best_stamp = None
    best_image = None

    for stamp, image in buffer:
        if stamp <= anchor_ns:
            if best_stamp is None or stamp > best_stamp:
                best_stamp = stamp
                best_image = image

    return best_stamp, best_image


def run_causal_inference(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    here = os.path.dirname(os.path.abspath(__file__))

    train_module = load_module_from_path(
        "train_mod", os.path.join(here, "2_train_model.py")
    )
    build_module = load_module_from_path(
        "build_mod", os.path.join(here, "1_build_dataset.py")
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model = load_model(train_module, checkpoint, device)

    db_embeddings = checkpoint["db_embeddings"].to(
        device=device, dtype=torch.float32
    )
    db_poses = (
        checkpoint["db_poses"].cpu().numpy().astype(np.float64)
    )

    if len(db_poses) == 0:
        raise ValueError("The checkpoint contains an empty reference map.")
    if len(db_poses) != len(db_embeddings):
        raise ValueError("Reference poses and embeddings have different lengths.")

    val_start_ns = 0
    if args.val_only:
        val_start_ns = int(
            checkpoint.get("val_start_timestamp_ns", 0)
        )
        if val_start_ns <= 0:
            raise ValueError(
                "--val-only requires a checkpoint with a held-out split."
            )

    camera_topics = build_module.CAMERA_TOPICS
    camera_count = len(build_module.CAMERAS)

    # Retain older images so a future-stamped arrival cannot overwrite
    # the most recent image eligible for the current anchor.
    buffers = [
        deque(maxlen=64) for _ in range(camera_count)
    ]

    first_gt_pose = None
    first_gt_stamp = None
    ekf = None
    previous_image = None
    previous_stamp = None

    n_frames = 0
    n_updates = 0
    n_skipped_stamps = 0
    timings_ms = []

    gate_radius_sq = args.gate_m ** 2
    gate_angle = math.radians(args.gate_deg)
    loss_weights = np.array([1.0, 2.0, 5.0], dtype=np.float64)

    for path in (args.output, args.log):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    telemetry_fields = [
        "frame", "timestamp_ns", "mode", "top1_sim",
        "anchor_ms", "dt", "max_camera_age_s", "missing_cameras",
        "dx_body_pred", "dy_body_pred", "dyaw_pred",
        "q_x", "q_y", "q_yaw",
        "pred_x", "pred_y", "pred_yaw",
        "gate_n",
        "retrieved_x", "retrieved_y", "retrieved_yaw",
        "innovation_x", "innovation_y", "innovation_yaw", "nis",
        "R_xx", "R_yy", "R_yawyaw",
        "P_xx", "P_yy", "P_yawyaw",
        "x", "y", "yaw",
    ]

    print(f"[INIT] Device: {device}")
    print(f"[INIT] Reference frames: {len(db_poses):,}")
    print(
        f"[CONFIG] q_x_floor={args.q_x_floor:g}, "
        f"r_scale={args.r_scale:g}, "
        f"gate={args.gate_m:g}m/{args.gate_deg:g}deg, "
        f"min_sim={args.min_sim:g}, "
        f"fix_q_scaling={args.fix_q_scaling}"
    )

    if args.val_only:
        print(
            "[VAL] Reset validation experiment; "
            f"start timestamp = {val_start_ns}"
        )

    topics = list(camera_topics) + ["/tf"]

    with (
        open(args.output, "w", newline="") as output_file,
        open(args.log, "w", newline="") as log_file,
        open(args.mcap_path, "rb") as bag_file,
        torch.inference_mode(),
    ):
        output_writer = csv.writer(output_file)
        output_writer.writerow(["timestamp_ns", "x", "y", "yaw"])

        log_writer = csv.DictWriter(
            log_file, fieldnames=telemetry_fields
        )
        log_writer.writeheader()

        reader = make_reader(
            bag_file, decoder_factories=[DecoderFactory()]
        )

        for _, channel, _, msg in reader.iter_decoded_messages(
            topics=topics
        ):
            topic = channel.topic

            # Capture once, never repeatedly overwrite the initial pose.
            if topic == "/tf":
                if ekf is None and first_gt_pose is None:
                    for transform in (
                        getattr(msg, "transforms", []) or []
                    ):
                        parent = transform.header.frame_id.lstrip("/")
                        child = transform.child_frame_id.lstrip("/")

                        if (
                            parent != "map"
                            or child != "constructor0/base_link"
                        ):
                            continue

                        stamp = build_module.stamp_to_ns(
                            transform.header.stamp
                        )
                        if stamp < val_start_ns:
                            continue

                        position = transform.transform.translation
                        yaw = build_module.yaw_from_quat(
                            transform.transform.rotation
                        )
                        first_gt_pose = (
                            float(position.x),
                            float(position.y),
                            float(yaw),
                        )
                        first_gt_stamp = stamp
                        break
                continue

            if not hasattr(msg, "data") or len(msg.data) == 0:
                continue

            camera_idx = camera_topics[topic]
            stamp_ns = build_module.stamp_to_ns(msg.header.stamp)

            start_time = time.perf_counter()
            image = build_module.crop_and_resize(msg.data)
            buffers[camera_idx].append((stamp_ns, image))

            # Output is anchored to camera_fl.
            if camera_idx != 0 or stamp_ns < val_start_ns:
                continue

            if previous_stamp is not None and stamp_ns <= previous_stamp:
                n_skipped_stamps += 1
                continue

            selected_images = []
            offsets = []
            missing_cameras = 0

            for camera_buffer in buffers:
                selected_stamp, selected_image = select_causal_image(
                    camera_buffer, stamp_ns
                )

                if selected_image is None:
                    # Same startup fallback as before, but never use
                    # a future-stamped secondary camera image.
                    selected_image = image
                    selected_stamp = stamp_ns
                    missing_cameras += 1

                selected_images.append(selected_image)
                offsets.append((stamp_ns - selected_stamp) * 1e-9)

            stacked = (
                np.stack(selected_images, axis=0).astype(np.float32)
                / 255.0
            )
            stacked = (
                stacked - stacked.mean(axis=(1, 2), keepdims=True)
            ) / (
                stacked.std(axis=(1, 2), keepdims=True) + 1e-5
            )
            current_image = (
                torch.from_numpy(stacked).unsqueeze(0).to(device)
            )

            record = {
                name: float("nan") for name in telemetry_fields
            }
            record.update(
                timestamp_ns=int(stamp_ns),
                mode="VO_ONLY",
                top1_sim=0.0,
                max_camera_age_s=float(max(offsets)),
                missing_cameras=missing_cameras,
            )

            delta = None
            q_var = None

            if previous_image is None:
                _, embedding, _ = model.forward_single(current_image)
            else:
                dt = max(
                    (stamp_ns - previous_stamp) * 1e-9, 0.01
                )
                time_features = torch.tensor(
                    [[dt] + offsets],
                    dtype=torch.float32,
                    device=device,
                )

                embedding, _, delta_tensor, log_var = model(
                    previous_image, current_image, time_features
                )
                delta = (
                    delta_tensor[0].cpu().numpy().astype(np.float64)
                )
                q_var = np.exp(
                    log_var[0].cpu().numpy()
                ).astype(np.float64)

                # OFF by default: the supplied sweep did not use it.
                if args.fix_q_scaling:
                    q_var /= loss_weights

                # Tested change 1: longitudinal process-noise floor.
                q_var[0] = max(
                    float(q_var[0]), args.q_x_floor ** 2
                )

                record.update(
                    dt=dt,
                    dx_body_pred=float(delta[0]),
                    dy_body_pred=float(delta[1]),
                    dyaw_pred=float(delta[2]),
                    q_x=float(q_var[0]),
                    q_y=float(q_var[1]),
                    q_yaw=float(q_var[2]),
                )

            if ekf is None:
                if (
                    first_gt_pose is not None
                    and first_gt_stamp <= stamp_ns
                ):
                    ekf = CausalSE2EKF(*first_gt_pose)
                    print("[START] Initialized from the captured initial pose.")
                else:
                    similarities = (
                        embedding @ db_embeddings.T
                    ).squeeze(0)
                    best_idx = int(torch.argmax(similarities).item())
                    ekf = CausalSE2EKF(*db_poses[best_idx])
                    print("[START] Initialized by global image retrieval.")

            if delta is not None:
                ekf.predict(
                    float(delta[0]),
                    float(delta[1]),
                    float(delta[2]),
                    q_var,
                )

            record.update(
                pred_x=float(ekf.state[0]),
                pred_y=float(ekf.state[1]),
                pred_yaw=float(ekf.state[2]),
            )

            dx = db_poses[:, 0] - ekf.state[0]
            dy = db_poses[:, 1] - ekf.state[1]
            heading_difference = np.abs(np.arctan2(
                np.sin(db_poses[:, 2] - ekf.state[2]),
                np.cos(db_poses[:, 2] - ekf.state[2]),
            ))

            gate_mask = (
                (dx * dx + dy * dy <= gate_radius_sq)
                & (heading_difference <= gate_angle)
            )
            gate_indices = np.flatnonzero(gate_mask)
            record["gate_n"] = int(len(gate_indices))

            if len(gate_indices) >= 5:
                gate_tensor = torch.as_tensor(
                    gate_indices, dtype=torch.long, device=device
                )
                gate_embeddings = db_embeddings.index_select(
                    0, gate_tensor
                )
                similarities = (
                    embedding @ gate_embeddings.T
                ).squeeze(0)

                k = min(8, len(gate_indices))
                top_values, top_local = torch.topk(
                    similarities, k=k
                )
                top1_sim = float(top_values[0].item())
                record["top1_sim"] = top1_sim

                if top1_sim > args.min_sim:
                    weights = (
                        torch.softmax(top_values * 15.0, dim=0)
                        .cpu().numpy().astype(np.float64)
                    )
                    # Normalize again after conversion for numerical hygiene.
                    weights /= weights.sum()

                    matched = db_poses[
                        gate_indices[top_local.cpu().numpy()]
                    ]

                    z_x = float(np.sum(weights * matched[:, 0]))
                    z_y = float(np.sum(weights * matched[:, 1]))
                    z_yaw = math.atan2(
                        float(np.sum(weights * np.sin(matched[:, 2]))),
                        float(np.sum(weights * np.cos(matched[:, 2]))),
                    )
                    measurement = np.array(
                        [z_x, z_y, z_yaw], dtype=np.float64
                    )

                    diff_xy = (
                        matched[:, :2] - np.array([[z_x, z_y]])
                    )
                    covariance_xy = (
                        (diff_xy.T * weights) @ diff_xy
                        + np.eye(2)
                        * (0.25 / max(top1_sim, 0.2)) ** 2
                    )

                    R = np.eye(3, dtype=np.float64) * (
                        math.radians(2.0) ** 2
                    )

                    # Tested change 2: scale POSITION covariance only.
                    # This matches the diagnostic sweep.
                    R[:2, :2] = covariance_xy * args.r_scale

                    innovation = measurement - ekf.state
                    innovation[2] = wrap_angle(innovation[2])
                    S = ekf.P + R
                    nis = float(
                        innovation @ np.linalg.solve(S, innovation)
                    )

                    record.update(
                        mode="EKF_FUSED",
                        retrieved_x=z_x,
                        retrieved_y=z_y,
                        retrieved_yaw=z_yaw,
                        innovation_x=float(innovation[0]),
                        innovation_y=float(innovation[1]),
                        innovation_yaw=float(innovation[2]),
                        nis=nis,
                        R_xx=float(R[0, 0]),
                        R_yy=float(R[1, 1]),
                        R_yawyaw=float(R[2, 2]),
                    )

                    ekf.update(measurement, R)
                    n_updates += 1

            previous_image = current_image
            previous_stamp = stamp_ns

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000.0
            timings_ms.append(elapsed_ms)

            n_frames += 1
            x, y, yaw = map(float, ekf.state)

            output_writer.writerow([
                int(stamp_ns),
                f"{x:.6f}",
                f"{y:.6f}",
                f"{yaw:.6f}",
            ])

            record.update(
                frame=n_frames,
                anchor_ms=elapsed_ms,
                P_xx=float(ekf.P[0, 0]),
                P_yy=float(ekf.P[1, 1]),
                P_yawyaw=float(ekf.P[2, 2]),
                x=x, y=y, yaw=yaw,
            )
            log_writer.writerow(record)

            if n_frames == 1 or n_frames % 100 == 0:
                print(
                    f"[{n_frames:5d}] {record['mode']:9s} | "
                    f"sim={record['top1_sim']:.3f} | "
                    f"gate={record['gate_n']:4d} | "
                    f"anchor={elapsed_ms:.1f}ms | "
                    f"x={x:.2f}, y={y:.2f}, "
                    f"yaw={math.degrees(yaw):.1f}deg"
                )
                output_file.flush()
                log_file.flush()

    if n_frames == 0:
        raise RuntimeError(
            "No predictions were produced. Check the bag and validation split."
        )

    timing_array = np.asarray(timings_ms)
    print("\n" + "=" * 70)
    print(f"[DONE] Wrote {n_frames} poses to {args.output}")
    print(f"[DONE] Telemetry: {args.log}")
    print(f"[INFO] Retrieval updates: {n_updates}/{n_frames}")
    print(f"[INFO] Non-increasing anchor stamps skipped: {n_skipped_stamps}")
    print(
        f"[TIMING] Anchor processing mean={timing_array.mean():.2f}ms, "
        f"p95={np.percentile(timing_array, 95):.2f}ms"
    )
    print(
        "[TIMING] Excludes bag decoding and secondary-camera preprocessing "
        "performed before the anchor."
    )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap_path")
    parser.add_argument(
        "--checkpoint", default="checkpoints/racing_model.pt"
    )
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--log", default="inference_telemetry.csv")
    parser.add_argument("--val-only", action="store_true")

    # Winning OVERALL-RMSE settings from the supplied sweep.
    parser.add_argument("--q-x-floor", type=float, default=1.0)
    parser.add_argument("--r-scale", type=float, default=0.03)
    parser.add_argument("--gate-m", type=float, default=35.0)
    parser.add_argument("--gate-deg", type=float, default=35.0)
    parser.add_argument("--min-sim", type=float, default=0.45)

    parser.add_argument(
        "--fix-q-scaling",
        action="store_true",
        help="Optional experiment; OFF for the supplied sweep winner.",
    )

    args = parser.parse_args()

    if args.q_x_floor < 0:
        parser.error("--q-x-floor must be nonnegative.")
    if args.r_scale <= 0:
        parser.error("--r-scale must be positive.")
    if args.gate_m <= 0 or args.gate_deg <= 0:
        parser.error("Gate limits must be positive.")

    run_causal_inference(args)


if __name__ == "__main__":
    main()
```

## Run this test

### 1. Back up your original inference file before replacing it

```bash
cp Scripts/3_causal_infer.py Scripts/3_causal_infer_original.py
```

Then save the replacement above as `Scripts/3_causal_infer.py`.

### 2. Run the updated inference

```bash
python Scripts/3_causal_infer.py \
  data/c0_2026-10-13_22-53-42_camloc_0.mcap \
  --checkpoint checkpoints/racing_model.pt \
  --val-only \
  --output val_submission_tuned.csv \
  --log inference_telemetry_tuned.csv
```

### 3. Score against the same validation ground truth

```bash
python tools/evaluate.py gt_val.csv val_submission_tuned.csv
```

**The result to look for is an overall RMSE near the sweep’s 9.37 m—not 2.15 m.** Exact agreement is not guaranteed because the diagnostic replay uses cached images and an interpolated initial GT pose, whereas streaming inference initializes and assembles images from the bag.

## Small correction to the diagnostic sweep

In `4_diagnostics.py`, change:

```python
res.sort(key=lambda r: r['cov_rmse'])
```

to:

```python
res.sort(key=lambda r: r['all_rmse'])
```

Also change the printed ranking description to:

```python
print(
    f'\nRanked by OVERALL position RMSE. '
    f'Covered frames: {covered.mean():.1%}.'
)
```

Keep the covered/uncovered columns—they are still useful—but do not select the submission settings using only the covered column.

**This update targets the improvement your sweep actually demonstrated.** It does not yet solve the poorly mapped final segment, and I would not add speculative motion corrections before confirming this result. Send the evaluator output from `val_submission_tuned.csv` next.

