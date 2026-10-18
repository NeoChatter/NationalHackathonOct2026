"""
Training data, held-out metrics and the measurements used for calibration (stages 2 and 3).

  RacingDataset, LocationPairBatchSampler, gt_body_deltas, evaluate_heldout     rmse-2.740 branch (2_train_model.py)
  embed_images, eval_retrieval, odom_bias_curve, robust_std                     rmse-4.363 branch (lib/data.py)

Every image goes through lib/models.preprocess with the checkpoint's preprocessing (CLAHE / normalisation), after
augmentation, so training, calibration and inference see identical network inputs.
"""
import math
import random
from collections import defaultdict

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset, Sampler

from .augment import PhotometricAugment, apply_structured, sample_rng, sample_structured
from .common import wrap
from .models import preprocess


def robust_std(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float('nan')


def gt_body_deltas(poses: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Ground-truth body-frame motion (dx, dy, dyaw) from frame idx-1 to frame idx. poses: [x, y, z, yaw]."""
    idx = np.asarray(idx, dtype=np.int64)
    x1, y1, yaw1 = poses[idx - 1, 0], poses[idx - 1, 1], poses[idx - 1, 3]
    x2, y2, yaw2 = poses[idx, 0], poses[idx, 1], poses[idx, 3]
    dxw, dyw = x2 - x1, y2 - y1
    c, s = np.cos(yaw1), np.sin(yaw1)
    return np.stack([c * dxw + s * dyw, -s * dxw + c * dyw, wrap(yaw2 - yaw1)], axis=1).astype(np.float32)


def valid_pairs(indices, timestamps_ns, dt_lo=0.01, dt_hi=0.5) -> np.ndarray:
    """Frames i of `indices` whose predecessor i-1 is in the SAME split and dt(i-1, i) is in [dt_lo, dt_hi] s."""
    indices = np.asarray(indices, dtype=np.int64)
    in_split = np.zeros(len(timestamps_ns), dtype=bool)
    in_split[indices] = True
    i = indices[indices > 0]
    dt = (timestamps_ns[i] - timestamps_ns[i - 1]) * 1e-9
    return i[in_split[i - 1] & (dt >= dt_lo) & (dt <= dt_hi)]


# ==============================================================================================
# training data (V, with S's structured layer and preprocessing)
# ==============================================================================================
class RacingDataset(Dataset):
    """
    Consecutive multi-camera pairs (t-1, t) with GT body-frame motion and global pose labels.

    still_prob   : probability that a sample becomes a synthetic STANDSTILL pair (prev == curr, zero motion), so the
                   VO head learns that "no image change" means "no motion".
    noise_std    : max std of Gaussian pixel noise added independently to both frames (normalised units).
    photo_aug    : PhotometricAugment or None; structured_p: probability of the structured layer (lib/augment.py).
    preproc      : the checkpoint's preprocessing (CLAHE / normalisation).
    """

    def __init__(self, images, dt_offsets, poses, timestamps_ns, indices, still_prob=0.0, noise_std=0.0,
                 photo_aug=None, structured_p=0.0, preproc=None):
        self.images, self.dt_offsets, self.poses, self.timestamps_ns = images, dt_offsets, poses, timestamps_ns
        self.still_prob, self.noise_std = float(still_prob), float(noise_std)
        self.photo_aug, self.structured_p, self.preproc = photo_aug, float(structured_p), preproc
        # a held-out block must not leak into training pairs: i-1 must be in the same split
        self.valid_indices = valid_pairs(indices, timestamps_ns, 0.01, 0.5)
        self.delta_targets = (gt_body_deltas(poses, self.valid_indices) if len(self.valid_indices)
                              else np.zeros((0, 3), np.float32))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, item):
        idx = int(self.valid_indices[item])
        img_curr = torch.from_numpy(self.images[idx]).float().div_(255.0)        # (4, H, W) in [0, 1]
        dt = float((self.timestamps_ns[idx] - self.timestamps_ns[idx - 1]) * 1e-9)
        time_feat = np.concatenate([[dt], self.dt_offsets[idx]]).astype(np.float32)   # (5,)

        if self.still_prob > 0.0 and random.random() < self.still_prob:
            img_prev = img_curr.clone()
            delta_target = np.zeros(3, dtype=np.float32)
        else:
            img_prev = torch.from_numpy(self.images[idx - 1]).float().div_(255.0)
            delta_target = self.delta_targets[item]

        # appearance augmentation BEFORE normalisation (which would cancel part of it), same draw for both frames
        if self.photo_aug is not None:
            prm = self.photo_aug.sample()
            if prm is not None:
                img_prev = self.photo_aug.apply(img_prev, prm)
                img_curr = self.photo_aug.apply(img_curr, prm)
        prev_np, curr_np = img_prev.numpy(), img_curr.numpy()
        if self.structured_p > 0.0 and random.random() < self.structured_p:
            sp = sample_structured(sample_rng(), prev_np.shape[0])
            prev_np, curr_np = apply_structured(prev_np, sp), apply_structured(curr_np, sp)
        img_prev = torch.from_numpy(preprocess(prev_np, self.preproc))
        img_curr = torch.from_numpy(preprocess(curr_np, self.preproc))
        if self.noise_std > 0.0:
            sigma = random.uniform(0.0, self.noise_std)
            img_prev = img_prev + torch.randn_like(img_prev) * sigma
            img_curr = img_curr + torch.randn_like(img_curr) * sigma

        x2, y2, _, yaw2 = self.poses[idx]
        pose_target = np.array([x2, y2, math.sin(yaw2), math.cos(yaw2), yaw2], dtype=np.float32)
        return (img_prev, img_curr, torch.from_numpy(time_feat),
                torch.from_numpy(np.asarray(delta_target, dtype=np.float32)), torch.from_numpy(pose_target))


class LocationPairBatchSampler(Sampler):
    """
    Every batch = B/2 random anchors + one partner per anchor taken from the SAME PLACE on a DIFFERENT pass
    (within `pos_radius` metres, `pos_yaw_deg` degrees, at least `min_gap_s` seconds apart). Anchors without any
    partner (e.g. a pit lane driven only once) get a random partner instead.
    max_batches > 0 caps the batches per epoch (smoke tests).
    """

    def __init__(self, xy, yaw, t_s, batch_size, pos_radius=3.0, pos_yaw_deg=10.0, min_gap_s=15.0, seed=0,
                 max_batches=0):
        assert batch_size % 2 == 0, 'batch size must be even for paired sampling'
        self.n = len(xy)
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.seed = seed
        self.max_batches = int(max_batches)
        self._epoch = 0

        keys = np.floor(xy / float(pos_radius)).astype(np.int64)
        grid = defaultdict(list)
        for i, (kx, ky) in enumerate(keys):
            grid[(int(kx), int(ky))].append(i)
        yaw_tol = math.radians(pos_yaw_deg)
        self.positives = []
        for i in range(self.n):
            kx, ky = int(keys[i, 0]), int(keys[i, 1])
            cand = []
            for ax in (-1, 0, 1):
                for ay in (-1, 0, 1):
                    cand.extend(grid.get((kx + ax, ky + ay), ()))
            cand = np.asarray(cand, dtype=np.int64)
            dist = np.hypot(xy[cand, 0] - xy[i, 0], xy[cand, 1] - xy[i, 1])
            dyaw = np.abs(wrap(yaw[cand] - yaw[i]))
            gap = np.abs(t_s[cand] - t_s[i])
            self.positives.append(cand[(dist <= pos_radius) & (dyaw <= yaw_tol) & (gap >= min_gap_s)])
        n_with = sum(1 for p in self.positives if len(p) > 0)
        print(f'[SAMPLER] {n_with}/{self.n} training items ({100.0 * n_with / max(self.n, 1):.1f}%) have a '
              f'same-place partner within {pos_radius:g} m / {pos_yaw_deg:g} deg on another pass')

    def __len__(self):
        n = self.n // self.batch_size
        return min(n, self.max_batches) if self.max_batches > 0 else n

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        order = rng.permutation(self.n)
        for b in range(len(self)):
            anchors = order[b * self.half:(b + 1) * self.half]
            partners = []
            for i in anchors:
                pos = self.positives[int(i)]
                partners.append(int(pos[rng.integers(len(pos))]) if len(pos) else int(rng.integers(self.n)))
            yield [int(i) for i in anchors] + partners


# ==============================================================================================
# embedding / prediction helpers
# ==============================================================================================
@torch.no_grad()
def embed_images(model, images, indices, device, bs=128, preproc=None) -> torch.Tensor:
    model.eval()
    out = []
    for s in range(0, len(indices), bs):
        x = torch.from_numpy(preprocess(images[indices[s:s + bs]], preproc)).to(device)
        out.append(model.forward_single(x)[1].float().cpu())
    return torch.cat(out, 0) if out else torch.zeros((0, 256))


@torch.no_grad()
def predict_vo(model, images, dt_offsets, timestamps_ns, pair_idx, device, preproc=None, bs=32) -> np.ndarray:
    """Learned-VO body-frame motion (dx, dy, dyaw) for the pairs (i-1, i), i in pair_idx."""
    model.eval()
    preds = []
    for s in range(0, len(pair_idx), bs):
        b = pair_idx[s:s + bs]
        dt_b = (timestamps_ns[b] - timestamps_ns[b - 1]) * 1e-9
        tf = np.concatenate([dt_b[:, None], dt_offsets[b]], axis=1).astype(np.float32)
        _, _, d, _ = model(torch.from_numpy(preprocess(images[b - 1], preproc)).to(device),
                           torch.from_numpy(preprocess(images[b], preproc)).to(device),
                           torch.from_numpy(tf).to(device))
        preds.append(d.float().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float64) if preds else np.zeros((0, 3))


# ==============================================================================================
# held-out metrics printed every epoch and written to <checkpoint>_history.csv (V)
# ==============================================================================================
EVAL_KEYS = [
    'val_covered_frac', 'val_ret_median_m', 'val_ret_within2', 'val_ret_within5', 'val_sim_covered',
    'val_sim_uncovered', 'val_ret_median_aug_m', 'val_ret_within5_aug',
    'val_vo_rmse_dx', 'val_vo_rmse_dy', 'val_vo_rmse_dyaw_deg', 'val_vo_scale',
    'val_vo_scale_lt10', 'val_vo_scale_10_20', 'val_vo_scale_20_30', 'val_vo_scale_gt30', 'val_standstill_speed',
]
SPEED_BINS = [('lt10', 0.0, 10.0), ('10_20', 10.0, 20.0), ('20_30', 20.0, 30.0), ('gt30', 30.0, 1e9)]


@torch.no_grad()
def evaluate_heldout(model, images, dt_offsets, poses, timestamps_ns, train_indices, val_indices, device,
                     preproc=None, ref_stride=2) -> dict:
    """Retrieval of held-out frames against the training map, a synthetic-lighting probe and VO accuracy by speed.
    Returns a dict with every key in EVAL_KEYS (NaN when not measurable)."""
    model.eval()
    metrics = {k: float('nan') for k in EVAL_KEYS}

    ref_idx = train_indices[::ref_stride]
    ref_emb = embed_images(model, images, ref_idx, device, preproc=preproc)
    ref_xy = poses[ref_idx, :2]
    val_emb = embed_images(model, images, val_indices, device, preproc=preproc)
    top_sim, top_j = (val_emb @ ref_emb.T).max(dim=1)
    top_sim, top_j = top_sim.numpy(), top_j.numpy()
    val_xy = poses[val_indices, :2]
    err = np.hypot(*(ref_xy[top_j] - val_xy).T)
    d_map, _ = cKDTree(poses[train_indices, :2]).query(val_xy)
    covered = d_map <= 5.0
    metrics['val_covered_frac'] = float(covered.mean())
    if covered.any():
        metrics.update(val_ret_median_m=float(np.median(err[covered])), val_ret_within2=float(np.mean(err[covered] <= 2)),
                       val_ret_within5=float(np.mean(err[covered] <= 5)), val_sim_covered=float(np.median(top_sim[covered])))
        if (~covered).any():
            metrics['val_sim_uncovered'] = float(np.median(top_sim[~covered]))
        print(f'    [VAL-RET] map covers {covered.mean():.1%} of val frames | on covered: median err '
              f'{metrics["val_ret_median_m"]:.2f} m, within 2 m {metrics["val_ret_within2"]:.1%}, '
              f'within 5 m {metrics["val_ret_within5"]:.1%} | top-1 sim median covered {metrics["val_sim_covered"]:.3f}'
              + (f' vs uncovered {metrics["val_sim_uncovered"]:.3f}' if (~covered).any() else ''))

    # lighting-shift probe: the SAME held-out frames with synthetic dusk-like lighting against the unchanged night map
    sub = np.arange(0, len(val_indices), 3)
    if len(sub) and covered[sub].any():
        aug = PhotometricAugment(strength=1.0, p=1.0)
        py_state, torch_state = random.getstate(), torch.get_rng_state()
        random.seed(1234)
        torch.manual_seed(1234)
        x_aug = np.stack([preprocess(aug.apply(torch.from_numpy(images[val_indices[i]]).float().div_(255.0),
                                               aug.sample()).numpy(), preproc) for i in sub])
        random.setstate(py_state)
        torch.set_rng_state(torch_state)
        emb_aug = torch.cat([model.forward_single(torch.from_numpy(x_aug[s:s + 128]).to(device))[1].float().cpu()
                             for s in range(0, len(x_aug), 128)], dim=0)
        j_aug = (emb_aug @ ref_emb.T).argmax(dim=1).numpy()
        err_aug = np.hypot(*(ref_xy[j_aug] - val_xy[sub]).T)
        cov_sub = covered[sub]
        metrics['val_ret_median_aug_m'] = float(np.median(err_aug[cov_sub]))
        metrics['val_ret_within5_aug'] = float(np.mean(err_aug[cov_sub] <= 5))
        print(f'    [VAL-RET-AUG] with synthetic dusk-like lighting on the query: median err '
              f'{metrics["val_ret_median_aug_m"]:.2f} m, within 5 m {metrics["val_ret_within5_aug"]:.1%}')

    pair_idx = valid_pairs(val_indices, timestamps_ns, 0.01, 0.5)
    if len(pair_idx) == 0:
        model.train()
        return metrics
    pred = predict_vo(model, images, dt_offsets, timestamps_ns, pair_idx, device, preproc)
    gt = gt_body_deltas(poses, pair_idx).astype(np.float64)
    dt = (timestamps_ns[pair_idx] - timestamps_ns[pair_idx - 1]) * 1e-9
    speed = np.abs(gt[:, 0]) / dt
    e = pred - gt
    metrics.update(val_vo_rmse_dx=float(np.sqrt((e[:, 0] ** 2).mean())), val_vo_rmse_dy=float(np.sqrt((e[:, 1] ** 2).mean())),
                   val_vo_rmse_dyaw_deg=float(math.degrees(np.sqrt((e[:, 2] ** 2).mean()))),
                   val_vo_scale=float(pred[:, 0].sum() / max(gt[:, 0].sum(), 1e-9)))
    print(f'    [VAL-VO] step RMSE dx {metrics["val_vo_rmse_dx"]:.3f} m | dy {metrics["val_vo_rmse_dy"]:.3f} m | '
          f'dyaw {metrics["val_vo_rmse_dyaw_deg"]:.3f} deg | forward scale {metrics["val_vo_scale"]:.3f}')
    parts = []
    for name, lo, hi in SPEED_BINS:
        m = (speed >= lo) & (speed < hi)
        if m.sum() > 5 and gt[m, 0].sum() > 0:
            metrics[f'val_vo_scale_{name}'] = float(pred[m, 0].sum() / gt[m, 0].sum())
            parts.append(f'{lo:.0f}-{hi:.0f}: {metrics[f"val_vo_scale_{name}"]:.2f} (n={int(m.sum())})')
    if parts:
        print('    [VAL-VO] forward scale by GT speed [m/s] -> ' + ' | '.join(parts))
    still = speed < 0.3
    if still.sum() > 3:
        metrics['val_standstill_speed'] = float(np.mean(np.abs(pred[still, 0]) / dt[still]))
    model.train()
    return metrics


# ==============================================================================================
# calibration measurements (S, adapted to RacingLocNet and to a database with time exclusion)
# ==============================================================================================
def eval_retrieval(model, images, poses, timestamps_ns, db_idx, q_idx, device, preproc=None, exclude_s=0.0,
                   gate_m=35.0, db_emb=None):
    """
    Global recall + calibration of the GATED retrieval error (what the tracker sees: candidates within gate_m of the
    true position). exclude_s > 0 removes every database frame within +-exclude_s of the query's own time, so
    training frames can be used as queries ("another pass of the same place") without touching validation laps.
    Returns (report, calib) with calib = retr_along_std / retr_lat_std / retr_yaw_std / sim_thresh.
    """
    db_e = db_emb.float() if db_emb is not None else embed_images(model, images, db_idx, device, preproc=preproc)
    q_e = embed_images(model, images, q_idx, device, preproc=preproc)
    sims = (q_e @ db_e.T).numpy().astype(np.float32)
    if exclude_s > 0:
        tq, td = timestamps_ns[q_idx] * 1e-9, timestamps_ns[db_idx] * 1e-9
        for s in range(0, len(q_idx), 256):
            blk = sims[s:s + 256]
            blk[np.abs(td[None, :] - tq[s:s + 256, None]) < exclude_s] = -np.inf
    db_xy, db_yaw = poses[db_idx, :2], poses[db_idx, 3]
    q_xy, q_yaw = poses[q_idx, :2], poses[q_idx, 3]

    g = sims.argmax(1)
    top = sims[np.arange(len(q_idx)), g]
    ok = np.isfinite(top)
    err_g = np.hypot(*(db_xy[g] - q_xy).T)[ok]
    rep = dict(n_queries=int(ok.sum()), n_db=int(len(db_idx)), exclude_s=float(exclude_s),
               recall5=float(np.mean(err_g < 5) * 100) if len(err_g) else float('nan'),
               recall20=float(np.mean(err_g < 20) * 100) if len(err_g) else float('nan'),
               gross100=float(np.mean(err_g > 100) * 100) if len(err_g) else float('nan'),
               median=float(np.median(err_g)) if len(err_g) else float('nan'),
               top1_sim_mean=float(top[ok].mean()) if ok.any() else float('nan'))

    tree = cKDTree(db_xy)
    along, lat, dyaw, top_sim, err = [], [], [], [], []
    for i in range(len(q_idx)):
        c = np.asarray(tree.query_ball_point(q_xy[i], gate_m), dtype=np.int64)
        if len(c):
            c = c[np.isfinite(sims[i, c])]
        if len(c) < 5:
            continue
        s = sims[i, c]
        j = c[int(s.argmax())]
        d = db_xy[j] - q_xy[i]
        cy, sy = math.cos(q_yaw[i]), math.sin(q_yaw[i])
        along.append(cy * d[0] + sy * d[1])
        lat.append(-sy * d[0] + cy * d[1])
        dyaw.append(float(wrap(db_yaw[j] - q_yaw[i])))
        top_sim.append(float(s.max()))
        err.append(float(np.hypot(*d)))
    err, top_sim = np.array(err), np.array(top_sim)
    calib = dict(retr_along_std=max(robust_std(along), 1.0) if along else 6.0,
                 retr_lat_std=max(robust_std(lat), 1.0) if lat else 2.5,
                 retr_yaw_std=max(robust_std(dyaw), math.radians(1.0)) if dyaw else 0.08,
                 sim_thresh=float(np.percentile(top_sim[err < 5.0], 5)) if (err < 5.0).sum() > 20 else 0.5)
    rep['gated_median_err'] = float(np.median(err)) if len(err) else float('nan')
    rep['gated_n'] = int(len(err))
    return rep, calib


def odom_bias_curve(p: np.ndarray, g: np.ndarray, n_bins: int = 8):
    """
    Speed-dependent forward-distance bias of the visual odometry (regression to the mean: over-reads slow, under-reads
    fast). p, g: predicted / true forward step per frame pair. Returns (xs, ys, residual_std, table); the tracker maps
    a predicted step through np.interp(dx, xs, ys).
    """
    mask = g > 0.2
    if mask.sum() < 20:
        return [], [], 0.30, dict(table=[], raw_median_abs=float('nan'), raw_bias_pct=float('nan'),
                                  cal_median_abs=float('nan'))
    edges = np.quantile(p[mask], np.linspace(0.0, 1.0, n_bins + 1))
    xs, ys, table = [0.0], [0.0], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = mask & (p >= lo) & (p <= hi)
        if m.sum() < 20:
            continue
        px, gy = float(np.median(p[m])), float(np.median(g[m]))
        if px <= xs[-1]:
            continue
        xs.append(px)
        ys.append(gy)
        table.append((px, px / 0.1, gy, gy / px))
    ys = np.maximum.accumulate(ys).tolist()
    resid = np.interp(p, xs, ys) - g
    std = max(robust_std(resid), 0.05)
    return xs, ys, std, dict(table=table, raw_median_abs=float(np.median(np.abs(p - g))),
                             raw_bias_pct=float(np.median((p - g)[mask] / g[mask]) * 100),
                             cal_median_abs=float(np.median(np.abs(resid))))
