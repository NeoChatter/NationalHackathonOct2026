"""
Training datasets and the measurements used for calibration (shared by 2_train.py and 3_calibrate.py).

Augmentation randomness is seeded from torch's per-worker seed, so a run with the same --seed is reproducible.
"""
import math

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from .common import wrap
from .models import apply_aug, preprocess, sample_aug


def robust_std(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float('nan')


def _rng(n: int):
    return np.random.default_rng([torch.initial_seed() % (2 ** 63), int(n)])


# ==============================================================================================
# datasets
# ==============================================================================================
class PlaceDataset(Dataset):
    """
    Returns (anchor, positive, hard_negative), each (4,H,W):
      positive : same place (<5 m, <15 deg) recorded in a DIFFERENT lap (>= 20 s apart)
      negative : 20-80 m away (along-track neighbour or other lap) -> teaches along-track discrimination
    Anchor and positive get INDEPENDENT augmentations, so the contrastive loss itself teaches
    "same place under different light = same descriptor".
    """

    def __init__(self, images, poses, timestamps_ns, indices, augment=True, preproc=None, aug_level='basic'):
        self.images, self.idx, self.augment = images, np.asarray(indices), augment
        self.preproc, self.aug_level = preproc or {}, aug_level
        self.xy = poses[self.idx, :2]
        yaw = poses[self.idx, 3]
        t = timestamps_ns[self.idx] * 1e-9
        self.tree = cKDTree(self.xy)
        self.pos = {}
        for k in range(len(self.idx)):
            c = np.asarray(self.tree.query_ball_point(self.xy[k], 5.0), dtype=np.int64)
            if len(c) == 0:
                continue
            ok = (np.abs(t[c] - t[k]) >= 20.0) & (np.abs(wrap(yaw[c] - yaw[k])) <= math.radians(15.0))
            if ok.any():
                self.pos[k] = c[ok]
        self.anchors = np.array(sorted(self.pos.keys()), dtype=np.int64)
        print(f'[PLACE] {len(self.anchors)} anchors have a same-place frame from another lap')

    def __len__(self):
        return len(self.anchors)

    def _load(self, k, rng):
        img = self.images[self.idx[k]]
        x = apply_aug(img, sample_aug(rng, self.aug_level), rng) if self.augment else img
        return preprocess(x, self.preproc)

    def __getitem__(self, n):
        rng = _rng(n)
        k = int(self.anchors[n])
        p = int(rng.choice(self.pos[k]))
        cand = np.asarray(self.tree.query_ball_point(self.xy[k], 80.0), dtype=np.int64)
        d = np.hypot(*(self.xy[cand] - self.xy[k]).T)
        cand = cand[d > 20.0]
        q = int(rng.choice(cand)) if len(cand) else int(rng.integers(len(self.idx)))
        return torch.from_numpy(np.stack([self._load(k, rng), self._load(p, rng), self._load(q, rng)]))


class OdomDataset(Dataset):
    """Consecutive frame pairs (0.05 s <= dt <= 0.2 s) with the GT body-frame motion (dx, dy, dyaw)."""

    def __init__(self, images, poses, timestamps_ns, indices, augment=True, preproc=None, aug_level='basic'):
        self.images, self.poses, self.ts, self.augment = images, poses, timestamps_ns, augment
        self.preproc, self.aug_level = preproc or {}, aug_level
        lo = int(indices[0])
        valid = []
        for i in indices:
            if i - 1 >= lo:
                dt = (timestamps_ns[i] - timestamps_ns[i - 1]) * 1e-9
                if 0.05 <= dt <= 0.2:
                    valid.append(i)
        self.valid = np.asarray(valid, dtype=np.int64)

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, n):
        i = int(self.valid[n])
        a, b = self.images[i - 1], self.images[i]
        if self.augment:
            rng = _rng(n)
            prm = sample_aug(rng, self.aug_level)            # same photometric change for both frames
            a, b = apply_aug(a, prm, rng), apply_aug(b, prm, rng)
        dt = float((self.ts[i] - self.ts[i - 1]) * 1e-9)
        x1, y1, _, yaw1 = self.poses[i - 1]
        x2, y2, _, yaw2 = self.poses[i]
        c, s = math.cos(yaw1), math.sin(yaw1)
        dxw, dyw = x2 - x1, y2 - y1
        delta = np.array([c * dxw + s * dyw, -s * dxw + c * dyw, math.atan2(math.sin(yaw2 - yaw1), math.cos(yaw2 - yaw1))],
                         dtype=np.float32)
        return (torch.from_numpy(preprocess(a, self.preproc)), torch.from_numpy(preprocess(b, self.preproc)),
                torch.tensor([dt], dtype=torch.float32), torch.from_numpy(delta))


# ==============================================================================================
# embedding / evaluation used for calibration
# ==============================================================================================
@torch.no_grad()
def embed_images(model, images, indices, device, bs=128, preproc=None):
    model.eval()
    out = []
    for s in range(0, len(indices), bs):
        x = preprocess(images[indices[s:s + bs]], preproc)
        out.append(model(torch.from_numpy(x).to(device)).cpu())
    return torch.cat(out, 0)


def eval_retrieval(place, images, poses, db_idx, q_idx, device, preproc=None):
    """Global recall (harder than what the tracker does) + calibration of the GATED retrieval error."""
    db_e = embed_images(place, images, db_idx, device, preproc=preproc)
    q_e = embed_images(place, images, q_idx, device, preproc=preproc)
    sims = (q_e @ db_e.T).numpy()
    db_xy, db_yaw = poses[db_idx, :2], poses[db_idx, 3]
    q_xy, q_yaw = poses[q_idx, :2], poses[q_idx, 3]

    g = sims.argmax(1)
    err_g = np.hypot(*(db_xy[g] - q_xy).T)
    rep = dict(recall5=float(np.mean(err_g < 5) * 100), recall20=float(np.mean(err_g < 20) * 100),
               gross100=float(np.mean(err_g > 100) * 100), median=float(np.median(err_g)))
    rep['top1_sim_mean'] = float(sims[np.arange(len(q_idx)), g].mean())

    tree = cKDTree(db_xy)
    along, lat, dyaw, top_sim, err = [], [], [], [], []
    for i in range(len(q_idx)):
        c = np.asarray(tree.query_ball_point(q_xy[i], 35.0), dtype=np.int64)
        if len(c) < 5:
            continue
        s = sims[i, c]
        j = c[int(s.argmax())]
        d = db_xy[j] - q_xy[i]
        cy, sy = math.cos(q_yaw[i]), math.sin(q_yaw[i])
        along.append(cy * d[0] + sy * d[1])
        lat.append(-sy * d[0] + cy * d[1])
        dyaw.append(wrap(db_yaw[j] - q_yaw[i]))
        top_sim.append(float(s.max()))
        err.append(float(np.hypot(*d)))
    err, top_sim = np.array(err), np.array(top_sim)
    calib = dict(retr_along_std=max(robust_std(along), 1.0), retr_lat_std=max(robust_std(lat), 1.0),
                 retr_yaw_std=max(robust_std(dyaw), math.radians(1.0)),
                 sim_thresh=float(np.percentile(top_sim[err < 5.0], 5)) if (err < 5.0).sum() > 20 else 0.5)
    rep['gated_median_err'] = float(np.median(err)) if len(err) else float('nan')
    return rep, calib


@torch.no_grad()
def predict_odom(odom, ds, device, bs=64):
    """(predicted, true) body-frame motion for every pair of an OdomDataset."""
    odom.eval()
    P, G = [], []
    for a, b, dt, d in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2):
        P.append(odom(a.to(device), b.to(device), dt.to(device)).cpu().numpy())
        G.append(d.numpy())
    return np.concatenate(P), np.concatenate(G)


def eval_odom(odom, ds, device):
    pred, gt = predict_odom(odom, ds, device)
    res = pred - gt
    moving = gt[:, 0] > 2.0
    rep = dict(dx_median_abs=float(np.median(np.abs(res[:, 0]))),
               dx_rel_bias_pct=float(np.median(res[moving, 0] / gt[moving, 0]) * 100) if moving.any() else float('nan'),
               dyaw_median_abs_deg=float(np.degrees(np.median(np.abs(res[:, 2])))))
    calib = dict(odom_dx_std=max(robust_std(res[:, 0]), 0.05), odom_dyaw_std=max(robust_std(res[:, 2]), 0.002))
    return rep, calib


def odom_bias_curve(p: np.ndarray, g: np.ndarray, n_bins: int = 8):
    """
    Speed-dependent forward-distance bias of the odometry (regression to the mean: over-reads slow, under-reads fast).
    p, g: predicted / true forward step per frame pair. Returns (xs, ys, residual_std, table) where the tracker
    maps a predicted step through np.interp(dx, xs, ys).
    """
    mask = g > 0.2
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
