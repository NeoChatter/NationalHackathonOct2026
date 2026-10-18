import argparse
import math
import os
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from locnet import (IMG_H, IMG_W, OdomNet, PlaceNet, apply_aug, normalize_cams, sample_aug)
from track_map import build_track_map, load_markers_h5


def wrap_np(a):
    return np.arctan2(np.sin(a), np.cos(a))


def robust_std(x):
    x = np.asarray(x, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float('nan')


# ==============================================================================================
# datasets
# ==============================================================================================
class PlaceDataset(Dataset):
    """
    Returns (anchor, positive, hard_negative), each (4,H,W):
      positive : same place (<5 m, <15 deg) recorded in a DIFFERENT lap (>= 20 s apart)
      negative : 20-80 m away (along-track neighbour or other lap) -> teaches along-track discrimination
    Random batches almost never contain a positive, which collapsed the old descriptor.
    """

    def __init__(self, images, poses, timestamps_ns, indices, augment=True):
        self.images, self.idx, self.augment = images, np.asarray(indices), augment
        self.xy = poses[self.idx, :2]
        yaw = poses[self.idx, 3]
        t = timestamps_ns[self.idx] * 1e-9
        self.tree = cKDTree(self.xy)
        self.pos = {}
        for k in range(len(self.idx)):
            c = np.asarray(self.tree.query_ball_point(self.xy[k], 5.0), dtype=np.int64)
            if len(c) == 0:
                continue
            ok = (np.abs(t[c] - t[k]) >= 20.0) & (np.abs(wrap_np(yaw[c] - yaw[k])) <= math.radians(15.0))
            if ok.any():
                self.pos[k] = c[ok]
        self.anchors = np.array(sorted(self.pos.keys()), dtype=np.int64)
        print(f'[PLACE] {len(self.anchors)} anchors have a same-place frame from another lap')

    def __len__(self):
        return len(self.anchors)

    def _load(self, k, rng):
        img = self.images[self.idx[k]]
        x = apply_aug(img, sample_aug(rng), rng) if self.augment else img.astype(np.float32) / 255.0
        return normalize_cams(x)

    def __getitem__(self, n):
        rng = np.random.default_rng()
        k = int(self.anchors[n])
        p = int(rng.choice(self.pos[k]))
        cand = np.asarray(self.tree.query_ball_point(self.xy[k], 80.0), dtype=np.int64)
        d = np.hypot(*(self.xy[cand] - self.xy[k]).T)
        cand = cand[d > 20.0]
        q = int(rng.choice(cand)) if len(cand) else int(rng.integers(len(self.idx)))
        return torch.from_numpy(np.stack([self._load(k, rng), self._load(p, rng), self._load(q, rng)]))


class OdomDataset(Dataset):
    def __init__(self, images, poses, timestamps_ns, indices, augment=True):
        self.images, self.poses, self.ts, self.augment = images, poses, timestamps_ns, augment
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
        rng = np.random.default_rng()
        a, b = self.images[i - 1], self.images[i]
        if self.augment:
            prm = sample_aug(rng)                       # same photometric change for both frames
            a, b = apply_aug(a, prm, rng), apply_aug(b, prm, rng)
        else:
            a, b = a.astype(np.float32) / 255.0, b.astype(np.float32) / 255.0
        dt = float((self.ts[i] - self.ts[i - 1]) * 1e-9)
        x1, y1, _, yaw1 = self.poses[i - 1]
        x2, y2, _, yaw2 = self.poses[i]
        c, s = math.cos(yaw1), math.sin(yaw1)
        dxw, dyw = x2 - x1, y2 - y1
        delta = np.array([c * dxw + s * dyw, -s * dxw + c * dyw, math.atan2(math.sin(yaw2 - yaw1), math.cos(yaw2 - yaw1))],
                         dtype=np.float32)
        return (torch.from_numpy(normalize_cams(a)), torch.from_numpy(normalize_cams(b)),
                torch.tensor([dt], dtype=torch.float32), torch.from_numpy(delta))


# ==============================================================================================
# losses / evaluation
# ==============================================================================================
def info_nce(a, p, n, temp=0.07):
    """Positives on the diagonal; negatives: every other positive + every hard negative in the batch."""
    logits = torch.cat([a @ p.T, a @ n.T], dim=1) / temp
    return F.cross_entropy(logits, torch.arange(len(a), device=a.device))


@torch.no_grad()
def embed_images(model, images, indices, device, bs=128):
    model.eval()
    out = []
    for s in range(0, len(indices), bs):
        x = normalize_cams(images[indices[s:s + bs]].astype(np.float32) / 255.0)
        out.append(model(torch.from_numpy(x).to(device)).cpu())
    return torch.cat(out, 0)


def eval_retrieval(place, images, poses, db_idx, q_idx, device):
    """Global recall (harder than what the tracker does) + calibration of the GATED retrieval error."""
    db_e = embed_images(place, images, db_idx, device)
    q_e = embed_images(place, images, q_idx, device)
    sims = (q_e @ db_e.T).numpy()
    db_xy, db_yaw = poses[db_idx, :2], poses[db_idx, 3]
    q_xy, q_yaw = poses[q_idx, :2], poses[q_idx, 3]

    g = sims.argmax(1)
    err_g = np.hypot(*(db_xy[g] - q_xy).T)
    rep = dict(recall5=float(np.mean(err_g < 5) * 100), recall20=float(np.mean(err_g < 20) * 100),
               gross100=float(np.mean(err_g > 100) * 100), median=float(np.median(err_g)))
    far = sims[np.arange(len(q_idx)), g]
    rep['top1_sim_mean'] = float(far.mean())

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
        dyaw.append(wrap_np(db_yaw[j] - q_yaw[i]))
        top_sim.append(float(s.max()))
        err.append(float(np.hypot(*d)))
    err, top_sim = np.array(err), np.array(top_sim)
    calib = dict(retr_along_std=max(robust_std(along), 1.0), retr_lat_std=max(robust_std(lat), 1.0),
                 retr_yaw_std=max(robust_std(dyaw), math.radians(1.0)),
                 sim_thresh=float(np.percentile(top_sim[err < 5.0], 5)) if (err < 5.0).sum() > 20 else 0.5)
    rep['gated_median_err'] = float(np.median(err)) if len(err) else float('nan')
    return rep, calib


@torch.no_grad()
def eval_odom(odom, ds, device):
    odom.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    res, gt = [], []
    for a, b, dt, d in loader:
        pred = odom(a.to(device), b.to(device), dt.to(device)).cpu()
        res.append((pred - d).numpy())
        gt.append(d.numpy())
    res, gt = np.concatenate(res), np.concatenate(gt)
    moving = gt[:, 0] > 2.0
    rep = dict(dx_median_abs=float(np.median(np.abs(res[:, 0]))),
               dx_rel_bias_pct=float(np.median(res[moving, 0] / gt[moving, 0]) * 100) if moving.any() else float('nan'),
               dyaw_median_abs_deg=float(np.degrees(np.median(np.abs(res[:, 2])))))
    calib = dict(odom_dx_std=max(robust_std(res[:, 0]), 0.05), odom_dyaw_std=max(robust_std(res[:, 2]), 0.002))
    return rep, calib


# ==============================================================================================
# training
# ==============================================================================================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[TRAIN] device: {device}')

    with h5py.File(args.data_h5, 'r') as h5f:
        timestamps_ns = h5f['timestamps_ns'][:]
        images = h5f['images'][:]
        poses = h5f['poses'][:]
        track_borders = h5f['track_borders'][:] if 'track_borders' in h5f else np.zeros((0, 3))
        track_markers = load_markers_h5(h5f)          # per-marker borders (legacy cache -> one flat marker)

    n_total = len(timestamps_ns)
    split = n_total if args.full_train else int(0.80 * n_total)
    train_idx, val_idx = np.arange(0, split), np.arange(split, n_total)
    print(f'[SPLIT] total {n_total} | train {len(train_idx)} | val {len(val_idx)}')

    calib = {}
    if args.full_train and args.calib_ckpt:
        calib = torch.load(args.calib_ckpt, map_location='cpu', weights_only=False).get('calib', {})
        print(f'[CALIB] reusing calibration from {args.calib_ckpt}: {calib}')

    # ---------------------------------------------------------------- place network
    place = PlaceNet(256).to(device)
    ds = PlaceDataset(images, poses, timestamps_ns, train_idx, augment=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(place.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.place_epochs)
    for ep in range(1, args.place_epochs + 1):
        place.train()
        t0, tot = time.perf_counter(), 0.0
        for batch in loader:                                   # (B,3,4,H,W)
            B = batch.shape[0]
            emb = place(batch.reshape(B * 3, 4, IMG_H, IMG_W).to(device, non_blocking=True)).reshape(B, 3, -1)
            a, p, n = emb[:, 0], emb[:, 1], emb[:, 2]
            loss = 0.5 * (info_nce(a, p, n) + info_nce(p, a, n))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(place.parameters(), 2.0)
            opt.step()
            tot += loss.item()
        sched.step()
        msg = f'[PLACE {ep:02d}/{args.place_epochs}] loss {tot/len(loader):.4f} ({time.perf_counter()-t0:.1f}s)'
        if len(val_idx) and (ep % 5 == 0 or ep == args.place_epochs):
            rep, _ = eval_retrieval(place, images, poses, train_idx, val_idx, device)
            msg += f' | VAL recall@5m {rep["recall5"]:.1f}% @20m {rep["recall20"]:.1f}% gross>100m {rep["gross100"]:.1f}% ' \
                   f'top1-sim {rep["top1_sim_mean"]:.2f}'
        print(msg)

    # ---------------------------------------------------------------- odometry network
    odom = OdomNet().to(device)
    ods = OdomDataset(images, poses, timestamps_ns, train_idx, augment=True)
    oload = DataLoader(ods, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    oopt = torch.optim.AdamW(odom.parameters(), lr=args.lr, weight_decay=1e-4)
    osched = torch.optim.lr_scheduler.CosineAnnealingLR(oopt, T_max=args.odom_epochs)
    scale = odom.scale.to(device)
    for ep in range(1, args.odom_epochs + 1):
        odom.train()
        t0, tot = time.perf_counter(), 0.0
        for a, b, dt, d in oload:
            a, b, dt, d = a.to(device), b.to(device), dt.to(device), d.to(device)
            pred = odom(a, b, dt)
            loss = F.smooth_l1_loss(pred / scale, d / scale, beta=0.05)
            oopt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(odom.parameters(), 2.0)
            oopt.step()
            tot += loss.item()
        osched.step()
        msg = f'[ODOM {ep:02d}/{args.odom_epochs}] loss {tot/len(oload):.4f} ({time.perf_counter()-t0:.1f}s)'
        if len(val_idx) and (ep % 5 == 0 or ep == args.odom_epochs):
            rep, _ = eval_odom(odom, OdomDataset(images, poses, timestamps_ns, val_idx, augment=False), device)
            msg += f' | VAL |dx err| median {rep["dx_median_abs"]:.3f} m, dx bias {rep["dx_rel_bias_pct"]:+.1f}%, ' \
                   f'dyaw err {rep["dyaw_median_abs_deg"]:.2f} deg'
        print(msg)

    # ---------------------------------------------------------------- calibration + database
    if len(val_idx):
        _, c1 = eval_retrieval(place, images, poses, train_idx, val_idx, device)
        _, c2 = eval_odom(odom, OdomDataset(images, poses, timestamps_ns, val_idx, augment=False), device)
        calib = {**c1, **c2, 'lat_clamp': 7.0}
        print(f'[CALIB] measured on validation laps: {calib}')
    db_emb = embed_images(place, images, train_idx, device)
    db_poses = torch.from_numpy(poses[train_idx][:, [0, 1, 3]]).float()

    # track frame (centerline, real borders, heading prior) from the same training frames as the database
    track_map = None
    try:
        tm = build_track_map(poses[train_idx][:, [0, 1, 3]], track_markers,
                             fallback_half_width=float(calib.get('lat_clamp', 7.0)))
        track_map = tm.to_dict()
        print(f'[TRACK] {tm.summary()}')
    except Exception as e:                                     # never lose a trained model over the track map
        print(f'[TRACK][WARN] track map not built ({e}); inference falls back to lat_clamp')

    os.makedirs(os.path.dirname(args.checkpoint) or '.', exist_ok=True)
    torch.save({'place_state': place.state_dict(), 'odom_state': odom.state_dict(),
                'db_embeddings': db_emb, 'db_poses': db_poses, 'calib': calib,
                'track_borders': torch.from_numpy(track_borders).float(), 'track_map': track_map,
                'val_start_timestamp_ns': int(timestamps_ns[val_idx[0]]) if len(val_idx) else 0},
               args.checkpoint)
    print(f'[DONE] saved {args.checkpoint} ({len(db_emb)} database descriptors)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-h5', default='data/training_cache.h5')
    ap.add_argument('--checkpoint', default='checkpoints/racing_model.pt')
    ap.add_argument('--place-epochs', type=int, default=40)
    ap.add_argument('--odom-epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=48)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--full-train', action='store_true', help='train/DB on 100% of frames (final submission)')
    ap.add_argument('--calib-ckpt', default='', help='checkpoint from the 80/20 run whose calibration to reuse')
    train(ap.parse_args())
