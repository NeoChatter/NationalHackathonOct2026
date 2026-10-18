"""
STAGE 2 - train PlaceNet (place descriptor) + OdomNet (frame-to-frame motion) and write a checkpoint.

  # condition-robust model (generic weather / time-of-day / illumination robustness)
  python Scripts/2_train.py --data-h5 data/training_cache.h5 --checkpoint checkpoints/robust.pt \
         --clahe --norm robust --aug structured --seed 0
  # final submission model: all frames, reuse the calibration of the 80/20 run
  python Scripts/2_train.py ... --full-train --calib-ckpt checkpoints/robust_cal.pt

Split: first 80 % of frames train + database, last 20 % (~2 laps) validation (used for calibration and for
--val-only inference). The checkpoint contains both networks, the database (embeddings + poses of the training
frames), the calibration measured on the validation laps, the track map, the preprocessing (ck['preproc'], which
inference follows automatically) and the provenance chain. Next: 3_calibrate.py.
"""
import argparse
import json
import os
import random
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib import provenance
from lib.common import IMG_H, IMG_W
from lib.data import OdomDataset, PlaceDataset, embed_images, eval_odom, eval_retrieval
from lib.models import OdomNet, PlaceNet, preproc_cfg
from lib.track_map import build_track_map, load_markers_h5


def info_nce(a, p, n, temp=0.07):
    """Positives on the diagonal; negatives: every other positive + every hard negative in the batch."""
    logits = torch.cat([a @ p.T, a @ n.T], dim=1) / temp
    return F.cross_entropy(logits, torch.arange(len(a), device=a.device))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args, record):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[TRAIN] device: {device} | seed {args.seed}')

    with h5py.File(args.data_h5, 'r') as h5f:
        timestamps_ns = h5f['timestamps_ns'][:]
        images = h5f['images'][:]
        poses = h5f['poses'][:]
        track_borders = h5f['track_borders'][:] if 'track_borders' in h5f else np.zeros((0, 3))
        track_markers = load_markers_h5(h5f)          # per-marker borders (legacy cache -> one flat marker)
        chain = [json.loads(h5f.attrs['provenance'])] if 'provenance' in h5f.attrs else []

    n_total = len(timestamps_ns)
    split = n_total if args.full_train else int(0.80 * n_total)
    train_idx, val_idx = np.arange(0, split), np.arange(split, n_total)
    print(f'[SPLIT] total {n_total} | train {len(train_idx)} | val {len(val_idx)}')

    calib = {}
    if args.full_train and args.calib_ckpt:
        calib = torch.load(args.calib_ckpt, map_location='cpu', weights_only=False).get('calib', {})
        print(f'[CALIB] reusing calibration from {args.calib_ckpt}: {calib}')

    preproc = preproc_cfg({'clahe': bool(args.clahe), 'norm': args.norm})
    print(f'[PREPROC] {preproc} | augmentation: {args.aug}')
    gen = torch.Generator().manual_seed(args.seed)

    # ---------------------------------------------------------------- place network
    place = PlaceNet(256).to(device)
    ds = PlaceDataset(images, poses, timestamps_ns, train_idx, augment=True, preproc=preproc, aug_level=args.aug)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                        drop_last=True, generator=gen)
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
        msg = f'[PLACE {ep:02d}/{args.place_epochs}] loss {tot / len(loader):.4f} ({time.perf_counter() - t0:.1f}s)'
        if len(val_idx) and (ep % 5 == 0 or ep == args.place_epochs):
            rep, _ = eval_retrieval(place, images, poses, train_idx, val_idx, device, preproc)
            msg += f' | VAL recall@5m {rep["recall5"]:.1f}% @20m {rep["recall20"]:.1f}% gross>100m {rep["gross100"]:.1f}% ' \
                   f'top1-sim {rep["top1_sim_mean"]:.2f}'
        print(msg)

    # ---------------------------------------------------------------- odometry network
    odom = OdomNet().to(device)
    ods = OdomDataset(images, poses, timestamps_ns, train_idx, augment=True, preproc=preproc, aug_level=args.aug)
    oload = DataLoader(ods, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                       drop_last=True, generator=gen)
    oopt = torch.optim.AdamW(odom.parameters(), lr=args.lr, weight_decay=1e-4)
    osched = torch.optim.lr_scheduler.CosineAnnealingLR(oopt, T_max=args.odom_epochs)
    scale = odom.scale.to(device)
    val_ods = OdomDataset(images, poses, timestamps_ns, val_idx, augment=False, preproc=preproc) if len(val_idx) else None
    for ep in range(1, args.odom_epochs + 1):
        odom.train()
        t0, tot = time.perf_counter(), 0.0
        for a, b, dt, d in oload:
            a, b, dt, d = a.to(device), b.to(device), dt.to(device), d.to(device)
            loss = F.smooth_l1_loss(odom(a, b, dt) / scale, d / scale, beta=0.05)
            oopt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(odom.parameters(), 2.0)
            oopt.step()
            tot += loss.item()
        osched.step()
        msg = f'[ODOM {ep:02d}/{args.odom_epochs}] loss {tot / len(oload):.4f} ({time.perf_counter() - t0:.1f}s)'
        if val_ods is not None and (ep % 5 == 0 or ep == args.odom_epochs):
            rep, _ = eval_odom(odom, val_ods, device)
            msg += f' | VAL |dx err| median {rep["dx_median_abs"]:.3f} m, dx bias {rep["dx_rel_bias_pct"]:+.1f}%, ' \
                   f'dyaw err {rep["dyaw_median_abs_deg"]:.2f} deg'
        print(msg)

    # ---------------------------------------------------------------- calibration + database + track map
    if len(val_idx):
        _, c1 = eval_retrieval(place, images, poses, train_idx, val_idx, device, preproc)
        _, c2 = eval_odom(odom, val_ods, device)
        calib = {**c1, **c2, 'lat_clamp': 7.0}
        print(f'[CALIB] measured on validation laps: {calib}')
    db_emb = embed_images(place, images, train_idx, device, preproc=preproc)
    db_poses = torch.from_numpy(poses[train_idx][:, [0, 1, 3]]).float()

    track_map = None
    try:
        tm = build_track_map(poses[train_idx][:, [0, 1, 3]], track_markers,
                             fallback_half_width=float(calib.get('lat_clamp', 7.0)))
        track_map = tm.to_dict()
        print(f'[TRACK] {tm.summary()}')
    except Exception as e:                                     # never lose a trained model over the track map
        print(f'[TRACK][WARN] track map not built ({e}); run 3_calibrate.py or inference falls back to lat_clamp')

    ck = {'place_state': place.state_dict(), 'odom_state': odom.state_dict(),
          'db_embeddings': db_emb, 'db_poses': db_poses, 'calib': calib,
          'track_borders': torch.from_numpy(track_borders).float(), 'track_map': track_map,
          'preproc': preproc, 'aug_level': args.aug,
          'val_start_timestamp_ns': int(timestamps_ns[val_idx[0]]) if len(val_idx) else 0,
          'provenance': chain}
    provenance.append(ck, record)
    os.makedirs(os.path.dirname(args.checkpoint) or '.', exist_ok=True)
    torch.save(ck, args.checkpoint)
    print(f'[DONE] saved {args.checkpoint} ({len(db_emb)} database descriptors). Next: 3_calibrate.py')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-h5', default='data/training_cache.h5')
    ap.add_argument('--checkpoint', default='checkpoints/racing_model.pt')
    ap.add_argument('--place-epochs', type=int, default=40)
    ap.add_argument('--odom-epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=48)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--full-train', action='store_true', help='train/DB on 100%% of frames (final submission)')
    ap.add_argument('--calib-ckpt', default='', help='checkpoint from the 80/20 run whose calibration to reuse')
    ap.add_argument('--clahe', action='store_true', help='CLAHE on every image (stored in the checkpoint)')
    ap.add_argument('--norm', choices=['std', 'robust'], default='std',
                    help='per-camera normalisation; robust = median/MAD (a bright sky band does not dominate)')
    ap.add_argument('--aug', choices=['basic', 'structured'], default='basic',
                    help='structured = lighting / shadow / glare / haze / sky / contrast / erase-top / JPEG')
    a = ap.parse_args()
    rec = provenance.run_record('2_train', a, inputs={'data_h5': a.data_h5, 'calib_ckpt': a.calib_ckpt})
    print(provenance.banner(rec))
    train(a, rec)
