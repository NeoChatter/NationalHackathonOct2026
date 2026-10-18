"""
Measure the odometry network's speed-dependent forward-distance bias on the HELD-OUT laps and store a correction
curve in the checkpoint (calib['dx_map_pred'] / calib['dx_map_true']). No retraining needed.

  python calibrate_odom.py --data-h5 data/training_cache.h5 --checkpoint checkpoints/racing_model.pt \
         --output checkpoints/racing_model_cal.pt

Then use checkpoints/racing_model_cal.pt for --val-only inference. For the final model, train with
--full-train --calib-ckpt checkpoints/racing_model_cal.pt : the curve is carried over automatically.
Requires a checkpoint trained WITHOUT --full-train (it needs the validation split).
"""
import argparse
import importlib.machinery
import importlib.util
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from locnet import OdomNet


def load_train_module(here):
    path = os.path.join(here, '2_train_model.py')
    loader = importlib.machinery.SourceFileLoader('train_mod', path)
    spec = importlib.util.spec_from_loader('train_mod', loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)          # the training run is guarded by __main__, so this only imports the classes
    return mod


def main(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    here = os.path.dirname(os.path.abspath(__file__))
    train_mod = load_train_module(here)

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    vs = int(ck.get('val_start_timestamp_ns', 0))
    if vs <= 0:
        raise SystemExit('This checkpoint has no validation split (was it trained with --full-train?).')
    odom = OdomNet().to(dev)
    odom.load_state_dict(ck['odom_state'])
    odom.eval()

    with h5py.File(args.data_h5, 'r') as f:
        ts = f['timestamps_ns'][:]
        images = f['images'][:]
        poses = f['poses'][:]
    val_idx = np.flatnonzero(ts >= vs)
    ds = train_mod.OdomDataset(images, poses, ts, val_idx, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    P, G = [], []
    with torch.no_grad():
        for a, b, dt, d in loader:
            out = odom(a.to(dev), b.to(dev), dt.to(dev)).cpu().numpy()
            P.append(out[:, 0])
            G.append(d.numpy()[:, 0])
    p, g = np.concatenate(P), np.concatenate(G)
    print(f'[CAL] {len(p)} held-out frame pairs')

    mask = g > 0.2
    edges = np.quantile(p[mask], np.linspace(0.0, 1.0, 9))     # 8 bins of equal population
    xs, ys = [0.0], [0.0]
    print('  predicted dx   ~speed    true dx   true/pred')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = mask & (p >= lo) & (p <= hi)
        if m.sum() < 20:
            continue
        px, gy = float(np.median(p[m])), float(np.median(g[m]))
        if px <= xs[-1]:
            continue
        xs.append(px)
        ys.append(gy)
        print(f'  {px:8.2f} m  {px/0.1:6.1f} m/s  {gy:8.2f} m  {gy/px:8.2f}')
    ys = np.maximum.accumulate(ys).tolist()

    resid = np.interp(p, xs, ys) - g
    raw = p - g
    mad = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))
    print(f'[CAL] forward-step error: raw median |err| {np.median(np.abs(raw)):.3f} m, bias {np.median(raw[mask]/g[mask])*100-100:+.1f}% '
          f'-> calibrated median |err| {np.median(np.abs(resid)):.3f} m')

    calib = dict(ck.get('calib', {}))
    calib['dx_map_pred'], calib['dx_map_true'] = xs, ys
    calib['odom_dx_std'] = max(mad(resid), 0.05)
    ck['calib'] = calib
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(ck, args.output)
    print(f'[DONE] wrote {args.output}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-h5', default='data/training_cache.h5')
    ap.add_argument('--checkpoint', default='checkpoints/racing_model.pt')
    ap.add_argument('--output', default='checkpoints/racing_model_cal.pt')
    main(ap.parse_args())
