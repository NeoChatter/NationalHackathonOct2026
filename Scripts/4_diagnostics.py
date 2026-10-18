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
    res.sort(key=lambda r: r['all_rmse'])
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
    print(
            f'\nRanked by OVERALL position RMSE. '
            f'Covered frames: {covered.mean():.1%}.'
        )
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