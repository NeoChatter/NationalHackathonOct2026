#!/usr/bin/env python3
"""
4_diagnostics.py -- OFFLINE diagnosis + parameter sweep of the camera-localization pipeline.

Uses ground truth ON PURPOSE, so this is an analysis tool only. It must never be part of the submitted
inference path.

    python Scripts/4_diagnostics.py --data-h5 data/training_cache.h5 --checkpoint checkpoints/racing_model.pt
    python Scripts/4_diagnostics.py --sweep            # finds the best filter settings, prints a ready command

The checkpoint must come from a run WITHOUT --full-train (otherwise there are no held-out validation frames).

The "fused" variant here is NOT a hand-copied replica any more: it drives the very same `FusionLocalizer`
class that `3_causal_infer.py` uses, so numbers measured here are what the submission run will do.

What it does (on the held-out validation frames, identical timestamps for all variants):
  1. One network pass over the validation frames (embeddings, aux pose, VO deltas, VO log-variances).
  2. Four variants: retrieval_top1, aux_head, vo_only, fused.
  3. Every error is split into along-track (longitudinal) and across-track (lateral) parts, and into frames the
     map covers vs. frames it does not (e.g. the final pit stop).
  4. Reference-map coverage, gate-exclusion, VO step-error / scale statistics, filter consistency (NIS / NEES).
  5. Standstill calibration: how well the frame-difference test separates a parked car from a moving one.
  6. --sweep: two-stage search (filter trust grid, then the new features one at a time).

Outputs (in --out-dir): variants_per_frame.csv, fused_telemetry.csv, diagnostics.png, sweep_results.csv,
recommended_args.txt
"""
import argparse
import csv
import dataclasses
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


def get_infer_module():
    """The inference script is the single source of truth for the filter (FusionLocalizer, presets)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return load_module_from_path('infer_mod', os.path.join(here, '3_causal_infer.py'))


# ----------------------------------------------------------------------------
# Step A: load checkpoint + cache, run the network once (the only torch-dependent part)
# ----------------------------------------------------------------------------
def load_inputs(args, infer_mod):
    import h5py
    import torch

    here = os.path.dirname(os.path.abspath(__file__))
    train_mod = load_module_from_path('train_mod', os.path.join(here, '2_train_model.py'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_cfg = train_mod.model_config_from_checkpoint(ckpt)
    model = train_mod.build_model_from_checkpoint(ckpt).to(device).eval()

    val_start_ns = int(ckpt.get('val_start_timestamp_ns', 0))
    if val_start_ns <= 0:
        sys.exit('This checkpoint has no held-out validation split (trained with --full-train?). '
                 'Retrain without --full-train to diagnose generalization.')

    with h5py.File(args.data_h5, 'r') as h5f:
        ts_all = h5f['timestamps_ns'][:]
        poses_all = h5f['poses'][:]          # [x, y, z, yaw]
        dtoff_all = h5f['dt_offsets'][:]
        causal_cache = bool(h5f.attrs.get('causal_sync', 0))
        i0 = int(np.searchsorted(ts_all, val_start_ns))
        val_end_ns = int(ckpt.get('val_end_timestamp_ns', 0))          # 0 = old checkpoint: validation ran to the end
        i_end = int(np.searchsorted(ts_all, val_end_ns, side='right')) if val_end_ns > 0 else len(ts_all)
        i1 = i_end if args.limit is None else min(i_end, i0 + args.limit)
        imgs = h5f['images'][i0:i1]
        odom = h5f['odom'][i0:i1] if 'odom' in h5f else None            # telemetry odometry (speed + yaw rate)
    if odom is None:
        print('[WARN] this cache has no telemetry odometry (built before the state_estimation support). '
              'Rebuild it with 1_build_dataset.py; falling back to the learned VO.')
        odom = np.full((i1 - i0, 7), np.nan)

    db_emb = ckpt['db_embeddings'].cpu().numpy().astype(np.float32)
    db_poses = ckpt['db_poses'].cpu().numpy().astype(np.float64)   # [x, y, yaw]
    if 'db_timestamps_ns' in ckpt:
        # New checkpoints: the map is every frame outside the held-out block (+ guard), not a prefix of the cache.
        db_ts = ckpt['db_timestamps_ns'].cpu().numpy().astype(np.int64)
        train_mask = np.isin(ts_all, db_ts)
        if int(train_mask.sum()) != len(db_emb):
            print(f'[WARN] DB has {len(db_emb)} references but {int(train_mask.sum())} cache frames match their '
                  f'timestamps. Was the cache rebuilt after training? Results may be meaningless.')
        train_poses, train_ts = poses_all[train_mask], ts_all[train_mask]
    else:
        if len(db_emb) != i0:
            print(f'[WARN] DB has {len(db_emb)} references but the cache has {i0} frames before the validation start. '
                  f'Was the cache rebuilt after training? Results may be meaningless.')
        train_poses, train_ts = poses_all[:i0], ts_all[:i0]
    if ts_all[i0] != val_start_ns:
        print('[WARN] validation start timestamp not found exactly in the cache.')
    if not causal_cache:
        print('[WARN] This cache predates the causal-sync fix in 1_build_dataset.py (side-camera images may be '
              'stamped AFTER the anchor). Replay results can then differ from the real inference run. Rebuild it.')

    ts = ts_all[i0:i1]
    gt = poses_all[i0:i1]
    dtoff = dtoff_all[i0:i1]
    N = len(ts)

    def prep(u8):
        x = u8.astype(np.float32) / 255.0
        x = (x - x.mean(axis=(2, 3), keepdims=True)) / (x.std(axis=(2, 3), keepdims=True) + 1e-5)
        return torch.from_numpy(x).to(device)

    emb = np.zeros((N, db_emb.shape[1]), np.float32)
    aux = np.zeros((N, 4), np.float32)
    delta = np.full((N, 3), np.nan)
    logvar = np.full((N, 3), np.nan)
    dt_raw = np.full(N, np.nan)

    print(f'[NET] Running the network on {N} validation frames on {device} (vo_mode={model_cfg["vo_mode"]}) ...')
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

    # Frame-difference score for the standstill test (identical code to inference).
    change = np.full(N, np.nan)
    prev_norm = infer_mod.normalize_stack(imgs[0])
    for i in range(1, N):
        cur_norm = infer_mod.normalize_stack(imgs[i])
        change[i] = infer_mod.frame_change_score(prev_norm, cur_norm)
        prev_norm = cur_norm

    return dict(
        ts=ts, gt=gt, emb=emb, aux=aux, delta=delta, logvar=logvar, dt_raw=dt_raw, change=change, odom=odom,
        db_emb=db_emb, db_poses=db_poses, model_cfg=model_cfg,
        xy_mean=ckpt['xy_mean'].cpu().numpy().astype(np.float64),
        xy_std=ckpt['xy_std'].cpu().numpy().astype(np.float64),
        train_poses=train_poses, train_ts=train_ts,
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


def compute_covered(gt, db_poses, close_m, close_deg):
    """GT-assisted: is a reference within close_m / close_deg of the true pose available at all?"""
    cy = math.radians(close_deg)
    out = np.zeros(len(gt), dtype=bool)
    for i in range(len(gt)):
        close = (np.hypot(db_poses[:, 0] - gt[i, 0], db_poses[:, 1] - gt[i, 1]) <= close_m) & \
                (np.abs(wrap(db_poses[:, 2] - gt[i, 3])) <= cy)
        out[i] = bool(close.any())
    return out


FUSED_COLS = [
    'frame', 'timestamp_ns', 'mode', 'gt_x', 'gt_y', 'gt_yaw',
    'dx_body_pred', 'dy_body_pred', 'dyaw_pred', 'q_x', 'q_y', 'q_yaw',
    'change_score', 'still', 'pred_x', 'pred_y', 'pred_yaw', 'gate_size_m', 'gate_n', 'top1_sim', 'global_sim',
    'conf_infl', 'retrieved_x', 'retrieved_y', 'retrieved_yaw',
    'innovation_x', 'innovation_y', 'innovation_yaw', 'nis',
    'P_xx', 'P_yy', 'P_yawyaw', 'R_xx', 'R_yy', 'R_yawyaw',
    'est_x', 'est_y', 'est_yaw', 'err_long', 'err_lat', 'err_yaw',
    'meas_err_long', 'meas_err_lat', 'nees_xy',
    'ref_close_to_gt', 'ref_close_in_gate', 'gate_excluded_close_ref',
    'odom_tel', 'n_hyp', 'map_dist', 'map_dyaw',
]


def run_fused(d, params, mod, args, lite=False):
    """
    Drive the real MultiStartLocalizer over the validation frames exactly as the inference script does: the start
    position is found from the images (no ground truth), the motion comes from telemetry odometry when available
    and plausible. lite=True skips the GT-assisted telemetry (used by the sweep, which only needs the estimates).
    --init gt starts from the true first pose instead (an optimistic upper bound, for comparison only).
    """
    gt, emb, delta, logvar, change, odom = d['gt'], d['emb'], d['delta'], d['logvar'], d['change'], d['odom']
    db = d['db_poses']
    N = len(gt)
    loc = mod.MultiStartLocalizer(d['db_emb'], db, params)
    loc.start(emb[0], pose=(gt[0, 0], gt[0, 1], gt[0, 3]) if args.init == 'gt' else None)
    health = mod.OdomHealth()
    gate_yaw = math.radians(params.gate_deg)
    close_yaw = math.radians(args.close_deg)

    est = np.zeros((N, 3))
    rows = []
    for i in range(N):
        if i == 0:
            rec = loc.step(emb[0])
        else:
            tel = tuple(odom[i]) if (args.odom != 'vo' and np.isfinite(odom[i, 0])) else None
            kind, motion, _, _ = mod.select_odometry(args.odom, tel, delta[i], health)
            rec = loc.step(emb[i], motion, np.exp(logvar[i]), change[i], kind)
        est[i] = (rec['x'], rec['y'], rec['yaw'])
        if lite:
            continue

        r = {c: float('nan') for c in FUSED_COLS}
        for k, v in rec.items():
            if k in r:
                r[k] = v
        r.update(frame=i, timestamp_ns=int(d['ts'][i]), gt_x=gt[i, 0], gt_y=gt[i, 1], gt_yaw=gt[i, 3],
                 gate_size_m=params.gate_m, est_x=rec['x'], est_y=rec['y'], est_yaw=rec['yaw'])

        # Ground-truth-assisted: was a reference close to the TRUE pose available, and was it inside the gate?
        gdx, gdy = db[:, 0] - rec['pred_x'], db[:, 1] - rec['pred_y']
        gate_mask = ((gdx ** 2 + gdy ** 2) <= params.gate_m ** 2) & \
                    (np.abs(wrap(db[:, 2] - rec['pred_yaw'])) <= gate_yaw)
        close = (np.hypot(db[:, 0] - gt[i, 0], db[:, 1] - gt[i, 1]) <= args.close_m) & \
                (np.abs(wrap(db[:, 2] - gt[i, 3])) <= close_yaw)
        has_close = bool(close.any())
        in_gate = bool((close & gate_mask).any())
        r.update(ref_close_to_gt=int(has_close), ref_close_in_gate=int(in_gate),
                 gate_excluded_close_ref=int(has_close and not in_gate))

        if np.isfinite(rec.get('retrieved_x', float('nan'))):
            ml, ma, _ = decompose(np.array([[rec['retrieved_x'], rec['retrieved_y']]]),
                                  np.array([rec['retrieved_yaw']]), gt[i:i + 1, :2], gt[i:i + 1, 3])
            r.update(meas_err_long=float(ml[0]), meas_err_lat=float(ma[0]))

        el, ea, ey = decompose(est[i:i + 1, :2], est[i:i + 1, 2], gt[i:i + 1, :2], gt[i:i + 1, 3])
        e_xy = np.array([est[i, 0] - gt[i, 0], est[i, 1] - gt[i, 1]])
        try:
            nees = float(e_xy @ np.linalg.solve(loc.ekf.P[:2, :2], e_xy))
        except np.linalg.LinAlgError:
            nees = float('nan')
        r.update(err_long=float(el[0]), err_lat=float(ea[0]), err_yaw=float(ey[0]), nees_xy=nees)
        rows.append(r)

    return est, rows, FUSED_COLS, loc


def batch_positive_stats(poses, ts, batch=64, n_batches=500, seed=0):
    """How much positive signal does the contrastive loss see in a RANDOM training batch (the old sampler)?"""
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


def still_calibration(d, speed):
    """How well does the frame-change score separate a parked car from a moving one? (GT-assisted)"""
    chg = d['change']
    ok = np.isfinite(chg)
    parked = ok & (speed < 0.2)
    moving = ok & (speed > 1.0)
    out = dict(n_parked=int(parked.sum()), n_moving=int(moving.sum()), suggested=None)
    if parked.sum() >= 5 and moving.sum() >= 5:
        out.update(
            parked_p50=float(np.percentile(chg[parked], 50)), parked_p99=float(np.percentile(chg[parked], 99)),
            moving_p01=float(np.percentile(chg[moving], 1)), moving_p05=float(np.percentile(chg[moving], 5)),
            moving_p50=float(np.percentile(chg[moving], 50)),
        )
        if out['parked_p99'] < out['moving_p01']:
            out['suggested'] = float(math.sqrt(out['parked_p99'] * out['moving_p01']))
    return out


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
    st = np.array([r['still'] == 1 for r in fused_rows])
    ax[1].plot(t, fe, lw=1.2, color='C3', label='fused estimate error')
    ax[1].plot(t, me, lw=0.8, color='C0', alpha=0.6, label='retrieval measurement error (fused loop)')
    for a, b in spans(ex):
        ax[1].axvspan(t[a], t[b], color='red', alpha=0.2, lw=0)
    for a, b in spans(nc):
        ax[1].axvspan(t[a], t[b], color='gray', alpha=0.25, lw=0)
    for a, b in spans(st):
        ax[1].axvspan(t[a], t[b], color='green', alpha=0.2, lw=0)
    ax[1].set_ylabel('fused: longitudinal [m]')
    ax[1].set_title('red = close reference existed but was outside the gate;  gray = no close reference in the map;  '
                    'green = standstill hold', fontsize=9)
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


# ----------------------------------------------------------------------------
# Sweep: stage 1 = filter trust grid, stage 2 = new features one at a time (greedy)
# ----------------------------------------------------------------------------
def evaluate_params(d, params, mod, args, covered):
    gt = d['gt']
    est, _, _, _ = run_fused(d, params, mod, args, lite=True)
    el, ea, ey = decompose(est[:, :2], est[:, 2], gt[:, :2], gt[:, 3])
    e = np.hypot(el, ea)
    unc = ~covered
    return dict(
        cov_rmse=float(np.sqrt((e[covered] ** 2).mean())) if covered.any() else float('nan'),
        cov_median=float(np.median(e[covered])) if covered.any() else float('nan'),
        cov_p95=float(np.percentile(e[covered], 95)) if covered.any() else float('nan'),
        cov_yaw_deg=float(np.degrees(np.sqrt((ey[covered] ** 2).mean()))) if covered.any() else float('nan'),
        unc_rmse=float(np.sqrt((e[unc] ** 2).mean())) if unc.any() else float('nan'),
        all_rmse=float(np.sqrt((e ** 2).mean())),
    )


def params_cli(params, ref=None):
    """Command-line flags for the settings that differ from `ref` (all of them if ref is None)."""
    parts = []
    for f in dataclasses.fields(params):
        v = getattr(params, f.name)
        if ref is not None and v == getattr(ref, f.name):
            continue
        flag = '--' + f.name.replace('_', '-')
        if isinstance(v, bool):
            if v:
                parts.append(flag)
        else:
            parts.append(f'{flag} {v:g}')
    return ' '.join(parts)


def sweep(d, args, mod, base_params):
    gt = d['gt']
    speed = np.r_[np.nan, np.hypot(*np.diff(gt[:, :2], axis=0).T) / np.maximum(np.diff(d['ts']) * 1e-9, 1e-3)]
    speed[0] = speed[1]
    covered = compute_covered(gt, d['db_poses'], args.close_m, args.close_deg)
    calib = still_calibration(d, speed)
    results = []

    def run(label, stage, params):
        r = evaluate_params(d, params, mod, args, covered)
        r.update(label=label, stage=stage, **dataclasses.asdict(params))
        results.append(r)
        print('.', end='', flush=True)
        return r

    def line(tag, r):
        print(f"  {tag:<34} covered {r['cov_rmse']:>6.2f} (median {r['cov_median']:>5.2f}, p95 {r['cov_p95']:>6.2f}) | "
              f"uncovered {r['unc_rmse']:>7.2f} | ALL {r['all_rmse']:>7.2f}")

    print(f'[SWEEP] {len(gt)} validation frames, {covered.mean():.1%} covered by the map. '
          f'Objective = overall position RMSE (that is what is scored), guarded so covered-track accuracy '
          f'cannot degrade by more than {args.guard:.0%}.')

    tel_mode = args.odom != 'vo' and bool(np.isfinite(d['odom'][1:, 0]).mean() > 0.5)
    print(f'[SWEEP] motion model: {"telemetry odometry (speed + yaw rate)" if tel_mode else "learned VO"} | '
          f'start: {"found from the images" if args.init == "global" else "TRUE first pose (--init gt)"}')

    # ---- reference rows -----------------------------------------------------------------------------------
    print('\nReference runs')
    legacy = mod.FusionParams(**mod.LEGACY_PRESET)
    ref_rows = [('LEGACY preset trust levels', legacy)]
    if base_params != legacy:
        ref_rows.append(('checkpoint preset (+ CLI overrides)', base_params))
    for label, p in ref_rows:
        line(label, run(label, 'reference', p))

    # ---- stage 1 ------------------------------------------------------------------------------------------
    if tel_mode:
        # accurate motion: the retrieval measurement is averaged over time, so its noise level is what matters
        if args.sweep_quick:
            rs_l, fl_l, gates = [1.0, 3.0], [1.0], [(35.0, 35.0), (50.0, 45.0)]
        else:
            rs_l, fl_l, gates = [0.3, 1.0, 3.0], [0.5, 1.0, 2.0], [(35.0, 35.0), (50.0, 45.0)]
        grid = [(dict(r_scale_tel=rs, ret_floor_tel_m=fl, gate_m=gm, gate_deg=gd), f'rt{rs:g} floor{fl:g} gate{gm:g}/{gd:g}')
                for rs in rs_l for fl in fl_l for gm, gd in gates]
    else:
        if args.sweep_quick:
            qs, rs_l, gates = [0.25, 0.5, 1.0], [0.1, 0.03], [(35.0, 35.0), (50.0, 45.0)]
        else:
            qs, rs_l, gates = [0.1, 0.25, 0.5, 1.0], [0.3, 0.1, 0.03], [(35.0, 35.0), (50.0, 45.0), (80.0, 60.0)]
        grid = [(dict(q_x_floor=qf, r_scale=rs, gate_m=gm, gate_deg=gd), f'q{qf:g} r{rs:g} gate{gm:g}/{gd:g}')
                for qf in qs for rs in rs_l for gm, gd in gates]
    print(f'\nStage 1: filter trust grid ({len(grid)} runs) around the checkpoint preset')
    stage1 = []
    for changes, label in grid:
        p = dataclasses.replace(base_params, **changes)
        stage1.append((p, run(label, 'stage1', p), label))
    print()
    finite_cov = [r['cov_rmse'] for _, r, _ in stage1 if np.isfinite(r['cov_rmse'])]
    best_cov = min(finite_cov) if finite_cov else float('inf')
    ok1 = [(p, r, lab) for p, r, lab in stage1 if not np.isfinite(best_cov) or r['cov_rmse'] <= best_cov * (1.0 + args.guard)]
    ok1.sort(key=lambda t: t[1]['all_rmse'])
    for rank, (p, r, lab) in enumerate(ok1[:8], 1):
        line(f'#{rank} {lab}', r)
    best_p, best_r = ok1[0][0], ok1[0][1]

    # ---- stage 2 ------------------------------------------------------------------------------------------
    still_values = [0.0, 0.02, 0.03, 0.05, 0.08, 0.12]
    if calib['suggested'] is not None:
        still_values.append(round(calib['suggested'], 4))
    knobs = [] if tel_mode else [('still_thresh', sorted(set(still_values)))]   # image standstill test: VO mode only
    knobs += [
        ('sim_tau', [0.0, 0.005, 0.01, 0.02]),
        ('sim_ref', [0.92, 0.95, 0.97]),
        ('sim_adapt_q', [0.0, 0.5, 0.75, 0.9]),
        ('cluster_m', [0.0, 15.0, 30.0]),
        ('reloc_sim', [2.0, 0.93, 0.95, 0.97]),
        ('nis_gate', [0.0, 30.0, 100.0]),
        ('topk', [4, 8, 12]),
    ]
    if tel_mode:
        knobs += [('q_tel_scale', [0.5, 1.0, 2.0, 4.0]), ('tel_dx_frac', [0.005, 0.01, 0.02])]
    else:
        knobs += [('q_scale', [0.5, 1.0, 2.0])]
    if args.init == 'global':
        knobs += [('init_max_frames', [30, 60, 100]), ('init_candidates', [48, 96, 192]),
                  ('init_margin', [0.01, 0.02, 0.05]), ('init_heading_w', [0.0, 0.004, 0.01])]
    anchor_cov = best_r['cov_rmse']      # covered-track accuracy of the stage-1 winner; stage 2 may not stray far from it
    print(f'\nStage 2: one feature at a time, greedy from the stage-1 winner (guard: covered RMSE <= '
          f'{1 + args.guard:.2f} x {anchor_cov:.2f} m)')
    for knob, values in knobs:
        cands = []
        for v in values:
            p = dataclasses.replace(best_p, **{knob: type(getattr(best_p, knob))(v)})
            r = best_r if p == best_p else run(f'{knob}={v:g}', 'stage2', p)
            cands.append((p, r, v))
        print()
        allowed = [c for c in cands
                   if not np.isfinite(anchor_cov) or c[1]['cov_rmse'] <= anchor_cov * (1.0 + args.guard)]
        for p, r, v in cands:
            line(f'{knob} = {v:g}' + ('  <- current' if p == best_p else ''), r)
        pick = min(allowed, key=lambda c: c[1]['all_rmse'])
        if pick[1]['all_rmse'] < best_r['all_rmse'] - 1e-9:
            best_p, best_r = pick[0], pick[1]
            print(f'  -> take {knob} = {pick[2]:g}  (overall {best_r["all_rmse"]:.2f} m)')

    # ---- result -------------------------------------------------------------------------------------------
    print('\n' + '=' * 78)
    print('RECOMMENDED SETTINGS (validation replay of the real FusionLocalizer)')
    line('recommended', best_r)
    line('LEGACY preset', next(r for r in results if r['label'].startswith('LEGACY')))
    print('\nflags for 3_causal_infer.py (only those differing from this checkpoint\'s preset):')
    cli = params_cli(best_p, mod.resolve_fusion_params(argparse.Namespace(), d['model_cfg']))
    print('  ' + (cli if cli else '(none: the preset is already the best found)'))
    print('=' * 78)
    print('Caution: the uncovered stretch is a few seconds of one recording, so its error is noisy. The guard keeps '
          'covered-track accuracy from being traded away for luck there; prefer round, plausible values.')

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'sweep_results.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    rec = os.path.join(args.out_dir, 'recommended_args.txt')
    with open(rec, 'w') as f:
        f.write(params_cli(best_p) + '\n')
    print(f'\n[FILES] {out}\n        {rec}')


# ----------------------------------------------------------------------------
# Full report
# ----------------------------------------------------------------------------
def analyze(d, args, mod, params):
    os.makedirs(args.out_dir, exist_ok=True)
    ts, gt = d['ts'], d['gt']
    N = len(ts)
    t = (ts - ts[0]) * 1e-9
    gt_xy, gt_yaw = gt[:, :2], gt[:, 3]
    dt = np.maximum(np.diff(ts) * 1e-9, 1e-3)
    speed = np.r_[np.nan, np.hypot(*np.diff(gt_xy, axis=0).T) / dt]
    speed[0] = speed[1]
    cfg = d['model_cfg']

    print(f'\n[INFO] filter driven by MultiStartLocalizer from 3_causal_infer.py | motion source: {args.odom} | '
          f'start: {"found from the images (as in the competition)" if args.init == "global" else "TRUE first pose (--init gt, optimistic)"}')
    print(f'[INFO] settings: {params}')
    print(f'[INFO] checkpoint: vo_mode={cfg["vo_mode"]}, q_scaling_fixed={cfg["q_scaling_fixed"]}')
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
    est, fused_rows, fused_cols, loc = run_fused(d, params, mod, args)

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

    rc = {5: float('nan')}
    if covered.any():
        rc = {m: float((ret_dist[covered] <= m).mean()) for m in (2, 5, 10)}
        print('On covered frames only, global top-1 retrieval within '
              + ', '.join(f'{m} m: {rc[m]:.1%}' for m in rc))
        if (~covered).any():
            print(f'Top-1 similarity: covered median {np.median(top1_sim[covered]):.3f} '
                  f'(10th pct {np.percentile(top1_sim[covered], 10):.3f}) | uncovered median '
                  f'{np.median(top1_sim[~covered]):.3f} (90th pct {np.percentile(top1_sim[~covered], 90):.3f})')

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
    reloc = np.array([r['mode'] == 'RELOCALIZED' for r in fused_rows])
    gate_n = np.array([r['gate_n'] for r in fused_rows])
    print(f'\n=== 3. Gate exclusion (gate {params.gate_m:.0f} m / {params.gate_deg:.0f} deg around the PREDICTED state) ===')
    print(f'frames where a close reference existed but the gate excluded it : {excl.mean():.1%}')
    if excl.any():
        first = int(np.argmax(excl))
        print(f'first such frame: #{first} (t = {t[first]:.1f} s); longest run: '
              f'{max(b - a + 1 for a, b in spans(excl))} frames')
    print(f'frames where a retrieval measurement was fused                  : {used.mean():.1%}')
    print(f'frames where the filter relocalized globally                    : {int(reloc.sum())}')
    print(f'frames with fewer than 5 references in the gate (no update)     : {(gate_n < 5).mean():.1%}')

    # ---- VO -----------------------------------------------------------------------------------
    c, s_ = np.cos(gt_yaw[:-1]), np.sin(gt_yaw[:-1])
    gx, gy = np.diff(gt_xy[:, 0]), np.diff(gt_xy[:, 1])
    gt_step = np.stack([c * gx + s_ * gy, -s_ * gx + c * gy, wrap(np.diff(gt_yaw))], axis=1)
    pr = d['delta'][1:]
    err = pr - gt_step
    ok = np.isfinite(err).all(axis=1)
    err, pr, gt_step_ok, spd, dt_ok = err[ok], pr[ok], gt_step[ok], speed[1:][ok], dt[ok]
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
    slow = spd < 0.3
    if slow.sum() > 3:
        print(f'standstill probe: {int(slow.sum())} frames with GT speed < 0.3 m/s -> VO reports a mean speed of '
              f'{np.mean(np.abs(pr[slow, 0]) / dt_ok[slow]):.2f} m/s (should be ~0)')
    dt_raw = d['dt_raw'][1:]
    bad_dt = (dt_raw < 0.01) | (dt_raw > 0.5)
    print(f'frame intervals outside the training range [0.01, 0.5] s: {int(bad_dt.sum())} of {len(dt_raw)}'
          f' (max interval {np.nanmax(dt_raw):.3f} s)')
    lv = d['logvar'][1:][ok]
    act = np.sqrt((err ** 2).mean(axis=0))
    pred_sd_now = np.sqrt(np.exp(lv).mean(axis=0))
    print('VO uncertainty calibration (actual RMS error / predicted std; about 1.0 is calibrated):')
    print(f'   as predicted exp(log_var)       : dx {act[0] / pred_sd_now[0]:.2f} | dy {act[1] / pred_sd_now[1]:.2f} | dyaw {act[2] / pred_sd_now[2]:.2f}')
    if not cfg['q_scaling_fixed']:
        pred_sd_w = np.sqrt((np.exp(lv) / np.array([1.0, 2.0, 5.0])).mean(axis=0))
        print(f'   with exp(log_var)/[1,2,5] (fix) : dx {act[0] / pred_sd_w[0]:.2f} | dy {act[1] / pred_sd_w[1]:.2f} | dyaw {act[2] / pred_sd_w[2]:.2f}')

    # ---- telemetry odometry (speed + yaw rate from state_estimation) --------------------------------
    od = d['odom'][1:]
    okt = np.isfinite(od[:, 0]) & (dt >= 0.01) & (dt <= 0.5)
    print('\n=== 4a. Telemetry odometry (speed + yaw rate), per step ===')
    if okt.sum() > 10:
        et = od[okt, :3] - gt_step[okt]
        t_scale = od[okt, 0].sum() / max(gt_step[okt, 0].sum(), 1e-9)
        print(f'valid for {okt.mean():.1%} of steps | forward-distance scale {t_scale:.4f} (1.000 = perfect)')
        print(f'step RMSE   : dx {np.sqrt((et[:, 0] ** 2).mean()):.3f} m | dy {np.sqrt((et[:, 1] ** 2).mean()):.3f} m | '
              f'dyaw {math.degrees(np.sqrt((et[:, 2] ** 2).mean())):.3f} deg   '
              f'(learned VO above: dx {np.sqrt((err[:, 0] ** 2).mean()):.3f} m)')
        used = np.mean([r['odom_tel'] == 1 for r in fused_rows])
        print(f'the fused run used telemetry odometry on {used:.1%} of frames (the rest fell back to the learned VO)')
    else:
        print('no usable telemetry odometry in this cache: the run used the learned VO only')

    # ---- finding the start position without an initial pose -----------------------------------------
    if args.init == 'global':
        print('\n=== 4c. Start position found from the images (no initial pose) ===')
        e_all = np.hypot(est[:, 0] - gt_xy[:, 0], est[:, 1] - gt_xy[:, 1])
        k = loc.commit_frame
        if k is None:
            print('the candidates were never resolved to one (run shorter than init_max_frames)')
        else:
            k = min(int(k), N - 1)
            print(f'committed after {k} frames ({t[k]:.1f} s) | position error at commit {e_all[k]:.2f} m | '
                  f'RMSE before commit {np.sqrt(np.mean(e_all[:k + 1] ** 2)):.2f} m | after {np.sqrt(np.mean(e_all[k:] ** 2)):.2f} m')
            if e_all[k] > 30.0:
                print('-> the commit landed on a wrong place (> 30 m off). Raise --init-max-frames, or check '
                      '[VAL-RET] in the training log: the descriptor may not separate places well enough.')

    # ---- standstill calibration ---------------------------------------------------------------
    cal = still_calibration(d, speed)
    print('\n=== 4b. Standstill detector calibration (frame-change score, GT-assisted) ===')
    if 'parked_p99' in cal:
        print(f"parked frames (GT speed < 0.2 m/s, n={cal['n_parked']}): score median {cal['parked_p50']:.4f}, 99th pct {cal['parked_p99']:.4f}")
        print(f"moving frames (GT speed > 1 m/s,   n={cal['n_moving']}): score 1st pct {cal['moving_p01']:.4f}, 5th pct {cal['moving_p05']:.4f}, median {cal['moving_p50']:.4f}")
        if cal['suggested'] is not None:
            print(f"-> cleanly separable. Suggested --still-thresh {cal['suggested']:.4f} "
                  f"(current setting: {params.still_thresh:g}{' = off' if params.still_thresh <= 0 else ''}).")
        else:
            print('-> parked and moving scores OVERLAP: keep the detector off (--still-thresh 0) or very conservative.')
        chg = d['change']
        if params.still_thresh > 0:
            fp = float(np.mean(chg[np.isfinite(chg) & (speed > 1.0)] < params.still_thresh))
            tp = float(np.mean(chg[np.isfinite(chg) & (speed < 0.2)] < params.still_thresh))
            print(f"at the current threshold: {tp:.1%} of parked frames flagged still, {fp:.2%} of moving frames wrongly flagged")
    else:
        print(f"not enough parked/moving frames to calibrate (parked {cal['n_parked']}, moving {cal['n_moving']}).")

    # ---- filter consistency -------------------------------------------------------------------
    nis = np.array([r['nis'] for r in fused_rows])
    nees = np.array([r['nees_xy'] for r in fused_rows])
    print('\n=== 5. Filter consistency (fused loop) ===')
    print(f'NIS  (innovation vs. predicted S, 3 dof, ideal mean 3): mean {np.nanmean(nis):.1f} | median {np.nanmedian(nis):.1f}')
    print(f'NEES (position error vs. P, 2 dof, ideal mean 2)      : mean {np.nanmean(nees):.1f} | median {np.nanmedian(nees):.1f}')

    # ---- descriptor-training batches ----------------------------------------------------------
    bs = batch_positive_stats(d['train_poses'], d['train_ts'])
    print(f'\n=== 6. What the contrastive loss sees in a RANDOM training batch of {bs["batch"]} '
          f'(old sampler; the new default sampler pairs every anchor with a same-place frame) ===')
    print(f'anchors with >= 1 other sample within 6 m and 20 deg: {bs["frac_pos_6m_20deg"]:.1%}')
    print(f'anchors with >= 1 other sample within 2 m and 10 deg: {bs["frac_pos_2m_10deg"]:.1%}')

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
                print(f'- {unc:.0%} of the fused squared error comes from frames the map does not cover: improve VO off-map '
                      '(standstill hold, early-fusion VO) rather than tuning covered-track settings.')
        if rc[5] < 0.5:
            print(f'- Only {rc[5]:.0%} of covered frames are retrieved within 5 m: the descriptor confuses places (improve training / sampling).')
    if excl.mean() > 0.05:
        print(f'- The gate hid a close reference in {excl.mean():.0%} of frames: the prediction drifted out of reach (widen the gate / enable relocalization).')
    if np.percentile(near_any, 95) > 5.0 or no_close.mean() > 0.05:
        print('- The map has coverage gaps on the validation route (check the plot, e.g. pit lane).')
    scale = pr[:, 0].sum() / gt_step_ok[:, 0].sum()
    if abs(scale - 1.0) > 0.02:
        print(f'- VO has a systematic forward-scale error ({scale:.3f}): this drifts along-track by itself.')
    if np.nanmean(nis) > 6 or np.nanmean(nees) > 4:
        print('- The filter is over-confident (NIS / NEES above the ideal): measurement or process covariances are too small.')
    if a_ < min(f_, r_, v_):
        print('- The auxiliary pose head beats everything else: keep it as a baseline / coarse recovery proposal.')
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
    print(f'\nOVERALL position RMSE of the fused output: {f_:.3f} m (covered frames: {covered.mean():.1%}).')
    return summ


def main():
    mod = get_infer_module()
    p = argparse.ArgumentParser(description='Offline, ground-truth-assisted diagnosis of the localization pipeline.')
    p.add_argument('--data-h5', default='data/training_cache.h5')
    p.add_argument('--checkpoint', default='checkpoints/racing_model.pt')
    p.add_argument('--out-dir', default='diagnostics')
    p.add_argument('--limit', type=int, default=None, help='only use the first N validation frames (quick test)')
    p.add_argument('--batch', type=int, default=32)
    # definition of "a reference close to the true pose"
    p.add_argument('--close-m', type=float, default=5.0)
    p.add_argument('--close-deg', type=float, default=20.0)
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--odom', choices=['auto', 'telemetry', 'vo'], default='auto',
                   help='motion source, as in 3_causal_infer.py (auto = telemetry when present and plausible)')
    p.add_argument('--init', choices=['global', 'gt'], default='global',
                   help="global = find the start from the images, as in the competition (default); "
                        "gt = start from the true first pose (optimistic, for comparison only)")
    p.add_argument('--sweep', action='store_true',
                   help='instead of the full report, search filter settings and print a ready-to-use command')
    p.add_argument('--sweep-quick', action='store_true', help='smaller stage-1 grid (12 instead of 36 runs)')
    p.add_argument('--guard', type=float, default=0.15,
                   help='sweep: max allowed relative loss of covered-track accuracy when chasing overall RMSE')
    mod.add_fusion_args(p)      # same flags as 3_causal_infer.py; unset = preset chosen from the checkpoint
    args = p.parse_args()

    data = load_inputs(args, mod)
    params = mod.resolve_fusion_params(args, data['model_cfg'])
    if args.sweep:
        sweep(data, args, mod, params)
        return
    analyze(data, args, mod, params)


if __name__ == '__main__':
    main()
