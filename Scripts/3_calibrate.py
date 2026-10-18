"""
STAGE 3 - post-training calibration of a checkpoint (no retraining). Three independent steps, all on by default:

  a) visual odometry bias : speed-dependent forward-distance bias of OdomNet, measured on the HELD-OUT laps and
                            stored as calib['dx_map_pred'] -> calib['dx_map_true'] (+ the residual std).
  b) state estimation     : speed scale, yaw-rate scale + bias and per-frame noise of the vehicle's speed / yaw-rate
                            signals (the only state-estimation fields used, see lib/state_est.py). Fitted on the
                            TRAINING split, reported on the held-out laps. Stored as calib['se'].
  c) track map            : centerline, real borders, pit corridor, heading prior from the database poses and the
                            per-marker track borders (from the cache, or straight from the bag with --mcap).

  python Scripts/3_calibrate.py --checkpoint checkpoints/racing_model_cal.pt --output checkpoints/se_cal.pt \
         --data-h5 data/training_cache.h5 --plot checkpoints/track.png
  python Scripts/3_calibrate.py ... --skip-odom --skip-se --mcap Data/<bag>.mcap     # only the track map

The cache must contain state_est/ (1_build_dataset.py, or --aux-only on an existing cache).
"""
import argparse
import math
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # make `lib` importable from any CWD

from lib import provenance  # noqa: E402
from lib.data import OdomDataset, odom_bias_curve, predict_odom  # noqa: E402
from lib.models import load_networks  # noqa: E402
from lib.state_est import DEFAULT_SE_CALIB, SE_FIELDS, evaluate_calibration, fit_calibration, pair_measurements  # noqa: E402
from lib.track_map import build_track_map, load_markers_h5, marker_xy, read_track_bounds_mcap  # noqa: E402


def calibrate_state_est(ck, args):
    with h5py.File(args.data_h5, 'r') as f:
        if 'state_est' not in f:
            print('[SE][WARN] cache has no state_est/: run 1_build_dataset.py <bag> --aux-only. Skipped.')
            return
        ts, poses = f['timestamps_ns'][:], f['poses'][:]
        se = {'t_ns': f['state_est/t_ns'][:]}
        for k in SE_FIELDS:
            se[k] = f[f'state_est/{k}'][:]
    ok = np.isfinite(se['v_mps']) & np.isfinite(se['wz_radps'])
    se = {k: v[ok] for k, v in se.items()}
    o = np.argsort(se['t_ns'])
    se = {k: v[o] for k, v in se.items()}
    vs = int(ck.get('val_start_timestamp_ns', 0))
    train_idx = np.flatnonzero(ts < vs) if vs > 0 else np.arange(len(ts))
    val_idx = np.flatnonzero(ts >= vs) if vs > 0 else np.zeros(0, dtype=int)

    m_tr = pair_measurements(se, ts, poses, train_idx)
    if len(m_tr) < 200:
        print(f'[SE][WARN] only {len(m_tr)} usable frame pairs: skipped')
        return
    cal = fit_calibration(m_tr)
    print(f'[SE] fitted on {cal["n_pairs"]} training frame pairs ({cal["n_moving"]} moving): '
          f'v_scale {cal["v_scale"]:.4f}, wz_scale {cal["wz_scale"]:.4f}, wz_bias {math.degrees(cal["wz_bias"]):+.3f} deg/s, '
          f'noise dx {cal["dx_std"]:.3f} m / dyaw {math.degrees(cal["dyaw_std"]):.3f} deg per frame')
    for name, idx in (('train', train_idx), ('held-out', val_idx)):
        if len(idx) < 50:
            continue
        r = evaluate_calibration(pair_measurements(se, ts, poses, idx), cal)
        print(f'[SE] {name:8s}: forward step median |err| {r["dx_median_abs"]:.3f} m, bias {r["dx_rel_bias_pct"]:+.2f}%, '
              f'distance drift {r["dist_drift_pct"]:+.2f}%, dyaw median |err| {r["dyaw_median_abs_deg"]:.3f} deg')
    if abs(cal['wz_scale']) < 0.5 or abs(cal['wz_scale']) > 2.0 or not 0.8 < cal['v_scale'] < 1.25:
        print('[SE][WARN] calibration looks implausible (sign / unit mismatch?) - check the fields before using it')
    calib = dict(ck.get('calib', {}) or {})
    calib['se'] = {**DEFAULT_SE_CALIB, **{k: cal[k] for k in ('v_scale', 'wz_scale', 'wz_bias', 'dx_std', 'dyaw_std')}}
    ck['calib'] = calib


def calibrate_odometry(ck, args, device):
    vs = int(ck.get('val_start_timestamp_ns', 0))
    if vs <= 0:
        print('[ODOM] checkpoint has no validation split (trained with --full-train): skipped')
        return
    _, odom = load_networks(ck, device)
    with h5py.File(args.data_h5, 'r') as f:
        ts, images, poses = f['timestamps_ns'][:], f['images'][:], f['poses'][:]
    val_idx = np.flatnonzero(ts >= vs)
    pred, gt = predict_odom(odom, OdomDataset(images, poses, ts, val_idx, augment=False, preproc=ck.get('preproc')),
                            device)
    xs, ys, std, rep = odom_bias_curve(pred[:, 0], gt[:, 0])
    print(f'[ODOM] {len(pred)} held-out frame pairs')
    print('  predicted dx   ~speed    true dx   true/pred')
    for px, v, gy, r in rep['table']:
        print(f'  {px:8.2f} m  {v:6.1f} m/s  {gy:8.2f} m  {r:8.2f}')
    print(f'[ODOM] forward step: raw median |err| {rep["raw_median_abs"]:.3f} m, median bias {rep["raw_bias_pct"]:+.1f}% '
          f'-> calibrated median |err| {rep["cal_median_abs"]:.3f} m')
    calib = dict(ck.get('calib', {}) or {})
    calib['dx_map_pred'], calib['dx_map_true'], calib['odom_dx_std'] = xs, ys, std
    ck['calib'] = calib


def build_map(ck, args):
    db = ck['db_poses'].cpu().numpy().astype(np.float64)           # time-ordered training frames
    if args.mcap:
        markers = read_track_bounds_mcap(args.mcap)
    else:
        with h5py.File(args.data_h5, 'r') as h5f:
            markers = load_markers_h5(h5f)
    if not markers:
        print('[TRACK][WARN] no track borders found: corridor falls back to +-lat_clamp around the reference lap')
    elif len(markers) == 1 and markers[0]['ns'] == 'legacy_flat':
        print('[TRACK][WARN] cache only has the flat border array; run 1_build_dataset.py --aux-only or use --mcap')
    calib = ck.get('calib', {}) or {}
    tm = build_track_map(db, markers, ds=args.ds, fallback_half_width=float(calib.get('lat_clamp', 7.0)))
    i = tm.info
    print(f'[TRACK] {tm.summary()}')
    print(f'[TRACK] reference lap: {i.get("n_laps")} laps, length {i.get("lap_len", float("nan")):.0f} m, p99 deviation '
          f'vs other laps {i.get("lap_p99_dev", float("nan")):.2f} m, DB poses on main track {i.get("db_on_main_pct", 0):.1f}%')
    for r in i.get('markers', []):
        print(f'  marker ns={r["ns"]!r:24s} id={r["id"]:4d} type={r["type"]:2d} points={r["n_points"]:6d} '
              f'-> {r["role"]:5s} (median n {r["median_n"]:+.1f} m)')
    print(f'[TRACK] heading spread p50/p90: {math.degrees(np.median(tm.head_std)):.1f}/'
          f'{math.degrees(np.percentile(tm.head_std, 90)):.1f} deg')
    ck['track_map'] = tm.to_dict()
    if args.plot:
        plot(tm, db[:, :2], markers, args.plot)


def plot(tm, db_xy, markers, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[PLOT] matplotlib not installed: skipped')
        return
    fig, ax = plt.subplots(figsize=(12, 12))
    for m in markers:
        p = marker_xy(m)
        ax.plot(p[:, 0], p[:, 1], '.', ms=0.6, color='0.6')
    ax.plot(db_xy[:, 0], db_xy[:, 1], ',', color='tab:blue', alpha=0.3, label='DB poses')
    ax.plot(*tm.to_xy(tm.s, tm.w_left).T, '-', lw=0.8, color='tab:green', label='left border (used)')
    ax.plot(*tm.to_xy(tm.s, -tm.w_right).T, '-', lw=0.8, color='tab:red', label='right border (used)')
    ax.plot(tm.xy[:, 0], tm.xy[:, 1], '-', lw=0.8, color='k', label='centerline')
    pit = np.isfinite(tm.pit_lo)
    if pit.any():
        ax.plot(*tm.to_xy(tm.s[pit], tm.pit_lo[pit]).T, '.', ms=1.5, color='tab:orange', label='pit/run-off corridor')
        ax.plot(*tm.to_xy(tm.s[pit], tm.pit_hi[pit]).T, '.', ms=1.5, color='tab:orange')
    ax.plot(*tm.xy[0], 'k^', ms=10, label='s = 0')
    ax.set_aspect('equal')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f'[PLOT] wrote {path}')


def main(args, record):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not args.skip_odom:
        calibrate_odometry(ck, args, device)
    if not args.skip_se:
        calibrate_state_est(ck, args)
    if not args.skip_track:
        build_map(ck, args)
    provenance.append(ck, record)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(ck, args.output)
    print(f'[DONE] wrote {args.output}. Next: 4_infer.py')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/racing_model.pt')
    ap.add_argument('--output', default='checkpoints/racing_model_cal.pt')
    ap.add_argument('--data-h5', default='data/training_cache.h5')
    ap.add_argument('--mcap', default='', help='read the track borders from the bag instead of the cache')
    ap.add_argument('--skip-odom', action='store_true', help='do not (re)measure the visual odometry bias curve')
    ap.add_argument('--skip-se', action='store_true', help='do not (re)calibrate the state-estimation odometry')
    ap.add_argument('--skip-track', action='store_true', help='do not (re)build the track map')
    ap.add_argument('--ds', type=float, default=1.0, help='centerline station spacing [m]')
    ap.add_argument('--plot', default='', help='optional PNG of centerline / borders / pit corridor (matplotlib)')
    a = ap.parse_args()
    rec = provenance.run_record('3_calibrate', a, inputs={'checkpoint': a.checkpoint, 'data_h5': a.data_h5,
                                                          'mcap': a.mcap})
    print(provenance.banner(rec))
    main(a, rec)
