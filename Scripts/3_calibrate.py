"""
STAGE 3 - post-training calibration of a checkpoint (no retraining). Every fit uses the TRAINING split only (the
checkpoint's database frames), never the validation block; the block is only used to print a held-out report.

  a) state estimation : speed scale, yaw-rate scale + bias and per-frame noise of the vehicle's speed / yaw-rate
                        signals (the only state-estimation fields used, see lib/state_est.py) -> calib['se']
  b) visual odometry  : speed-dependent forward-distance bias curve + noise of the learned VO (fallback odometry)
                        -> calib['dx_map_pred'], calib['dx_map_true'], calib['odom_dx_std'], calib['odom_dyaw_std']
  c) track map        : centre line, real borders, pit corridor, heading prior from the database poses and the
                        per-marker borders (from the cache, or from the bag with --mcap) -> ck['track_map']
  d) retrieval        : only if the checkpoint has no retrieval calibration yet (e.g. the rmse-2.740
                        racing_model_val.pt): same measurement as stage 2 ([calib] in config.toml)

  python Scripts/3_calibrate.py                 # [paths].val_checkpoint -> [paths].val_checkpoint_cal
  python Scripts/3_calibrate.py --full          # [paths].checkpoint     -> [paths].checkpoint_cal
  python Scripts/3_calibrate.py --checkpoint X.pt --output X_cal.pt --plot results/track_map.png

The cache must contain state_est/ (1_build_dataset.py, or --aux-only on an older cache).
"""
import argparse
import math
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # make `lib` importable from any CWD

from lib import config, provenance  # noqa: E402
from lib.data import eval_retrieval, gt_body_deltas, odom_bias_curve, predict_vo, robust_std, valid_pairs  # noqa: E402
from lib.models import build_model_from_checkpoint, load_checkpoint, pick_device, save_checkpoint  # noqa: E402
from lib.split import indices_from_checkpoint  # noqa: E402
from lib.state_est import DEFAULT_SE_CALIB, SE_FIELDS, evaluate_calibration, fit_calibration, pair_measurements  # noqa: E402
from lib.track_map import build_track_map, load_markers_h5, marker_xy, read_track_bounds_mcap  # noqa: E402

RETRIEVAL_KEYS = ('retr_along_std', 'retr_lat_std', 'retr_yaw_std', 'sim_thresh')


def _calib(ck) -> dict:
    return dict(ck.get('calib', {}) or {})


def calibrate_state_est(ck, args, ts, poses, train_idx, val_idx):
    with h5py.File(args.data_h5, 'r') as f:
        if 'state_est' not in f:
            print('[SE][WARN] cache has no state_est/: run 1_build_dataset.py <bag> --aux-only. Skipped.')
            return
        se = {'t_ns': f['state_est/t_ns'][:].astype(np.int64)}
        for k in SE_FIELDS:
            se[k] = f[f'state_est/{k}'][:]
    ok = np.isfinite(se['v_mps']) & np.isfinite(se['wz_radps'])
    se = {k: v[ok] for k, v in se.items()}
    o = np.argsort(se['t_ns'], kind='stable')
    se = {k: v[o] for k, v in se.items()}

    m_tr = pair_measurements(se, ts, poses, train_idx)
    if len(m_tr) < 200:
        print(f'[SE][WARN] only {len(m_tr)} usable training frame pairs: skipped')
        return
    cal = fit_calibration(m_tr)
    print(f'[SE] fitted on {cal["n_pairs"]} TRAINING frame pairs ({cal["n_moving"]} moving): v_scale {cal["v_scale"]:.4f}, '
          f'wz_scale {cal["wz_scale"]:.4f}, wz_bias {math.degrees(cal["wz_bias"]):+.3f} deg/s ({cal["wz_fit"]}), '
          f'noise dx {cal["dx_std"]:.3f} m / dyaw {math.degrees(cal["dyaw_std"]):.3f} deg per frame')
    if cal['wz_fit'] != 'scale+bias':
        print('[SE][WARN] yaw-rate scale fit rejected: check the yaw-rate field before trusting the calibration')
    for name, idx in (('train', train_idx), ('held-out', val_idx)):
        if len(idx) < 50:
            continue
        r = evaluate_calibration(pair_measurements(se, ts, poses, idx), cal)
        print(f'[SE] {name:8s}: forward step median |err| {r["dx_median_abs"]:.3f} m, bias {r["dx_rel_bias_pct"]:+.2f}%, '
              f'distance drift {r["dist_drift_pct"]:+.2f}%, dyaw median |err| {r["dyaw_median_abs_deg"]:.3f} deg'
              + ('  (report only, not fitted)' if name == 'held-out' else ''))
    if abs(cal['wz_scale']) < 0.5 or abs(cal['wz_scale']) > 2.0 or not 0.8 < cal['v_scale'] < 1.25:
        print('[SE][WARN] calibration looks implausible (sign / unit mismatch?) - check the fields before using it')
    calib = _calib(ck)
    calib['se'] = {**DEFAULT_SE_CALIB, **{k: cal[k] for k in ('v_scale', 'wz_scale', 'wz_bias', 'dx_std', 'dyaw_std')}}
    ck['calib'] = calib


def calibrate_odometry(ck, args, model, device, cache, train_idx):
    ts, images, poses, dt_offsets = cache['ts'], cache['images'], cache['poses'], cache['dt_offsets']
    pairs = valid_pairs(train_idx, ts, 0.05, 0.2)
    if len(pairs) > args.max_pairs:                        # evenly spread subsample keeps the stage fast
        pairs = pairs[np.linspace(0, len(pairs) - 1, args.max_pairs).astype(np.int64)]
    if len(pairs) < 50:
        print(f'[ODOM][WARN] only {len(pairs)} training frame pairs: skipped')
        return
    pred = predict_vo(model, images, dt_offsets, ts, pairs, device, ck.get('preproc'))
    gt = gt_body_deltas(poses, pairs).astype(np.float64)
    xs, ys, std, rep = odom_bias_curve(pred[:, 0], gt[:, 0])
    print(f'[ODOM] {len(pred)} TRAINING frame pairs (the VO trained on them, so this curve is close to identity)')
    print('  predicted dx   ~speed    true dx   true/pred')
    for px, v, gy, r in rep['table']:
        print(f'  {px:8.2f} m  {v:6.1f} m/s  {gy:8.2f} m  {r:8.2f}')
    print(f'[ODOM] forward step: raw median |err| {rep["raw_median_abs"]:.3f} m, median bias {rep["raw_bias_pct"]:+.1f}% '
          f'-> calibrated median |err| {rep["cal_median_abs"]:.3f} m')
    calib = _calib(ck)
    calib['dx_map_pred'], calib['dx_map_true'], calib['odom_dx_std'] = list(xs), list(ys), float(std)
    calib['odom_dyaw_std'] = float(max(robust_std(pred[:, 2] - gt[:, 2]), 0.002))
    ck['calib'] = calib


def calibrate_retrieval(ck, args, model, device, cache, train_idx):
    calib = _calib(ck)
    have = all(k in calib for k in RETRIEVAL_KEYS)
    if args.retrieval == 'never' or (args.retrieval == 'missing' and have):
        return
    ts, images, poses = cache['ts'], cache['images'], cache['poses']
    db_ts = np.asarray(ck['db_timestamps_ns'], dtype=np.int64)
    db_idx = np.searchsorted(ts, db_ts)
    if (db_idx >= len(ts)).any() or not np.array_equal(ts[np.minimum(db_idx, len(ts) - 1)], db_ts):
        raise SystemExit('[ERROR] the cache does not contain the checkpoint database frames (different cache?)')
    C = cfg_calib = args.calib_cfg
    q_idx = db_idx[::max(int(C.get('query_stride', 5)), 1)]
    rep, rc = eval_retrieval(model, images, poses, ts, db_idx, q_idx, device, ck.get('preproc'),
                             exclude_s=float(C.get('exclude_s', 20.0)), gate_m=float(C.get('gate_m', 35.0)),
                             db_emb=ck['db_embeddings'])
    calib.update(rc)
    calib.setdefault('lat_clamp', 7.0)
    ck['calib'] = calib
    ck['calib_info'] = dict(source='train_crosspass (stage 3)', report=rep, config=dict(cfg_calib))
    print(f'[RETRIEVAL] {"re-measured" if have else "checkpoint had no retrieval calibration -> measured"}: {rc} | {rep}')


def build_map(ck, args):
    db = np.asarray(ck['db_poses'], dtype=np.float64)              # time-ordered training frames [x, y, yaw]
    if args.mcap:
        markers = read_track_bounds_mcap(args.mcap)
    else:
        with h5py.File(args.data_h5, 'r') as h5f:
            markers = load_markers_h5(h5f)
    if not markers:
        print('[TRACK][WARN] no track borders found: corridor falls back to +-lat_clamp around the reference lap')
    elif len(markers) == 1 and markers[0]['ns'] == 'legacy_flat':
        print('[TRACK][WARN] cache only has the flat border array; run 1_build_dataset.py --aux-only or use --mcap')
    calib = _calib(ck)
    tm = build_track_map(db, markers, ds=args.ds, fallback_half_width=float(calib.get('lat_clamp', 7.0)),
                         min_lap_len=args.min_lap_len_m)
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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
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
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'[PLOT] wrote {path}')


def main(args, record):
    device = pick_device(args.device)
    ck = load_checkpoint(args.checkpoint)
    with h5py.File(args.data_h5, 'r') as f:
        cache = dict(ts=f['timestamps_ns'][:].astype(np.int64), poses=f['poses'][:])
        need_images = args.odom or args.retrieval != 'never'
        if need_images:
            cache.update(images=f['images'][:], dt_offsets=f['dt_offsets'][:])
    train_idx, val_idx = indices_from_checkpoint(ck, cache['ts'])
    print(f'[SPLIT] training frames {len(train_idx)} (fits) | validation frames {len(val_idx)} (report only)')
    if len(train_idx) == 0:
        raise SystemExit('[ERROR] no training frames of this checkpoint found in the cache (different cache?)')
    model = build_model_from_checkpoint(ck, device) if need_images else None

    if args.se:
        calibrate_state_est(ck, args, cache['ts'], cache['poses'], train_idx, val_idx)
    if args.odom:
        calibrate_odometry(ck, args, model, device, cache, train_idx)
    if args.retrieval != 'never':
        calibrate_retrieval(ck, args, model, device, cache, train_idx)
    if args.track:
        build_map(ck, args)
    provenance.append(ck, record)
    save_checkpoint(ck, args.output)
    print(f'[DONE] wrote {args.output}. Next: 4_infer.py')


if __name__ == '__main__':
    cfg = config.bootstrap()
    provenance.configure(cfg)
    K = cfg['calibrate']
    B = argparse.BooleanOptionalAction
    ap = config.parser(__doc__, cfg)
    ap.add_argument('--full', action='store_true', help='default to the submission checkpoint pair instead of the validation one')
    ap.add_argument('--checkpoint', default=None, help='input (default [paths].val_checkpoint, --full: [paths].checkpoint)')
    ap.add_argument('--output', default=None, help='output (default [paths].val_checkpoint_cal / checkpoint_cal)')
    ap.add_argument('--data-h5', default=cfg.path('cache'))
    ap.add_argument('--mcap', default='', help='read the track borders from this bag instead of the cache')
    ap.add_argument('--device', default=cfg.get('run', 'device', 'auto'))
    ap.add_argument('--se', action=B, default=bool(K.get('se', True)), help='state-estimation calibration')
    ap.add_argument('--odom', action=B, default=bool(K.get('odom', True)), help='visual-odometry bias curve')
    ap.add_argument('--track', action=B, default=bool(K.get('track', True)), help='(re)build the track map')
    ap.add_argument('--retrieval', choices=['missing', 'always', 'never'], default='missing',
                    help='retrieval calibration: only if the checkpoint has none (default), always, never')
    ap.add_argument('--max-pairs', type=int, default=int(K.get('vo_max_pairs', 4000)),
                    help='training frame pairs used for the VO bias curve (subsampled)')
    ap.add_argument('--ds', type=float, default=float(K.get('ds', 1.0)), help='centerline station spacing [m]')
    ap.add_argument('--min-lap-len-m', type=float, default=float(K.get('min_lap_len_m', 500.0)))
    ap.add_argument('--plot', default='', help='optional PNG of centerline / borders / pit corridor')
    a = ap.parse_args()
    a.calib_cfg = dict(cfg['calib'])
    if a.checkpoint is None:
        a.checkpoint = cfg.path('checkpoint') if a.full else cfg.path('val_checkpoint')
    if a.output is None:
        a.output = cfg.path('checkpoint_cal') if a.full else cfg.path('val_checkpoint_cal')
    config.require_file(a.checkpoint, 'checkpoint', 'run Scripts/2_train.py first')
    config.require_file(a.data_h5, 'cache', 'run Scripts/1_build_dataset.py first')
    rec = provenance.run_record('3_calibrate', a, inputs={'checkpoint': a.checkpoint, 'data_h5': a.data_h5,
                                                          'mcap': a.mcap}, cfg=cfg)
    print(provenance.banner(rec))
    main(a, rec)
