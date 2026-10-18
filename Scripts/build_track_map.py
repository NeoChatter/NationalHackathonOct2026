"""
Add a track map (centerline, real borders, pit corridor, heading prior) to an EXISTING checkpoint. No retraining.

  # borders from the HDF5 cache (after: python Scripts/1_build_dataset.py <bag> --output <h5> --borders-only)
  python Scripts/build_track_map.py --data-h5 data/training_cache.h5 \
         --checkpoint checkpoints/racing_model_cal.pt --output checkpoints/racing_model_track.pt

  # or straight from the bag (reads only the track_bounds topic)
  python Scripts/build_track_map.py --mcap data/c0_2026-10-13_22-53-42_camloc_0.mcap \
         --checkpoint checkpoints/racing_model_cal.pt --output checkpoints/racing_model_track.pt --plot track_map.png

The map is built from ck['db_poses'] (the training frames of the place database), so --val-only stays clean.
"""
import argparse
import math
import os

import h5py
import numpy as np
import torch

from track_map import build_track_map, load_markers_h5, read_track_bounds_mcap


def plot(tm, db_xy, markers, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[PLOT] matplotlib not installed: skipped')
        return
    from track_map import marker_xy
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


def main(args):
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    db = ck['db_poses'].cpu().numpy().astype(np.float64)           # [x, y, yaw], time-ordered training frames
    if args.mcap:
        markers = read_track_bounds_mcap(args.mcap)
    else:
        with h5py.File(args.data_h5, 'r') as h5f:
            markers = load_markers_h5(h5f)
    if not markers:
        print('[WARN] no track borders found: the corridor falls back to +-lat_clamp around the reference lap')
    elif len(markers) == 1 and markers[0]['ns'] == 'legacy_flat':
        print('[WARN] cache only has the flat border array (line structure lost). Run 1_build_dataset.py '
              '--borders-only or pass --mcap for per-marker borders.')

    calib = ck.get('calib', {}) or {}
    tm = build_track_map(db, markers, ds=args.ds, fallback_half_width=float(calib.get('lat_clamp', 7.0)))
    print(f'[TRACK] {tm.summary()}')
    print(f'[TRACK] reference lap: {tm.info.get("n_laps")} laps found, lap length {tm.info.get("lap_len", float("nan")):.0f} m, '
          f'p99 deviation vs other laps {tm.info.get("lap_p99_dev", float("nan")):.2f} m, '
          f'DB poses on main track {tm.info.get("db_on_main_pct", float("nan")):.1f}%')
    for r in tm.info.get('markers', []):
        print(f'  marker ns={r["ns"]!r:24s} id={r["id"]:4d} type={r["type"]:2d} points={r["n_points"]:6d} '
              f'-> {r["role"]:5s} (median n {r["median_n"]:+.1f} m)')
    print(f'[TRACK] heading spread (racing line vs tangent, lap-to-lap) p50/p90: '
          f'{math.degrees(np.median(tm.head_std)):.1f}/{math.degrees(np.percentile(tm.head_std, 90)):.1f} deg')

    ck['track_map'] = tm.to_dict()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(ck, args.output)
    print(f'[DONE] wrote {args.output}')
    if args.plot:
        plot(tm, db[:, :2], markers, args.plot)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/racing_model_cal.pt')
    ap.add_argument('--output', default='checkpoints/racing_model_track.pt')
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--data-h5', default='data/training_cache.h5')
    src.add_argument('--mcap', default='', help='read the borders straight from the bag instead of the cache')
    ap.add_argument('--ds', type=float, default=1.0, help='centerline station spacing [m]')
    ap.add_argument('--plot', default='', help='optional PNG of centerline / borders / pit corridor (matplotlib)')
    main(ap.parse_args())
