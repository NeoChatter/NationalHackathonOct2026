#!/usr/bin/env python3
"""
Trajectory visualisation, error analysis and metrics for a pose file (the jury asked for a visualisation).

With ground truth (validation laps):
    python tools/visualize.py --pred submission.csv --gt gt.csv \
        --checkpoint checkpoints/racing_model.pt --telemetry inference_telemetry.csv --out-dir results/figures

Without ground truth (a run on a recording that has no pose topic): omit --gt. The plot then shows the estimate on
the track borders only, with no error numbers.

Writes into --out-dir:
    trajectory.png        track borders + ground truth + estimate, estimate coloured by position error
    error_timeline.png    position / longitudinal / lateral / yaw error over time, with filter mode and similarity
    error_distribution.png  CDF + histogram of the position error
    trajectory.mp4|gif    (with --animate) the car driving the lap
    metrics.txt/json      same metrics as tools/evaluate.py, ready to commit as "validation results"
"""
import argparse
import json
import os

import numpy as np

MAX_GT_GAP_NS = 50_000_000
BIN_NS = 1_000_000_000


def load_csv(path):
    d = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding=None)
    out = {}
    for name in d.dtype.names:
        try:
            out[name] = np.asarray(d[name], dtype=np.int64 if name == 'timestamp_ns' else float)
        except (ValueError, TypeError):
            out[name] = np.asarray(d[name])
    order = np.argsort(out['timestamp_ns'])
    return {k: v[order] for k, v in out.items()}


def interpolate_gt(gt, ts):
    """Same rule as tools/evaluate.py: linear interpolation, skip stamps with a GT gap > 50 ms."""
    tg = gt['timestamp_ns']
    i = np.searchsorted(tg, ts)
    ok = (i > 0) & (i < len(tg))
    i = np.clip(i, 1, len(tg) - 1)
    ok &= (tg[i] - tg[i - 1]) <= MAX_GT_GAP_NS
    a = (ts - tg[i - 1]) / np.maximum(tg[i] - tg[i - 1], 1)
    yaw_u = np.unwrap(gt['yaw'])
    gx = gt['x'][i - 1] + a * (gt['x'][i] - gt['x'][i - 1])
    gy = gt['y'][i - 1] + a * (gt['y'][i] - gt['y'][i - 1])
    gyaw = yaw_u[i - 1] + a * (yaw_u[i] - yaw_u[i - 1])
    return ok, gx, gy, gyaw


def load_borders(checkpoint):
    if not checkpoint:
        return None
    try:
        import torch
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
        b = ckpt.get('track_borders')
        return b.numpy()[:, :2] if b is not None and len(b) else None
    except Exception as exc:  # noqa: BLE001
        print(f'[WARN] could not read track borders from {checkpoint}: {exc!r}')
        return None


def rms(v):
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else float('nan')


def compute_metrics(ts, pos, lon, lat, dyaw_deg, gt_t0, gt_t1):
    bins = np.arange(gt_t0, gt_t1, BIN_NS)
    covered = np.unique(np.searchsorted(bins, ts, side='right') - 1)
    m = dict(
        poses_evaluated=int(len(ts)),
        coverage_pct=float(100.0 * len(covered) / max(len(bins), 1)),
        position_rmse_m=rms(pos),
        position_median_m=float(np.median(pos)),
        position_p95_m=float(np.percentile(pos, 95)),
        lateral_rmse_m=rms(lat),
        longitudinal_rmse_m=rms(lon),
        yaw_rmse_deg=rms(dyaw_deg),
    )
    for thr in (0.25, 0.5, 1.0, 2.0, 5.0):
        m[f'within_{thr:g}m_pct'] = float(100.0 * np.mean(pos < thr))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', required=True, help='submission csv: timestamp_ns,x,y,yaw')
    ap.add_argument('--gt', default=None, help='ground truth csv from tools/extract_gt.py')
    ap.add_argument('--checkpoint', default=None, help='checkpoint (only used to draw the track borders)')
    ap.add_argument('--telemetry', default=None, help='inference_telemetry.csv (adds filter mode + similarity)')
    ap.add_argument('--out-dir', default='results/figures')
    ap.add_argument('--animate', action='store_true', help='also render the lap as mp4 (gif if ffmpeg is missing)')
    ap.add_argument('--title', default='')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    pred = load_csv(args.pred)
    ts = pred['timestamp_ns']
    px, py, pyaw = pred['x'], pred['y'], pred['yaw']
    t = (ts - ts[0]) * 1e-9
    borders = load_borders(args.checkpoint)
    tele = load_csv(args.telemetry) if args.telemetry else None
    title = args.title or os.path.basename(args.pred)

    have_gt = args.gt is not None
    metrics = {}
    if have_gt:
        gt = load_csv(args.gt)
        ok, gx, gy, gyaw = interpolate_gt(gt, ts)
        dx, dy = px - gx, py - gy
        lon = np.cos(gyaw) * dx + np.sin(gyaw) * dy
        lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy
        pos = np.hypot(dx, dy)
        dyaw = np.degrees(np.angle(np.exp(1j * (pyaw - gyaw))))
        # Coverage is measured over the span that was actually predicted, so a validation block (a few laps of a
        # longer bag) is not penalized; for a full-bag run this equals the evaluator's window.
        window = (max(int(gt['timestamp_ns'][0]), int(ts[0])), min(int(gt['timestamp_ns'][-1]), int(ts[-1])))
        metrics = compute_metrics(ts[ok], pos[ok], lon[ok], lat[ok], dyaw[ok], window[0], window[1])
        ev = ok
    else:
        gx = gy = pos = lon = lat = dyaw = None
        ev = np.ones(len(ts), dtype=bool)

    # ---- 1. trajectory --------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 9))
    if borders is not None:
        ax.scatter(borders[:, 0], borders[:, 1], s=1.5, c='0.7', label='track borders (map prior)', zorder=1)
    if have_gt:
        ax.plot(gt['x'], gt['y'], 'k-', lw=1.0, alpha=0.7, label='ground truth', zorder=2)
        vmax = max(float(np.percentile(pos[ev], 95)), 1.0)
        sc = ax.scatter(px[ev], py[ev], c=pos[ev], s=6, cmap='turbo', vmin=0, vmax=vmax, zorder=3,
                        label='estimate (colour = position error)')
        cb = fig.colorbar(sc, ax=ax, shrink=0.75)
        cb.set_label(f'position error [m] (colour scale capped at the 95th percentile, {vmax:.1f} m)')
    else:
        sc = ax.scatter(px, py, c=t, s=6, cmap='viridis', zorder=3, label='estimate (colour = time)')
        fig.colorbar(sc, ax=ax, shrink=0.75).set_label('time [s]')
    ax.scatter([px[0]], [py[0]], c='lime', s=90, edgecolors='k', zorder=5, label='start')
    ax.scatter([px[-1]], [py[-1]], c='red', s=90, edgecolors='k', zorder=5, label='end')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m] (map frame)')
    ax.set_ylabel('y [m] (map frame)')
    sub = f'position RMSE {metrics["position_rmse_m"]:.2f} m, median {metrics["position_median_m"]:.2f} m' if have_gt else 'no ground truth'
    ax.set_title(f'{title}\n{sub}')
    ax.legend(loc='best', fontsize=8, markerscale=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'trajectory.png'), dpi=150)
    plt.close(fig)

    # ---- 2. error over time -----------------------------------------------------------------------------------
    n_rows = 4 if have_gt else 1
    fig, axs = plt.subplots(n_rows, 1, figsize=(13, 2.6 * n_rows + 1), sharex=True, squeeze=False)
    axs = axs[:, 0]
    if have_gt:
        axs[0].plot(t[ev], pos[ev], 'C3', lw=1)
        axs[0].axhline(metrics['position_rmse_m'], color='k', ls='--', lw=0.8, label=f'RMSE {metrics["position_rmse_m"]:.2f} m')
        axs[0].set_ylabel('position error [m]')
        axs[0].legend(fontsize=8)
        axs[1].plot(t[ev], lon[ev], 'C0', lw=0.9, label='longitudinal (along heading)')
        axs[1].plot(t[ev], lat[ev], 'C1', lw=0.9, label='lateral (across heading)')
        axs[1].axhline(0, color='k', lw=0.5)
        axs[1].set_ylabel('error [m]')
        axs[1].legend(fontsize=8, ncol=2)
        axs[2].plot(t[ev], dyaw[ev], 'C2', lw=0.9)
        axs[2].axhline(0, color='k', lw=0.5)
        axs[2].set_ylabel('yaw error [deg]')
        sim_ax = axs[3]
    else:
        sim_ax = axs[0]
    if tele is not None and 'top1_sim' in tele:
        tt = (tele['timestamp_ns'] - ts[0]) * 1e-9
        sim_ax.plot(tt, tele['global_sim'] if 'global_sim' in tele else tele['top1_sim'], 'C0', lw=0.9, label='best similarity to the map')
        sim_ax.set_ylabel('similarity')
        sim_ax.set_ylim(0.3, 1.0)
        if 'still' in tele:
            still = tele['still'] == 1
            sim_ax.fill_between(tt, 0.3, 1.0, where=still, color='green', alpha=0.15, label='standstill hold')
        if 'mode' in tele:
            reloc = np.array([m == 'RELOCALIZED' for m in tele['mode']])
            sim_ax.fill_between(tt, 0.3, 1.0, where=reloc, color='red', alpha=0.3, label='global relocalization')
            vo_only = np.array([m == 'VO_ONLY' for m in tele['mode']])
            sim_ax.fill_between(tt, 0.3, 1.0, where=vo_only, color='gray', alpha=0.15, label='odometry only (no map match)')
        sim_ax.legend(fontsize=8, ncol=3, loc='lower left')
    else:
        sim_ax.text(0.5, 0.5, 'pass --telemetry to see similarity and filter mode', ha='center', va='center',
                    transform=sim_ax.transAxes)
    sim_ax.set_xlabel('time [s]')
    for a in axs:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'error_timeline.png'), dpi=140)
    plt.close(fig)

    # ---- 3. distribution -----------------------------------------------------------------------------------------
    if have_gt:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        s = np.sort(pos[ev])
        ax[0].plot(s, 100 * np.arange(1, len(s) + 1) / len(s), 'C3')
        for thr in (0.5, 1.0, 2.0, 5.0):
            ax[0].axvline(thr, color='0.6', lw=0.6)
            ax[0].text(thr, 3, f'{metrics[f"within_{thr:g}m_pct"]:.0f}%', fontsize=8, ha='left')
        ax[0].set_xscale('log')
        ax[0].set_xlabel('position error [m] (log scale)')
        ax[0].set_ylabel('share of poses with error below x [%]')
        ax[0].set_title('Cumulative error distribution')
        ax[0].grid(alpha=0.3, which='both')
        ax[1].hist(pos[ev], bins=np.linspace(0, max(float(np.percentile(pos[ev], 99)), 1.0), 60), color='C3', alpha=0.8)
        ax[1].set_xlabel('position error [m]')
        ax[1].set_ylabel('poses')
        ax[1].set_title(f'median {metrics["position_median_m"]:.2f} m | p95 {metrics["position_p95_m"]:.2f} m')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, 'error_distribution.png'), dpi=140)
        plt.close(fig)

    # ---- metrics files -------------------------------------------------------------------------------------------
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out_dir, 'metrics.txt'), 'w') as f:
        f.write(f'pose file: {args.pred}\n')
        for k, v in metrics.items():
            f.write(f'{k:32s} {v:.4f}\n' if isinstance(v, float) else f'{k:32s} {v}\n')
    print('\n'.join(f'{k:32s} {v:.4f}' if isinstance(v, float) else f'{k:32s} {v}' for k, v in metrics.items()))

    # ---- optional animation ------------------------------------------------------------------------------------------
    if args.animate:
        from matplotlib.animation import FuncAnimation
        fig, ax = plt.subplots(figsize=(8, 8))
        if borders is not None:
            ax.scatter(borders[:, 0], borders[:, 1], s=1, c='0.75')
        if have_gt:
            ax.plot(gt['x'], gt['y'], 'k-', lw=0.6, alpha=0.4)
        trail, = ax.plot([], [], 'C3-', lw=1.5, label='estimate')
        dot_p, = ax.plot([], [], 'C3o', ms=9)
        dot_g, = ax.plot([], [], 'ks', ms=7, label='ground truth') if have_gt else (None,)
        txt = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top', fontsize=10)
        ax.set_aspect('equal')
        ax.legend(loc='lower right', fontsize=8)
        stride = max(1, len(ts) // 600)
        frames = list(range(0, len(ts), stride))

        def update(k):
            i = frames[k]
            trail.set_data(px[:i + 1], py[:i + 1])
            dot_p.set_data([px[i]], [py[i]])
            if have_gt and ok[i]:
                dot_g.set_data([gx[i]], [gy[i]])
                txt.set_text(f't = {t[i]:5.1f} s   error = {pos[i]:5.2f} m')
            else:
                txt.set_text(f't = {t[i]:5.1f} s')
            return trail, dot_p, txt

        anim = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
        out = os.path.join(args.out_dir, 'trajectory.mp4')
        try:
            anim.save(out, writer='ffmpeg', fps=20)
        except Exception:  # noqa: BLE001
            out = os.path.join(args.out_dir, 'trajectory.gif')
            anim.save(out, writer='pillow', fps=20)
        plt.close(fig)
        print(f'animation: {out}')

    print(f'\nfigures written to {args.out_dir}')


if __name__ == '__main__':
    main()
