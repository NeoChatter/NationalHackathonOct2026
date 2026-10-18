#!/usr/bin/env python3
"""
Plot how training improved, epoch by epoch.

<<<<<<< HEAD
    python tools/plot_training.py checkpoints/racing_model_val_history.csv --out results/figures/training_progress.png

Reads the history CSV that Scripts/2_train.py rewrites after every epoch (so it also works on a run that is
=======
    python tools/plot_training.py checkpoints/racing_model_history.csv --out results/figures/training_progress.png

Reads the history CSV that Scripts/2_train_model.py rewrites after every epoch (so it also works on a run that is
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510
still going or crashed). Panels that need held-out metrics are skipped when the run used --full-train.
"""
import argparse
import csv
import math
import os

import numpy as np


def load(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for key in rows[0].keys():
        cols[key] = np.array([float(r[key]) if r[key] not in ('', None) else float('nan') for r in rows])
    return cols


def has(cols, key):
    return key in cols and np.isfinite(cols[key]).any()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('history_csv')
    ap.add_argument('--out', default='results/figures/training_progress.png')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    c = load(args.history_csv)
    ep = c['epoch']
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle('Training progress (every point is one epoch)', fontsize=14)

    # 1. training losses
    a = ax[0, 0]
    a.plot(ep, c['loss_contrastive'], label='place (contrastive)', color='C0')
    a.plot(ep, c['loss_aux'], label='aux pose', color='C2')
    a.set_ylabel('training loss')
    a2 = a.twinx()
    a2.plot(ep, c['loss_vo'], label='odometry (beta-NLL + relative)', color='C3')
    a2.set_ylabel('odometry loss')
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8)
    a.set_title('What the model minimizes')

    # 2. retrieval error on held-out laps
    a = ax[0, 1]
    if has(c, 'val_ret_median_m'):
        a.plot(ep, c['val_ret_median_m'], 'o-', color='C0', label='median error, clean query')
        if has(c, 'val_ret_median_aug_m'):
            a.plot(ep, c['val_ret_median_aug_m'], 's--', color='C1', label='median error, dusk-like lighting')
        a.set_ylabel('retrieval error [m]')
        a.legend(fontsize=8)
    else:
        a.text(0.5, 0.5, 'no held-out metrics\n(--full-train)', ha='center', va='center')
    a.set_title('Place retrieval accuracy (held-out laps, lower is better)')

    # 3. retrieval recall
    a = ax[0, 2]
    if has(c, 'val_ret_within5'):
        a.plot(ep, 100 * c['val_ret_within2'], 'o-', label='within 2 m', color='C2')
        a.plot(ep, 100 * c['val_ret_within5'], 'o-', label='within 5 m', color='C0')
        if has(c, 'val_ret_within5_aug'):
            a.plot(ep, 100 * c['val_ret_within5_aug'], 's--', label='within 5 m, dusk-like lighting', color='C1')
        a.set_ylabel('share of held-out frames [%]')
        a.set_ylim(0, 100)
        a.legend(fontsize=8)
    a.set_title('Retrieval recall (higher is better)')

    # 4. VO step error
    a = ax[1, 0]
    if has(c, 'val_vo_rmse_dx'):
        a.plot(ep, c['val_vo_rmse_dx'], 'o-', color='C3', label='forward dx [m]')
        a.plot(ep, c['val_vo_rmse_dy'], 'o-', color='C4', label='lateral dy [m]')
        a.set_ylabel('per-frame RMSE [m]')
        a2 = a.twinx()
        a2.plot(ep, c['val_vo_rmse_dyaw_deg'], 's--', color='C5', label='yaw [deg]')
        a2.set_ylabel('yaw RMSE [deg]')
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, fontsize=8)
    a.set_title('Odometry step error (held-out laps)')

    # 5. VO scale by speed
    a = ax[1, 1]
    if has(c, 'val_vo_scale'):
        a.axhline(1.0, color='k', lw=0.8)
        a.plot(ep, c['val_vo_scale'], 'o-', color='k', label='all speeds')
        for key, label in (('lt10', '< 10 m/s'), ('10_20', '10-20'), ('20_30', '20-30'), ('gt30', '> 30')):
            if has(c, f'val_vo_scale_{key}'):
                a.plot(ep, c[f'val_vo_scale_{key}'], '.-', label=label)
        a.set_ylabel('predicted / true distance')
        a.legend(fontsize=8, ncol=2)
    a.set_title('Odometry scale (1.0 is perfect)')

    # 6. standstill + similarity separation
    a = ax[1, 2]
    if has(c, 'val_sim_covered'):
        a.plot(ep, c['val_sim_covered'], 'o-', color='C0', label='top-1 similarity, on mapped track')
        if has(c, 'val_sim_uncovered'):
            a.plot(ep, c['val_sim_uncovered'], 'o-', color='C3', label='top-1 similarity, off the map')
        a.set_ylabel('cosine similarity')
        a.legend(fontsize=8, loc='lower left')
    if has(c, 'val_standstill_speed'):
        a2 = a.twinx()
        a2.plot(ep, c['val_standstill_speed'], 's--', color='C1', label='VO speed when parked [m/s]')
        a2.set_ylabel('speed reported while parked [m/s]')
        a2.legend(fontsize=8, loc='upper right')
    a.set_title('Does the map separate known from unknown places?')

    for a in ax.ravel():
        a.set_xlabel('epoch')
        a.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f'saved {args.out}')

    # Short text summary for the README / slides
    last = {k: v[-1] for k, v in c.items()}
    first_eval = next((i for i in range(len(ep)) if math.isfinite(c.get('val_ret_median_m', np.array([np.nan]))[i])), None)
    if first_eval is not None:
        print(f"\nretrieval median error: epoch {int(ep[first_eval])} {c['val_ret_median_m'][first_eval]:.2f} m -> "
              f"epoch {int(ep[-1])} {last['val_ret_median_m']:.2f} m")
        if math.isfinite(last.get('val_vo_rmse_dx', float('nan'))):
            print(f"VO forward step RMSE:   epoch {int(ep[first_eval])} {c['val_vo_rmse_dx'][first_eval]:.3f} m -> "
                  f"epoch {int(ep[-1])} {last['val_vo_rmse_dx']:.3f} m | final scale {last['val_vo_scale']:.3f}")


if __name__ == '__main__':
    main()
