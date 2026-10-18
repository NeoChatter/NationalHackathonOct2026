"""
STAGE 6 - robustness suite: run 4_infer.py --val-only under perturbations / kidnaps and tabulate the degradation.

  python Scripts/6_stress_test.py --mcap Data/<bag>.mcap --checkpoint checkpoints/robust_cal.pt --gt data/gt.csv \
         --out stress/robust
  # A/B: the milestone-2 behaviour on the same scenarios
  python Scripts/6_stress_test.py ... --out stress/m2 --extra "--reloc legacy --no-adaptive-sim --no-memory"
  # subsets: --only clean,overcast2,overcast2_kidnap   or   --group weather

Images are perturbed, the database is not: "another condition against the recorded map". The scenarios cover a
balanced matrix of generic weather / time-of-day / image-quality / sensor faults. They approximate conditions as a
robustness check; they do not reproduce any specific run and none is tuned against evaluation data.

Per scenario: score (same rules as tools/evaluate.py), x clean, % frames lost, re-seeds, memory size, median
global top-1 similarity and its shift vs clean, kidnap recovery time. Writes <out>/<scenario>.csv (+ telemetry,
log, meta) and <out>/summary.csv / summary.md / summary.meta.json.
"""
import argparse
import math
import os
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # make `lib` importable from any CWD

from lib import provenance  # noqa: E402
from lib.metrics import align, load_gt, recovery, score  # noqa: E402

# name -> (group, perturb spec, kidnap spec[, extra 4_infer args]). A balanced matrix: no condition is weighted
# over any other. Every run starts WITHOUT a start pose (global initialisation), as on the test run.
SCENARIOS = {
    'clean': ('base', '', ''),
    'clean_vo': ('base', '', '', '--odom vo'),                # camera-only odometry: what state estimation buys
    # weather / time of day (both brighter and darker; diffuse and harsh)
    'overcast2': ('weather', 'overcast:2', ''),
    'overcast3': ('weather', 'overcast:3', ''),
    'low_sun2': ('weather', 'low_sun:2', ''),
    'daylight2': ('weather', 'daylight:2', ''),
    'rain2': ('weather', 'rain:2', ''),
    'haze2': ('weather', 'haze:2', ''),
    'overcast2_kidnap': ('weather', 'overcast:2', '300:80'),
    # illumination level / local light (symmetric: bright AND dark)
    'bright2': ('light', 'bright:2', ''),
    'dark2': ('light', 'dark:2', ''),
    'gamma2': ('light', 'gamma:2', ''),
    'contrast2': ('light', 'contrast:2', ''),
    'clip2': ('light', 'clip:2', ''),
    'glare2': ('light', 'glare:2', ''),
    'shadow2': ('light', 'shadow:2', ''),
    'sky2': ('light', 'sky:2', ''),
    'vignette2': ('light', 'vignette:2', ''),
    # image quality
    'blur2': ('quality', 'blur:2', ''),
    'noise2': ('quality', 'noise:2', ''),
    'jpeg2': ('quality', 'jpeg:2', ''),
    # sensor faults
    'cam_drop1': ('sensor', 'cam_drop:1', ''),
    'cam_freeze1': ('sensor', 'cam_freeze:1', ''),
    'frame_drop2': ('sensor', 'frame_drop:2', ''),
    # confidently-wrong recovery
    'kidnap': ('kidnap', '', '300:80'),
    'kidnap_far': ('kidnap', '', '400:250'),
}
# ~1/3 of the time: one scenario per family
QUICK = ['clean', 'clean_vo', 'overcast2', 'low_sun2', 'rain2', 'daylight2', 'contrast2', 'noise2', 'cam_drop1',
         'kidnap']
RECOVER_MAX_S = 30.0


def evaluate(gt, sub, tel) -> dict:
    al = align(gt, sub)
    s = score(al)
    mode = tel['mode'].values[np.minimum(al['rows'], len(tel) - 1)]
    init = tel['mode'].values == 'INITIALIZING'
    out = dict(rmse=s['rmse'], lat=s['lat'], lon=s['lon'], yaw_deg=s['yaw_deg'], p95=s['p95'],
               coverage=s['coverage'], lost_pct=float(100.0 * np.mean(mode == 'RELOCALIZING')),
               init_s=float(init.sum() / 10.0) if not init.all() else float('inf'),
               reseeds=int(tel['reseed'].sum()) if 'reseed' in tel else 0,
               mem_size=int(tel['mem_size'].max()) if 'mem_size' in tel else 0,
               top1_global_med=float(np.nanmedian(tel['top1_global'])) if 'top1_global' in tel else float('nan'))
    if 'kidnap' in tel and tel['kidnap'].sum() > 0:
        err = np.full(len(tel), np.inf)                        # error per telemetry row (= per submission row)
        err[al['rows']] = al['err']
        rec = recovery(err, tel['kidnap'].values)
        out.update(kidnaps=len(rec), recover_med_s=float(np.median(rec)),
                   recovered_pct=float(100.0 * np.mean(rec <= RECOVER_MAX_S)))
    return out


def main(args, record):
    infer = os.path.join(os.path.dirname(os.path.abspath(__file__)), '4_infer.py')
    os.makedirs(args.out, exist_ok=True)
    if args.only:
        names = [s.strip() for s in args.only.split(',') if s.strip()]
    elif args.group:
        names = [n for n, spec in SCENARIOS.items() if spec[0] in args.group.split(',') or n == 'clean']
    elif args.quick:
        names = list(QUICK)
    else:
        names = list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f'unknown scenarios {unknown}; choose from {list(SCENARIOS)}')

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    gt = load_gt(args.gt, int(ck.get('val_start_timestamp_ns', 0)))   # all runs are --val-only
    del ck

    rows = []
    for name in names:
        group, pert, kid = SCENARIOS[name][:3]
        scen_extra = SCENARIOS[name][3] if len(SCENARIOS[name]) > 3 else ''
        sub_csv = os.path.join(args.out, f'{name}.csv')
        tel_csv = os.path.join(args.out, f'{name}_telemetry.csv')
        if not (args.skip_existing and os.path.exists(sub_csv) and os.path.exists(tel_csv)):
            cmd = [sys.executable, infer, args.mcap, '--checkpoint', args.checkpoint, '--val-only',
                   '--output', sub_csv, '--log', tel_csv, '--perturb-seed', str(args.seed)]
            if pert:
                cmd += ['--perturb', pert]
            if kid:
                cmd += ['--kidnap', kid]
            cmd += shlex.split(scen_extra) + shlex.split(args.extra)
            print(f'\n[RUN] {name}: {" ".join(cmd)}', flush=True)
            with open(os.path.join(args.out, f'{name}.log'), 'w') as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                print(f'[FAIL] {name} (exit {rc}), see {name}.log')
                rows.append(dict(scenario=name, group=group, perturb=pert, kidnap=kid, error=f'exit {rc}'))
                continue
        r = evaluate(gt, pd.read_csv(sub_csv), pd.read_csv(tel_csv))
        rows.append(dict(scenario=name, group=group, perturb=pert, kidnap=kid, **r))
        print(f'[OK] {name:14s} RMSE {r["rmse"]:7.2f}  lat {r["lat"]:6.2f}  lon {r["lon"]:6.2f}  init {r["init_s"]:5.1f}s  '
              f'lost {r["lost_pct"]:5.1f}%  '
              f'reseeds {r["reseeds"]:3d}  mem {r["mem_size"]:5d}  top1 {r["top1_global_med"]:.3f}'
              + (f'  recover {r["recover_med_s"]:.1f}s ({r["recovered_pct"]:.0f}%)' if 'kidnaps' in r else ''), flush=True)

    df = pd.DataFrame(rows)
    if 'rmse' in df and (df.scenario == 'clean').any():
        base = df.loc[df.scenario == 'clean'].iloc[0]
        df['rmse_x_clean'] = df['rmse'] / base['rmse']
        df['top1_shift'] = df['top1_global_med'] - base['top1_global_med']
    df.to_csv(os.path.join(args.out, 'summary.csv'), index=False)
    cols = [c for c in ['scenario', 'group', 'rmse', 'rmse_x_clean', 'lat', 'lon', 'yaw_deg', 'p95', 'init_s', 'lost_pct',
                        'reseeds', 'mem_size', 'top1_global_med', 'top1_shift', 'recover_med_s', 'recovered_pct',
                        'error'] if c in df]
    with open(os.path.join(args.out, 'summary.md'), 'w') as f:
        f.write(f'# Stress suite\n\ncheckpoint `{args.checkpoint}`, extra args `{args.extra or "-"}`, '
                f'git `{record["git"].get("describe")}`\n\n')
        try:
            f.write(df[cols].to_markdown(index=False, floatfmt='.3g'))
        except ImportError:                                   # to_markdown needs the optional 'tabulate'
            f.write('```\n' + df[cols].to_string(index=False) + '\n```')
        f.write('\n')
    provenance.write_meta(os.path.join(args.out, 'summary'), record,
                          torch.load(args.checkpoint, map_location='cpu', weights_only=False).get('provenance'))
    print('\n' + df[cols].to_string(index=False, float_format=lambda v: f'{v:.3g}' if math.isfinite(v) else str(v)))
    print(f'\n[DONE] {os.path.join(args.out, "summary.csv")}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mcap', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--gt', required=True, help='GT CSV (full or validation part; cropped to the validation window)')
    ap.add_argument('--out', default='stress/run')
    ap.add_argument('--only', default='', help='comma-separated scenario names')
    ap.add_argument('--group', default='', help='comma-separated groups: base,weather,light,quality,sensor,kidnap')
    ap.add_argument('--quick', action='store_true', help=f'one scenario per family: {",".join(QUICK)}')
    ap.add_argument('--extra', default='', help='extra args for 4_infer.py, e.g. "--reloc legacy --no-memory"')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip-existing', action='store_true', help='re-score existing outputs instead of re-running')
    a = ap.parse_args()
    rec = provenance.run_record('6_stress_test', a, inputs={'mcap': a.mcap, 'checkpoint': a.checkpoint, 'gt': a.gt})
    print(provenance.banner(rec))
    main(a, rec)
