"""
STAGE 5 - score a submission and explain where the error comes from (one report, one JSON).

  python Scripts/5_evaluate.py --gt data/gt.csv --sub runs/val.csv --checkpoint checkpoints/robust_cal.pt --val-only

  1. SCORE     same rules as tools/evaluate.py (position RMSE, lon / lat / yaw, coverage of 1 s bins)
  2. SLICES    share of the squared error by tracker mode (TRACKING / RELOCALIZING) and by block of frames
               (needs the telemetry: default <sub>_telemetry.csv)
  3. COVERAGE  is every evaluated pose near a database pose with a compatible heading? Lost frames in places the
               database never saw are a data problem, lost frames in covered places a relocalisation problem
               (needs --checkpoint)
  4. RUN       re-seeds, experience memory growth, top-1 similarity level (how far conditions moved the descriptor)

--val-only crops the GT to the checkpoint's validation window, so the full GT from
1_build_dataset.py --export-gt (or tools/extract_gt.py) can be used directly instead of a hand-made gt_val.csv.
Writes <sub>.eval.json with all numbers + the provenance of the submission (its .meta.json).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # make `lib` importable from any CWD

from lib import provenance  # noqa: E402
from lib.metrics import align, db_coverage, format_score, load_gt, score, slices  # noqa: E402
from lib.track_map import TrackMap

UNMAPPED_M = 10.0


def main(args, record):
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False) if args.checkpoint else None
    min_ns = int(ck.get('val_start_timestamp_ns', 0)) if (ck is not None and args.val_only) else 0
    if args.val_only and ck is None:
        raise SystemExit('--val-only needs --checkpoint (it holds the validation start time)')
    gt = load_gt(args.gt, min_ns)
    sub = pd.read_csv(args.sub)
    tel_path = args.tel or os.path.splitext(args.sub)[0] + '_telemetry.csv'
    tel = pd.read_csv(tel_path) if os.path.exists(tel_path) else None

    al = align(gt, sub)
    sc = score(al)
    report = dict(score=sc, submission=os.path.abspath(args.sub), gt=os.path.abspath(args.gt), gt_min_ns=min_ns)
    print('=== 1. SCORE ' + '=' * 57)
    print(format_score(sc))

    if tel is not None:
        sl = slices(al, tel, args.block)
        report['slices'] = sl
        print('\n=== 2. WHERE THE ERROR COMES FROM ' + '=' * 36)
        print('by mode:')
        for r in sl['by_mode']:
            print(f'  {r["label"]:13s} frames {r["frames"]:5d} ({r["pct_frames"]:4.1f}%)  RMSE {r["rmse"]:7.2f}  '
                  f'lon {r["lon"]:7.2f}  lat {r["lat"]:6.2f}  share of squared error {r["share_pct"]:5.1f}%')
        print(f'by block of {args.block} frames:')
        for r in sl['by_block']:
            print(f'  {r["label"]}  RMSE {r["rmse"]:7.2f}  lon {r["lon"]:7.2f}  lat {r["lat"]:6.2f}  share {r["share_pct"]:5.1f}%')
    else:
        print(f'\n(no telemetry at {tel_path}: slices skipped)')

    if ck is not None:
        db = ck['db_poses'].cpu().numpy().astype(np.float64)
        track = TrackMap.from_dict(ck['track_map']) if ck.get('track_map') is not None else None
        cov = db_coverage(al['gx'], al['gy'], al['gyaw'], db, track)
        unm = cov['near_head'] > UNMAPPED_M
        fin = cov['near_head'][np.isfinite(cov['near_head'])]
        report['coverage'] = dict(near_head_median=float(np.median(fin)) if len(fin) else None,
                                  unmapped_pct=float(100 * unm.mean()), off_main_pct=float(100 * cov['off_main'].mean()))
        print('\n=== 3. DATABASE COVERAGE OF THE EVALUATED POSES ' + '=' * 22)
        print(f'  nearest heading-compatible DB pose: median {report["coverage"]["near_head_median"]} m | '
              f'unmapped (> {UNMAPPED_M:g} m) {report["coverage"]["unmapped_pct"]:.1f}% | '
              f'off main track {report["coverage"]["off_main_pct"]:.1f}%')
        idx = np.flatnonzero(unm)
        runs = [r for r in (np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []) if len(r) >= 5]
        report['coverage']['unmapped_stretches'] = []
        for r in runs:
            a, b = int(r[0]), int(r[-1])
            item = dict(eval_rows=[a, b], seconds=(b - a + 1) / 10.0, off_main_pct=float(100 * cov['off_main'][a:b + 1].mean()))
            report['coverage']['unmapped_stretches'].append(item)
            print(f'  unmapped stretch: evaluated rows {a}-{b} ({item["seconds"]:.1f} s), off-main {item["off_main_pct"]:.0f}%')
        if tel is not None and 'mode' in tel:
            lost = tel['mode'].values[np.minimum(al['rows'], len(tel) - 1)] == 'RELOCALIZING'
            if lost.any():
                frac = float(unm[lost].mean())
                report['coverage']['lost_unmapped_pct'] = 100 * frac
                print(f'  verdict: {100 * frac:.0f}% of the lost frames are in places the database does not cover -> '
                      + ('mostly a COVERAGE problem (track-constrained dead reckoning / more data)' if frac > 0.5
                         else 'mostly a RELOCALISATION problem (the places are mapped)'))

    if tel is not None:
        run = {}
        for col, fn in (('reseed', 'sum'), ('mem_event', None), ('top1_global', 'median'), ('mem_size', 'max')):
            if col in tel:
                v = pd.to_numeric(tel[col], errors='coerce')
                run[col] = float(getattr(v, fn)()) if fn else int((v > 0).sum())
        report['run'] = run
        print('\n=== 4. RUN ' + '=' * 59)
        print('  ' + ' | '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in run.items()))

    report['submission_meta'] = provenance.read_meta(args.sub)
    report['evaluation'] = record
    out = args.json or args.sub + '.eval.json'
    with open(out, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\n[DONE] {out}')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True, help='GT CSV (timestamp_ns,x,y,[z,]yaw)')
    ap.add_argument('--sub', required=True, help='submission CSV from 4_infer.py')
    ap.add_argument('--tel', default='', help='telemetry CSV (default: <sub>_telemetry.csv)')
    ap.add_argument('--checkpoint', default='', help='enables the database coverage analysis and --val-only')
    ap.add_argument('--val-only', action='store_true', help='crop the GT to the checkpoint validation window')
    ap.add_argument('--block', type=int, default=200)
    ap.add_argument('--json', default='', help='report path (default: <sub>.eval.json)')
    a = ap.parse_args()
    rec = provenance.run_record('5_evaluate', a, inputs={'gt': a.gt, 'sub': a.sub, 'checkpoint': a.checkpoint})
    main(a, rec)
