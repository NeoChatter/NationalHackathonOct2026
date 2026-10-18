"""
STAGE 5 - score a pose file and explain where the error comes from (one report, one JSON).

  python Scripts/5_evaluate.py --val-only        # results/poses/val_block_poses.csv vs [paths].gt_csv, val_checkpoint_cal
  python Scripts/5_evaluate.py --gt <test_gt.csv> --sub results/poses/test_poses.csv --checkpoint <ckpt>

  1. SCORE     same rules as tools/evaluate.py (position RMSE, lon / lat / yaw, coverage of 1 s bins)
  2. SLICES    share of the squared error by tracker mode (INITIALIZING / TRACKING / RELOCALIZING) and by block of
               frames (needs the telemetry: default <sub>_telemetry.csv)
  3. COVERAGE  is every evaluated pose near a database pose with a compatible heading? Lost frames in places the
               database never saw are a data problem, lost frames in covered places a relocalisation problem
  4. RUN       re-seeds, experience memory growth, top-1 similarity level

--val-only cuts the GT exactly like tools/gt_window.py (to the predicted span inside the checkpoint's validation
block), so the numbers equal `tools/gt_window.py` + `tools/evaluate.py`, i.e. the rmse-2.740 reference protocol.
Writes <sub>.eval.json with all numbers + the provenance of the pose file (its .meta.json).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # make `lib` importable from any CWD

from lib import config, provenance  # noqa: E402
from lib.metrics import align, crop_gt_to_span, db_coverage, format_score, load_gt, score, slices  # noqa: E402
from lib.track_map import TrackMap  # noqa: E402

UNMAPPED_M = 10.0


def main(args, record):
    ck = None
    if args.checkpoint:
        from lib.models import load_checkpoint                     # torch only when a checkpoint is given
        ck = load_checkpoint(args.checkpoint)
    if args.val_only and ck is None:
        raise SystemExit('--val-only needs --checkpoint (it holds the validation window)')
    gt = load_gt(args.gt)
    sub = pd.read_csv(args.sub)
    if len(sub) == 0:
        raise SystemExit(f'[ERROR] no poses in {args.sub}')
    window = None
    if args.val_only:
        vs = int(ck.get('val_start_timestamp_ns', 0) or 0)
        ve = int(ck.get('val_end_timestamp_ns', 0) or 0)
        t_first = max(int(sub.timestamp_ns.min()), vs)
        t_last = min(int(sub.timestamp_ns.max()), ve) if ve else int(sub.timestamp_ns.max())
        gt = crop_gt_to_span(gt, t_first, t_last)
        window = [t_first, t_last]
        print(f'[GT] cut to the predicted span of the validation block: {len(gt)} GT poses '
              f'({(gt.timestamp_ns.iloc[-1] - gt.timestamp_ns.iloc[0]) * 1e-9:.1f} s), same as tools/gt_window.py')
    tel_path = args.tel or os.path.splitext(args.sub)[0] + '_telemetry.csv'
    tel = pd.read_csv(tel_path) if os.path.exists(tel_path) else None

    al = align(gt, sub)
    sc = score(al)
    report = dict(score=sc, submission=os.path.abspath(args.sub), gt=os.path.abspath(args.gt), gt_window_ns=window)
    print('=== 1. SCORE ' + '=' * 57)
    print(format_score(sc))

    if tel is not None and len(al['err']):
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

    if ck is not None and len(al['err']):
        db = np.asarray(ck['db_poses'], dtype=np.float64)
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
        for col, fn in (('reseed', 'sum'), ('mem_event', None), ('top1_global', 'median'), ('mem_size', 'max'),
                        ('visual_update', 'mean')):
            if col in tel:
                v = pd.to_numeric(tel[col], errors='coerce')
                run[col] = float(getattr(v, fn)()) if fn else int((v > 0).sum())
        if 'odom_src' in tel:
            run.update({f'odom_{k}': int(v) for k, v in tel['odom_src'].fillna('-').value_counts().items()})
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
    cfg = config.bootstrap()
    provenance.configure(cfg)
    ap = config.parser(__doc__, cfg)
    ap.add_argument('--gt', default=cfg.path('gt_csv'), help='GT CSV (timestamp_ns,x,y,[z,]yaw); default [paths].gt_csv')
    ap.add_argument('--sub', default=None, help='pose CSV from 4_infer.py (default results/poses/val_block_poses.csv '
                                                'with --val-only)')
    ap.add_argument('--tel', default='', help='telemetry CSV (default: <sub>_telemetry.csv)')
    ap.add_argument('--checkpoint', default=None, help='enables the database coverage analysis and --val-only '
                                                       '(default [paths].val_checkpoint_cal with --val-only)')
    ap.add_argument('--val-only', action='store_true', help='cut the GT to the validation block (gt_window.py rule)')
    ap.add_argument('--block', type=int, default=int(cfg.get('evaluate', 'block', 200)))
    ap.add_argument('--json', default='', help='report path (default: <sub>.eval.json)')
    a = ap.parse_args()
    if a.sub is None:
        if not a.val_only:
            ap.error('--sub is required without --val-only')
        a.sub = os.path.join(cfg.path('results_dir'), 'poses', 'val_block_poses.csv')
    if a.checkpoint is None and a.val_only:
        a.checkpoint = cfg.path('val_checkpoint_cal')
    config.require_file(a.gt, 'ground-truth CSV', 'stage 1 writes it ([paths].gt_csv), or use tools/extract_gt.py')
    config.require_file(a.sub, 'pose file', 'run Scripts/4_infer.py first')
    rec = provenance.run_record('5_evaluate', a, inputs={'gt': a.gt, 'sub': a.sub, 'checkpoint': a.checkpoint or ''},
                                cfg=cfg)
    main(a, rec)
