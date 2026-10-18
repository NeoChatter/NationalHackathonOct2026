#!/usr/bin/env python3
"""
Single run script: raw bag -> validation score and/or submission pose file, in the original stage order.

  python run_submission.py smoke                         # 1st: synthetic end-to-end check of the whole chain (minutes)
  python run_submission.py validate                      # reproduce the validation number (2-lap block, seed 0)
  python run_submission.py submit --test-bag <TEST.mcap> # submission model (all frames) + test pose file
  python run_submission.py all --test-bag <TEST.mcap>    # validate, then submit

Options: --config PATH (default $RACING_CONFIG or config.toml), --dry-run (print the commands only),
         --rebuild (re-run stage 1 even if the cache exists), --from-stage N (validate / submit: skip stages < N),
         --stress (validate: also run the quick robustness suite, stage 6).

Every stage reads the same config file, so paths, seeds and hyperparameters live in one place (config.toml).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'Scripts'))

from lib import config  # noqa: E402

PY = sys.executable


def S(name):
    return os.path.join(HERE, 'Scripts', name)


def T(name):
    return os.path.join(HERE, 'tools', name)


class Runner:
    def __init__(self, cfg_file: str, dry_run: bool):
        self.cfg_file, self.dry = cfg_file, dry_run
        self.t0 = time.perf_counter()

    def run(self, label, cmd, tee=None):
        print(f'\n{"=" * 100}\n[{time.perf_counter() - self.t0:7.0f}s] {label}\n  $ {" ".join(cmd)}\n{"=" * 100}',
              flush=True)
        if self.dry:
            return ''
        env = dict(os.environ, PYTHONUNBUFFERED='1')               # live output when piped to tee / a log file
        if tee:
            res = subprocess.run(cmd, capture_output=True, text=True, env=env)
            print(res.stdout + res.stderr, flush=True)
            os.makedirs(os.path.dirname(tee) or '.', exist_ok=True)
            with open(tee, 'w') as f:
                f.write(res.stdout)
            out, rc = res.stdout, res.returncode
        else:
            rc, out = subprocess.run(cmd, env=env).returncode, ''
        if rc != 0:
            raise SystemExit(f'\n[FAILED] {label} (exit code {rc}). Fix the error above and re-run with '
                             f'--from-stage to continue.')
        return out


def stage1(r, cfg, args):
    cache = cfg.path('cache')
    if os.path.exists(cache) and os.path.exists(cfg.path('gt_csv')) and not args.rebuild:
        print(f'\n[SKIP] stage 1: {cache} and {cfg.path("gt_csv")} exist (use --rebuild to re-create them)')
        return
    r.run('STAGE 1  bag -> cache + ground-truth CSV', [PY, S('1_build_dataset.py'), '--config', r.cfg_file])


def validate(r, cfg, args):
    res = cfg.path('results_dir')
    poses = os.path.join(res, 'poses', 'val_block_poses.csv')
    tel = os.path.splitext(poses)[0] + '_telemetry.csv'
    hist = os.path.splitext(cfg.path('val_checkpoint'))[0] + '_history.csv'
    gt_cut = os.path.join(res, 'gt_val_block.csv')
    steps = [
        (1, lambda: stage1(r, cfg, args)),
        (2, lambda: r.run('STAGE 2  train the validation model (2 laps held out)',
                          [PY, S('2_train.py'), '--config', r.cfg_file])),
        (2, lambda: r.run('         training curves', [PY, T('plot_training.py'), hist,
                                                       '--out', os.path.join(res, 'figures', 'training_progress.png')])),
        (3, lambda: r.run('STAGE 3  calibrate (training split only) + track map',
                          [PY, S('3_calibrate.py'), '--config', r.cfg_file,
                           '--plot', os.path.join(res, 'figures', 'track_map.png')])),
        (4, lambda: r.run('STAGE 4  causal inference on the validation block',
                          [PY, S('4_infer.py'), '--config', r.cfg_file, '--val-only', '--output', poses])),
        (5, lambda: r.run('         organisers scorer: cut GT to the predicted span',
                          [PY, T('gt_window.py'), cfg.path('gt_csv'), poses, gt_cut])),
        (5, lambda: r.run('         organisers scorer: tools/evaluate.py  <- REPORTED RMSE',
                          [PY, T('evaluate.py'), gt_cut, poses], tee=os.path.join(res, 'validation', 'official_score.txt'))),
        (5, lambda: r.run('STAGE 5  error slices + database coverage',
                          [PY, S('5_evaluate.py'), '--config', r.cfg_file, '--val-only', '--sub', poses])),
        (5, lambda: r.run('         figures + metrics.txt/json',
                          [PY, T('visualize.py'), '--pred', poses, '--gt', cfg.path('gt_csv'),
                           '--checkpoint', cfg.path('val_checkpoint_cal'), '--telemetry', tel,
                           '--out-dir', os.path.join(res, 'validation')])),
    ]
    if args.stress:
        steps.append((6, lambda: r.run('STAGE 6  robustness suite (quick set)',
                                       [PY, S('6_stress_test.py'), '--config', r.cfg_file] + list(args.stress_args))))
    for stage, fn in steps:
        if stage >= args.from_stage:
            fn()
    return poses


def submit(r, cfg, args):
    test_bag = args.test_bag or cfg.path('test_bag')
    if not test_bag:
        raise SystemExit('[ERROR] submit needs the test recording: --test-bag PATH (or [paths].test_bag)')
    if not r.dry:
        config.require_file(test_bag, 'test bag')
    res = cfg.path('results_dir')
    poses = os.path.join(res, 'poses', 'test_poses.csv')
    steps = [
        (1, lambda: stage1(r, cfg, args)),
        (2, lambda: r.run('STAGE 2  train the submission model (all frames)',
                          [PY, S('2_train.py'), '--config', r.cfg_file, '--full-train'])),
        (3, lambda: r.run('STAGE 3  calibrate + track map (submission model)',
                          [PY, S('3_calibrate.py'), '--config', r.cfg_file, '--full'])),
        (4, lambda: r.run('STAGE 4  causal inference on the test recording -> SUBMISSION FILE',
                          [PY, S('4_infer.py'), test_bag, '--config', r.cfg_file, '--output', poses])),
        (5, lambda: r.run('         trajectory figure (no ground truth)',
                          [PY, T('visualize.py'), '--pred', poses, '--checkpoint', cfg.path('checkpoint_cal'),
                           '--telemetry', os.path.splitext(poses)[0] + '_telemetry.csv',
                           '--out-dir', os.path.join(res, 'test')])),
    ]
    for stage, fn in steps:
        if stage >= args.from_stage:
            fn()
    print(f'\n[SUBMISSION] {poses}  (timestamp_ns,x,y,yaw; one pose per camera_fl frame)')
    return poses


# ----------------------------------------------------------------------------------------------
# smoke test: the full chain on a synthetic bag with a tiny configuration
# ----------------------------------------------------------------------------------------------
SMOKE_OVERRIDES = """
[run]
seed = 0
[train]
epochs = 2
batch_size = 8
workers = 0
eval_every = 2
max_batches_per_epoch = 3
pretrained = false
[calib]
query_stride = 3
[calibrate]
min_lap_len_m = 300.0
[stress]
quick = true
[provenance]
full_sha256 = true
"""


def smoke(r, args):
    import csv
    base = os.path.abspath(args.config or os.environ.get(config.ENV_VAR) or config.DEFAULT_FILE)
    out = os.path.join(HERE, 'runs', 'smoke')
    if os.path.isdir(out) and not r.dry:
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    bag = os.path.join(out, 'synthetic.mcap')
    lit = lambda p: "'" + os.path.join(out, p) + "'"             # TOML literal string: no escaping on Windows
    with open(os.path.join(out, 'smoke.toml'), 'w') as f:
        f.write(f"extends = '{base}'\n\n[paths]\nroot = '{out}'\nbag = {lit('synthetic.mcap')}\n"
                f"test_bag = {lit('synthetic.mcap')}\ncache = {lit('cache.h5')}\ngt_csv = {lit('gt.csv')}\n"
                f"val_checkpoint = {lit('ckpt_val.pt')}\nval_checkpoint_cal = {lit('ckpt_val_cal.pt')}\n"
                f"checkpoint = {lit('ckpt_full.pt')}\ncheckpoint_cal = {lit('ckpt_full_cal.pt')}\n"
                f"results_dir = {lit('results')}\nruns_dir = {lit('runs')}\nstress_dir = {lit('stress')}\n"
                f"hash_cache = {lit('sha256_cache.json')}\n" + SMOKE_OVERRIDES)
    r.cfg_file = os.path.join(out, 'smoke.toml')
    cfg = config.load(r.cfg_file)
    r.run('SMOKE  synthetic bag (5 laps on a 628 m circle)', [PY, T('make_synthetic_bag.py'), bag])
    args.rebuild, args.from_stage = True, 1
    poses = validate(r, cfg, args)
    r.run('SMOKE  stage 6 (2 scenarios)', [PY, S('6_stress_test.py'), '--config', r.cfg_file,
                                          '--only', 'clean,cam_drop1'])
    args.test_bag, args.rebuild = bag, False
    test_poses = submit(r, cfg, args)
    if r.dry:
        return

    # ---- checks on the outputs --------------------------------------------------------------
    problems = []
    for p in (poses, test_poses):
        with open(p, newline='') as f:
            rows = list(csv.DictReader(f))
        ts = [int(x['timestamp_ns']) for x in rows]
        if not rows or list(rows[0].keys()) != ['timestamp_ns', 'x', 'y', 'yaw']:
            problems.append(f'{p}: wrong header or empty')
        if any(b <= a for a, b in zip(ts, ts[1:])):
            problems.append(f'{p}: timestamps not strictly increasing')
        if any(not all(map(lambda v: v == v, (float(x['x']), float(x['y']), float(x['yaw'])))) for x in rows):
            problems.append(f'{p}: NaN in the poses')
        print(f'[CHECK] {os.path.relpath(p, out)}: {len(rows)} poses')
    score = open(os.path.join(cfg.path('results_dir'), 'validation', 'official_score.txt')).read()
    if 'NOT RANKED' in score or 'RMSE' not in score:
        problems.append('validation score file is not ranked / incomplete')
    for p in (cfg.path('val_checkpoint_cal'), cfg.path('checkpoint_cal'), poses + '.meta.json', poses + '.eval.json',
              os.path.join(cfg.path('results_dir'), 'validation', 'trajectory.png'),
              os.path.join(cfg.path('stress_dir'), 'run', 'summary.csv')):
        if not os.path.exists(p):
            problems.append(f'missing output {p}')
    print('\n' + score)
    if problems:
        raise SystemExit('[SMOKE TEST FAILED]\n  ' + '\n  '.join(problems))
    print(f'[SMOKE TEST PASSED] every stage ran end to end on the synthetic bag ({time.perf_counter() - r.t0:.0f} s). '
          f'Outputs in {out} (synthetic data: the score above means nothing).')
    if not args.keep:
        shutil.rmtree(out, ignore_errors=True)
        print(f'[SMOKE] removed {out} (use --keep to inspect it)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['smoke', 'validate', 'submit', 'all'])
    ap.add_argument('--config', default=None, help='config file (default $RACING_CONFIG or config.toml)')
    ap.add_argument('--test-bag', default='', help='test recording for submit / all (default [paths].test_bag)')
    ap.add_argument('--dry-run', action='store_true', help='print the commands without running them')
    ap.add_argument('--rebuild', action='store_true', help='re-run stage 1 even if the cache exists')
    ap.add_argument('--from-stage', type=int, default=1, help='skip stages before N (validate / submit)')
    ap.add_argument('--stress', action='store_true', help='validate: also run stage 6 (quick set)')
    ap.add_argument('--keep', action='store_true', help='smoke: keep runs/smoke/ afterwards')
    args = ap.parse_args()
    args.stress_args = []
    if args.mode == 'smoke':
        smoke(Runner('', args.dry_run), args)
        return
    cfg = config.load(args.config)
    r = Runner(cfg.file, args.dry_run)
    print(f'[CONFIG] {cfg.file} | root {cfg.root}')
    if args.mode in ('validate', 'all'):
        validate(r, cfg, args)
    if args.mode in ('submit', 'all'):
        if args.mode == 'all':
            args.from_stage = 1
        submit(r, cfg, args)
    print(f'\n[DONE] {args.mode} finished in {time.perf_counter() - r.t0:.0f} s')


if __name__ == '__main__':
    main()
