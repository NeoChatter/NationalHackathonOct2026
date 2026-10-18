"""
STAGE 4 - causal inference: bag -> pose CSV (+ per-frame telemetry CSV, + provenance sidecar).

  python Scripts/4_infer.py <test.mcap> --output results/poses/test_poses.csv      # [paths].checkpoint_cal
  python Scripts/4_infer.py --val-only                                              # [paths].bag, val_checkpoint_cal,
                                                                                    # -> results/poses/val_block_poses.csv
Inputs read from the bag: the 4 camera topics and /constructor0/state_estimation (ONLY speed, yaw rate and
accelerations, see lib/state_est.py). Ground truth (/tf) and the state-estimation pose fields are NEVER read.
No start pose: the tracker localises itself globally (INITIALIZING -> TRACKING).

Per camera_fl frame (the anchor), strictly causal (only messages with stamps <= the anchor stamp):
  1. companion images: per camera the newest image stamped <= the anchor (lib/common.select_causal_image)   [V]
  2. preprocessing of the checkpoint (ck['preproc']) -> RacingLocNet descriptor -> similarity to every DB entry
  3. odometry since the previous anchor: calibrated speed + yaw-rate integrals (preferred); the learned VO when the
     state estimation is missing or implausible; otherwise coast. The VO also cross-checks the speed signal
     (OdomHealth, monitoring only, never switches the source)                                             [S + V]
  4. gated retrieval -> 4-state EKF update [x, y, yaw, odometry scale]                                    [S]
  5. along-track grid filter on all similarities: start finding without a start pose, re-seeding          [S]
  6. track constraints: heading prior, real borders, lost-mode pull to the racing line                    [S]
  7. one pose per anchor, stamped with the camera_fl header stamp. A frame whose companion images are missing
     or older than [infer].max_cam_age_ms gets no visual update but still a pose (no frame is dropped)    [V]

Diagnostics / A-B: --odom vo|se, --reloc legacy, --init-pose X Y YAW (a start pose handed in by the caller, e.g. an
organiser harness; never read from the bag), --memory, --adaptive-sim, --adabn, --perturb, --kidnap (6_stress_test).
"""
import argparse
import csv
import math
import os
import sys
import time
from collections import deque

import numpy as np
import torch
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # make `lib` importable from any CWD

from lib import config, provenance  # noqa: E402
from lib.common import CAMERA_TOPICS, CAMERAS, IMG_H, IMG_W, crop_and_resize, select_causal_image, stamp_to_ns  # noqa: E402
from lib.memory import ExperienceMemory  # noqa: E402
from lib.models import (build_model_from_checkpoint, enable_test_time_bn, load_checkpoint,  # noqa: E402
                        model_config_from_checkpoint, pick_device, preproc_cfg, preprocess)
from lib.perturb import Perturber  # noqa: E402
from lib.state_est import SE_TOPIC, OdomHealth, SEBuffer, odometry_plausible, se_sample  # noqa: E402
from lib.track_map import TrackMap  # noqa: E402
from lib.tracker import Tracker  # noqa: E402

TEL_COLS = ['frame', 'timestamp_ns', 'mode', 'lost', 'top1_sim', 'total_ms', 'x', 'y', 'yaw', 's', 'n',
            'top1_global', 'sf_s', 'sf_mass', 'sf_margin', 'reseed', 'kidnap', 'mem_size', 'mem_event',
            'odom_src', 'speed_mps', 'k_scale', 'max_cam_age_ms', 'missing_cams', 'visual_update', 'vo_dx',
            'odom_health_ratio']


def run(args, record):
    device = pick_device(args.device)
    ck = load_checkpoint(args.checkpoint)
    model = build_model_from_checkpoint(ck, device)
    model_cfg = model_config_from_checkpoint(ck)
    db_emb = ck['db_embeddings'].to(device).float()
    db_poses = np.asarray(ck['db_poses'], dtype=np.float64)        # [x, y, yaw] of the database frames
    if len(db_poses) == 0 or len(db_poses) != len(db_emb):
        raise SystemExit('[ERROR] the checkpoint has an empty or inconsistent place database')

    calib = dict(ck.get('calib', {}) or {})
    calib.update(args.tracker_cfg)                                 # config.toml [tracker]
    if args.sim_thresh is not None:
        calib['sim_thresh'] = args.sim_thresh
    if args.border_margin is not None:
        calib['border_margin'] = args.border_margin
    calib['heading_prior'] = bool(args.heading_prior)
    calib['reloc'] = args.reloc
    calib['adaptive_sim'] = bool(args.adaptive_sim)
    preproc = preproc_cfg(ck.get('preproc'))
    val_start_ns = val_end_ns = 0
    if args.val_only:
        val_start_ns = int(ck.get('val_start_timestamp_ns', 0) or 0)
        val_end_ns = int(ck.get('val_end_timestamp_ns', 0) or 0)   # 0 = until the end of the bag
        if val_start_ns <= 0:
            raise SystemExit('[ERROR] --val-only needs a checkpoint trained with a validation split (not --full-train)')
    print(f'[INIT] {len(db_emb):,} database frames on {device} | VO mode {model_cfg["vo_mode"]} | preproc {preproc}')
    if not all(k in calib for k in ('retr_along_std', 'retr_lat_std', 'sim_thresh')):
        print('[WARN] checkpoint has no retrieval calibration (run 3_calibrate.py): tracker defaults are used')
    if args.val_only:
        print(f'[VAL] validation block of the training run: timestamps {val_start_ns} .. {val_end_ns or "end of bag"}')

    track = None
    if args.no_track_map:
        print('[TRACK] disabled (--no-track-map): lat_clamp around the training path')
    elif ck.get('track_map') is not None:
        track = TrackMap.from_dict(ck['track_map'])
        print(f'[TRACK] {track.summary()}')
    else:
        print('[TRACK][WARN] checkpoint has no track_map (run 3_calibrate.py): no global initialisation possible')

    se_cal = calib.get('se')
    use_se = args.odom in ('auto', 'se')
    if use_se and not se_cal:
        print('[SE][WARN] checkpoint has no state-estimation calibration (run 3_calibrate.py): using raw signals')
    se_buf = SEBuffer(se_cal)
    health = OdomHealth()
    print(f'[ODOM] mode {args.odom}' + (f' | state estimation {se_buf.c}' if use_se else '')
          + (' | VO cross-check on' if args.vo_health_check and args.odom == 'auto' else ''))

    memory = None
    if args.memory:
        if track is None:
            print('[MEMORY] needs a track map: disabled')
        else:
            memory = ExperienceMemory(db_emb, track, calib)
            print(f'[MEMORY] on: up to {memory.c["mem_max"]} entries (experimental, off for submissions)')
    perturber = Perturber(args.perturb, seed=args.perturb_seed) if args.perturb else None
    if perturber is not None:
        print(f'[STRESS] perturbation: {perturber.describe()}')
    kidnap_every, kidnap_m = 0, 0.0
    if args.kidnap:
        a, b = args.kidnap.split(':')
        kidnap_every, kidnap_m = int(a), float(b)
        print(f'[STRESS] kidnap: every {kidnap_every} frames, +-{kidnap_m:g} m along the track (alternating)')

    with torch.no_grad():                                          # warm-up: cuDNN autotune, allocator
        z = torch.zeros(1, 4, IMG_H, IMG_W, device=device)
        tf0 = torch.full((1, 5), 0.05, device=device)
        for _ in range(3):
            model(z, z, tf0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    if args.adabn > 0:                                             # AFTER the warm-up (zeros would skew the stats)
        enable_test_time_bn(model, args.adabn)
        print(f'[ADABN] BatchNorm statistics follow the incoming frames (momentum {args.adabn:g}) - experimental')

    pose0 = tuple(float(v) for v in args.init_pose) if args.init_pose else None
    if pose0 is not None:
        print(f'[START] start pose handed in with --init-pose (not from the bag): {pose0}')
    elif track is None:
        print('[START][WARN] global initialisation needs a track map: falling back to the best global match')
    else:
        print('[START] no start pose: global initialisation from the along-track filter')

    for p in (args.output, args.log):
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)

    tracker = None
    buffers = [deque(maxlen=args.buffer_len) for _ in CAMERAS]
    n_nonmono = n_dropped = n_sets = n_se = n_vo = n_coast = n_degraded = n_implausible = 0
    kidnap_sign = 1.0
    prev_t, prev_anchor_ns = None, 0
    warned_health = False
    decode_ms = 0.0
    preds, timings = [], []
    topics = list(CAMERA_TOPICS.keys()) + ([SE_TOPIC] if use_se else [])
    preload_ns = int(args.preload_s * 1e9)
    start_time = max(val_start_ns - preload_ns, 0) if args.val_only else None

    with open(args.log, 'w', newline='') as logf, open(args.mcap_path, 'rb') as f:
        lw = csv.writer(logf)
        lw.writerow(TEL_COLS)
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, ch, _, msg in reader.iter_decoded_messages(topics=topics, start_time=start_time):
            if ch.topic == SE_TOPIC:
                se_buf.add(se_sample(msg))                         # speed / yaw rate / accelerations only
                continue
            if not getattr(msg, 'data', None):
                continue
            cam = CAMERA_TOPICS[ch.topic]
            s_ns = stamp_to_ns(msg.header.stamp)
            if args.val_only and s_ns < val_start_ns - preload_ns:
                continue                                           # can never be a companion of a validation anchor
            t0 = time.perf_counter()
            buffers[cam].append((s_ns, crop_and_resize(bytes(msg.data))))
            decode_ms += (time.perf_counter() - t0) * 1000.0
            if cam != 0:
                continue                                           # camera_fl anchors a pose
            if s_ns < val_start_ns:
                decode_ms = 0.0
                continue
            if val_end_ns and s_ns > val_end_ns:
                break                                              # end of the validation block
            if prev_anchor_ns and s_ns <= prev_anchor_ns:
                n_nonmono += 1
                decode_ms = 0.0
                continue
            if perturber is not None and perturber.drop_frame():
                n_dropped += 1                                     # stress test: the whole image set is lost
                decode_ms = 0.0
                continue

            t1 = time.perf_counter()
            picks = [select_causal_image(b, s_ns) for b in buffers]
            imgs, ages, missing = [], [], 0
            for st, im in picks:
                if im is None:                                     # start-up: never a future-stamped image
                    st, im = s_ns, picks[0][1]
                    missing += 1
                imgs.append(im)
                ages.append((s_ns - st) * 1e-9)
            visual_ok = missing == 0 and max(ages) * 1000.0 <= args.max_cam_age_ms
            imgs = np.stack(imgs)
            if perturber is not None:
                imgs = perturber(imgs)
            cur = torch.from_numpy(preprocess(imgs, preproc)).unsqueeze(0).to(device)
            dt = max((s_ns - prev_anchor_ns) * 1e-9, 0.01) if prev_anchor_ns else 0.1
            top1, kidnapped, mem_event = float('nan'), 0, 0

            se_odo = se_buf.odometry(prev_anchor_ns, s_ns) if (use_se and prev_anchor_ns) else None
            se_ok = se_odo is not None and odometry_plausible(se_odo[0], se_odo[1], dt)
            if se_odo is not None and not se_ok:
                n_implausible += 1
            need_vo = prev_t is not None and args.odom != 'se' and (
                args.odom == 'vo' or not se_ok or args.vo_health_check)
            vo = None
            with torch.no_grad():                                  # no_grad (not inference_mode): the optional
                if need_vo:                                        # memory / AdaBN write into normal tensors
                    tfeat = torch.tensor([[dt] + ages], dtype=torch.float32, device=device)
                    emb, _, delta, _ = model(prev_t, cur, tfeat)
                    vo = delta[0].float().cpu().numpy().astype(np.float64)
                else:
                    emb = model.forward_single(cur)[1]
                db_all = memory.emb if memory is not None else db_emb
                sims_all = db_all @ emb[0].float()                 # similarity to every database entry
                if tracker is None:
                    start = pose0
                    if start is None and track is None:
                        start = tuple(db_poses[int(sims_all.argmax().item())])
                    tracker = Tracker(start, db_poses, calib, track=track)
                elif prev_anchor_ns:
                    if kidnap_every and n_sets > 0 and n_sets % kidnap_every == 0 and tracker.initialized:
                        tracker.kidnap(kidnap_sign * kidnap_m)
                        kidnap_sign, kidnapped = -kidnap_sign, 1
                    if se_ok and vo is not None:
                        health.update(se_odo[0], float(vo[0]))
                        if not health.ok and not warned_health:
                            warned_health = True
                            print(f'[INFO] the learned VO disagrees with the speed signal (median distance ratio '
                                  f'{health.median_ratio:.2f}); the speed signal is kept, the VO is only a fallback')
                    if args.odom != 'vo' and se_ok:
                        tracker.predict_se(se_odo[0], se_odo[1], dt)
                        n_se += 1
                    elif args.odom != 'se' and vo is not None:
                        tracker.predict(float(vo[0]), float(vo[2]), dt)
                        n_vo += 1
                    else:
                        tracker.coast(dt)
                        n_coast += 1

                sims_np = None
                if visual_ok:
                    gi = tracker.gate_indices()
                    if len(gi) >= args.min_gate:
                        sims = sims_all[torch.from_numpy(gi).to(device)]
                        vals, loc = torch.topk(sims, k=min(args.topk, len(gi)))
                        top1 = float(vals[0].item())
                        tracker.update(gi[loc.cpu().numpy()], vals.float().cpu().numpy())
                    elif tracker.initialized:
                        tracker.miss()
                    sims_np = sims_all.float().cpu().numpy()
                else:
                    n_degraded += 1
                    if tracker.initialized:
                        tracker.miss()
            if sims_np is not None:
                tracker.global_update(sims_np)                     # initialisation / re-seeding
            tracker.end_frame()
            if memory is not None and sims_np is not None and not kidnapped and tracker.initialized:
                mem_event = memory.maybe_add(emb[0].float(), tracker, s_ns * 1e-9, float(sims_np.max()))
            n_sets += 1
            prev_t, prev_anchor_ns = cur, s_ns

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            total_ms = decode_ms + (time.perf_counter() - t1) * 1000.0
            decode_ms = 0.0
            timings.append(total_ms)
            x, y, yaw = tracker.pose_at(0.0)
            preds.append((s_ns, x, y, yaw))
            n = len(preds)
            sf = tracker.sf
            lw.writerow([n, s_ns, tracker.mode, tracker.lost, f'{top1:.4f}', f'{total_ms:.2f}',
                         f'{x:.4f}', f'{y:.4f}', f'{yaw:.4f}', f'{tracker.sn[0]:.2f}', f'{tracker.sn[1]:.3f}',
                         f'{float(sims_np.max()):.4f}' if sims_np is not None else 'nan',
                         f'{sf.s_hat:.2f}' if sf else 'nan', f'{sf.mass:.3f}' if sf else 'nan',
                         f'{sf.margin:.2f}' if sf else 'nan', int(tracker.last_reseed), kidnapped,
                         memory.size if memory else 0, mem_event, tracker.odom_src, f'{tracker.v:.2f}',
                         f'{tracker.k:.4f}', f'{max(ages) * 1000.0:.1f}', missing, int(visual_ok),
                         f'{vo[0]:.4f}' if vo is not None else 'nan', f'{health.median_ratio:.4f}'])
            if n == 1 or n % 100 == 0 or (tracker.last_reseed and tracker.init_frame == tracker.n_frames):
                print(f'[{n:5d}] {tracker.mode:12s} lost={tracker.lost:3d} sim={top1:.3f} | {total_ms:5.1f} ms | '
                      f'x={x:8.2f} y={y:8.2f} yaw={math.degrees(yaw):7.1f}deg | v={tracker.v:5.1f} m/s '
                      f'({tracker.odom_src or "-"}) k={tracker.k:.3f}' + (f' | mem {memory.size}' if memory else ''))

    if not preds:
        raise SystemExit('[ERROR] no poses were produced: check the bag, its camera topics and the validation split')
    with open(args.output, 'w', newline='') as fo:
        w = csv.writer(fo)
        w.writerow(['timestamp_ns', 'x', 'y', 'yaw'])
        for s_ns, x, y, yaw in preds:
            w.writerow([s_ns, f'{x:.6f}', f'{y:.6f}', f'{yaw:.6f}'])

    t = np.asarray(timings[3:] if len(timings) > 10 else timings)
    summary = dict(poses=len(preds), nonmonotonic_anchors=n_nonmono, dropped_sets=n_dropped, odom_se=n_se,
                   odom_vo=n_vo, odom_coast=n_coast, se_implausible=n_implausible, no_visual_update=n_degraded,
                   se_samples=se_buf.n, ms_mean=float(t.mean()), ms_median=float(np.median(t)),
                   ms_p95=float(np.percentile(t, 95)), device=str(device),
                   init='init-pose' if pose0 is not None else 'global',
                   odom_health_ratio=health.median_ratio, odom_health_ok=bool(health.ok))
    print('=' * 70)
    print(f'{len(preds)} poses written to {args.output} (one per camera_fl frame; {n_nonmono} non-increasing '
          f'anchor stamps skipped)')
    print(f'[TIMING] per frame incl. JPEG decoding: mean {t.mean():.1f} ms | median {np.median(t):.1f} ms | '
          f'p95 {np.percentile(t, 95):.1f} ms | device {device}')
    print(f'Odometry: {n_se} frames from state estimation ({se_buf.n} samples read, {n_implausible} implausible), '
          f'{n_vo} from visual odometry, {n_coast} coasted | VO/speed distance ratio {health.median_ratio:.3f}')
    print(f'Frames without a visual update (missing / stale companion image): {n_degraded}')
    if tracker is not None:
        init_s = tracker.init_frame / 10.0 if tracker.init_frame >= 0 else float('nan')
        summary.update(reseeds=tracker.n_reseed, clamped=tracker.n_clamped, heading_prior=tracker.n_heading_prior,
                       init_frame=tracker.init_frame, initialized=tracker.initialized)
        print('Start: ' + ('handed in (--init-pose)' if pose0 is not None else
                           f'global initialisation after {tracker.init_frame} frames (~{init_s:.1f} s)'
                           if tracker.initialized else 'NEVER initialised (grid filter never confident)'))
        print(f'Relocalisation ({args.reloc}): {tracker.n_reseed} grid-filter seeds | border clamp '
              f'{tracker.n_clamped} | heading prior {tracker.n_heading_prior}')
    if memory is not None:
        summary.update(memory_size=memory.size, memory_added=memory.n_added, memory_replaced=memory.n_replaced)
    record['summary'] = summary
    for p in (args.output, args.log):
        provenance.write_meta(p, record, ck.get('provenance'))
    print(f'Telemetry: {args.log} | provenance: {args.output}.meta.json')
    print('=' * 70)


if __name__ == '__main__':
    cfg = config.bootstrap()
    provenance.configure(cfg)
    I = cfg['infer']
    B = argparse.BooleanOptionalAction
    ap = config.parser(__doc__, cfg)
    ap.add_argument('mcap_path', nargs='?', default=None,
                    help='bag to localise (default: [paths].bag with --val-only, else [paths].test_bag)')
    ap.add_argument('--checkpoint', default=None,
                    help='default [paths].val_checkpoint_cal with --val-only, else [paths].checkpoint_cal')
    ap.add_argument('--output', default=None, help='pose CSV (default results/poses/{val_block,test}_poses.csv)')
    ap.add_argument('--log', default='', help='telemetry CSV (default: <output>_telemetry.csv)')
    ap.add_argument('--val-only', action='store_true', help="only the checkpoint's validation block of the training run")
    ap.add_argument('--device', default=cfg.get('run', 'device', 'auto'))
    ap.add_argument('--init', choices=['global'], default=I.get('init', 'global'),
                    help='global = find the start position from the images (the only mode; see --init-pose)')
    ap.add_argument('--init-pose', type=float, nargs=3, metavar=('X', 'Y', 'YAW'), default=None,
                    help='start pose handed in by the caller (e.g. an organiser harness). Never read from the bag.')
    ap.add_argument('--odom', choices=['auto', 'se', 'vo'], default=I.get('odom', 'auto'),
                    help='auto = state estimation, visual odometry where it is missing / implausible; '
                         'se = state estimation only (coast through gaps); vo = camera-only')
    ap.add_argument('--vo-health-check', action=B, default=bool(I.get('vo_health_check', True)))
    ap.add_argument('--topk', type=int, default=int(I.get('topk', 8)))
    ap.add_argument('--min-gate', type=int, default=int(I.get('min_gate_candidates', 5)))
    ap.add_argument('--max-cam-age-ms', type=float, default=float(I.get('max_cam_age_ms', 300)))
    ap.add_argument('--preload-s', type=float, default=float(I.get('preload_s', 10.0)))
    ap.add_argument('--buffer-len', type=int, default=int(cfg.get('build', 'buffer_len', 64)))
    ap.add_argument('--sim-thresh', type=float, default=None, help='override the calibrated similarity threshold')
    ap.add_argument('--heading-prior', action=B, default=bool(cfg.get('tracker', 'heading_prior', True)))
    ap.add_argument('--border-margin', type=float, default=None)
    ap.add_argument('--no-track-map', action='store_true')
    ap.add_argument('--reloc', choices=['sfilter', 'legacy'], default=I.get('reloc', 'sfilter'))
    ap.add_argument('--adaptive-sim', action=B, default=bool(I.get('adaptive_sim', False)))
    ap.add_argument('--memory', action=B, default=bool(I.get('memory', False)), help='experience memory (experimental)')
    ap.add_argument('--adabn', type=float, default=float(I.get('adabn', 0.0)), help='test-time BatchNorm; 0 = off')
    ap.add_argument('--perturb', default='', help="stress test, e.g. 'overcast:2' (lib/perturb.py)")
    ap.add_argument('--perturb-seed', type=int, default=0)
    ap.add_argument('--kidnap', default='', help="stress test 'EVERY_FRAMES:METRES', e.g. '300:80'")
    a = ap.parse_args()
    a.tracker_cfg = dict(cfg['tracker'])
    if a.mcap_path is None:
        a.mcap_path = cfg.path('bag') if a.val_only else cfg.path('test_bag')
    if a.checkpoint is None:
        a.checkpoint = cfg.path('val_checkpoint_cal') if a.val_only else cfg.path('checkpoint_cal')
    if a.output is None:
        a.output = os.path.join(cfg.path('results_dir'), 'poses', 'val_block_poses.csv' if a.val_only else 'test_poses.csv')
    if not a.log:
        a.log = os.path.splitext(a.output)[0] + '_telemetry.csv'
    config.require_file(a.mcap_path, 'bag', 'pass it as the first argument or set [paths].bag / [paths].test_bag')
    config.require_file(a.checkpoint, 'checkpoint', 'run Scripts/3_calibrate.py first')
    rec = provenance.run_record('4_infer', a, inputs={'mcap': a.mcap_path, 'checkpoint': a.checkpoint}, cfg=cfg)
    print(provenance.banner(rec))
    run(a, rec)
