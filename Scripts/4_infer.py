"""
STAGE 4 - causal inference: bag -> submission CSV (+ per-frame telemetry, + provenance sidecar).

  python Scripts/4_infer.py Data/<bag>.mcap --checkpoint checkpoints/robust_cal.pt --output submission.csv
  python Scripts/4_infer.py Data/<bag>.mcap --checkpoint checkpoints/robust_cal.pt --val-only --output runs/val.csv

Per 4-camera image set (closed by camera_fl), strictly causal (only images with stamps <= the output stamp):
  1. preprocess exactly as at training time (ck['preproc'])                         lib/models.py
  2. PlaceNet descriptor -> similarity to every database entry (night DB + memory)   lib/models.py, lib/memory.py
  3. OdomNet motion since the previous set -> EKF predict (bias-calibrated)          lib/tracker.py
  4. gated retrieval -> EKF update (thresholds adapt to today's similarity level)    lib/tracker.py
  5. along-track filter on all similarities; re-seeds the EKF when lost              lib/sfilter.py
  6. track constraints: heading prior, real borders, lost-mode pull to the line      lib/track_map.py
  7. experience memory stores the descriptor when the pose is trustworthy            lib/memory.py
The only ground truth used is the first pose of the run (allowed).

A/B and stress flags: --reloc legacy --no-adaptive-sim --no-memory (= milestone-2 behaviour), --no-track-map,
--no-heading-prior, --adabn, --perturb, --kidnap (see 6_stress_test.py).
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from lib import provenance
from lib.common import CAMERA_TOPICS, IMG_H, IMG_W, MAX_SET_SPREAD_NS, crop_and_resize, read_first_pose, stamp_to_ns
from lib.memory import ExperienceMemory
from lib.models import enable_test_time_bn, load_networks, preproc_cfg, preprocess
from lib.perturb import Perturber
from lib.track_map import TrackMap
from lib.tracker import Tracker

TEL_COLS = ['frame', 'timestamp_ns', 'mode', 'lost', 'top1_sim', 'total_ms', 'x', 'y', 'yaw', 's', 'n',
            'top1_global', 'sf_s', 'sf_mass', 'sf_margin', 'reseed', 'kidnap', 'mem_size', 'mem_event']


def run(args, record):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    place, odom = load_networks(ck, device)
    db_emb = ck['db_embeddings'].to(device).float()
    db_poses = ck['db_poses'].cpu().numpy()                       # [x, y, yaw]

    calib = dict(ck.get('calib', {}))
    if args.sim_thresh is not None:
        calib['sim_thresh'] = args.sim_thresh
    if args.no_heading_prior:
        calib['heading_prior'] = False
    if args.border_margin is not None:
        calib['border_margin'] = args.border_margin
    calib['reloc'] = args.reloc
    calib['adaptive_sim'] = bool(args.adaptive_sim)               # off unless explicitly requested
    preproc = preproc_cfg(ck.get('preproc'))                      # must match training (old checkpoints: default)
    val_start_ns = int(ck.get('val_start_timestamp_ns', 0)) if args.val_only else 0
    print(f'[INIT] {len(db_emb):,} database frames on {device}; preproc {preproc}; calibration {calib}')

    track = None
    if args.no_track_map:
        print('[TRACK] disabled (--no-track-map): lat_clamp around the training path')
    elif ck.get('track_map') is not None:
        track = TrackMap.from_dict(ck['track_map'])
        print(f'[TRACK] {track.summary()}')
    else:
        print('[TRACK] checkpoint has no track_map: lat_clamp around the training path (run 3_calibrate.py)')

    memory = None
    if args.memory:
        if track is None:
            print('[MEMORY] needs a track map: disabled')
        else:
            memory = ExperienceMemory(db_emb, track, calib)
            print(f'[MEMORY] on: up to {memory.c["mem_max"]} entries, {memory.c["mem_per_bin"]} per '
                  f'{memory.c["mem_bin_m"]:g} m of track; stores only when top-1 sim < {memory.c["mem_sim_hi"]:g}')

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
        for _ in range(3):
            place(z)
            odom(z, z, torch.full((1, 1), 0.1, device=device))
        if device.type == 'cuda':
            torch.cuda.synchronize()
    if args.adabn > 0:                                             # AFTER the warm-up (zeros would skew the stats)
        enable_test_time_bn(place, args.adabn)
        enable_test_time_bn(odom, args.adabn)
        print(f'[ADABN] BatchNorm statistics follow the incoming frames (momentum {args.adabn:g})')

    first_pose = read_first_pose(args.mcap_path, val_start_ns)
    if first_pose is None:
        print('[WARN] no first pose found: will initialise from global retrieval on the first frame')
    else:
        print(f'[START] first pose x={first_pose[0]:.2f} y={first_pose[1]:.2f} yaw={math.degrees(first_pose[2]):.1f}deg')

    for p in (args.output, args.log):
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)

    tracker = None
    n_skipped = n_dropped = n_sets = 0
    kidnap_sign = 1.0
    prev_t, prev_stamp_ns = None, 0
    latest_img, latest_stamp = [None] * 4, [0] * 4
    decode_ms = 0.0
    preds, timings = [], []

    with open(args.log, 'w', newline='') as logf:
        lw = csv.writer(logf)
        lw.writerow(TEL_COLS)
        with open(args.mcap_path, 'rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, ch, _, msg in reader.iter_decoded_messages(topics=list(CAMERA_TOPICS.keys())):
                cam = CAMERA_TOPICS[ch.topic]
                s_ns = stamp_to_ns(msg.header.stamp)
                if s_ns < val_start_ns or not getattr(msg, 'data', None):
                    continue
                t0 = time.perf_counter()
                latest_img[cam] = crop_and_resize(bytes(msg.data))
                latest_stamp[cam] = s_ns
                decode_ms += (time.perf_counter() - t0) * 1000.0
                if cam != 0:
                    continue                                       # camera_fl closes an image set

                t1 = time.perf_counter()
                # Same pairing rule as 1_build_dataset (latest image of each camera). The state is anchored at the
                # camera_fl stamp; the SUBMITTED stamp is the newest image stamp used (strictly causal) and the pose
                # is extrapolated by that small lag.
                t_set = max(latest_stamp)
                if any(im is None for im in latest_img) or (t_set - min(latest_stamp)) > MAX_SET_SPREAD_NS:
                    n_skipped += 1
                    decode_ms = 0.0
                    continue
                if perturber is not None and perturber.drop_frame():         # stress: lost image set
                    n_dropped += 1
                    decode_ms = 0.0
                    continue
                dt = max((s_ns - prev_stamp_ns) * 1e-9, 0.01) if prev_stamp_ns else 0.1
                top1, kidnapped, mem_event = 0.0, 0, 0

                imgs = np.stack(latest_img)
                if perturber is not None:
                    imgs = perturber(imgs)
                cur = torch.from_numpy(preprocess(imgs, preproc)).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = place(cur)                               # (1,256)
                    db_all = memory.emb if memory is not None else db_emb
                    sims_all = db_all @ emb[0]                     # similarity to every database entry
                    if tracker is None:
                        if first_pose is not None:
                            pose0 = first_pose
                        else:
                            j = int(sims_all.argmax().item())
                            pose0 = tuple(db_poses[j])
                        tracker = Tracker(pose0, db_poses, calib, track=track)
                    elif prev_t is not None:
                        if kidnap_every and n_sets > 0 and n_sets % kidnap_every == 0:
                            tracker.kidnap(kidnap_sign * kidnap_m)  # stress: estimator made confidently wrong
                            kidnap_sign, kidnapped = -kidnap_sign, 1
                        d = odom(prev_t, cur, torch.full((1, 1), dt, device=device))[0].cpu().numpy()
                        tracker.predict(float(d[0]), float(d[2]), dt)        # dy_body ignored: cars move along their heading

                    gi = tracker.gate_indices()
                    if len(gi) >= 5:
                        sims = sims_all[torch.from_numpy(gi).to(device)]
                        vals, loc = torch.topk(sims, k=min(8, len(gi)))
                        top1 = float(vals[0].item())
                        tracker.update(gi[loc.cpu().numpy()], vals.cpu().numpy())
                    else:
                        tracker.miss()
                    sims_np = sims_all.float().cpu().numpy()
                tracker.global_update(sims_np)                     # along-track filter; may re-seed when lost
                tracker.end_frame()
                if memory is not None and not kidnapped:
                    mem_event = memory.maybe_add(emb[0], tracker, s_ns * 1e-9, float(sims_np.max()))
                n_sets += 1
                prev_t, prev_stamp_ns = cur, s_ns

                total_ms = decode_ms + (time.perf_counter() - t1) * 1000.0
                decode_ms = 0.0
                timings.append(total_ms)
                x, y, yaw = tracker.pose_at(max((t_set - s_ns) * 1e-9, 0.0))
                preds.append((t_set, x, y, yaw))
                n = len(preds)
                sf = tracker.sf
                lw.writerow([n, t_set, tracker.mode, tracker.lost, f'{top1:.3f}', f'{total_ms:.2f}',
                             f'{x:.4f}', f'{y:.4f}', f'{yaw:.4f}', f'{tracker.sn[0]:.2f}', f'{tracker.sn[1]:.3f}',
                             f'{float(sims_np.max()):.4f}', f'{sf.s_hat:.2f}' if sf else 'nan',
                             f'{sf.mass:.3f}' if sf else 'nan', f'{sf.margin:.2f}' if sf else 'nan',
                             int(tracker.last_reseed), kidnapped, memory.size if memory else 0, mem_event])
                if n == 1 or n % 100 == 0:
                    print(f'[{n:5d}] {tracker.mode:12s} lost={tracker.lost:3d} sim={top1:.3f} | {total_ms:5.1f} ms | '
                          f'x={x:8.2f} y={y:8.2f} yaw={math.degrees(yaw):7.1f}deg | k={tracker.k:.2f}'
                          + (f' | mem {memory.size}' if memory else ''))

    with open(args.output, 'w', newline='') as fo:
        w = csv.writer(fo)
        w.writerow(['timestamp_ns', 'x', 'y', 'yaw'])
        for s_ns, x, y, yaw in preds:
            w.writerow([s_ns, f'{x:.6f}', f'{y:.6f}', f'{yaw:.6f}'])

    t = np.asarray(timings[3:] if len(timings) > 10 else timings)    # drop the first frames (warm-up)
    summary = dict(poses=len(preds), skipped_sets=n_skipped, dropped_sets=n_dropped,
                   ms_median=float(np.median(t)), ms_p95=float(np.percentile(t, 95)), device=str(device))
    print('=' * 70)
    print(f'{len(preds)} poses written to {args.output} ({n_skipped} incomplete image sets skipped)')
    print(f'End-to-end time per image set (decode + networks + filter): median {np.median(t):.1f} ms | '
          f'mean {t.mean():.1f} ms | p95 {np.percentile(t, 95):.1f} ms | device {device}')
    if tracker is not None:
        thr = tuple(round(v, 3) for v in tracker._thresholds())
        summary.update(reseeds=tracker.n_reseed, clamped=tracker.n_clamped, heading_prior=tracker.n_heading_prior,
                       thresholds=list(thr))
        if track is not None:
            print(f'Track frame: border clamp fired on {tracker.n_clamped} frames, heading prior on '
                  f'{tracker.n_heading_prior} frames')
        print(f'Relocalisation ({args.reloc}): {tracker.n_reseed} along-track re-seeds | adaptive thresholds now {thr}'
              + (f' | {n_dropped} image sets dropped by the stress test' if n_dropped else ''))
    if memory is not None:
        summary.update(memory_size=memory.size, memory_added=memory.n_added, memory_replaced=memory.n_replaced)
        print(f'Experience memory: {memory.size} entries ({memory.n_added} added, {memory.n_replaced} replaced)')
    record['summary'] = summary
    for p in (args.output, args.log):
        provenance.write_meta(p, record, ck.get('provenance'))
    print(f'Provenance: {args.output}.meta.json')
    print('=' * 70)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mcap_path')
    ap.add_argument('--checkpoint', default='checkpoints/racing_model_cal.pt')
    ap.add_argument('--output', default='submission.csv')
    ap.add_argument('--log', default='', help='telemetry CSV (default: <output>_telemetry.csv)')
    ap.add_argument('--val-only', action='store_true', help='run only on the held-out last 20%% of the run')
    ap.add_argument('--sim-thresh', type=float, default=None, help='override the calibrated similarity threshold')
    ap.add_argument('--no-track-map', action='store_true', help='A/B: ignore the track map, use the old lat_clamp')
    ap.add_argument('--no-heading-prior', action='store_true', help='A/B: track map borders but no heading prior')
    ap.add_argument('--border-margin', type=float, default=None, help='override the border margin [m]')
    ap.add_argument('--reloc', choices=['sfilter', 'legacy'], default='sfilter',
                    help='sfilter = along-track grid filter re-seeding (needs a track map); legacy = 5-match chain only')
    ap.add_argument('--adaptive-sim', action='store_true',
                    help='lower the similarity gate when conditions differ from the DB (off by default; dead-banded)')
    ap.add_argument('--no-adaptive-sim', action='store_true', help=argparse.SUPPRESS)  # back-compat no-op
    ap.add_argument('--memory', action='store_true',
                    help='experience memory: add today\'s descriptors when the DB matches poorly (off by default)')
    ap.add_argument('--no-memory', action='store_true', help=argparse.SUPPRESS)        # back-compat no-op
    ap.add_argument('--adabn', type=float, default=0.0,
                    help='test-time BatchNorm adaptation momentum (e.g. 0.01); 0 = off (experimental)')
    ap.add_argument('--perturb', default='', help="stress test, e.g. 'overcast:2' or 'glare:1,cam_drop:1' (lib/perturb.py)")
    ap.add_argument('--perturb-seed', type=int, default=0)
    ap.add_argument('--kidnap', default='', help="stress test 'EVERY_FRAMES:METRES', e.g. '300:80'")
    a = ap.parse_args()
    if not a.log:
        a.log = os.path.splitext(a.output)[0] + '_telemetry.csv'
    rec = provenance.run_record('4_infer', a, inputs={'mcap': a.mcap_path, 'checkpoint': a.checkpoint})
    print(provenance.banner(rec))
    run(a, rec)
