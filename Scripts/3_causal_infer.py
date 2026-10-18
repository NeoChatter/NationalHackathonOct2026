import argparse
import csv
import math
import os
import time

import numpy as np
import torch
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from locnet import (CAMERA_TOPICS, IMG_H, IMG_W, OdomNet, PlaceNet, crop_and_resize,
                    normalize_cams, stamp_to_ns, yaw_from_quat)
from track_map import TrackMap
from tracker import Tracker

MAX_SET_SPREAD_NS = 300_000_000   # 300 ms: max stamp spread inside one image set


def read_first_pose(mcap_path, min_ns):
    """The ONLY use of /tf: the first pose of the run (allowed). The main pass never reads /tf again."""
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in reader.iter_decoded_messages(topics=['/tf']):
            for t in msg.transforms:
                if t.header.frame_id.lstrip('/') == 'map' and t.child_frame_id.lstrip('/') == 'constructor0/base_link':
                    if stamp_to_ns(t.header.stamp) >= min_ns:
                        p = t.transform.translation
                        return float(p.x), float(p.y), yaw_from_quat(t.transform.rotation)
    return None


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    place = PlaceNet(256).to(device)
    place.load_state_dict(ck['place_state'])
    place.eval()
    odom = OdomNet().to(device)
    odom.load_state_dict(ck['odom_state'])
    odom.eval()
    db_emb = ck['db_embeddings'].to(device)
    db_poses = ck['db_poses'].cpu().numpy()                       # [x, y, yaw]
    calib = dict(ck.get('calib', {}))
    if args.sim_thresh is not None:
        calib['sim_thresh'] = args.sim_thresh
    if args.no_heading_prior:
        calib['heading_prior'] = False
    if args.border_margin is not None:
        calib['border_margin'] = args.border_margin
    val_start_ns = ck.get('val_start_timestamp_ns', 0) if args.val_only else 0
    print(f'[INIT] {len(db_emb):,} database frames on {device}; calibration: {calib}')

    track = None
    if args.no_track_map:
        print('[TRACK] disabled (--no-track-map): lat_clamp around the training path')
    elif ck.get('track_map') is not None:
        track = TrackMap.from_dict(ck['track_map'])
        print(f'[TRACK] {track.summary()}')
    else:
        print('[TRACK] checkpoint has no track_map: lat_clamp around the training path '
              '(add one with Scripts/build_track_map.py)')

    with torch.no_grad():                                          # warm-up: cuDNN autotune, allocator
        z = torch.zeros(1, 4, IMG_H, IMG_W, device=device)
        for _ in range(3):
            place(z)
            odom(z, z, torch.full((1, 1), 0.1, device=device))
        if device.type == 'cuda':
            torch.cuda.synchronize()

    first_pose = read_first_pose(args.mcap_path, val_start_ns)
    if first_pose is None:
        print('[WARN] no first pose found: will initialise from global retrieval on the first frame')
    else:
        print(f'[START] first pose x={first_pose[0]:.2f} y={first_pose[1]:.2f} yaw={math.degrees(first_pose[2]):.1f}deg')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.log) or '.', exist_ok=True)

    tracker = None
    n_skipped = 0
    prev_t, prev_stamp_ns = None, 0
    latest_img = [None] * 4
    latest_stamp = [0] * 4
    decode_ms = 0.0
    preds, timings = [], []

    with open(args.log, 'w', newline='') as logf:
        lw = csv.writer(logf)
        lw.writerow(['frame', 'timestamp_ns', 'mode', 'lost', 'top1_sim', 'total_ms', 'x', 'y', 'yaw', 's', 'n'])
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
                # camera_fl stamp (what the training labels mean); the SUBMITTED timestamp is the newest stamp of
                # any image used (strictly causal) and the pose is extrapolated by that small lag.
                t_set = max(latest_stamp)
                if any(im is None for im in latest_img) or (t_set - min(latest_stamp)) > MAX_SET_SPREAD_NS:
                    n_skipped += 1
                    decode_ms = 0.0
                    continue
                dt = max((s_ns - prev_stamp_ns) * 1e-9, 0.01) if prev_stamp_ns else 0.1
                top1 = 0.0

                cur = torch.from_numpy(normalize_cams(np.stack(latest_img).astype(np.float32) / 255.0)).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = place(cur)                               # (1,256)
                    if tracker is None:
                        if first_pose is not None:
                            pose0 = first_pose
                        else:
                            j = int((emb @ db_emb.T).argmax().item())
                            pose0 = tuple(db_poses[j])
                        tracker = Tracker(pose0, db_poses, calib, track=track)
                    elif prev_t is not None:
                        d = odom(prev_t, cur, torch.full((1, 1), dt, device=device))[0].cpu().numpy()
                        tracker.predict(float(d[0]), float(d[2]), dt)        # dy_body ignored: cars move along their heading

                    gi = tracker.gate_indices()
                    if len(gi) >= 5:
                        sims = db_emb[torch.from_numpy(gi).to(device)] @ emb[0]
                        k = min(8, len(gi))
                        vals, loc = torch.topk(sims, k=k)
                        top1 = float(vals[0].item())
                        tracker.update(gi[loc.cpu().numpy()], vals.cpu().numpy())
                    else:
                        tracker.lost += 1
                tracker.end_frame()
                prev_t, prev_stamp_ns = cur, s_ns

                total_ms = decode_ms + (time.perf_counter() - t1) * 1000.0
                decode_ms = 0.0
                timings.append(total_ms)
                x, y, yaw = tracker.pose_at(max((t_set - s_ns) * 1e-9, 0.0))
                preds.append((t_set, x, y, yaw))
                n = len(preds)
                lw.writerow([n, t_set, tracker.mode, tracker.lost, f'{top1:.3f}', f'{total_ms:.2f}', f'{x:.4f}', f'{y:.4f}', f'{yaw:.4f}',
                             f'{tracker.sn[0]:.2f}', f'{tracker.sn[1]:.3f}'])
                if n == 1 or n % 100 == 0:
                    print(f'[{n:5d}] {tracker.mode:12s} lost={tracker.lost:3d} sim={top1:.3f} | {total_ms:5.1f} ms | '
                          f'x={x:8.2f} y={y:8.2f} yaw={math.degrees(yaw):7.1f}deg | odom scale k={tracker.k:.2f}')

    with open(args.output, 'w', newline='') as fo:
        w = csv.writer(fo)
        w.writerow(['timestamp_ns', 'x', 'y', 'yaw'])
        for s_ns, x, y, yaw in preds:
            w.writerow([s_ns, f'{x:.6f}', f'{y:.6f}', f'{yaw:.6f}'])

    t = np.asarray(timings[3:] if len(timings) > 10 else timings)    # drop the first frames (warm-up)
    print('=' * 70)
    print(f'{len(preds)} poses written to {args.output} ({n_skipped} incomplete image sets skipped)')
    print(f'End-to-end time per image set (decode + networks + filter): median {np.median(t):.1f} ms | '
          f'mean {t.mean():.1f} ms | p95 {np.percentile(t, 95):.1f} ms | device {device}')
    if tracker is not None and track is not None:
        print(f'Track frame: border clamp fired on {tracker.n_clamped} frames, heading prior applied on '
              f'{tracker.n_heading_prior} frames')
    print('=' * 70)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mcap_path')
    ap.add_argument('--checkpoint', default='checkpoints/racing_model.pt')
    ap.add_argument('--output', default='val_submission.csv')
    ap.add_argument('--log', default='inference_telemetry.csv')
    ap.add_argument('--sim-thresh', type=float, default=None, help='override the calibrated similarity threshold')
    ap.add_argument('--val-only', action='store_true', help='run only on the held-out last 20%% of the run')
    ap.add_argument('--no-track-map', action='store_true', help='A/B: ignore the track map, use the old lat_clamp')
    ap.add_argument('--no-heading-prior', action='store_true', help='A/B: track map borders but no heading prior')
    ap.add_argument('--border-margin', type=float, default=None, help='override the border margin [m]')
    run(ap.parse_args())
