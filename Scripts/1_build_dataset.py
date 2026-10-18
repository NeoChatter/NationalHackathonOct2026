<<<<<<< HEAD
"""
STAGE 1 - bag -> training cache (+ the full-run ground-truth CSV used for scoring).

  python Scripts/1_build_dataset.py                          # [paths].bag -> [paths].cache and [paths].gt_csv
  python Scripts/1_build_dataset.py <bag> --output <cache.h5> --export-gt <gt.csv>
  python Scripts/1_build_dataset.py <bag> --aux-only         # only (re)write borders + state estimation into a cache

Camera sync (rmse-2.740 branch): one sample per camera_fl frame; from every camera the newest image whose stamp is
<= that anchor (lib/common.select_causal_image, the SAME rule as inference). Anchors whose stamp does not increase,
or where a camera has no eligible image yet, are skipped. Extras (rmse-4.363 branch): per-marker track borders,
whitelisted state estimation, --export-gt, --aux-only, provenance.

Writes the cache:
  timestamps_ns (N,)          camera_fl stamp of every sample
  images        (N,4,H,W)     cropped grayscale, uint8 (lib/common.crop_and_resize)
  dt_offsets    (N,4)         anchor stamp minus each camera's image stamp [s] (always >= 0)
  poses         (N,4)         GT [x, y, z, yaw] interpolated at the anchor (training / calibration targets only)
  track_borders (M,3)         all border points (flat layout, marker pose applied)
  track_bounds/marker_XXX     one dataset per border marker with ns / id / type / frame_id attributes
  state_est/{t_ns, v_mps, wz_radps, ax_mps2, ay_mps2}
                              ONLY the allowed state-estimation fields (lib/state_est.SE_FIELDS), full rate
  attrs: causal_sync = 1, provenance (JSON run record)
--export-gt writes every /tf pose as CSV (timestamp_ns,x,y,z,yaw), the same format as tools/extract_gt.py.
"""
import csv
import json
import math
import os
import sys
import time
from collections import deque

=======
import argparse
import math
import os
import time
from collections import deque

import cv2
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510
import h5py
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

<<<<<<< HEAD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # make `lib` importable from any CWD

from lib import config, provenance  # noqa: E402
from lib.common import (CAMERA_TOPICS, CAMERAS, IMG_H, IMG_W, TF_TOPIC, crop_and_resize, is_base_tf,  # noqa: E402
                        select_causal_image, stamp_to_ns, yaw_from_quat)
from lib.state_est import SE_FIELDS, SE_TOPIC, pair_measurements, se_sample  # noqa: E402
from lib.track_map import (TRACK_BOUNDS_TOPIC, flat_points, markers_from_msg, read_track_bounds_mcap,  # noqa: E402
                           save_markers_h5)


def _save_state_est(h5f, samples):
    if 'state_est' in h5f:
        del h5f['state_est']
    if not samples:
        return
    a = np.asarray(sorted(samples), dtype=np.float64)
    g = h5f.create_group('state_est')
    g.create_dataset('t_ns', data=np.asarray([s[0] for s in sorted(samples)], dtype=np.int64))
    for j, f in enumerate(SE_FIELDS):
        g.create_dataset(f, data=a[:, 1 + j])
    g.attrs['topic'] = SE_TOPIC
    g.attrs['fields'] = ','.join(SE_FIELDS)


def _print_state_est(samples):
    if not samples:
        print(f'[STATE_EST] no {SE_TOPIC} messages found')
        return
    a = np.asarray(samples, dtype=np.float64)
    dur = (a[:, 0].max() - a[:, 0].min()) * 1e-9
    print(f'[STATE_EST] {len(a)} samples over {dur:.0f} s ({len(a) / max(dur, 1e-9):.0f} Hz) | fields {SE_FIELDS} | '
          f'speed max {np.nanmax(a[:, 1]):.1f} m/s')


def _print_markers(markers):
    print(f'[BORDERS] {len(markers)} track_bounds markers')
    for m in markers:
        print(f'  ns={m["ns"]!r:24s} id={m["id"]:4d} type={m["type"]:2d} points={len(m["points"]):6d} frame={m["frame_id"]!r}')


def report_odometry(stamps, poses, samples):
    """Sanity check of the raw speed / yaw-rate signals against ground truth (units, signs and scale show up here).
    The calibration itself is fitted on the training split only, in stage 3."""
    if not samples or len(stamps) < 3:
        print('[ODOM] no state estimation: stage 4 will fall back to visual odometry')
        return
    s = sorted(samples)
    se = {'t_ns': np.asarray([r[0] for r in s], dtype=np.int64), 'v_mps': np.asarray([r[1] for r in s]),
          'wz_radps': np.asarray([r[2] for r in s])}
    ok = np.isfinite(se['v_mps']) & np.isfinite(se['wz_radps'])
    se = {k: v[ok] for k, v in se.items()}
    m = pair_measurements(se, stamps, poses, np.arange(len(stamps)))
    if len(m) < 10:
        print('[ODOM] state estimation does not cover the camera frames')
        return
    g_dx, g_dy, iv, iw, _ = m.T
    mv = g_dx > 0.5
    scale = float(iv[mv].sum() / max(g_dx[mv].sum(), 1e-9)) if mv.any() else float('nan')
    print(f'[ODOM] speed / yaw-rate integrals valid for {len(m)}/{len(stamps) - 1} frame pairs | raw distance scale '
          f'vs GT {scale:.4f} (1.000 = perfect) | dyaw RMSE {math.degrees(float(np.sqrt(np.mean((iw - g_dy) ** 2)))):.3f} deg')
    if np.isfinite(scale) and abs(scale - 1.0) > 0.03:
        print('[ODOM] WARNING: raw distance scale off by more than 3 % - check the units / meaning of v_mps')


def build_dataset(args, record):
    print(f'[1/3] Scanning {args.mcap_path} for GT poses (/tf), track_bounds, state estimation and 4 cameras...')
    t0 = time.perf_counter()
    gt_stamps, gt_poses, track_markers, se = [], [], [], []
    buffers = [deque(maxlen=args.buffer_len) for _ in CAMERAS]
    last_anchor_ns = None
    s_stamps, s_imgs, s_offs = [], [], []
    n_future_logged = n_incomplete = n_nonmonotonic = 0

    topics = list(CAMERA_TOPICS) + [TF_TOPIC, TRACK_BOUNDS_TOPIC, SE_TOPIC]
    with open(args.mcap_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, _, msg in reader.iter_decoded_messages(topics=topics):
            topic = channel.topic
            if topic == TF_TOPIC:
                for t in getattr(msg, 'transforms', []) or []:
                    if is_base_tf(t):
                        p = t.transform.translation
                        gt_stamps.append(stamp_to_ns(t.header.stamp))
                        gt_poses.append([float(p.x), float(p.y), float(p.z), yaw_from_quat(t.transform.rotation)])
            elif topic == SE_TOPIC:
                se.append(se_sample(msg))                                 # whitelisted fields only
            elif topic == TRACK_BOUNDS_TOPIC and not track_markers:
                track_markers = markers_from_msg(msg)                    # keep each border line separately
            elif topic in CAMERA_TOPICS:
                if not getattr(msg, 'data', None):
                    continue
                cam = CAMERA_TOPICS[topic]
                s_ns = stamp_to_ns(msg.header.stamp)
                buffers[cam].append((s_ns, crop_and_resize(msg.data)))
                if cam != 0:                                             # camera_fl anchors a sample
=======
CAMERAS = ('camera_fl', 'camera_fr', 'camera_cl', 'camera_cr')
CAMERA_TOPICS = {f'/constructor0/sensor/{c}/compressed_image': i for i, c in enumerate(CAMERAS)}
IMG_W, IMG_H = 256, 128

# How many recent images per camera are kept for causal selection (same value as inference).
BUFFER_LEN = 64


def crop_and_resize(jpeg_bytes: bytes) -> np.ndarray:
    """
    Decodes 1506x728 JPEG, crops vertical track ROI [y: 18% to 74%] to remove
    night sky floodlights (top) and race car nosecone/tires (bottom), and resizes to 256x128.
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    h, w = img.shape[:2]
    y0, y1 = int(0.18 * h), int(0.74 * h)
    cropped = img[y0:y1, :]
    resized = cv2.resize(cropped, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return resized


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec * 1_000_000_000 + stamp.nanosec)


def select_causal_image(buffer, anchor_ns):
    """
    Newest already-received image whose exposure stamp is <= anchor_ns.

    This is the single definition used by BOTH the dataset builder and the inference script, so the
    images the model is trained / mapped with are exactly the images it sees at run time.
    """
    best_stamp = None
    best_image = None
    for stamp, image in buffer:
        if stamp <= anchor_ns and (best_stamp is None or stamp > best_stamp):
            best_stamp = stamp
            best_image = image
    return best_stamp, best_image


STATE_ESTIMATION_TOPIC = '/constructor0/state_estimation'
ODOM_FIELDS = ('dx_body', 'dy_body', 'dyaw', 'speed', 'yaw_rate', 'ax', 'ay')


class TelemetryTrack:
    """
    Speed, yaw rate and accelerations from /constructor0/state_estimation.

    The organisers allow exactly these telemetry fields as an input at test time ("speed, accelerations and yaw
    rate"). The pose fields of that message (x_m, y_m, z_m, roll/pitch/yaw, sn_map_state, ...) are deliberately
    NEVER read: the test file does not contain them. Fields used: v_mps, wz_radps, ax_mps2, ay_mps2.

    The dataset builder and the inference script both use this class, so training/validation caches contain
    the odometry the submission will actually see.
    """

    def __init__(self, keep=1024):
        self.t = deque(maxlen=keep)     # stamps [ns]
        self.v = deque(maxlen=keep)     # speed [m/s]
        self.w = deque(maxlen=keep)     # yaw rate [rad/s]
        self.ax = deque(maxlen=keep)    # longitudinal acceleration [m/s^2]
        self.ay = deque(maxlen=keep)    # lateral acceleration [m/s^2]
        self.n_added = 0
        self.n_rejected = 0

    def add(self, msg) -> bool:
        try:
            t = stamp_to_ns(msg.header.stamp)
            v = float(msg.v_mps)
            w = float(msg.wz_radps)
            ax = float(getattr(msg, 'ax_mps2', 0.0))
            ay = float(getattr(msg, 'ay_mps2', 0.0))
        except (AttributeError, TypeError, ValueError):
            self.n_rejected += 1
            return False
        if not (math.isfinite(v) and math.isfinite(w)) or (self.t and t <= self.t[-1]):
            self.n_rejected += 1
            return False
        self.t.append(t)
        self.v.append(v)
        self.w.append(w)
        self.ax.append(ax if math.isfinite(ax) else 0.0)
        self.ay.append(ay if math.isfinite(ay) else 0.0)
        self.n_added += 1
        return True

    def integrate(self, t0_ns, t1_ns, max_stale_s=0.25):
        """
        Body-frame motion between two camera stamps, in the body frame at t0: (dx, dy, dyaw), followed by speed,
        yaw rate and accelerations at t1. Only samples that have ALREADY arrived are used (causal); the last value is
        held up to t1. Returns None if the stream does not cover the interval.
        """
        n = len(self.t)
        if n < 2 or t1_ns <= t0_ns:
            return None
        ts = np.fromiter(self.t, dtype=np.int64, count=n)
        if ts[0] > t0_ns + 20_000_000 or (t1_ns - ts[-1]) * 1e-9 > max_stale_s:
            return None
        rel = (ts - t0_ns).astype(np.float64)
        span = float(t1_ns - t0_ns)
        grid = np.linspace(0.0, span, max(2, int(math.ceil(span / 5e6))) + 1)
        v = np.interp(grid, rel, np.fromiter(self.v, dtype=np.float64, count=n))
        w = np.interp(grid, rel, np.fromiter(self.w, dtype=np.float64, count=n))
        dt = np.diff(grid) * 1e-9
        psi = np.concatenate([[0.0], np.cumsum(0.5 * (w[:-1] + w[1:]) * dt)])     # heading relative to t0
        psi_mid = 0.5 * (psi[:-1] + psi[1:])
        speed_mid = 0.5 * (v[:-1] + v[1:])
        ax = float(np.interp(grid[-1], rel, np.fromiter(self.ax, dtype=np.float64, count=n)))
        ay = float(np.interp(grid[-1], rel, np.fromiter(self.ay, dtype=np.float64, count=n)))
        return (float(np.sum(speed_mid * dt * np.cos(psi_mid))), float(np.sum(speed_mid * dt * np.sin(psi_mid))),
                float(psi[-1]), float(v[-1]), float(w[-1]), ax, ay)


def report_odometry(stamps, poses, odom):
    """Sanity check of the telemetry odometry against ground truth (units, signs and frames show up here)."""
    n = len(stamps)
    if n < 3:
        return
    dt = np.diff(stamps) * 1e-9
    gx, gy = np.diff(poses[:, 0]), np.diff(poses[:, 1])
    yaw = poses[:-1, 3]
    gt_dx = np.cos(yaw) * gx + np.sin(yaw) * gy
    gt_dy = -np.sin(yaw) * gx + np.cos(yaw) * gy
    gt_dyaw = np.arctan2(np.sin(np.diff(poses[:, 3])), np.cos(np.diff(poses[:, 3])))
    o = odom[1:]
    ok = np.isfinite(o[:, 0]) & (dt >= 0.01) & (dt <= 0.5)
    if ok.sum() < 10:
        print('[ODOM] telemetry odometry could not be computed for this recording (no usable state_estimation data)')
        return
    scale = o[ok, 0].sum() / max(gt_dx[ok].sum(), 1e-9)
    print(f'[ODOM] telemetry odometry valid for {ok.sum()}/{n - 1} steps')
    print(f'[ODOM] vs ground truth: forward-distance scale {scale:.4f} (1.000 = perfect) | step RMSE '
          f'dx {np.sqrt(np.mean((o[ok, 0] - gt_dx[ok]) ** 2)):.3f} m, dy {np.sqrt(np.mean((o[ok, 1] - gt_dy[ok]) ** 2)):.3f} m, '
          f'dyaw {math.degrees(np.sqrt(np.mean((o[ok, 2] - gt_dyaw[ok]) ** 2))):.3f} deg')
    if abs(scale - 1.0) > 0.03:
        print('[ODOM] WARNING: forward scale is off by more than 3%. Check the units / meaning of v_mps before trusting '
              'telemetry odometry (the inference script cross-checks it against the learned VO at run time).')


def build_dataset(mcap_path: str, output_h5: str):
    print(f'[1/3] Scanning {mcap_path} for GT poses (/tf), telemetry (/state_estimation: speed, yaw rate, accel), '
          f'track_bounds, and 4 cameras...')
    t0 = time.perf_counter()

    telemetry = TelemetryTrack()
    sample_odom = []        # (N, 7): dx_body, dy_body, dyaw, speed, yaw_rate, ax, ay  (NaN = unavailable)
    gt_stamps = []
    gt_poses = []  # [x, y, z, yaw]
    track_borders = []

    # Per-camera buffers of (exposure stamp, image). A sample is anchored on every camera_fl frame and
    # takes, from every camera, the newest image with stamp <= the anchor stamp (strictly causal).
    buffers = [deque(maxlen=BUFFER_LEN) for _ in CAMERAS]
    last_anchor_ns = None

    sample_stamps = []
    sample_imgs = []        # (N, 4, H, W) uint8
    sample_dt_offsets = []  # (N, 4) float32 in seconds: (t_fl - t_cam) >= 0

    n_future_logged = 0     # anchors where a camera had already logged a FUTURE-stamped image
    n_incomplete = 0        # anchors skipped because a camera had no eligible image yet
    n_nonmonotonic = 0      # anchors skipped because the stamp did not increase

    topics_to_read = list(CAMERA_TOPICS.keys()) + ['/tf', '/constructor0/track_bounds', STATE_ESTIMATION_TOPIC]

    with open(mcap_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, _, msg in reader.iter_decoded_messages(topics=topics_to_read):
            topic = channel.topic

            if topic == STATE_ESTIMATION_TOPIC:
                telemetry.add(msg)

            elif topic == '/tf':
                for t in getattr(msg, 'transforms', []) or []:
                    if t.header.frame_id.lstrip('/') == 'map' and t.child_frame_id.lstrip('/') == 'constructor0/base_link':
                        s_ns = stamp_to_ns(t.header.stamp)
                        p = t.transform.translation
                        yaw = yaw_from_quat(t.transform.rotation)
                        gt_stamps.append(s_ns)
                        gt_poses.append([float(p.x), float(p.y), float(p.z), yaw])

            elif topic == '/constructor0/track_bounds' and not track_borders:
                for marker in getattr(msg, 'markers', []) or []:
                    for pt in getattr(marker, 'points', []) or []:
                        track_borders.append([float(pt.x), float(pt.y), float(pt.z)])

            elif topic in CAMERA_TOPICS:
                cam_idx = CAMERA_TOPICS[topic]
                s_ns = stamp_to_ns(msg.header.stamp)
                if not hasattr(msg, 'data') or not msg.data:
                    continue

                buffers[cam_idx].append((s_ns, crop_and_resize(msg.data)))

                # Trigger a synchronized causal sample every time camera_fl (idx 0) arrives
                if cam_idx != 0:
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510
                    continue
                if last_anchor_ns is not None and s_ns <= last_anchor_ns:
                    n_nonmonotonic += 1
                    continue
<<<<<<< HEAD
=======

>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510
                picks = [select_causal_image(b, s_ns) for b in buffers]
                if any(img is None for _, img in picks):
                    n_incomplete += 1
                    continue
<<<<<<< HEAD
                if any(st > s_ns for b in buffers for st, _ in b):      # the old "newest logged" rule would cheat
                    n_future_logged += 1
                last_anchor_ns = s_ns
                s_stamps.append(s_ns)
                s_imgs.append(np.stack([img for _, img in picks], axis=0))
                s_offs.append(np.array([(s_ns - st) * 1e-9 for st, _ in picks], dtype=np.float32))
                if len(s_stamps) % 1000 == 0:
                    print(f'  {len(s_stamps)} multi-camera frames ({time.perf_counter() - t0:.1f}s)...')

    if not gt_stamps:
        raise SystemExit('[ERROR] no map -> constructor0/base_link poses on /tf: this bag cannot be used for training')
    gt_stamps = np.asarray(gt_stamps, dtype=np.int64)
    gt_poses = np.asarray(gt_poses, dtype=np.float64)
    o = np.argsort(gt_stamps, kind='stable')
    gt_stamps, gt_poses = gt_stamps[o], gt_poses[o]

    if args.export_gt and not args.no_export_gt:
        os.makedirs(os.path.dirname(args.export_gt) or '.', exist_ok=True)
        with open(args.export_gt, 'w', newline='') as fo:
            w = csv.writer(fo)
            w.writerow(['timestamp_ns', 'x', 'y', 'z', 'yaw'])
            for s, (x, y, z, yaw) in zip(gt_stamps, gt_poses):
                w.writerow([int(s), f'{x:.4f}', f'{y:.4f}', f'{z:.4f}', f'{yaw:.6f}'])
        provenance.write_meta(args.export_gt, record)
        print(f'[GT] {len(gt_stamps)} poses -> {args.export_gt}')

    s_stamps = np.asarray(s_stamps, dtype=np.int64)
    valid = (s_stamps >= gt_stamps[0]) & (s_stamps <= gt_stamps[-1])        # camera samples inside the GT window
    s_stamps = s_stamps[valid]
    s_imgs = np.asarray(s_imgs, dtype=np.uint8)[valid]
    s_offs = np.asarray(s_offs, dtype=np.float32)[valid]

    ix = np.interp(s_stamps, gt_stamps, gt_poses[:, 0])
    iy = np.interp(s_stamps, gt_stamps, gt_poses[:, 1])
    iz = np.interp(s_stamps, gt_stamps, gt_poses[:, 2])
    iyaw = np.interp(s_stamps, gt_stamps, np.unwrap(gt_poses[:, 3]))       # unwrap: no +-pi glitch
    poses = np.stack([ix, iy, iz, np.arctan2(np.sin(iyaw), np.cos(iyaw))], axis=1).astype(np.float64)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with h5py.File(args.output, 'w') as h5f:
        h5f.create_dataset('timestamps_ns', data=s_stamps)
        h5f.create_dataset('images', data=s_imgs, chunks=(min(64, max(len(s_imgs), 1)), 4, IMG_H, IMG_W))
        h5f.create_dataset('dt_offsets', data=s_offs)
        h5f.create_dataset('poses', data=poses)
        if track_markers:
            h5f.create_dataset('track_borders', data=flat_points(track_markers))
            save_markers_h5(h5f, track_markers)
        _save_state_est(h5f, se)
        h5f.attrs['causal_sync'] = 1                    # built with the stamp <= anchor rule used at inference
        h5f.attrs['provenance'] = json.dumps(record, default=str)

    n = max(len(s_stamps), 1)
    print(f'[DONE] {len(s_stamps)} synchronized 4-camera samples -> {args.output} ({time.perf_counter() - t0:.1f}s)')
    print(f'[SYNC] anchors where a camera had already logged a future-stamped image (never used): '
          f'{n_future_logged} ({100.0 * n_future_logged / n:.1f}%)')
    print(f'[SYNC] anchors skipped (no eligible image yet): {n_incomplete} | non-increasing stamps: {n_nonmonotonic}')
    for i, name in enumerate(CAMERAS):
        off = s_offs[:, i] if len(s_offs) else np.zeros(1)
        print(f'[SYNC] {name}: age mean {off.mean() * 1e3:.1f} ms | p95 {np.percentile(off, 95) * 1e3:.1f} ms | '
              f'max {off.max() * 1e3:.1f} ms')
    _print_markers(track_markers)
    _print_state_est(se)
    report_odometry(s_stamps, poses, se)


def add_aux_only(args, record):
    """Upgrade an existing cache with per-marker borders + state estimation, without re-decoding the images."""
    config.require_file(args.output, 'cache', 'run stage 1 without --aux-only first')
    markers = read_track_bounds_mcap(args.mcap_path)
    se = []
    with open(args.mcap_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in reader.iter_decoded_messages(topics=[SE_TOPIC]):
            se.append(se_sample(msg))
    with h5py.File(args.output, 'a') as h5f:
        if not int(h5f.attrs.get('causal_sync', 0)):
            print('[WARN] this cache was NOT built with the causal camera sync: rebuild it (stage 1 without --aux-only)')
        if markers:
            if 'track_borders' in h5f:
                del h5f['track_borders']
            h5f.create_dataset('track_borders', data=flat_points(markers))
            save_markers_h5(h5f, markers)
        _save_state_est(h5f, se)
        h5f.attrs['provenance_aux'] = json.dumps(record, default=str)
    print(f'[DONE] wrote borders + state estimation into {args.output}')
    _print_markers(markers)
    _print_state_est(se)


if __name__ == '__main__':
    cfg = config.bootstrap()
    provenance.configure(cfg)
    ap = config.parser(__doc__, cfg)
    ap.add_argument('mcap_path', nargs='?', default=cfg.path('bag'), help='training bag (default [paths].bag)')
    ap.add_argument('--output', default=cfg.path('cache'))
    ap.add_argument('--export-gt', default=cfg.path('gt_csv'), help='write all /tf GT poses to this CSV')
    ap.add_argument('--no-export-gt', action='store_true')
    ap.add_argument('--buffer-len', type=int, default=int(cfg.get('build', 'buffer_len', 64)))
    ap.add_argument('--aux-only', action='store_true',
                    help='only (re)write per-marker borders + state estimation into an existing cache')
    a = ap.parse_args()
    config.require_file(a.mcap_path, 'bag', 'set [paths].bag in config.toml or pass it as the first argument')
    rec = provenance.run_record('1_build_dataset', a, inputs={'mcap': a.mcap_path}, cfg=cfg)
    print(provenance.banner(rec))
    (add_aux_only if a.aux_only else build_dataset)(a, rec)
=======

                # The previous version of this script used the newest LOGGED image of every camera,
                # which can be stamped after the anchor. Count how often that would have happened.
                if any(st > s_ns for b in buffers for st, _ in b):
                    n_future_logged += 1

                # Telemetry odometry from the previous anchor to this one, from samples that arrived so far.
                odo = telemetry.integrate(last_anchor_ns, s_ns) if last_anchor_ns is not None else None
                sample_odom.append(odo if odo is not None else (float('nan'),) * len(ODOM_FIELDS))

                last_anchor_ns = s_ns
                stacked = np.stack([img for _, img in picks], axis=0)  # (4, 128, 256)
                dt_offs = np.array([(s_ns - st) * 1e-9 for st, _ in picks], dtype=np.float32)
                sample_stamps.append(s_ns)
                sample_imgs.append(stacked)
                sample_dt_offsets.append(dt_offs)

                if len(sample_stamps) % 1000 == 0:
                    print(f'  Extracted {len(sample_stamps)} multi-camera frames ({time.perf_counter()-t0:.1f}s)...')

    gt_stamps = np.asarray(gt_stamps, dtype=np.int64)
    gt_poses = np.asarray(gt_poses, dtype=np.float64)
    sort_idx = np.argsort(gt_stamps)
    gt_stamps = gt_stamps[sort_idx]
    gt_poses = gt_poses[sort_idx]

    sample_stamps = np.asarray(sample_stamps, dtype=np.int64)
    # Keep only camera samples within valid GT time range
    valid_mask = (sample_stamps >= gt_stamps[0]) & (sample_stamps <= gt_stamps[-1])
    sample_stamps = sample_stamps[valid_mask]
    sample_imgs = np.asarray(sample_imgs, dtype=np.uint8)[valid_mask]
    sample_dt_offsets = np.asarray(sample_dt_offsets, dtype=np.float32)[valid_mask]
    sample_odom = np.asarray(sample_odom, dtype=np.float64)[valid_mask]

    # Linearly interpolate exact GT (x, y, z, yaw) at each camera_fl timestamp
    interp_x = np.interp(sample_stamps, gt_stamps, gt_poses[:, 0])
    interp_y = np.interp(sample_stamps, gt_stamps, gt_poses[:, 1])
    interp_z = np.interp(sample_stamps, gt_stamps, gt_poses[:, 2])
    # Unwrap yaw before linear interpolation so +/- pi crossings don't glitch
    unwrapped_yaw = np.unwrap(gt_poses[:, 3])
    interp_yaw = np.interp(sample_stamps, gt_stamps, unwrapped_yaw)
    interp_yaw = np.arctan2(np.sin(interp_yaw), np.cos(interp_yaw))

    poses_xyzyaw = np.stack([interp_x, interp_y, interp_z, interp_yaw], axis=1).astype(np.float64)

    os.makedirs(os.path.dirname(output_h5) or '.', exist_ok=True)
    with h5py.File(output_h5, 'w') as h5f:
        h5f.create_dataset('timestamps_ns', data=sample_stamps)
        h5f.create_dataset('images', data=sample_imgs, chunks=(64, 4, IMG_H, IMG_W))
        h5f.create_dataset('dt_offsets', data=sample_dt_offsets)
        h5f.create_dataset('poses', data=poses_xyzyaw)
        h5f.create_dataset('odom', data=sample_odom)          # telemetry odometry per anchor, see ODOM_FIELDS
        h5f.attrs['odom_fields'] = ','.join(ODOM_FIELDS)
        if track_borders:
            h5f.create_dataset('track_borders', data=np.asarray(track_borders, dtype=np.float64))
        # Marks caches built with the causal (stamp <= anchor) selection used at inference time.
        h5f.attrs['causal_sync'] = 1

    n = max(len(sample_stamps), 1)
    print(f'[DONE] Saved {len(sample_stamps)} synchronized 4-camera samples to {output_h5} in {time.perf_counter()-t0:.1f}s.')
    print(f'[SYNC] anchors where a camera had already logged a future-stamped image '
          f'(old builder would have used it): {n_future_logged} ({100.0 * n_future_logged / n:.1f}%)')
    print(f'[SYNC] anchors skipped (no eligible image yet): {n_incomplete} | non-increasing stamps: {n_nonmonotonic}')
    for i, name in enumerate(CAMERAS):
        off = sample_dt_offsets[:, i]
        print(f'[SYNC] {name}: age mean {off.mean() * 1e3:.1f} ms | p95 {np.percentile(off, 95) * 1e3:.1f} ms | '
              f'max {off.max() * 1e3:.1f} ms')
    print(f'[ODOM] state_estimation samples used: {telemetry.n_added} | rejected: {telemetry.n_rejected}')
    report_odometry(sample_stamps, poses_xyzyaw, sample_odom)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mcap_path', help='Path to c0_2026-10-13_22-53-42_camloc_0.mcap')
    parser.add_argument('--output', default='data/training_cache.h5', help='Output HDF5 cache')
    args = parser.parse_args()
    build_dataset(args.mcap_path, args.output)
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510
