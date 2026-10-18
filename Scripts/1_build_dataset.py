"""
STAGE 1 - bag -> training cache.

  python Scripts/1_build_dataset.py Data/c0_2026-10-13_22-53-42_camloc_0.mcap --output data/training_cache.h5 \
         --export-gt data/gt.csv
  python Scripts/1_build_dataset.py <bag> --output data/training_cache.h5 --aux-only   # borders + state estimation
                                                                                         # only, no image re-decode
Writes data/training_cache.h5:
  timestamps_ns (N,)          camera_fl stamp of every 4-camera sample
  images        (N,4,H,W)     cropped grayscale, uint8 (lib/common.crop_and_resize)
  dt_offsets    (N,4)         camera_fl stamp minus each camera's stamp [s]
  poses         (N,4)         GT [x, y, z, yaw] interpolated at the camera_fl stamp (yaw unwrapped first)
                              (training / calibration targets only, never an inference input)
  track_borders (M,3)         all border points (legacy flat layout)
  track_bounds/marker_XXX     one dataset per border marker (left / right / pit ...) with ns / id / type attrs
  state_est/{t_ns,v_mps,wz_radps,ax_mps2,ay_mps2}
                              ONLY the allowed state-estimation fields (lib/state_est.SE_FIELDS), full rate
  attrs['provenance']         JSON run record (git, args, bag fingerprint)
--export-gt writes every /tf pose as CSV (timestamp_ns,x,y,z,yaw), the same format as tools/extract_gt.py.
"""
import argparse
import csv
import json
import os
import sys
import time

import h5py
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # make `lib` importable from any CWD

from lib import provenance  # noqa: E402
from lib.common import (CAMERA_TOPICS, CAMERAS, IMG_H, IMG_W, TF_TOPIC, _is_base_tf, crop_and_resize,  # noqa: E402
                        stamp_to_ns, yaw_from_quat)
from lib.state_est import SE_FIELDS, SE_TOPIC, se_sample  # noqa: E402
from lib.track_map import (TRACK_BOUNDS_TOPIC, flat_points, markers_from_msg, read_track_bounds_mcap,  # noqa: E402
                           save_markers_h5)


def _save_state_est(h5f, samples):
    if 'state_est' in h5f:
        del h5f['state_est']
    if not samples:
        return
    a = np.asarray(sorted(samples), dtype=np.float64)
    g = h5f.create_group('state_est')
    g.create_dataset('t_ns', data=a[:, 0].astype(np.int64))
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


def build_dataset(args, record):
    print(f'[1/3] Scanning {args.mcap_path} for GT poses (/tf), track_bounds, state estimation and 4 cameras...')
    t0 = time.perf_counter()
    gt_stamps, gt_poses, track_markers, se = [], [], [], []
    latest_img = [None] * len(CAMERAS)
    latest_stamp = [0] * len(CAMERAS)
    s_stamps, s_imgs, s_offs = [], [], []

    topics = list(CAMERA_TOPICS) + [TF_TOPIC, TRACK_BOUNDS_TOPIC, SE_TOPIC]
    with open(args.mcap_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, _, msg in reader.iter_decoded_messages(topics=topics):
            topic = channel.topic
            if topic == TF_TOPIC:
                for t in getattr(msg, 'transforms', []) or []:
                    if _is_base_tf(t):
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
                latest_img[cam] = crop_and_resize(msg.data)
                latest_stamp[cam] = s_ns
                if cam == 0 and all(im is not None for im in latest_img):   # camera_fl closes a sample
                    s_stamps.append(s_ns)
                    s_imgs.append(np.stack(latest_img, axis=0))
                    s_offs.append(np.array([(s_ns - st) * 1e-9 for st in latest_stamp], dtype=np.float32))
                    if len(s_stamps) % 1000 == 0:
                        print(f'  {len(s_stamps)} multi-camera frames ({time.perf_counter() - t0:.1f}s)...')

    gt_stamps = np.asarray(gt_stamps, dtype=np.int64)
    gt_poses = np.asarray(gt_poses, dtype=np.float64)
    o = np.argsort(gt_stamps)
    gt_stamps, gt_poses = gt_stamps[o], gt_poses[o]

    if args.export_gt:
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
    iyaw = np.arctan2(np.sin(iyaw), np.cos(iyaw))
    poses = np.stack([ix, iy, iz, iyaw], axis=1).astype(np.float64)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with h5py.File(args.output, 'w') as h5f:
        h5f.create_dataset('timestamps_ns', data=s_stamps)
        h5f.create_dataset('images', data=s_imgs, chunks=(64, 4, IMG_H, IMG_W))
        h5f.create_dataset('dt_offsets', data=s_offs)
        h5f.create_dataset('poses', data=poses)
        if track_markers:
            h5f.create_dataset('track_borders', data=flat_points(track_markers))
            save_markers_h5(h5f, track_markers)
        _save_state_est(h5f, se)
        h5f.attrs['provenance'] = json.dumps(record, default=str)
    print(f'[DONE] {len(s_stamps)} synchronized 4-camera samples -> {args.output} ({time.perf_counter() - t0:.1f}s)')
    _print_markers(track_markers)
    _print_state_est(se)


def _print_markers(markers):
    print(f'[BORDERS] {len(markers)} track_bounds markers')
    for m in markers:
        print(f'  ns={m["ns"]!r:24s} id={m["id"]:4d} type={m["type"]:2d} points={len(m["points"]):6d} frame={m["frame_id"]!r}')


def add_aux_only(args, record):
    """Upgrade an existing cache with per-marker borders + state estimation, without re-decoding the images."""
    markers = read_track_bounds_mcap(args.mcap_path)
    se = []
    with open(args.mcap_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in reader.iter_decoded_messages(topics=[SE_TOPIC]):
            se.append(se_sample(msg))
    with h5py.File(args.output, 'a') as h5f:
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
    ap = argparse.ArgumentParser()
    ap.add_argument('mcap_path', help='e.g. Data/c0_2026-10-13_22-53-42_camloc_0.mcap')
    ap.add_argument('--output', default='data/training_cache.h5')
    ap.add_argument('--export-gt', default='', help='also write all /tf GT poses to this CSV')
    ap.add_argument('--aux-only', action='store_true',
                    help='only (re)write per-marker borders + state estimation into an existing cache')
    ap.add_argument('--borders-only', action='store_true', help=argparse.SUPPRESS)   # old name of --aux-only
    a = ap.parse_args()
    if not a.mcap_path or not os.path.exists(a.mcap_path):
        raise SystemExit(f'bag not found: {a.mcap_path!r} (is $BAG set?)')
    rec = provenance.run_record('1_build_dataset', a, inputs={'mcap': a.mcap_path})
    print(provenance.banner(rec))
    (add_aux_only if (a.aux_only or a.borders_only) else build_dataset)(a, rec)
