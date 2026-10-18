import argparse
import math
import os
import time
import cv2
import h5py
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from track_map import TRACK_BOUNDS_TOPIC, flat_points, markers_from_msg, read_track_bounds_mcap, save_markers_h5

CAMERAS = ('camera_fl', 'camera_fr', 'camera_cl', 'camera_cr')
CAMERA_TOPICS = {f'/constructor0/sensor/{c}/compressed_image': i for i, c in enumerate(CAMERAS)}
IMG_W, IMG_H = 256, 128


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


def build_dataset(mcap_path: str, output_h5: str):
    print(f'[1/3] Scanning {mcap_path} for GT poses (/tf, /state_estimation), track_bounds, and 4 cameras...')
    t0 = time.perf_counter()

    gt_stamps = []
    gt_poses = []  # [x, y, z, yaw]
    track_markers = []  # one dict per border marker (left / right / pit lane ...), see track_map.py

    # Buffer for causal multi-camera sync anchored on camera_fl (index 0)
    latest_cam_img = [None] * len(CAMERAS)
    latest_cam_stamp = [0] * len(CAMERAS)

    sample_stamps = []
    sample_imgs = []      # (N, 4, H, W) uint8
    sample_dt_offsets = []  # (N, 4) float32 in seconds: (t_fl - t_cam)

    topics_to_read = list(CAMERA_TOPICS.keys()) + ['/tf', TRACK_BOUNDS_TOPIC]

    with open(mcap_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, _, msg in reader.iter_decoded_messages(topics=topics_to_read):
            topic = channel.topic

            if topic == '/tf':
                for t in getattr(msg, 'transforms', []) or []:
                    if t.header.frame_id.lstrip('/') == 'map' and t.child_frame_id.lstrip('/') == 'constructor0/base_link':
                        s_ns = stamp_to_ns(t.header.stamp)
                        p = t.transform.translation
                        yaw = yaw_from_quat(t.transform.rotation)
                        gt_stamps.append(s_ns)
                        gt_poses.append([float(p.x), float(p.y), float(p.z), yaw])

            elif topic == TRACK_BOUNDS_TOPIC and not track_markers:
                track_markers = markers_from_msg(msg)   # keep each border line separately

            elif topic in CAMERA_TOPICS:
                cam_idx = CAMERA_TOPICS[topic]
                s_ns = stamp_to_ns(msg.header.stamp)
                if not hasattr(msg, 'data') or not msg.data:
                    continue

                latest_cam_img[cam_idx] = crop_and_resize(msg.data)
                latest_cam_stamp[cam_idx] = s_ns

                # Trigger a synchronized causal sample every time camera_fl (idx 0) arrives
                if cam_idx == 0 and all(im is not None for im in latest_cam_img):
                    stacked = np.stack(latest_cam_img, axis=0)  # (4, 128, 256)
                    dt_offs = np.array(
                        [(s_ns - st) * 1e-9 for st in latest_cam_stamp], dtype=np.float32
                    )
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
        if track_markers:
            # flat array kept for older readers; the per-marker group keeps the line structure
            h5f.create_dataset('track_borders', data=flat_points(track_markers))
            save_markers_h5(h5f, track_markers)

    print(f'[DONE] Saved {len(sample_stamps)} synchronized 4-camera samples to {output_h5} in {time.perf_counter()-t0:.1f}s.')
    _print_markers(track_markers)


def _print_markers(markers):
    print(f'[BORDERS] {len(markers)} track_bounds markers')
    for m in markers:
        print(f'  ns={m["ns"]!r:24s} id={m["id"]:4d} type={m["type"]:2d} points={len(m["points"]):6d} frame={m["frame_id"]!r}')


def add_borders_only(mcap_path: str, output_h5: str):
    """Upgrade an existing cache with per-marker borders without re-decoding the images (reads one topic)."""
    markers = read_track_bounds_mcap(mcap_path)
    if not markers:
        raise SystemExit(f'No {TRACK_BOUNDS_TOPIC} markers found in {mcap_path}')
    with h5py.File(output_h5, 'a') as h5f:
        if 'track_borders' in h5f:
            del h5f['track_borders']
        h5f.create_dataset('track_borders', data=flat_points(markers))
        save_markers_h5(h5f, markers)
    print(f'[DONE] wrote per-marker borders into {output_h5}')
    _print_markers(markers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mcap_path', help='Path to c0_2026-10-13_22-53-42_camloc_0.mcap')
    parser.add_argument('--output', default='data/training_cache.h5', help='Output HDF5 cache')
    parser.add_argument('--borders-only', action='store_true',
                        help='only (re)write the per-marker track borders into an existing cache')
    args = parser.parse_args()
    if args.borders_only:
        add_borders_only(args.mcap_path, args.output)
    else:
        build_dataset(args.mcap_path, args.output)