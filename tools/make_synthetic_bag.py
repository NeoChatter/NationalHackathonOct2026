#!/usr/bin/env python3
"""
Write a small SYNTHETIC MCAP bag with the same topics and message fields as the organisers' recording, so the whole
pipeline (stages 1-6 + tools) can be smoke-tested end to end in a few minutes on any machine, before touching the
real 7 GB bag. Used by `python run_submission.py smoke`. Never use it for anything that is scored.

    python tools/make_synthetic_bag.py runs/smoke/synthetic.mcap [--laps 5] [--seed 0]

Content: a car drives laps on a circular track (radius 100 m, ~36 m/s, ~17.5 s per lap).
  /constructor0/sensor/camera_{fl,fr,cl,cr}/compressed_image   10 Hz JPEGs of a position-dependent texture; the
                                    side cameras are stamped slightly before / AFTER camera_fl and logged before it,
                                    so the causal camera selection is exercised
  /constructor0/state_estimation    100 Hz speed / yaw rate / accelerations (+ pose fields that must never be read)
  /tf                               100 Hz map -> constructor0/base_link (ground truth)
  /constructor0/track_bounds        two LINE_STRIP border markers (inner / outer edge)
"""
import argparse
import math
import os

import cv2
import numpy as np
from mcap_ros2.writer import Writer

SEP = '=' * 80
TIME = 'MSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n'
HEADER = 'MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n'
VEC3 = 'MSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n'
POINT = 'MSG: geometry_msgs/Point\nfloat64 x\nfloat64 y\nfloat64 z\n'
QUAT = 'MSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n'


def _msgdef(main: str, *deps: str) -> str:
    return main + ''.join(f'{SEP}\n{d}' for d in deps)


IMAGE_DEF = _msgdef('std_msgs/Header header\nstring format\nuint8[] data\n', HEADER, TIME)
TF_DEF = _msgdef('geometry_msgs/TransformStamped[] transforms\n',
                 'MSG: geometry_msgs/TransformStamped\nstd_msgs/Header header\nstring child_frame_id\n'
                 'geometry_msgs/Transform transform\n',
                 'MSG: geometry_msgs/Transform\ngeometry_msgs/Vector3 translation\ngeometry_msgs/Quaternion rotation\n',
                 VEC3, QUAT, HEADER, TIME)
MARKERS_DEF = _msgdef('visualization_msgs/Marker[] markers\n',
                      'MSG: visualization_msgs/Marker\nstd_msgs/Header header\nstring ns\nint32 id\nint32 type\n'
                      'int32 action\ngeometry_msgs/Pose pose\ngeometry_msgs/Point[] points\n',
                      'MSG: geometry_msgs/Pose\ngeometry_msgs/Point position\ngeometry_msgs/Quaternion orientation\n',
                      POINT, QUAT, HEADER, TIME)
SE_DEF = _msgdef('std_msgs/Header header\nfloat64 v_mps\nfloat64 wz_radps\nfloat64 ax_mps2\nfloat64 ay_mps2\n'
                 'float64 x_m\nfloat64 y_m\nfloat64 yaw_rad\n', HEADER, TIME)

CAM_TOPICS = [f'/constructor0/sensor/{c}/compressed_image' for c in ('camera_fl', 'camera_fr', 'camera_cl', 'camera_cr')]
CAM_STAMP_OFF = (0.0, -0.020, -0.045, +0.030)      # side-camera exposure relative to camera_fl [s]
CAM_LOG_DELAY = (0.060, 0.010, 0.010, 0.010)       # transport delay: camera_fl is logged last
CAM_VIEW = (0.0, 0.5 * math.pi, 0.15, -0.5 * math.pi)   # viewing direction relative to the heading


def _stamp(ns: int) -> dict:
    return {'sec': int(ns // 1_000_000_000), 'nanosec': int(ns % 1_000_000_000)}


def _header(ns: int, frame: str) -> dict:
    return {'stamp': _stamp(ns), 'frame_id': frame}


class World:
    """Position-dependent texture: a ring of random landmarks around the track, seen through a pinhole-ish strip."""

    def __init__(self, radius: float, seed: int):
        rng = np.random.default_rng(seed)
        n = 900
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        side = rng.choice([-1.0, 1.0], n)
        self.lm = np.stack([(radius + side * rng.uniform(9, 30, n)) * np.cos(ang),
                            (radius + side * rng.uniform(9, 30, n)) * np.sin(ang)], axis=1)
        self.h = rng.uniform(0.2, 1.0, n)
        self.b = rng.uniform(0.3, 1.0, n)

    def render(self, x, y, heading, w=192, h=96, rng=None) -> np.ndarray:
        img = np.full((h, w), 40.0, np.float32)
        img[h // 2:, :] = 70.0                                         # road
        d = self.lm - np.array([x, y])
        dist = np.hypot(d[:, 0], d[:, 1])
        bearing = np.arctan2(d[:, 1], d[:, 0]) - heading
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        vis = (np.abs(bearing) < 0.6) & (dist < 120) & (dist > 2)
        for i in np.flatnonzero(vis)[np.argsort(-dist[vis])]:
            u = int((0.5 - bearing[i] / 1.2) * w)
            size = max(int(400.0 / dist[i]), 1)
            top = int(h * (0.55 - 0.45 * self.h[i] * min(30.0 / dist[i], 1.0)))
            cv2.rectangle(img, (u - size, top), (u + size, h // 2 + size // 2), float(255 * self.b[i]), -1)
        if rng is not None:
            img += rng.normal(0, 4.0, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output')
    ap.add_argument('--laps', type=float, default=5.0)
    ap.add_argument('--radius', type=float, default=100.0)
    ap.add_argument('--speed', type=float, default=36.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    world = World(args.radius, args.seed)
    R, v = args.radius, args.speed
    w = v / R
    duration = args.laps * 2 * math.pi * R / v
    t0 = 1_760_000_000_000_000_000                                     # ns, arbitrary epoch

    def pose(t):                                                       # counter-clockwise circle
        th = w * t
        return R * math.cos(th), R * math.sin(th), th + 0.5 * math.pi

    msgs = []                                                          # (log_ns, kind, payload)
    for k in range(int(duration * 100)):                               # 100 Hz GT + state estimation
        t = k * 0.01
        ns = t0 + int(t * 1e9)
        x, y, yaw = pose(t)
        msgs.append((ns, 'tf', (ns, x, y, yaw)))
        msgs.append((ns + 2_000_000, 'se', (ns, v + rng.normal(0, 0.05), w + rng.normal(0, 0.002), x, y, yaw)))
    for k in range(int(duration * 10)):                                # 10 Hz cameras
        t_fl = k * 0.1 + 0.05
        for c in range(4):
            t = t_fl + CAM_STAMP_OFF[c]
            if t < 0:
                continue
            x, y, yaw = pose(t)
            img = world.render(x, y, yaw + CAM_VIEW[c], rng=rng)
            ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            ns = t0 + int(t * 1e9)
            msgs.append((ns + int(CAM_LOG_DELAY[c] * 1e9), 'cam', (c, ns, jpg.tobytes())))
    msgs.sort(key=lambda m: m[0])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'wb') as f:
        writer = Writer(f)
        s_img = writer.register_msgdef('sensor_msgs/msg/CompressedImage', IMAGE_DEF)
        s_tf = writer.register_msgdef('tf2_msgs/msg/TFMessage', TF_DEF)
        s_mk = writer.register_msgdef('visualization_msgs/msg/MarkerArray', MARKERS_DEF)
        s_se = writer.register_msgdef('synthetic_msgs/msg/StateEstimation', SE_DEF)

        th = np.linspace(0, 2 * np.pi, 721)
        markers = []
        for i, r in enumerate((R - 7.0, R + 7.0)):
            markers.append({'header': _header(t0, 'map'), 'ns': 'track_bounds', 'id': i, 'type': 4, 'action': 0,
                            'pose': {'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                                     'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}},
                            'points': [{'x': float(r * math.cos(a)), 'y': float(r * math.sin(a)), 'z': 0.0} for a in th]})
        writer.write_message('/constructor0/track_bounds', s_mk, {'markers': markers}, log_time=t0, publish_time=t0)

        for log_ns, kind, p in msgs:
            if kind == 'tf':
                ns, x, y, yaw = p
                m = {'transforms': [{'header': _header(ns, 'map'), 'child_frame_id': 'constructor0/base_link',
                                     'transform': {'translation': {'x': x, 'y': y, 'z': 0.0},
                                                   'rotation': {'x': 0.0, 'y': 0.0, 'z': math.sin(yaw / 2),
                                                                'w': math.cos(yaw / 2)}}}]}
                writer.write_message('/tf', s_tf, m, log_time=log_ns, publish_time=log_ns)
            elif kind == 'se':
                ns, sp, wz, x, y, yaw = p
                m = {'header': _header(ns, 'constructor0/base_link'), 'v_mps': float(sp), 'wz_radps': float(wz),
                     'ax_mps2': 0.0, 'ay_mps2': float(sp * wz), 'x_m': x, 'y_m': y, 'yaw_rad': yaw}
                writer.write_message('/constructor0/state_estimation', s_se, m, log_time=log_ns, publish_time=log_ns)
            else:
                c, ns, data = p
                m = {'header': _header(ns, 'camera'), 'format': 'jpeg', 'data': data}
                writer.write_message(CAM_TOPICS[c], s_img, m, log_time=log_ns, publish_time=log_ns)
        writer.finish()
    n_cam = sum(1 for m in msgs if m[1] == 'cam')
    print(f'[SYNTHETIC] {args.output}: {duration:.1f} s, {args.laps:g} laps of {2 * math.pi * R:.0f} m, '
          f'{n_cam} camera images, {sum(1 for m in msgs if m[1] == "tf")} GT poses')


if __name__ == '__main__':
    main()
