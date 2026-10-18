"""
One place for everything every stage must agree on: camera topics, the image crop, timestamps, angles and the
ground-truth topic. (Previously copied between 1_build_dataset.py and locnet.py.)
"""
import math

import cv2
import numpy as np

CAMERAS = ('camera_fl', 'camera_fr', 'camera_cl', 'camera_cr')
CAMERA_TOPICS = {f'/constructor0/sensor/{c}/compressed_image': i for i, c in enumerate(CAMERAS)}
TF_TOPIC = '/tf'
MAP_FRAME, BASE_FRAME = 'map', 'constructor0/base_link'

IMG_W, IMG_H = 256, 128
CROP_Y = (0.18, 0.74)               # vertical ROI: drops sky / floodlights (top) and nosecone (bottom)
MAX_SET_SPREAD_NS = 300_000_000     # max stamp spread inside one 4-camera image set


def crop_and_resize(jpeg_bytes: bytes) -> np.ndarray:
    """JPEG -> grayscale, vertical track ROI, resized to IMG_W x IMG_H (uint8)."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    h = img.shape[0]
    y0, y1 = int(CROP_Y[0] * h), int(CROP_Y[1] * h)
    return cv2.resize(img[y0:y1, :], (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec * 1_000_000_000 + stamp.nanosec)


def wrap(a):
    """Angle(s) to (-pi, pi]; works for floats and numpy arrays."""
    return np.arctan2(np.sin(a), np.cos(a))


def _is_base_tf(t) -> bool:
    return t.header.frame_id.lstrip('/') == MAP_FRAME and t.child_frame_id.lstrip('/') == BASE_FRAME


def iter_gt_poses(mcap_path: str, min_ns: int = 0):
    """Yields (stamp_ns, x, y, z, yaw) of map -> base_link from /tf."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in reader.iter_decoded_messages(topics=[TF_TOPIC]):
            for t in getattr(msg, 'transforms', []) or []:
                if not _is_base_tf(t):
                    continue
                s = stamp_to_ns(t.header.stamp)
                if s >= min_ns:
                    p = t.transform.translation
                    yield s, float(p.x), float(p.y), float(p.z), yaw_from_quat(t.transform.rotation)


def read_first_pose(mcap_path: str, min_ns: int = 0):
    """DIAGNOSTIC ONLY (4_infer.py --init first-pose on the training bag): the first GT pose of the run.
    No start pose is given on the test run, so submissions must use global initialisation. None if no /tf."""
    for _, x, y, _, yaw in iter_gt_poses(mcap_path, min_ns):
        return x, y, yaw
    return None
