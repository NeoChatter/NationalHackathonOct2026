"""
One place for everything every stage must agree on: camera topics, the image crop, the causal camera-sync rule,
timestamps, angles and the ground-truth topic.

Camera sync (from the rmse-2.740 branch): a sample is anchored on every camera_fl frame and takes, from every camera,
the newest image whose header stamp is <= the anchor stamp (select_causal_image). The SAME function is used by the
cache builder (stage 1) and by inference (stage 4), so the model is trained and mapped on exactly the images it sees
at run time, and no pose ever uses an image stamped after its own timestamp.
"""
import math

import cv2
import numpy as np

CAMERAS = ('camera_fl', 'camera_fr', 'camera_cl', 'camera_cr')
CAMERA_TOPICS = {f'/constructor0/sensor/{c}/compressed_image': i for i, c in enumerate(CAMERAS)}
TF_TOPIC = '/tf'                    # ground truth: read by stage 1 (training labels) and the scoring tools ONLY
MAP_FRAME, BASE_FRAME = 'map', 'constructor0/base_link'

IMG_W, IMG_H = 256, 128
CROP_Y = (0.18, 0.74)               # vertical ROI: drops sky / floodlights (top) and nosecone (bottom)
BUFFER_LEN = 64                     # images kept per camera for the causal selection (config [build].buffer_len)


def crop_and_resize(jpeg_bytes: bytes) -> np.ndarray:
    """JPEG -> grayscale, vertical track ROI, resized to IMG_W x IMG_H (uint8)."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    h = img.shape[0]
    y0, y1 = int(CROP_Y[0] * h), int(CROP_Y[1] * h)
    return cv2.resize(img[y0:y1, :], (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def select_causal_image(buffer, anchor_ns: int):
    """Newest already-received (stamp, image) with stamp <= anchor_ns, or (None, None). Shared by stages 1 and 4."""
    best_stamp, best_image = None, None
    for stamp, image in buffer:
        if stamp <= anchor_ns and (best_stamp is None or stamp > best_stamp):
            best_stamp, best_image = stamp, image
    return best_stamp, best_image


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec * 1_000_000_000 + stamp.nanosec)


def wrap(a):
    """Angle(s) to (-pi, pi]; works for floats and numpy arrays."""
    return np.arctan2(np.sin(a), np.cos(a))


def is_base_tf(t) -> bool:
    """map -> constructor0/base_link transform (ground truth)."""
    return t.header.frame_id.lstrip('/') == MAP_FRAME and t.child_frame_id.lstrip('/') == BASE_FRAME
