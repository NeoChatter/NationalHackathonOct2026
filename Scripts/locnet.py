"""
Shared code for 2_train_model.py and 3_causal_infer.py (replaces the importlib hack).

NOTE: crop_and_resize / yaw_from_quat / stamp_to_ns are copies of the functions in
1_build_dataset.py. Script 1 itself is unchanged. If you ever change the crop, change it in both places.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

CAMERAS = ('camera_fl', 'camera_fr', 'camera_cl', 'camera_cr')
CAMERA_TOPICS = {f'/constructor0/sensor/{c}/compressed_image': i for i, c in enumerate(CAMERAS)}
IMG_W, IMG_H = 256, 128


# ----------------------------------------------------------------------------------------------
# preprocessing (identical to script 1)
# ----------------------------------------------------------------------------------------------
def crop_and_resize(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    h, w = img.shape[:2]
    y0, y1 = int(0.18 * h), int(0.74 * h)
    return cv2.resize(img[y0:y1, :], (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec * 1_000_000_000 + stamp.nanosec)


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def normalize_cams(x: np.ndarray) -> np.ndarray:
    """x: float32 in [0,1], shape (..., 4, H, W). Per-camera zero-mean / unit-variance."""
    m = x.mean(axis=(-2, -1), keepdims=True)
    s = x.std(axis=(-2, -1), keepdims=True)
    return ((x - m) / (s + 1e-5)).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# augmentation (the hidden run can be faster / differently lit: simulate gamma, gain, blur, noise)
# ----------------------------------------------------------------------------------------------
def sample_aug(rng):
    return [dict(gamma=rng.uniform(0.7, 1.5), gain=rng.uniform(0.6, 1.6),
                 blur=int(rng.choice([1, 1, 3, 5, 9])), noise=rng.uniform(0.0, 0.03),
                 drop=bool(rng.random() < 0.05)) for _ in range(4)]


def apply_aug(img_u8: np.ndarray, params, rng) -> np.ndarray:
    """img_u8: (4,H,W) uint8 -> (4,H,W) float32 in ~[0,1]."""
    x = img_u8.astype(np.float32) / 255.0
    out = np.empty_like(x)
    for c, p in enumerate(params):
        if p['drop']:
            out[c] = 0.0
            continue
        y = np.power(np.clip(x[c], 0.0, 1.0), p['gamma']) * p['gain']
        if p['blur'] > 1:
            y = cv2.blur(y, (p['blur'], 1))  # horizontal smear ~ side-camera motion blur
        if p['noise'] > 0:
            y = y + rng.normal(0.0, p['noise'], y.shape).astype(np.float32)
        out[c] = y
    return out


# ----------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------
def _resnet_trunk(in_ch: int) -> nn.Module:
    bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    old = bb.conv1
    bb.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        w = old.weight.mean(dim=1, keepdim=True)            # (64,1,7,7)
        bb.conv1.weight.copy_(w.repeat(1, in_ch, 1, 1) / in_ch)
    return nn.Sequential(*list(bb.children())[:-1])          # -> (N,512,1,1)


class PlaceNet(nn.Module):
    """4 cameras -> one L2-normalised place descriptor. (No BatchNorm in the head: no train/eval gap.)"""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.encoder = _resnet_trunk(1)
        self.cam_emb = nn.Parameter(torch.randn(1, 4, 512) * 0.02)
        self.attn_gate = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.Linear(512 * 4, 512), nn.LayerNorm(512), nn.GELU(),
                                  nn.Linear(512, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (B,4,H,W)
        B, C, H, W = x.shape
        f = self.encoder(x.reshape(B * C, 1, H, W)).reshape(B, C, 512) + self.cam_emb
        f = f * torch.sigmoid(self.attn_gate(f))
        return F.normalize(self.head(f.reshape(B, C * 512)), p=2, dim=-1)


class OdomNet(nn.Module):
    """
    Frame-to-frame ego-motion. EARLY fusion: the first conv sees [prev, curr, curr-prev] of the same
    camera, so it can measure pixel displacement. (Your old head only saw separately pooled features,
    which cannot encode displacement.) Output: dx_body [m], dy_body [m], dyaw [rad] over the interval dt.
    """
    SCALE = (4.0, 0.5, 0.05)

    def __init__(self):
        super().__init__()
        self.encoder = _resnet_trunk(3)
        self.cam_emb = nn.Parameter(torch.randn(1, 4, 512) * 0.02)
        self.head = nn.Sequential(nn.Linear(512 * 4 + 1, 512), nn.LayerNorm(512), nn.GELU(),
                                  nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 3))
        self.register_buffer('scale', torch.tensor(self.SCALE, dtype=torch.float32))

    def forward(self, prev: torch.Tensor, curr: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        B, C, H, W = curr.shape                               # dt: (B,1) seconds
        x = torch.stack([prev, curr, curr - prev], dim=2)     # (B,4,3,H,W)
        f = self.encoder(x.reshape(B * C, 3, H, W)).reshape(B, C, 512) + self.cam_emb
        out = self.head(torch.cat([f.reshape(B, C * 512), dt * 10.0], dim=-1))
        return out * self.scale
