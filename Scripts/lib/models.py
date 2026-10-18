"""
Networks, network-input preprocessing, training augmentation and test-time adaptation.

Preprocessing is a property of the CHECKPOINT (ck['preproc']), so training, calibration and inference can never
disagree. Old checkpoints have no entry -> {'clahe': False, 'norm': 'std'} = the original behaviour.

  clahe : local contrast equalisation. Robust to global contrast changes (haze, overcast); keeps local structure.
  norm  : 'std'    per-camera mean / std (original)
          'robust' per-camera median / MAD, clipped. A large bright or dark region (e.g. a visible sky or a shadow
                   that is absent in other conditions) dominates the mean / std and squashes the road; the median /
                   MAD are insensitive to it.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .common import IMG_H, IMG_W  # noqa: F401  (re-exported for the scripts)

DEFAULT_PREPROC = dict(clahe=False, norm='std')


def preproc_cfg(preproc=None) -> dict:
    return {**DEFAULT_PREPROC, **(preproc or {})}


# ----------------------------------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------------------------------
def normalize_cams(x: np.ndarray, mode: str = 'std') -> np.ndarray:
    """x: float32 in [0,1], shape (..., 4, H, W). Per-camera normalisation."""
    if mode == 'robust':
        med = np.median(x, axis=(-2, -1), keepdims=True)
        mad = np.median(np.abs(x - med), axis=(-2, -1), keepdims=True) * 1.4826
        return np.clip((x - med) / (mad + 1e-3), -4.0, 4.0).astype(np.float32)
    m = x.mean(axis=(-2, -1), keepdims=True)
    s = x.std(axis=(-2, -1), keepdims=True)
    return ((x - m) / (s + 1e-5)).astype(np.float32)


_CLAHE = None


def _clahe():
    global _CLAHE
    if _CLAHE is None:                                        # created lazily: one per process / DataLoader worker
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8))
    return _CLAHE


def clahe_u8(x: np.ndarray) -> np.ndarray:
    """Local contrast equalisation of every (H,W) image in a (..., H, W) uint8 array."""
    flat = x.reshape(-1, x.shape[-2], x.shape[-1])
    out = np.empty_like(flat)
    c = _clahe()
    for i in range(len(flat)):
        out[i] = c.apply(flat[i])
    return out.reshape(x.shape)


def preprocess(x: np.ndarray, preproc=None) -> np.ndarray:
    """(..., 4, H, W) uint8, or float in [0,1] (after augmentation) -> float32 network input."""
    cfg = preproc_cfg(preproc)
    if cfg['clahe']:
        u8 = x if x.dtype == np.uint8 else (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        x = clahe_u8(u8).astype(np.float32) / 255.0
    elif x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    return normalize_cams(x, cfg['norm'])


# ----------------------------------------------------------------------------------------------
# training augmentation
#   basic      : the original photometric augmentation (unchanged, same random draws)
#   structured : a BROAD, SYMMETRIC set of generic weather / time-of-day / illumination effects, so the place and
#                odometry models do not depend on any one condition. It targets no particular run: brightening and
#                darkening are equally likely (gamma both sides of 1, gain both sides of 1), and the effects below
#                are drawn independently at similar rates. Only effects that per-camera normalisation does NOT
#                already remove are included (a uniform gain/offset is normalised away; spatially varying light,
#                saturation, haze and compression are not).
#                  grad      directional illumination ramp (sun / lights from a side)
#                  shadow    a soft-edged shadow band across the frame
#                  glare     a bright bloom (sun, floodlight, reflection)
#                  haze      contrast fades with distance (toward the top of the crop)
#                  skyband   a brighter OR darker band at the top (sky visible or not, either way)
#                  contrast  global contrast lower OR higher (flat overcast .. harsh light)
#                  erase_top randomly blank the upper band so the descriptor relies on the track surface / kerbs /
#                            barriers (present in all conditions), not on the background above it (condition-specific)
#                  jpeg      compression artefacts
#   The stress suite (perturb.py) uses different formulas / ranges, so it is never the training distribution.
# ----------------------------------------------------------------------------------------------
def sample_aug(rng, level: str = 'basic'):
    out = []
    for _ in range(4):
        p = dict(gamma=rng.uniform(0.7, 1.5), gain=rng.uniform(0.6, 1.6),
                 blur=int(rng.choice([1, 1, 3, 5, 9])), noise=rng.uniform(0.0, 0.03),
                 drop=bool(rng.random() < 0.05))
        if level == 'structured':
            p.update(gamma=rng.uniform(0.4, 2.2), gain=rng.uniform(0.4, 2.4), clip=True)   # both sides of 1
            if rng.random() < 0.4:
                p['grad'] = (rng.uniform(0.2, 0.8), rng.uniform(-math.pi, math.pi))
            if rng.random() < 0.4:
                p['shadow'] = (rng.uniform(-0.6, 0.6), rng.uniform(0.1, 0.4), rng.uniform(-math.pi, math.pi),
                               rng.uniform(0.25, 0.7))                       # centre, width, angle, factor
            if rng.random() < 0.4:
                p['glare'] = (rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.08, 0.3),
                              rng.uniform(0.3, 1.0))                         # cx, cy, radius, amplitude
            if rng.random() < 0.4:
                p['haze'] = (rng.uniform(0.5, 2.5), rng.uniform(0.3, 0.95))   # density, airlight (dark .. bright)
            if rng.random() < 0.4:
                # brighter OR darker band at the top; mix of either sign, so the model never assumes which
                p['skyband'] = (rng.uniform(0.1, 0.45), rng.uniform(0.0, 1.0), rng.uniform(-0.8, 0.8))  # height, level, mix
            if rng.random() < 0.4:
                p['contrast'] = rng.uniform(0.35, 1.6)                        # lower OR higher than 1
            if rng.random() < 0.3:
                p['erase_top'] = (rng.uniform(0.1, 0.4), rng.uniform(0.0, 1.0))  # band height, fill level
            if rng.random() < 0.3:
                p['jpeg'] = int(rng.integers(10, 60))
        out.append(p)
    return out


def _uv(h: int, w: int):
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    return (u / max(w - 1, 1)) * 2.0 - 1.0, (v / max(h - 1, 1)) * 2.0 - 1.0     # [-1,1] x [-1,1]


def apply_aug(img_u8: np.ndarray, params, rng) -> np.ndarray:
    """img_u8: (4,H,W) uint8 -> (4,H,W) float32 in ~[0,1]."""
    x = img_u8.astype(np.float32) / 255.0
    out = np.empty_like(x)
    H, W = x.shape[-2:]
    for c, p in enumerate(params):
        if p['drop']:
            out[c] = 0.0
            continue
        y = np.power(np.clip(x[c], 0.0, 1.0), p['gamma']) * p['gain']
        if any(k in p for k in ('grad', 'shadow', 'glare', 'haze', 'skyband', 'erase_top')):
            u, v = _uv(H, W)
            v01 = 0.5 * (v + 1.0)                             # 0 = top of the crop (far), 1 = bottom (near)
            if 'grad' in p:                                   # directional illumination ramp
                amp, ang = p['grad']
                y = y * (1.0 + amp * (math.cos(ang) * u + math.sin(ang) * v))
            if 'shadow' in p:                                 # slanted shadow band (soft edges)
                c0, wid, ang, fac = p['shadow']
                d = np.abs(math.cos(ang) * u + math.sin(ang) * v - c0)
                m = np.clip((wid - d) / 0.05, 0.0, 1.0)
                y = y * (1.0 - (1.0 - fac) * m)
            if 'glare' in p:                                  # bright bloom (sun / light / reflection)
                cx, cy, r, a = p['glare']
                y = y + a * np.exp(-((u - (2 * cx - 1)) ** 2 + (v - (2 * cy - 1)) ** 2) / (2.0 * r * r))
            if 'haze' in p:                                   # atmospheric scattering: far rows fade to airlight
                beta, A = p['haze']
                t = np.exp(-beta * (1.0 - v01) ** 2)
                y = y * t + A * (1.0 - t)
            if 'skyband' in p:                                # a brighter OR darker band at the top (mix sign)
                hz, lvl, mix = p['skyband']
                m = np.clip((hz - v01) / 0.04, 0.0, 1.0)
                y = y * (1.0 - abs(mix) * m) + (mix > 0) * abs(mix) * m * lvl
            if 'erase_top' in p:                              # do not rely on the upper band at all
                hgt, fill = p['erase_top']
                y = np.where(v01 < hgt, fill, y)
        if 'contrast' in p:                                   # global contrast change (flat .. harsh)
            mu = float(y.mean())
            y = mu + (y - mu) * p['contrast']
        if p['blur'] > 1:
            y = cv2.blur(y, (p['blur'], 1))                   # horizontal smear ~ side-camera motion blur
        if p['noise'] > 0:
            y = y + rng.normal(0.0, p['noise'], y.shape).astype(np.float32)
        if p.get('clip'):
            y = np.clip(y, 0.0, 1.0)                          # sensor saturation is real: keep it
        if 'jpeg' in p:
            u8 = (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            ok, enc = cv2.imencode('.jpg', u8, [cv2.IMWRITE_JPEG_QUALITY, int(p['jpeg'])])
            if ok:
                y = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        out[c] = y
    return out


# ----------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------
def _resnet_trunk(in_ch: int, pretrained: bool = True) -> nn.Module:
    bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    old = bb.conv1
    bb.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        w = old.weight.mean(dim=1, keepdim=True)            # (64,1,7,7)
        bb.conv1.weight.copy_(w.repeat(1, in_ch, 1, 1) / in_ch)
    return nn.Sequential(*list(bb.children())[:-1])          # -> (N,512,1,1)


class PlaceNet(nn.Module):
    """4 cameras -> one L2-normalised place descriptor. (No BatchNorm in the head: no train/eval gap.)"""

    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.encoder = _resnet_trunk(1, pretrained)
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
    Frame-to-frame ego-motion. EARLY fusion: the first conv sees [prev, curr, curr-prev] of the same camera, so it
    can measure pixel displacement. Output: dx_body [m], dy_body [m], dyaw [rad] over the interval dt.
    """
    SCALE = (4.0, 0.5, 0.05)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = _resnet_trunk(3, pretrained)
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


def load_networks(ck: dict, device):
    """Both networks from a checkpoint, in eval mode (no ImageNet download: every weight comes from ck)."""
    place = PlaceNet(256, pretrained=False).to(device)
    place.load_state_dict(ck['place_state'])
    odom = OdomNet(pretrained=False).to(device)
    odom.load_state_dict(ck['odom_state'])
    return place.eval(), odom.eval()


# ----------------------------------------------------------------------------------------------
# test-time adaptation (optional, --adabn in 4_infer.py)
# ----------------------------------------------------------------------------------------------
def enable_test_time_bn(model: nn.Module, momentum: float = 0.01):
    """
    AdaBN-style causal adaptation: every BatchNorm keeps eval-mode behaviour but its running mean / variance follow
    the incoming frames (exponential average updated just before the current frame is normalised, so only past and
    current images are used). Moves the feature statistics from the recorded conditions toward the run's own, for
    any condition. The database was embedded with the recorded statistics, so use it together with the experience
    memory and verify with the stress suite. Returns the hook handles (call .remove() to stop adapting).
    """
    handles = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            def hook(mod, inp, mom=momentum):
                x = inp[0]
                with torch.no_grad():
                    mod.running_mean.mul_(1.0 - mom).add_(mom * x.mean(dim=(0, 2, 3)))
                    mod.running_var.mul_(1.0 - mom).add_(mom * x.var(dim=(0, 2, 3), unbiased=False))
            handles.append(m.register_forward_pre_hook(hook))
    return handles
