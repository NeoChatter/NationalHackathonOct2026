"""
Training-time appearance augmentation. Never changes image geometry: the camera mount is fixed and the pose is
encoded in the image geometry, so moving pixels would corrupt the labels.

Two layers, both drawn ONCE per training pair (same draw for the previous and the current frame, so the VO head
still sees a photometrically consistent pair; different draws for different samples, so an anchor and its
same-place partner usually look differently lit):

  1. PhotometricAugment (rmse-2.740 branch, torch): gamma, exposure + clipping, raised black level, sky glow,
     smooth shadow field, blur, with per-camera jitter. Probability [augment].photo_aug_p, strength photo_aug.
  2. Structured effects (rmse-4.363 branch, lib/models.sample_aug 'structured', numpy): directional illumination
     ramp, shadow band, glare, haze, brighter/darker sky band, global contrast, erased upper band, JPEG artefacts,
     camera drop-out. Applied with probability [augment].structured_p on top of layer 1. The base gamma / gain /
     blur / noise of the original 'structured' level are NOT drawn again here, because layer 1 already covers them.

The stress suite (perturb.py) uses different formulas / ranges, so it is never the training distribution.
"""
import math
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------
# layer 1: PhotometricAugment (V, unchanged)
# ----------------------------------------------------------------------------------------------
class PhotometricAugment:
    """strength scales every range (1.0 = defaults, 0 = off). p = probability a sample is augmented at all."""

    def __init__(self, strength=1.0, p=0.85):
        self.s = float(strength)
        self.p = float(p)

    def sample(self, n_cams=4):
        if self.s <= 0.0 or random.random() > self.p:
            return None
        s = self.s

        def log_uniform(lo, hi):
            return math.exp(random.uniform(s * math.log(lo), s * math.log(hi)))

        return dict(
            gamma=log_uniform(0.5, 2.2) * torch.exp(0.10 * s * torch.randn(n_cams, 1, 1)),
            gain=log_uniform(0.5, 3.0) * torch.exp(0.15 * s * torch.randn(n_cams, 1, 1)),
            offset=random.uniform(0.0, 0.20) * s if random.random() < 0.5 else 0.0,
            sky=random.uniform(0.0, 0.25) * s if random.random() < 0.5 else 0.0,
            field=(0.25 * s * torch.randn(n_cams, 1, 4, 8)) if random.random() < 0.5 else None,
            blur=random.choice([0, 0, 3, 5]),
        )

    @staticmethod
    def apply(x: torch.Tensor, prm) -> torch.Tensor:
        """x: (4, H, W) float in [0, 1]."""
        H, W = x.shape[-2:]
        x = x.clamp(0.0, 1.0) ** prm['gamma']
        x = x * prm['gain']
        if prm['field'] is not None:
            field = F.interpolate(prm['field'], size=(H, W), mode='bilinear', align_corners=False)
            x = x * torch.exp(field.squeeze(1))
        x = x + prm['offset']
        if prm['sky'] > 0.0:
            profile = torch.linspace(1.0, 0.0, H).view(1, H, 1) ** 1.5
            x = x + prm['sky'] * profile
        x = x.clamp(0.0, 1.0)
        k = prm['blur']
        if k > 1:
            x = F.avg_pool2d(x.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)[0]
        return x


# ----------------------------------------------------------------------------------------------
# layer 2: structured effects (S, sample_aug / apply_aug of the 'structured' level without the base draws)
# ----------------------------------------------------------------------------------------------
def sample_structured(rng, n_cams=4):
    out = []
    for _ in range(n_cams):
        p = dict(drop=bool(rng.random() < 0.05), clip=True)
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


def apply_structured(x: np.ndarray, params) -> np.ndarray:
    """x: (4,H,W) float32 in [0,1] -> (4,H,W) float32 in [0,1]."""
    out = np.empty_like(x, dtype=np.float32)
    H, W = x.shape[-2:]
    for c, p in enumerate(params):
        if p['drop']:
            out[c] = 0.0
            continue
        y = np.asarray(x[c], dtype=np.float32)
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
        if p.get('clip'):
            y = np.clip(y, 0.0, 1.0)                          # sensor saturation is real: keep it
        if 'jpeg' in p:
            u8 = (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            ok, enc = cv2.imencode('.jpg', u8, [cv2.IMWRITE_JPEG_QUALITY, int(p['jpeg'])])
            if ok:
                y = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        out[c] = y
    return out


def sample_rng():
    """Generator for one sample, drawn from Python's `random` (seeded by seed_everything in the main process and by
    PyTorch in every DataLoader worker), so draws differ between epochs and a run with the same seed and the same
    number of workers is reproducible."""
    return np.random.default_rng(random.getrandbits(64))
