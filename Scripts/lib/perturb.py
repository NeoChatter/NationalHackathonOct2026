"""
Stress perturbations for 4_infer.py (--perturb 'name:severity[,name:severity...]').

Applied ONLY to the incoming camera image sets (after crop + resize, before the network). The place database stays
as it was built, i.e. this mimics running in conditions that differ from the recorded ones. Severity is 0 (none)
.. 3 (severe); fractional values interpolate. Seeded, so runs are reproducible.

Implementations / ranges intentionally differ from the training augmentation in lib/augment.py, so a good stress
score is not just "the model saw exactly this during training".

  photometric   bright, dark, gamma, contrast, clip (highlight saturation), noise, blur (motion), jpeg
  structured    glare (blooming blobs), shadow (hard-edged shadow bands), vignette,
                haze (contrast fades toward the horizon), sky (a bright OR dark band at the top of the frame)
  composite     daylight  (= gamma + gain + lower contrast + shadow bands + glare)
                overcast  (= haze + flat contrast + a brightened sky band + mild vignette): diffuse, low-contrast
                           light with the background made visible
                low_sun   (= haze + strong directional shadow + horizon glare): sun low in the sky
                rain      (= blur + contrast loss + noise + droplets)
  sensor        cam_drop (cameras black), cam_freeze (cameras stuck), frame_drop (whole image sets lost)

These are generic weather / time-of-day perturbations for a robustness check. They approximate conditions, they
do not reproduce any specific run, and none was tuned against evaluation data.
"""
import cv2
import numpy as np

_SEV = np.array([0.0, 1.0, 2.0, 3.0])
# parameter at severity 0, 1, 2, 3
_TABLE = {
    'bright': [1.0, 1.5, 2.0, 2.8],        # gain (saturating)
    'dark': [1.0, 0.55, 0.35, 0.2],        # gain
    'gamma': [1.0, 0.7, 0.5, 0.35],        # x ** g  (< 1 lifts shadows: flat daylight look)
    'contrast': [1.0, 0.7, 0.5, 0.3],      # factor around the mean
    'clip': [1.0, 0.85, 0.7, 0.55],        # values above this quantile saturate
    'noise': [0.0, 6.0, 12.0, 22.0],       # sigma, grey levels
    'blur': [1.0, 5.0, 9.0, 15.0],         # horizontal kernel length, pixels
    'jpeg': [100.0, 35.0, 18.0, 8.0],      # quality
    'glare': [0.0, 1.0, 2.0, 4.0],         # number of blobs
    'shadow': [0.0, 1.0, 2.0, 3.0],        # number of bands
    'vignette': [0.0, 0.3, 0.55, 0.8],     # corner darkening
    'haze': [1.0, 0.6, 0.4, 0.25],         # transmission at the top of the crop (1 = clear)
    'sky': [0.0, 0.15, 0.25, 0.35],        # height of the bright sky band (fraction of the crop)
    'cam_drop': [0.0, 1.0, 2.0, 3.0],      # cameras forced black (never camera_fl)
    'cam_freeze': [0.0, 1.0, 2.0, 3.0],    # cameras stuck on their first frame (never camera_fl)
    'frame_drop': [0.0, 0.05, 0.15, 0.3],  # probability of losing an image set
    'daylight': [0.0, 1.0, 2.0, 3.0],
    'overcast': [0.0, 1.0, 2.0, 3.0],
    'low_sun': [0.0, 1.0, 2.0, 3.0],
    'rain': [0.0, 1.0, 2.0, 3.0],
}
_CAM_ORDER = (1, 3, 2)                     # fr, cr, cl: which secondary cameras fail first


def _p(name, sev):
    return float(np.interp(sev, _SEV, _TABLE[name]))


class Perturber:
    def __init__(self, spec: str, seed: int = 0):
        self.ops = []
        for item in [t.strip() for t in spec.split(',') if t.strip()]:
            name, _, sev = item.partition(':')
            name = name.strip()
            if name not in _TABLE:
                raise ValueError(f'unknown perturbation {name!r}; choose from {sorted(_TABLE)}')
            self.ops.append((name, float(sev) if sev else 1.0))
        self.rng = np.random.default_rng(seed)
        self.frozen = {}
        self._bands = None                 # shadow bands drift slowly instead of jumping every frame

    def describe(self) -> str:
        return ', '.join(f'{n}:{s:g}' for n, s in self.ops) or 'none'

    def drop_frame(self) -> bool:
        p = sum(_p('frame_drop', s) for n, s in self.ops if n == 'frame_drop')
        return bool(p > 0 and self.rng.random() < p)

    # ------------------------------------------------------------------------------------------
    def __call__(self, imgs_u8: np.ndarray) -> np.ndarray:
        """(4,H,W) uint8 -> (4,H,W) uint8."""
        x = imgs_u8.astype(np.float32) / 255.0
        for name, sev in self.ops:
            if sev <= 0 or name == 'frame_drop':
                continue
            x = getattr(self, '_' + name)(x, sev)
        return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    # ---- photometric -------------------------------------------------------------------------
    def _bright(self, x, s):
        return np.clip(x * _p('bright', s), 0.0, 1.0)

    def _dark(self, x, s):
        return x * _p('dark', s)

    def _gamma(self, x, s):
        return np.power(np.clip(x, 0.0, 1.0), _p('gamma', s))

    def _contrast(self, x, s):
        m = x.mean(axis=(-2, -1), keepdims=True)
        return m + (x - m) * _p('contrast', s)

    def _clip(self, x, s):
        q = np.quantile(x, _p('clip', s), axis=(-2, -1), keepdims=True)
        return np.clip(x / np.maximum(q, 1e-3), 0.0, 1.0)

    def _noise(self, x, s):
        return x + self.rng.normal(0.0, _p('noise', s) / 255.0, x.shape).astype(np.float32)

    def _blur(self, x, s):
        k = max(int(round(_p('blur', s))), 1)
        if k <= 1:
            return x
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        return np.stack([cv2.filter2D(c, -1, kern) for c in x])

    def _jpeg(self, x, s):
        q = int(round(_p('jpeg', s)))
        out = []
        for c in x:
            ok, enc = cv2.imencode('.jpg', (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8),
                                   [cv2.IMWRITE_JPEG_QUALITY, q])
            out.append(cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 if ok else c)
        return np.stack(out)

    # ---- structured --------------------------------------------------------------------------
    def _grid(self, H, W):
        v, u = np.mgrid[0:H, 0:W].astype(np.float32)
        return u / W, v / H

    def _glare(self, x, s):
        n = int(round(_p('glare', s)))
        H, W = x.shape[-2:]
        u, v = self._grid(H, W)
        for c in range(len(x)):
            for _ in range(n):
                cx, cy = self.rng.uniform(0, 1), self.rng.uniform(0, 0.6)       # sun / lights: upper part
                r = self.rng.uniform(0.05, 0.2)
                x[c] = x[c] + self.rng.uniform(0.5, 1.2) * np.exp(-((u - cx) ** 2 + ((v - cy) * 0.5) ** 2) / (2 * r * r))
        return x

    def _shadow(self, x, s):
        n = int(round(_p('shadow', s)))
        if n <= 0:
            return x
        H, W = x.shape[-2:]
        u, v = self._grid(H, W)
        if self._bands is None or len(self._bands) != n:
            self._bands = [[self.rng.uniform(0, 1), self.rng.uniform(-1.5, 1.5), self.rng.uniform(0.05, 0.25),
                            self.rng.uniform(0.25, 0.55)] for _ in range(n)]
        for b in self._bands:                                     # pos, slope, width, factor; drift per frame
            b[0] = (b[0] + self.rng.normal(0.0, 0.03)) % 1.0
            band = np.abs(u - (b[0] + b[1] * (v - 0.5))) < b[2]
            x = np.where(band[None], x * b[3], x)
        return x

    def _vignette(self, x, s):
        H, W = x.shape[-2:]
        u, v = self._grid(H, W)
        r2 = (u - 0.5) ** 2 + (v - 0.5) ** 2
        return x * (1.0 - _p('vignette', s) * r2 / 0.5)

    def _haze(self, x, s):
        H, W = x.shape[-2:]
        _, v = self._grid(H, W)
        t_far = _p('haze', s)
        t = t_far + (1.0 - t_far) * np.power(v, 0.7)            # linear-ish in row: far (top) = most haze
        A = 0.72 + 0.04 * s                                      # grey airlight
        return x * t[None] + A * (1.0 - t[None])

    def _sky(self, x, s):
        H, W = x.shape[-2:]
        _, v = self._grid(H, W)
        h = _p('sky', s)
        if h <= 0:
            return x
        level = 0.85 - 0.25 * v / max(h, 1e-3)                   # bright at the top, a bit darker at the horizon
        m = np.clip((h - v) * H / 5.0, 0.0, 1.0)                 # ~5 px soft horizon
        return x * (1.0 - m[None]) + (level * m)[None]

    def _overcast(self, x, s):
        x = self._sky(x, s)
        x = self._haze(x, s)
        x = self._gamma(x, 0.6 * s)
        return self._vignette(self._contrast(x, 0.5 * s), 0.3 * s)

    def _low_sun(self, x, s):
        x = self._haze(x, 0.6 * s)
        x = self._shadow(x, max(1.0, s - 0.5))
        H, W = x.shape[-2:]
        u, v = self._grid(H, W)
        c = int(self.rng.integers(0, len(x)))                    # glare from a low sun near the horizon in one camera
        cx, cy = self.rng.uniform(0.2, 0.8), 0.25
        x[c] = x[c] + 0.4 * (s / 3.0) * np.exp(-((u - cx) ** 2 + ((v - cy) * 0.4) ** 2) / (2 * 0.12 ** 2))
        return x

    # ---- composites --------------------------------------------------------------------------
    def _daylight(self, x, s):
        x = self._gamma(x, 0.8 * s)
        x = self._bright(x, 0.5 * s)
        x = self._contrast(x, 0.6 * s)
        x = self._shadow(x, max(1.0, s - 0.5))
        return self._glare(x, 0.5 * s)

    def _rain(self, x, s):
        x = self._blur(x, 0.8 * s)
        x = self._contrast(x, s)
        x = self._noise(x, 0.6 * s)
        H, W = x.shape[-2:]
        for c in range(len(x)):                                   # droplets: small local blurry blobs
            for _ in range(int(6 * s)):
                cx, cy, r = int(self.rng.integers(0, W)), int(self.rng.integers(0, H)), int(self.rng.integers(3, 9))
                y0, y1, x0, x1 = max(cy - r, 0), min(cy + r, H), max(cx - r, 0), min(cx + r, W)
                if y1 - y0 > 2 and x1 - x0 > 2:
                    x[c, y0:y1, x0:x1] = cv2.GaussianBlur(x[c, y0:y1, x0:x1], (0, 0), r / 2.0) * 1.1
        return x

    # ---- sensor faults -----------------------------------------------------------------------
    def _cam_drop(self, x, s):
        for c in _CAM_ORDER[:int(round(_p('cam_drop', s)))]:
            x[c] = 0.0
        return x

    def _cam_freeze(self, x, s):
        for c in _CAM_ORDER[:int(round(_p('cam_freeze', s)))]:
            if c not in self.frozen:
                self.frozen[c] = x[c].copy()
            x[c] = self.frozen[c]
        return x
