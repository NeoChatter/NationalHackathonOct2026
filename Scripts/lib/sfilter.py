"""
Along-track grid filter: a 1-D belief over arc length s, one bin per TrackMap station (~1 m).

Why: the EKF is a single Gaussian. Once it is wrong it can only recover through a hard rule ("5 matches above an
absolute similarity"), and absolute similarities are exactly what changes with lighting / weather / sensor state.
This filter runs ALL the time next to the EKF:

  predict : shift the belief by the calibrated odometry distance (nominal scale, NOT the EKF's online scale,
            which can drift while lost) and blur it by a fraction of the step (covers a wrong scale).
  update  : similarity of the current descriptor to EVERY database frame -> best similarity per s bin ->
            robust z-score over the whole lap (median / MAD) -> tempered likelihood. The z-score is relative,
            so a global drop of all similarities (daylight vs. a night database) does not change it.
  anchor  : while the EKF is tracking, its s is fed in as a soft measurement, so the belief starts from the last
            good position the moment the EKF gets lost.
  init    : with no start pose the belief starts UNIFORM over the lap; the same ready() test then finds the start
            position (global localisation), so no ground truth is needed at runtime.

Re-seed signal (ready()): a clear peak (probability mass in a +-10 m window), that is DISTINCTIVE (z of the best
bin minus z of the best bin > 50 m away along the track) and that has moved consistently with the odometry for
several consecutive frames. All of that is relative, nothing depends on an absolute similarity level.
"""
import math
from collections import deque

import numpy as np

DEFAULT_SF = dict(
    sf_beta=0.3,             # likelihood temperature (z units); < 1 because consecutive frames are correlated
    sf_z_lo=-2.0,            # z clipping: one bad frame cannot erase the belief ...
    sf_z_hi=10.0,            # ... and one lucky frame cannot dominate it
    sf_frac_sigma=0.10,      # odometry distance std per step as a fraction of the step (covers a wrong scale)
    sf_base_sigma=0.3,       # m per step
    sf_eps=1e-3,             # mass spread uniformly every step: the belief can always recover from a kidnap
    sf_anchor_sigma=5.0,     # m   EKF -> filter anchoring while tracking
    sf_window=10.0,          # m   peak window (mass, mean, spread)
    sf_far=50.0,             # m   distinctiveness competitor must be at least this far along s
    sf_mass_min=0.6,         #     minimum peak mass to count as confident
    sf_margin_min=2.0,       #     minimum distinctiveness (z units)
    sf_confirm=5,            #     consecutive confident, odometry-consistent frames before a re-seed
    sf_consist_m=4.0,        # m   per-frame tolerance: peak motion vs. odometry distance
)


def _blur_circ(p: np.ndarray, sigma_bins: float) -> np.ndarray:
    half = int(math.ceil(3.0 * sigma_bins))
    if half < 1:
        return p
    if 2 * half + 1 >= len(p):
        return np.full_like(p, 1.0 / len(p))
    x = np.arange(-half, half + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    return np.convolve(np.r_[p[-half:], p, p[:half]], k, mode='valid')


class AlongTrackFilter:
    def __init__(self, track, db_k: np.ndarray, cfg=None, s0=None):
        """track: TrackMap; db_k: station index of every database frame (TrackMap.project)."""
        self.c = {**DEFAULT_SF, **(cfg or {})}
        self.tr = track
        self.K = track.K
        self.ds = track.length / track.K
        self.db_k = np.asarray(db_k, dtype=np.int64)
        cov = np.zeros(self.K, dtype=bool)
        cov[self.db_k] = True
        self.covered = cov | np.roll(cov, 1) | np.roll(cov, -1)
        self.p = np.full(self.K, 1.0 / self.K)
        self.hist = deque(maxlen=int(self.c['sf_confirm']))
        self._step = 0.0                         # odometry distance since the last update
        self.s_hat, self.s_std, self.mass = float('nan'), float('nan'), 0.0
        self.z_max, self.margin = float('nan'), float('nan')
        win = max(int(round(self.c['sf_window'] / self.ds)), 1)
        self._box = np.ones(2 * win + 1)
        self._win = win
        if s0 is not None:
            self.reset(s0)

    # ------------------------------------------------------------------------------------------
    def reset(self, s: float, sigma: float = 5.0):
        d = self.tr.wrap_s(self.tr.s - s)
        self.p = np.exp(-0.5 * (d / sigma) ** 2) + 1e-12
        self.p /= self.p.sum()
        self.hist.clear()
        self._step = 0.0
        self._estimate()

    def shift(self, ds_m: float):
        """Move the whole belief by ds_m (used by the kidnap test to keep filter and EKF consistent)."""
        self.p = np.roll(self.p, int(round(ds_m / self.ds)))
        self._estimate()

    # ------------------------------------------------------------------------------------------
    def predict(self, dist: float, frac: float = None):
        """Shift by the odometry distance; blur by base + frac * dist (frac: the odometry source's relative error,
        default sf_frac_sigma for visual odometry; the speed signal is much tighter)."""
        if not np.isfinite(dist) or dist <= 0.0:
            return
        frac = self.c['sf_frac_sigma'] if frac is None else frac
        self._step += dist
        shift = dist / self.ds
        i = int(math.floor(shift))
        f = shift - i
        p = (1.0 - f) * np.roll(self.p, i) + f * np.roll(self.p, i + 1)
        sig = math.hypot(self.c['sf_base_sigma'], frac * dist) / self.ds
        p = _blur_circ(p, sig)
        eps = self.c['sf_eps']
        p = (1.0 - eps) * p + eps / self.K
        self.p = p / p.sum()

    def update(self, sims_all: np.ndarray):
        """sims_all: cosine similarity of the current descriptor to every database frame (N,)."""
        b = np.full(self.K, -np.inf)
        np.maximum.at(b, self.db_k, np.asarray(sims_all, dtype=np.float64))
        b = np.maximum(b, np.maximum(np.roll(b, 1), np.roll(b, -1)))      # bridge 1-bin gaps
        ok = self.covered & np.isfinite(b)
        v = b[ok]
        if len(v) < 20:
            return
        med = float(np.median(v))
        scale = 1.4826 * float(np.median(np.abs(v - med))) + 1e-4
        z = np.zeros(self.K)
        z[ok] = (b[ok] - med) / scale                                     # uncovered bins stay neutral (z = 0)
        z = np.clip(z, self.c['sf_z_lo'], self.c['sf_z_hi'])

        logl = self.c['sf_beta'] * z
        p = self.p * np.exp(logl - logl.max())
        tot = p.sum()
        self.p = p / tot if tot > 0 and np.isfinite(tot) else np.full(self.K, 1.0 / self.K)

        j = int(np.argmax(z))
        far = np.abs(self.tr.wrap_s(self.tr.s - self.tr.s[j])) > self.c['sf_far']
        self.z_max = float(z[j])
        self.margin = float(z[j] - (z[far].max() if far.any() else self.c['sf_z_lo']))
        self._estimate()
        self.hist.append((self.s_hat, self.mass, self.margin, self._step))
        self._step = 0.0

    def anchor(self, s: float, sigma: float = None):
        sigma = self.c['sf_anchor_sigma'] if sigma is None else sigma
        d = self.tr.wrap_s(self.tr.s - s)
        p = self.p * np.exp(-0.5 * (d / sigma) ** 2)
        tot = p.sum()
        if tot > 0 and np.isfinite(tot):
            self.p = p / tot
            self._estimate()

    # ------------------------------------------------------------------------------------------
    def _estimate(self):
        w = self._win
        pw = np.convolve(np.r_[self.p[-w:], self.p, self.p[:w]], self._box, mode='valid')
        j = int(np.argmax(pw))
        idx = (np.arange(j - w, j + w + 1)) % self.K
        pj = self.p[idx]
        m = float(pj.sum())
        off = self.tr.wrap_s(self.tr.s[idx] - self.tr.s[j])
        mu = float(np.sum(pj * off) / max(m, 1e-12))
        self.mass = m
        self.s_hat = float((self.tr.s[j] + mu) % self.tr.length)
        self.s_std = float(math.sqrt(max(np.sum(pj * (off - mu) ** 2) / max(m, 1e-12), 0.0)))

    def ready(self) -> bool:
        """Confident, distinctive and odometry-consistent for sf_confirm consecutive updates."""
        n = int(self.c['sf_confirm'])
        if len(self.hist) < n:
            return False
        h = list(self.hist)
        for s, mass, margin, _ in h:
            if not (mass >= self.c['sf_mass_min'] and margin >= self.c['sf_margin_min']):
                return False
        for (s0, _, _, _), (s1, _, _, step) in zip(h[:-1], h[1:]):
            if abs(float(self.tr.wrap_s(s1 - s0 - step))) > self.c['sf_consist_m']:
                return False
        return True

    def set_entry(self, j: int, k: int):
        """Database entry j (new or overwritten by the experience memory) now lies at station k."""
        k = int(k)
        if j == len(self.db_k):
            self.db_k = np.append(self.db_k, k).astype(np.int64)
        else:
            self.db_k[j] = k
        self.covered[[k - 1, k, (k + 1) % self.K]] = True

    def station(self, s: float) -> int:
        return int(np.clip(np.searchsorted(self.tr.s, s % self.tr.length, side='right') - 1, 0, self.K - 1))
