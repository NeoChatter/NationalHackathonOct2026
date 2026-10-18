"""
Vehicle state-estimation inputs from /constructor0/state_estimation - ONLY the fields the rules allow.

Rules: speed, accelerations and yaw rate may be used as inputs. Pose fields are removed from the test file and must
not be depended on. This module is the ONLY place in the code base that reads this topic, and SE_FIELDS below is the
complete list of fields that are ever read from it.

  used        v_mps (speed), wz_radps (yaw rate), ax_mps2 / ay_mps2 (longitudinal / lateral acceleration)
  never read  x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad, pos_accuracy        (pose)
              yaw_vel_rad                                                   (a heading, equals yaw_rad)
              sn_map_state                                                  (position in track coordinates)
              vx_mps, vy_mps, beta_rad, kappa_radpm                         (frame unclear / heading-derived)

The odometry is integrated causally: for the interval between two camera frames only samples with a stamp <= the
later frame are used.
"""
import math
from collections import deque

import numpy as np

from .common import stamp_to_ns

SE_TOPIC = '/constructor0/state_estimation'
SE_FIELDS = ('v_mps', 'wz_radps', 'ax_mps2', 'ay_mps2')

DEFAULT_SE_CALIB = dict(
    v_scale=1.0,         # true distance = v_scale * integral(v)
    wz_scale=1.0,        # true heading change = wz_scale * (integral(wz) - wz_bias * T)
    wz_bias=0.0,         # rad/s
    dx_std=0.05,         # m   per 0.1 s frame (residual after calibration)
    dyaw_std=0.003,      # rad per 0.1 s frame
    max_gap_s=0.25,      # a gap between samples longer than this -> no state-estimation odometry for that frame
    still_mps=0.5,       # below this speed the car counts as stationary
)


def se_sample(msg):
    """(stamp_ns, v_mps, wz_radps, ax_mps2, ay_mps2) from one StateEstimation message. Whitelisted fields only."""
    return (stamp_to_ns(msg.header.stamp),) + tuple(float(getattr(msg, f, float('nan'))) for f in SE_FIELDS)


def integrate_raw(T, V, W, t0_ns: int, t1_ns: int, max_gap_s: float):
    """Uncalibrated (integral(v) [m], integral(wz) [rad], duration [s]) over [t0, t1] from samples T (ns, sorted).
    Linear interpolation between samples; after the last sample <= t1 its value is held (causal: no sample later
    than t1 is ever used). None if the interval is not covered (no sample within max_gap_s of t0, or a gap)."""
    if len(T) < 2 or t1_ns <= t0_ns:
        return None
    gap_ns = max_gap_s * 1e9
    a = int(np.searchsorted(T, t0_ns, side='right'))       # first sample > t0
    b = int(np.searchsorted(T, t1_ns, side='right'))       # first sample > t1  (never used)
    if a == 0 or b == 0 or t0_ns - T[a - 1] > gap_ns or t1_ns - T[b - 1] > gap_ns:
        return None
    Ti, Vi, Wi = T[a - 1:b], V[a - 1:b], W[a - 1:b]          # last sample <= t0 ... last sample <= t1
    tg = np.concatenate([[t0_ns], Ti[1:], [t1_ns]]).astype(np.float64)
    if np.max(np.diff(tg)) > gap_ns:
        return None
    Tf = Ti.astype(np.float64)
    vg = np.interp(tg, Tf, Vi)                               # beyond the last sample: held (np.interp clamps)
    wg = np.interp(tg, Tf, Wi)
    dt = np.diff(tg) * 1e-9
    return float(np.sum(0.5 * (vg[1:] + vg[:-1]) * dt)), float(np.sum(0.5 * (wg[1:] + wg[:-1]) * dt)), float(dt.sum())


class SEBuffer:
    """Causal buffer of state-estimation samples (filled while reading the bag in time order)."""

    def __init__(self, calib=None, maxlen: int = 4000):
        self.c = {**DEFAULT_SE_CALIB, **(calib or {})}
        self.t, self.v, self.w = deque(maxlen=maxlen), deque(maxlen=maxlen), deque(maxlen=maxlen)
        self.n = 0

    def add(self, sample):
        t, v, w = sample[0], sample[1], sample[2]
        if not (math.isfinite(v) and math.isfinite(w)):
            return
        if self.t and t <= self.t[-1]:                        # out-of-order / duplicate stamp: keep it monotonic
            return
        self.t.append(int(t))
        self.v.append(v)
        self.w.append(w)
        self.n += 1

    def speed(self) -> float:
        return float(self.v[-1]) if self.v else float('nan')

    def odometry(self, t0_ns: int, t1_ns: int):
        """Calibrated (distance [m], heading change [rad], mean speed [m/s]) over (t0, t1], or None."""
        if len(self.t) < 2:
            return None
        T = np.fromiter(self.t, dtype=np.int64, count=len(self.t))
        r = integrate_raw(T, np.fromiter(self.v, float, len(self.v)), np.fromiter(self.w, float, len(self.w)),
                          t0_ns, t1_ns, self.c['max_gap_s'])
        if r is None:
            return None
        iv, iw, dur = r
        dist = max(self.c['v_scale'] * iv, 0.0)
        dyaw = self.c['wz_scale'] * (iw - self.c['wz_bias'] * dur)
        return dist, dyaw, dist / max(dur, 1e-6)


# ----------------------------------------------------------------------------------------------
# calibration against GT on the TRAINING split (used by 3_calibrate.py)
# ----------------------------------------------------------------------------------------------
def _robust_std(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float('nan')


def pair_measurements(se: dict, ts: np.ndarray, poses: np.ndarray, idx: np.ndarray, lag_ns: int = 0):
    """For consecutive camera frames (i-1, i), i in idx (0.05 s <= dt <= 0.2 s): GT forward step / heading change in
    the body frame of frame i-1, and the raw state-estimation integrals over the same interval."""
    T = se['t_ns'] + lag_ns
    V, W = se['v_mps'], se['wz_radps']
    lo = int(idx[0])
    rows = []
    for i in idx:
        if i - 1 < lo:
            continue
        dt = (ts[i] - ts[i - 1]) * 1e-9
        if not 0.05 <= dt <= 0.2:
            continue
        r = integrate_raw(T, V, W, int(ts[i - 1]), int(ts[i]), 0.25)
        if r is None:
            continue
        x1, y1, _, yaw1 = poses[i - 1]
        x2, y2, _, yaw2 = poses[i]
        g_dx = math.cos(yaw1) * (x2 - x1) + math.sin(yaw1) * (y2 - y1)
        g_dyaw = math.atan2(math.sin(yaw2 - yaw1), math.cos(yaw2 - yaw1))
        rows.append((g_dx, g_dyaw, r[0], r[1], r[2]))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def fit_calibration(m: np.ndarray) -> dict:
    """m: rows (gt_dx, gt_dyaw, int_v, int_wz, T). Speed scale (median ratio while moving), yaw-rate scale + bias
    (least squares), residual noise per frame."""
    g_dx, g_dy, iv, iw, T = m.T
    moving = (g_dx > 0.5) & (iv > 0.2)
    v_scale = float(np.median(g_dx[moving] / iv[moving])) if moving.sum() > 50 else 1.0
    A = np.stack([iw, T], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, g_dy, rcond=None)
    wz_fit = 'scale+bias'
    if not 0.5 <= abs(a) <= 2.0:
        # Degenerate (e.g. an almost constant yaw rate makes integral(wz) and T collinear) or implausible: keep the
        # scale at 1 and fit the bias alone. (The original code reset the scale to 1 but kept the bias of the other
        # solution, which doubled the heading change on such data.)
        wz_fit = 'bias only (scale fit %.3g rejected)' % a
        a = 1.0
        b = float(np.dot(T, g_dy - iw) / max(float(np.dot(T, T)), 1e-12))
    wz_scale = float(a)
    wz_bias = float(-b / wz_scale)
    res_dx = v_scale * iv - g_dx
    res_dy = wz_scale * (iw - wz_bias * T) - g_dy
    return dict(v_scale=v_scale, wz_scale=wz_scale, wz_bias=wz_bias, wz_fit=wz_fit,
                dx_std=max(_robust_std(res_dx[moving]) if moving.any() else 0.05, 0.01),
                dyaw_std=max(_robust_std(res_dy), 5e-4), n_pairs=int(len(m)), n_moving=int(moving.sum()))


def evaluate_calibration(m: np.ndarray, cal: dict) -> dict:
    g_dx, g_dy, iv, iw, T = m.T
    moving = g_dx > 0.5
    res_dx = cal['v_scale'] * iv - g_dx
    res_dy = cal['wz_scale'] * (iw - cal['wz_bias'] * T) - g_dy
    return dict(dx_median_abs=float(np.median(np.abs(res_dx[moving]))) if moving.any() else float('nan'),
                dx_rel_bias_pct=float(np.median(res_dx[moving] / g_dx[moving]) * 100) if moving.any() else float('nan'),
                dyaw_median_abs_deg=float(np.degrees(np.median(np.abs(res_dy)))),
                dist_drift_pct=float(100 * (np.sum(cal['v_scale'] * iv[moving]) / np.sum(g_dx[moving]) - 1))
                if moving.any() else float('nan'))


# ----------------------------------------------------------------------------------------------
# run-time checks (from the rmse-2.740 branch, 3_causal_infer.py: telemetry_plausible / OdomHealth)
# ----------------------------------------------------------------------------------------------
SE_MAX_SPEED = 150.0       # m/s; the car tops out near 40 m/s, so far above this the stream is broken
SE_MAX_YAW_RATE = 5.0      # rad/s


def odometry_plausible(dist: float, dyaw: float, dt: float) -> bool:
    """Physically possible state-estimation odometry for one frame interval (finite, speed and yaw rate in range).
    Implausible values make stage 4 fall back to the learned visual odometry for that frame."""
    if not all(math.isfinite(v) for v in (dist, dyaw, dt)) or dt <= 0.0:
        return False
    return -5.0 <= dist / dt <= SE_MAX_SPEED and abs(dyaw / dt) <= SE_MAX_YAW_RATE


class OdomHealth:
    """
    Monitoring only: compares the forward distance of the state estimation with the learned VO's while the car is
    clearly moving (VO step > 1 m) and reports a gross disagreement (median ratio outside [lo, hi]).

    It never switches the motion source. The VO was trained on night images only, so in a daylight test it is the
    VO, not the speed signal, that is likely to be wrong. The speed signal is only dropped when its own values are
    implausible (odometry_plausible).
    """

    def __init__(self, window=60, lo=0.5, hi=2.0, min_vo_dx=1.0, min_samples=20):
        self.ratios = deque(maxlen=window)
        self.lo, self.hi = lo, hi
        self.min_vo_dx = min_vo_dx
        self.min_samples = min_samples
        self.ok = True
        self.median_ratio = float('nan')

    def update(self, se_dx: float, vo_dx: float) -> bool:
        if vo_dx > self.min_vo_dx:
            self.ratios.append(se_dx / vo_dx)
        if len(self.ratios) >= self.min_samples:
            self.median_ratio = float(np.median(self.ratios))
            self.ok = self.lo <= self.median_ratio <= self.hi
        return self.ok
