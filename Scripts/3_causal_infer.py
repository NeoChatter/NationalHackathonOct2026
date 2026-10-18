#!/usr/bin/env python3
"""
Causal camera localization (image retrieval + odometry + EKF) that finds its own start position.

Rules this script follows (organisers' final clarification):
  * No initial pose is given, so none is used. The start position is found from the images (MultiStartLocalizer).
    /tf (the ground-truth pose topic) is NEVER read here, and neither are the pose fields of
    /constructor0/state_estimation.
  * From /constructor0/state_estimation only the speed (v_mps), the yaw rate (wz_radps) and the accelerations are
    read. They drive the motion model; if they are missing or out of range, the learned visual odometry takes over.
  * Causal: a pose at time t uses only images with header.stamp <= t, and telemetry that has already arrived.

All filter logic lives in numpy classes below (FusionLocalizer, MultiStartLocalizer). `4_diagnostics.py` drives the
SAME classes on the held-out validation laps, so what its sweep measures is what a submission run does.

Odometry (motion between two camera frames)
  telemetry   dead reckoning from speed + yaw rate, integrated at the telemetry rate (~100 Hz) between the two camera
              stamps. Accurate and independent of lighting, so it also works for a different time of day or speed.
  vo          the learned two-frame network. Used only when telemetry is missing, stale or out of range. It was trained
              on night images, so it is never allowed to overrule the telemetry.

Start-up (no initial pose)
  Several candidate start positions (the best-matching, mutually distant map entries) are tracked in parallel for the
  first seconds. Each is propagated with the odometry and scored by how well the images keep matching the map around
  it and by how well its dead-reckoned path stays on the mapped track. The best candidate is committed to.

Filter extras (carried over): similarity-weighted trust, cluster-safe averaging, global relocalization when lost,
image-based standstill hold (VO mode only). Settings come from a preset chosen by the checkpoint; any command-line
flag overrides the preset.
"""

import argparse
import csv
import dataclasses
import importlib.util
import math
import os
import time
from collections import deque

import numpy as np

LOSS_WEIGHTS = np.array([1.0, 2.0, 5.0], dtype=np.float64)  # weights the OLD VO loss used (legacy checkpoints)


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def wrap_array(a):
    return np.arctan2(np.sin(a), np.cos(a))


# --------------------------------------------------------------------------------------------------------------
# EKF on SE(2)
# --------------------------------------------------------------------------------------------------------------
class CausalSE2EKF:
    """State: [x, y, yaw]. Q/R trust adjustments are applied by FusionLocalizer, not hidden in here."""

    def __init__(self, x0, y0, yaw0):
        self.state = np.array([x0, y0, yaw0], dtype=np.float64)
        self.P = np.diag([
            0.15 ** 2,
            0.15 ** 2,
            math.radians(1.0) ** 2,
        ])

    def predict(self, dx_body, dy_body, dyaw, q_var):
        x, y, yaw = self.state
        c, s = math.cos(yaw), math.sin(yaw)

        self.state[0] = x + c * dx_body - s * dy_body
        self.state[1] = y + s * dx_body + c * dy_body
        self.state[2] = wrap_angle(yaw + dyaw)

        F = np.array([
            [1.0, 0.0, -s * dx_body - c * dy_body],
            [0.0, 1.0,  c * dx_body - s * dy_body],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        rotation = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        Q_body = np.diag([
            max(float(q_var[0]), 0.05 ** 2),
            max(float(q_var[1]), 0.02 ** 2),
            max(float(q_var[2]), math.radians(0.3) ** 2),
        ])

        self.P = F @ self.P @ F.T + rotation @ Q_body @ rotation.T
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, measurement, R):
        residual = np.array([
            measurement[0] - self.state[0],
            measurement[1] - self.state[1],
            wrap_angle(measurement[2] - self.state[2]),
        ], dtype=np.float64)

        S = self.P + R

        # Equivalent to P @ inv(S), without explicitly taking an inverse.
        K = np.linalg.solve(S.T, self.P.T).T

        self.state += K @ residual
        self.state[2] = wrap_angle(self.state[2])

        # Joseph covariance update for numerical stability.
        I_K = np.eye(3) - K
        self.P = I_K @ self.P @ I_K.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)


# --------------------------------------------------------------------------------------------------------------
# Image helpers shared with the diagnostics
# --------------------------------------------------------------------------------------------------------------
def normalize_stack(u8_stack):
    """(4, H, W) uint8 -> float32, per-camera zero mean / unit variance (same prep as training)."""
    x = u8_stack.astype(np.float32) / 255.0
    return (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-5)


def frame_change_score(prev_norm, curr_norm):
    """
    How much the picture changed between two consecutive frames: the LARGEST over the four cameras of the mean
    absolute difference of the normalized images. A parked car only shows sensor noise; a moving car changes every
    camera (the side cameras see strong parallax), so the max over cameras is a robust "is it moving?" signal.
    """
    return float(np.abs(curr_norm - prev_norm).mean(axis=(1, 2)).max())


# --------------------------------------------------------------------------------------------------------------
# Fusion parameters + presets
# --------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class FusionParams:
    # ---- learned-VO motion model (used when telemetry odometry is unavailable) --------------------------------
    q_x_floor: float = 1.0      # minimum along-track VO process noise, std in metres per frame
    q_scale: float = 1.0        # multiplier on the learned VO std (before the floor)
    # ---- telemetry odometry (speed + yaw rate) ----------------------------------------------------------------
    q_tel_scale: float = 1.0    # multiplier on the telemetry noise model below
    tel_dx_frac: float = 0.01   # forward / lateral noise std as a fraction of the distance driven per frame
    tel_dx_floor: float = 0.03  # forward noise floor [m per frame]
    tel_dy_floor: float = 0.03  # lateral noise floor [m per frame]
    tel_yaw_floor_deg: float = 0.2   # yaw noise floor [deg per frame]
    # ---- retrieval measurement ----------------------------------------------------------------------------------
    r_scale: float = 0.03       # multiplies the retrieval POSITION covariance (<1 trusts retrieval more); VO mode
    r_scale_tel: float = 1.0    # same, when the motion model is telemetry (accurate motion: retrieval is averaged)
    ret_floor_tel_m: float = 0.0  # telemetry mode: adds (this)^2 to the retrieval position variance [m]
    gate_m: float = 35.0        # retrieval candidates must be within this distance of the predicted state ...
    gate_deg: float = 35.0      # ... and this heading difference
    min_sim: float = 0.45       # hard floor on the best in-gate similarity
    sim_ref: float = 0.95       # similarity at/above which a match is fully trusted (upper bound, see sim_adapt_q)
    sim_tau: float = 0.0        # R inflates by exp((sim_ref - sim) / sim_tau) below sim_ref; 0 = off
    sim_adapt_q: float = 0.0    # >0: the reference is min(sim_ref, this quantile of the last 600 global top-1
                                # similarities), i.e. "weak" is judged against this session's own good matches
    cluster_m: float = 0.0      # average only references within this distance of the best match; 0 = off
    topk: int = 8               # references averaged into one measurement
    nis_gate: float = 0.0       # if NIS exceeds this, R is scaled by NIS / nis_gate (robust update); 0 = off
    # ---- global relocalization once tracking ------------------------------------------------------------------------
    reloc_sim: float = 2.0      # global relocalization needs a global best similarity >= this (>1 = disabled)
    reloc_frames: int = 3       # ... on this many consecutive consistent frames
    reloc_margin: float = 0.03  # ... and this much better than the best in-gate match
    # ---- image-based standstill hold (VO mode only) --------------------------------------------------------------------
    still_thresh: float = 0.0   # frame_change_score below this means "not moving"; 0 = off
    still_frames: int = 3       # consecutive still frames required before motion is forced to zero
    fix_q_scaling: bool = False  # legacy checkpoints only: exp(log_var) / [1, 2, 5]
    # ---- finding the start position (no initial pose is given) -------------------------------------------------------------
    init_candidates: int = 96   # candidate start positions tracked in parallel (cheap: they share one similarity pass)
    init_nms_m: float = 20.0    # candidates must be at least this far apart (or head the other way)
    init_min_frames: int = 15   # earliest commit (frames at 10 Hz)
    init_max_frames: int = 60   # latest commit: the best candidate wins
    init_margin: float = 0.02   # commit early when the leader's mean score beats the runner-up by this much
    init_offtrack_m: float = 8.0   # a candidate further than this from every mapped pose is "off the track" ...
    init_offtrack_w: float = 0.02  # ... and loses this much score per metre beyond it, per frame
    init_heading_free_deg: float = 25.0  # a candidate heading differently from the nearest mapped pose by more than this ...
    init_heading_w: float = 0.004        # ... loses this much score per degree beyond it, per frame


_COMMON = dict(
    q_scale=1.0, q_tel_scale=1.0, tel_dx_frac=0.01, tel_dx_floor=0.03, tel_dy_floor=0.03, tel_yaw_floor_deg=0.2,
    r_scale=0.03, r_scale_tel=1.0, min_sim=0.45, sim_ref=0.95, topk=8, nis_gate=0.0, reloc_frames=3,
    reloc_margin=0.03, still_frames=3, fix_q_scaling=False,
    init_candidates=96, init_nms_m=20.0, init_min_frames=15, init_max_frames=60, init_margin=0.02,
    init_offtrack_m=8.0, init_offtrack_w=0.02, init_heading_free_deg=25.0, init_heading_w=0.004,
)

# Checkpoints trained with the previous recipe (learned-VO trust levels of the rmse-13.688 milestone).
LEGACY_PRESET = dict(
    _COMMON, q_x_floor=1.0, ret_floor_tel_m=1.0, gate_m=35.0, gate_deg=35.0, sim_tau=0.0, sim_adapt_q=0.0,
    cluster_m=0.0, reloc_sim=2.0, still_thresh=0.0,
)

# Checkpoints trained with the current recipe. These are the settings `4_diagnostics.py --sweep` chose on the held-out
# laps of racing_model_val.pt (diagnostics/recommended_args.txt), so the plain command reproduces the validated
# 2.740 m without extra flags. Re-run the sweep after changing the recipe.
# still_thresh is 0 here on purpose: new checkpoints carry a threshold CALIBRATED ON THE TRAINING DATA
# (stored in model_config by 2_train_model.py), which resolve_fusion_params() uses instead.
NEW_PRESET = dict(
    _COMMON, q_x_floor=0.5, q_tel_scale=0.5, tel_dx_frac=0.02, r_scale_tel=0.3, ret_floor_tel_m=2.0,
    gate_m=50.0, gate_deg=45.0, sim_ref=0.97, sim_tau=0.005, sim_adapt_q=0.75, cluster_m=15.0, topk=4,
    nis_gate=30.0, reloc_sim=0.95, still_thresh=0.0,
)


def add_fusion_args(parser):
    g = parser.add_argument_group('fusion settings (unset = preset chosen from the checkpoint)')
    g.add_argument('--q-x-floor', type=float, default=None, help='learned-VO mode: min along-track process-noise std [m/frame]')
    g.add_argument('--q-scale', type=float, default=None, help='learned-VO mode: multiplier on the learned VO std')
    g.add_argument('--q-tel-scale', type=float, default=None, help='telemetry mode: multiplier on the odometry noise model')
    g.add_argument('--tel-dx-frac', type=float, default=None, help='telemetry mode: noise as a fraction of distance per frame')
    g.add_argument('--tel-dx-floor', type=float, default=None)
    g.add_argument('--tel-dy-floor', type=float, default=None)
    g.add_argument('--tel-yaw-floor-deg', type=float, default=None)
    g.add_argument('--r-scale', type=float, default=None, help='learned-VO mode: multiplier on the retrieval position covariance')
    g.add_argument('--r-scale-tel', type=float, default=None, help='telemetry mode: multiplier on the retrieval position covariance')
    g.add_argument('--ret-floor-tel-m', type=float, default=None, help='telemetry mode: retrieval position std floor [m]')
    g.add_argument('--gate-m', type=float, default=None)
    g.add_argument('--gate-deg', type=float, default=None)
    g.add_argument('--min-sim', type=float, default=None)
    g.add_argument('--sim-ref', type=float, default=None, help='similarity at/above which a match is fully trusted')
    g.add_argument('--sim-tau', type=float, default=None, help='similarity trust softness (0 = off)')
    g.add_argument('--sim-adapt-q', type=float, default=None,
                   help='judge similarity against this quantile of recent global matches (0 = fixed --sim-ref)')
    g.add_argument('--cluster-m', type=float, default=None, help='average only references this close to the best (0 = off)')
    g.add_argument('--topk', type=int, default=None)
    g.add_argument('--nis-gate', type=float, default=None, help='robust update threshold on NIS (0 = off)')
    g.add_argument('--reloc-sim', type=float, default=None, help='global relocalization similarity (>1 = off)')
    g.add_argument('--reloc-frames', type=int, default=None)
    g.add_argument('--reloc-margin', type=float, default=None)
    g.add_argument('--still-thresh', type=float, default=None, help='standstill frame-change threshold (0 = off)')
    g.add_argument('--still-frames', type=int, default=None)
    g.add_argument('--fix-q-scaling', action='store_true', default=None,
                   help='legacy checkpoints only: divide exp(log_var) by the old loss weights [1, 2, 5]')
    g.add_argument('--init-candidates', type=int, default=None, help='start-up: candidate start positions')
    g.add_argument('--init-nms-m', type=float, default=None)
    g.add_argument('--init-min-frames', type=int, default=None)
    g.add_argument('--init-max-frames', type=int, default=None)
    g.add_argument('--init-margin', type=float, default=None)
    g.add_argument('--init-offtrack-m', type=float, default=None)
    g.add_argument('--init-offtrack-w', type=float, default=None)
    g.add_argument('--init-heading-free-deg', type=float, default=None)
    g.add_argument('--init-heading-w', type=float, default=None)


def resolve_fusion_params(args, model_cfg):
    """Preset from the checkpoint, overridden by any explicitly given command-line value."""
    preset = dict(NEW_PRESET if model_cfg.get('q_scaling_fixed', False) else LEGACY_PRESET)
    if model_cfg.get('q_scaling_fixed', False) and model_cfg.get('still_thresh') is not None:
        preset['still_thresh'] = float(model_cfg['still_thresh'])   # calibrated on the training data
    for field in dataclasses.fields(FusionParams):
        value = getattr(args, field.name, None)
        if value is not None:
            preset[field.name] = value
    if model_cfg.get('q_scaling_fixed', False):
        preset['fix_q_scaling'] = False  # new checkpoints already predict a proper variance
    return FusionParams(**preset)


# --------------------------------------------------------------------------------------------------------------
# Cross-check of the telemetry odometry against the learned VO
# --------------------------------------------------------------------------------------------------------------
class OdomHealth:
    """
    Monitoring only: compares the forward distance of the telemetry with the learned VO's while the car is clearly
    moving (VO step > 1 m) and reports a gross disagreement (median ratio outside [lo, hi]).

    It never switches the motion source. The VO was trained on night images only, so in a daylight test it is the
    VO, not the telemetry, that is likely to be wrong. Telemetry is only dropped when its own values are implausible
    (see telemetry_plausible).
    """

    def __init__(self, window=60, lo=0.5, hi=2.0, min_vo_dx=1.0, min_samples=20):
        self.ratios = deque(maxlen=window)
        self.lo, self.hi = lo, hi
        self.min_vo_dx = min_vo_dx
        self.min_samples = min_samples
        self.ok = True
        self.median_ratio = float('nan')

    def update(self, tel_dx, vo_dx):
        if vo_dx > self.min_vo_dx:
            self.ratios.append(tel_dx / vo_dx)
        if len(self.ratios) >= self.min_samples:
            self.median_ratio = float(np.median(self.ratios))
            self.ok = self.lo <= self.median_ratio <= self.hi
        return self.ok


TEL_MAX_SPEED = 150.0      # m/s; the car tops out near 40 m/s, so far above this the stream is broken
TEL_MAX_YAW_RATE = 5.0     # rad/s


def telemetry_plausible(tel):
    """Physically possible values for one step of telemetry odometry (finite, speed and yaw rate in range)."""
    dx, dy, dyaw, speed, yaw_rate = (float(v) for v in tel[:5])
    return (all(math.isfinite(v) for v in (dx, dy, dyaw, speed, yaw_rate))
            and -5.0 <= speed <= TEL_MAX_SPEED and abs(yaw_rate) <= TEL_MAX_YAW_RATE)


def select_odometry(mode, tel, vo_delta, health):
    """
    Which odometry drives this step? Shared by the inference script and the diagnostics replay.
      mode      : "auto" | "telemetry" | "vo"
      tel       : tuple from TelemetryTrack.integrate (dx, dy, dyaw, speed, yaw_rate, ...) or None
      vo_delta  : learned-VO motion for this step, or None on the first frame
    auto uses the telemetry whenever it is present and plausible, and the learned VO otherwise.
    Returns (kind, motion, telemetry_speed, fell_back) with kind in {"telemetry", "vo"};
    fell_back = telemetry was present but implausible.
    """
    if mode == "vo" or tel is None:
        return "vo", vo_delta, float("nan"), False
    if vo_delta is not None:
        health.update(tel[0], vo_delta[0])          # monitoring only, never overrides the telemetry
    if mode == "telemetry" or telemetry_plausible(tel):
        return "telemetry", np.array(tel[:3], dtype=np.float64), float(tel[3]), False
    return "vo", vo_delta, float(tel[3]), True


# --------------------------------------------------------------------------------------------------------------
# The filter used by BOTH inference and diagnostics
# --------------------------------------------------------------------------------------------------------------
class FusionLocalizer:
    TELEMETRY_FIELDS = [
        "frame", "timestamp_ns", "mode", "top1_sim", "global_sim", "sim_ref_eff", "conf_infl",
        "anchor_ms", "dt", "max_camera_age_s", "missing_cameras",
        "odom_tel", "tel_speed", "n_hyp", "map_dist", "map_dyaw",
        "change_score", "still",
        "dx_body_pred", "dy_body_pred", "dyaw_pred",
        "q_x", "q_y", "q_yaw",
        "pred_x", "pred_y", "pred_yaw",
        "gate_n",
        "retrieved_x", "retrieved_y", "retrieved_yaw",
        "innovation_x", "innovation_y", "innovation_yaw", "nis",
        "R_xx", "R_yy", "R_yawyaw",
        "P_xx", "P_yy", "P_yawyaw",
        "x", "y", "yaw",
    ]

    def __init__(self, db_embeddings, db_poses, params):
        self.db_emb = np.ascontiguousarray(db_embeddings, dtype=np.float32)   # (N, D)
        self.db = np.asarray(db_poses, dtype=np.float64)                      # (N, 3): x, y, yaw
        self.p = params
        self.ekf = None
        self.n_updates = 0
        self.n_reloc = 0
        self.n_still = 0
        self._still_run = 0
        self._reloc_streak = 0
        self._reloc_prev = None
        self._sim_hist = deque(maxlen=600)   # recent global top-1 similarities (~60 s at 10 Hz)

    # -- initialisation ------------------------------------------------------------------------------------
    def initialize(self, pose):
        self.ekf = CausalSE2EKF(float(pose[0]), float(pose[1]), float(pose[2]))

    # -- helpers -------------------------------------------------------------------------------------------
    def _measurement(self, cand_idx, cand_sims):
        """Similarity-weighted mean pose of the best `topk` candidates (optionally only those near the best)."""
        p = self.p
        k = min(p.topk, len(cand_idx))
        if k < len(cand_idx):
            part = np.argpartition(-cand_sims, k - 1)[:k]
            order = part[np.argsort(-cand_sims[part])]
        else:
            order = np.argsort(-cand_sims)
        top_idx = cand_idx[order]
        top_sims = cand_sims[order].astype(np.float64)
        matched = self.db[top_idx]

        if p.cluster_m > 0.0 and len(top_idx) > 1:
            dist = np.hypot(matched[:, 0] - matched[0, 0], matched[:, 1] - matched[0, 1])
            keep = dist <= p.cluster_m          # entry 0 (the best match) is always kept
            matched, top_sims = matched[keep], top_sims[keep]

        top1_sim = float(top_sims[0])
        weights = np.exp((top_sims - top_sims.max()) * 15.0)
        weights /= weights.sum()

        z_x = float(np.sum(weights * matched[:, 0]))
        z_y = float(np.sum(weights * matched[:, 1]))
        z_yaw = math.atan2(
            float(np.sum(weights * np.sin(matched[:, 2]))),
            float(np.sum(weights * np.cos(matched[:, 2]))),
        )
        diff_xy = matched[:, :2] - np.array([[z_x, z_y]])
        cov_xy = (diff_xy.T * weights) @ diff_xy + np.eye(2) * (0.25 / max(top1_sim, 0.2)) ** 2
        return np.array([z_x, z_y, z_yaw], dtype=np.float64), cov_xy, top1_sim

    def _stillness(self, change_score):
        p = self.p
        if p.still_thresh <= 0.0 or change_score is None or not np.isfinite(change_score):
            self._still_run = 0
            return False
        self._still_run = self._still_run + 1 if change_score < p.still_thresh else 0
        return self._still_run >= p.still_frames

    # -- one frame -----------------------------------------------------------------------------------------
    def step(self, embedding, delta=None, q_var=None, change_score=None, odom="vo", sims_all=None):
        """
        embedding    : (D,) L2-normalized descriptor of the current frame
        delta        : motion (dx, dy, dyaw) in the previous body frame; None for the first frame
        q_var        : variance of `delta` when odom == "vo" (from the network); ignored for telemetry
        change_score : frame_change_score(prev, curr) for the VO-mode standstill hold; None for the first frame
        odom         : "vo" (learned visual odometry) or "telemetry" (speed + yaw rate dead reckoning)
        sims_all     : precomputed db_emb @ embedding (shared between start-up candidates)
        Returns a telemetry dict; the new pose is in self.ekf.state.
        """
        p, ekf = self.p, self.ekf
        embedding = np.asarray(embedding, dtype=np.float32)
        telemetry_mode = odom == "telemetry"
        rec = {"mode": "VO_ONLY", "top1_sim": 0.0, "odom_tel": 1 if telemetry_mode else 0}

        still = False
        if delta is not None:
            if telemetry_mode:
                motion = np.asarray(delta, dtype=np.float64)
                dist = math.hypot(motion[0], motion[1])
                q = np.array([
                    (p.tel_dx_frac * dist) ** 2 + p.tel_dx_floor ** 2,
                    (p.tel_dx_frac * dist) ** 2 + p.tel_dy_floor ** 2,
                    math.radians(p.tel_yaw_floor_deg) ** 2 + (0.02 * abs(motion[2])) ** 2,
                ]) * (p.q_tel_scale ** 2)
            else:
                still = self._stillness(change_score)
                if still:
                    self.n_still += 1
                    motion = np.zeros(3)
                    q = np.zeros(3)                      # the EKF's own small floors apply
                else:
                    motion = np.asarray(delta, dtype=np.float64)
                    q = np.array(q_var, dtype=np.float64)
                    if p.fix_q_scaling:
                        q = q / LOSS_WEIGHTS
                    q = q * (p.q_scale ** 2)
                    q[0] = max(q[0], p.q_x_floor ** 2)
            ekf.predict(float(motion[0]), float(motion[1]), float(motion[2]), q)
            rec.update(dx_body_pred=float(motion[0]), dy_body_pred=float(motion[1]), dyaw_pred=float(motion[2]),
                       q_x=float(q[0]), q_y=float(q[1]), q_yaw=float(q[2]))
        rec["still"] = int(still)
        rec["change_score"] = float(change_score) if change_score is not None else float("nan")
        rec.update(pred_x=float(ekf.state[0]), pred_y=float(ekf.state[1]), pred_yaw=float(ekf.state[2]))

        # ---- retrieval against the whole map, then the gate around the predicted state --------------------
        sims_all = self.db_emb @ embedding if sims_all is None else sims_all
        gx = self.db[:, 0] - ekf.state[0]
        gy = self.db[:, 1] - ekf.state[1]
        d2 = gx * gx + gy * gy
        nearest = int(np.argmin(d2))
        # how far from every mapped pose, and how differently headed than the nearest one: the "is this on the
        # mapped track, going the right way?" test used when choosing between candidate starts
        rec["map_dist"] = float(math.sqrt(float(d2[nearest])))
        rec["map_dyaw"] = abs(math.degrees(wrap_angle(float(self.db[nearest, 2] - ekf.state[2]))))
        gate_yaw = math.radians(p.gate_deg)
        near = np.flatnonzero(d2 <= p.gate_m ** 2)
        gate_idx = near[np.abs(wrap_array(self.db[near, 2] - ekf.state[2])) <= gate_yaw] if len(near) else near
        rec["gate_n"] = int(len(gate_idx))

        best_global = int(np.argmax(sims_all))
        global_sim = float(sims_all[best_global])
        rec["global_sim"] = global_sim
        best_in_gate = bool(d2[best_global] <= p.gate_m ** 2
                            and abs(wrap_angle(float(self.db[best_global, 2] - ekf.state[2]))) <= gate_yaw)
        gate_best = float(sims_all[gate_idx].max()) if len(gate_idx) else -1.0

        # What counts as a "good" match is judged against this session's own recent good matches, so a model whose
        # similarity scale differs from the one the 0.95 was read off (or a different time of day) is not mistrusted.
        self._sim_hist.append(global_sim)
        sim_ref = p.sim_ref
        if p.sim_adapt_q > 0.0 and len(self._sim_hist) >= 50:
            sim_ref = min(p.sim_ref, float(np.quantile(np.fromiter(self._sim_hist, dtype=np.float64), p.sim_adapt_q)))
        rec["sim_ref_eff"] = sim_ref

        # ---- global recovery when lost ---------------------------------------------------------------------
        if (p.reloc_sim <= 1.0 and global_sim >= p.reloc_sim and not best_in_gate
                and global_sim - gate_best >= p.reloc_margin):
            candidate = self.db[best_global]
            if (self._reloc_prev is not None
                    and math.hypot(candidate[0] - self._reloc_prev[0], candidate[1] - self._reloc_prev[1]) <= 15.0):
                self._reloc_streak += 1
            else:
                self._reloc_streak = 1
            self._reloc_prev = candidate.copy()

            if self._reloc_streak >= p.reloc_frames:
                z, _, top1 = self._measurement(np.arange(len(self.db)), sims_all)
                ekf.state = z.copy()
                ekf.P = np.diag([3.0 ** 2, 3.0 ** 2, math.radians(3.0) ** 2])
                self._reloc_streak, self._reloc_prev = 0, None
                self.n_reloc += 1
                rec.update(mode="RELOCALIZED", top1_sim=top1, retrieved_x=float(z[0]), retrieved_y=float(z[1]),
                           retrieved_yaw=float(z[2]))
                return self._finish(rec)
        else:
            self._reloc_streak, self._reloc_prev = 0, None

        # ---- gated retrieval update -------------------------------------------------------------------------
        if len(gate_idx) >= 5:
            z, cov_xy, top1_sim = self._measurement(gate_idx, sims_all[gate_idx])
            rec["top1_sim"] = top1_sim

            if top1_sim > p.min_sim:
                infl = 1.0
                if p.sim_tau > 0.0:
                    infl = min(math.exp(max(0.0, sim_ref - top1_sim) / p.sim_tau), 1e6)

                R = np.eye(3, dtype=np.float64) * (math.radians(2.0) ** 2) * infl
                if telemetry_mode:
                    R[:2, :2] = (cov_xy * p.r_scale_tel + np.eye(2) * p.ret_floor_tel_m ** 2) * infl
                else:
                    R[:2, :2] = cov_xy * p.r_scale * infl

                innovation = z - ekf.state
                innovation[2] = wrap_angle(innovation[2])
                nis = float(innovation @ np.linalg.solve(ekf.P + R, innovation))
                if p.nis_gate > 0.0 and nis > p.nis_gate:
                    R = R * (nis / p.nis_gate)       # robust: an outlier gets less say, not zero

                rec.update(
                    mode="EKF_FUSED", conf_infl=infl,
                    retrieved_x=float(z[0]), retrieved_y=float(z[1]), retrieved_yaw=float(z[2]),
                    innovation_x=float(innovation[0]), innovation_y=float(innovation[1]),
                    innovation_yaw=float(innovation[2]), nis=nis,
                    R_xx=float(R[0, 0]), R_yy=float(R[1, 1]), R_yawyaw=float(R[2, 2]),
                )
                ekf.update(z, R)
                self.n_updates += 1

        return self._finish(rec)

    def _finish(self, rec):
        ekf = self.ekf
        rec.update(
            P_xx=float(ekf.P[0, 0]), P_yy=float(ekf.P[1, 1]), P_yawyaw=float(ekf.P[2, 2]),
            x=float(ekf.state[0]), y=float(ekf.state[1]), yaw=float(ekf.state[2]),
        )
        return rec


# --------------------------------------------------------------------------------------------------------------
# Finding the start position without an initial pose
# --------------------------------------------------------------------------------------------------------------
class MultiStartLocalizer:
    """
    No initial pose is given, so the start position has to come from the images. A single "best match" on the first
    frame is fragile (look-alike straights, a different time of day), so several candidate starts are tracked in
    parallel and one is committed to after a few seconds.

      candidates : the best-matching map entries, at least `init_nms_m` apart (or heading the other way)
      each one   : its own FusionLocalizer, propagated with the same odometry and corrected by its own retrievals
      score      : per frame, the best in-gate similarity  minus  a penalty for being away from every mapped pose
                   (a candidate whose dead-reckoned path leaves the track cannot be the true start)
      commit     : when the leader's mean score beats the runner-up by `init_margin` (after init_min_frames), or at
                   init_max_frames at the latest. Candidates that merge onto the same pose are collapsed.

    Until it commits, the pose of the current leader is reported, so every frame has an estimate.
    After the commit it behaves exactly like a single FusionLocalizer (including global relocalization).
    """

    def __init__(self, db_embeddings, db_poses, params):
        self.p = params
        self.db_emb = np.ascontiguousarray(db_embeddings, dtype=np.float32)
        self.db = np.asarray(db_poses, dtype=np.float64)
        self.hyps = []
        self.util = []
        self.active = None
        self.frames = 0
        self.commit_frame = None
        self.started = False
        self._leader = 0

    # -- state shared with the single filter ---------------------------------------------------------------
    @property
    def ekf(self):
        if self.active is not None:
            return self.active.ekf
        return self.hyps[self._leader].ekf if self.hyps else None

    @property
    def n_updates(self):
        return self.active.n_updates if self.active is not None else 0

    @property
    def n_reloc(self):
        return self.active.n_reloc if self.active is not None else 0

    @property
    def n_still(self):
        return self.active.n_still if self.active is not None else 0

    # -- start ---------------------------------------------------------------------------------------------
    def start(self, embedding, pose=None):
        """pose: optional known start (experiments only). Otherwise candidates come from the first frame."""
        self.started = True
        p = self.p
        if pose is not None:
            self.active = FusionLocalizer(self.db_emb, self.db, p)
            self.active.initialize(pose)
            self.commit_frame = 0
            return 1

        sims = self.db_emb @ np.asarray(embedding, dtype=np.float32)
        n = len(sims)
        pool = np.arange(n)
        if n > 8000:
            pool = np.argpartition(-sims, 8000)[:8000]
        pool = pool[np.argsort(-sims[pool])]
        chosen = []
        for j in pool:
            pj = self.db[int(j)]
            if all(math.hypot(pj[0] - self.db[c][0], pj[1] - self.db[c][1]) > p.init_nms_m
                   or abs(wrap_angle(pj[2] - self.db[c][2])) > math.radians(60.0) for c in chosen):
                chosen.append(int(j))
                if len(chosen) >= max(p.init_candidates, 1):
                    break
        hyp_params = dataclasses.replace(p, reloc_sim=2.0)      # candidates must stay distinct: no global jumps
        for j in chosen:
            h = FusionLocalizer(self.db_emb, self.db, hyp_params)
            h.initialize(self.db[j])
            self.hyps.append(h)
        self.util = [0.0] * len(self.hyps)
        return len(self.hyps)

    # -- one frame -----------------------------------------------------------------------------------------
    def step(self, embedding, delta=None, q_var=None, change_score=None, odom="vo"):
        if self.active is not None:
            rec = self.active.step(embedding, delta, q_var, change_score, odom)
            rec["n_hyp"] = 1
            return rec

        p = self.p
        sims_all = self.db_emb @ np.asarray(embedding, dtype=np.float32)
        recs = [h.step(embedding, delta, q_var, change_score, odom, sims_all=sims_all) for h in self.hyps]
        for i, r in enumerate(recs):
            off = max(0.0, r.get("map_dist", 0.0) - p.init_offtrack_m)
            skew = max(0.0, r.get("map_dyaw", 0.0) - p.init_heading_free_deg)
            self.util[i] += float(r.get("top1_sim", 0.0) or 0.0) - p.init_offtrack_w * off - p.init_heading_w * skew
        self.frames += 1

        # Collapse candidates that converged onto the same pose: keep the better one.
        keep = list(range(len(self.hyps)))
        order = sorted(keep, key=lambda i: -self.util[i])
        kept = []
        for i in order:
            si = self.hyps[i].ekf.state
            if any(math.hypot(si[0] - self.hyps[j].ekf.state[0], si[1] - self.hyps[j].ekf.state[1]) < 5.0
                   and abs(wrap_angle(si[2] - self.hyps[j].ekf.state[2])) < math.radians(20.0) for j in kept):
                continue
            kept.append(i)
        self.hyps = [self.hyps[i] for i in kept]
        self.util = [self.util[i] for i in kept]
        recs = [recs[i] for i in kept]

        # kept is sorted by descending utility, so index 0 leads
        self._leader = 0
        mean_lead = self.util[0] / self.frames
        mean_second = self.util[1] / self.frames if len(self.util) > 1 else float("-inf")
        commit = (len(self.hyps) == 1
                  or (self.frames >= p.init_min_frames and mean_lead - mean_second >= p.init_margin)
                  or self.frames >= p.init_max_frames)
        rec = dict(recs[0])
        rec["n_hyp"] = len(self.hyps)
        if commit:
            self.active = self.hyps[0]
            self.active.p = p                                   # re-enable global relocalization
            self.hyps, self.util = [], []
            self.commit_frame = self.frames
        return rec


# --------------------------------------------------------------------------------------------------------------
# Inference over a bag
# --------------------------------------------------------------------------------------------------------------
def run_causal_inference(args):
    import torch
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    here = os.path.dirname(os.path.abspath(__file__))

    train_module = load_module_from_path("train_mod", os.path.join(here, "2_train_model.py"))
    build_module = load_module_from_path("build_mod", os.path.join(here, "1_build_dataset.py"))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_cfg = train_module.model_config_from_checkpoint(checkpoint)
    model = train_module.build_model_from_checkpoint(checkpoint).to(device).eval()
    params = resolve_fusion_params(args, model_cfg)

    db_embeddings = checkpoint["db_embeddings"].cpu().numpy().astype(np.float32)
    db_poses = checkpoint["db_poses"].cpu().numpy().astype(np.float64)

    if len(db_poses) == 0:
        raise ValueError("The checkpoint contains an empty reference map.")
    if len(db_poses) != len(db_embeddings):
        raise ValueError("Reference poses and embeddings have different lengths.")

    val_start_ns = 0
    val_end_ns = 0
    if args.val_only:
        val_start_ns = int(checkpoint.get("val_start_timestamp_ns", 0))
        val_end_ns = int(checkpoint.get("val_end_timestamp_ns", 0))     # 0 = until the end (old checkpoints)
        if val_start_ns <= 0:
            raise ValueError("--val-only requires a checkpoint with a held-out split.")

    init_pose = tuple(float(v) for v in args.init_pose) if args.init_pose else None

    camera_topics = build_module.CAMERA_TOPICS
    camera_count = len(build_module.CAMERAS)

    # Retain older images so a future-stamped arrival cannot overwrite the most recent eligible image.
    buffers = [deque(maxlen=build_module.BUFFER_LEN) for _ in range(camera_count)]

    localizer = MultiStartLocalizer(db_embeddings, db_poses, params)
    telemetry = build_module.TelemetryTrack()
    health = OdomHealth()
    warned_odom = False
    warned_vo = False

    previous_image = None
    previous_norm = None
    previous_stamp = None
    reported_commit = False

    n_frames = 0
    n_skipped_stamps = 0
    n_tel_frames = 0
    timings_ms = []

    for path in (args.output, args.log):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    print(f"[INIT] Device: {device}")
    print(f"[INIT] Reference frames: {len(db_poses):,} | VO mode: {model_cfg['vo_mode']} | "
          f"preset: {'NEW' if model_cfg.get('q_scaling_fixed') else 'LEGACY'}")
    print(f"[CONFIG] {params}")
    print(f"[CONFIG] odometry source: {args.odom} (telemetry = speed + yaw rate from /constructor0/state_estimation)")
    if args.val_only:
        print(f"[VAL] Held-out block of the training run; timestamps {val_start_ns} .. {val_end_ns or 'end of bag'}")
    if init_pose is not None:
        print(f"[INIT] --init-pose given (experiments only): x={init_pose[0]:.3f} y={init_pose[1]:.3f} yaw={init_pose[2]:.4f}")

    # /tf and the pose fields of state_estimation are never used; only the telemetry fields are read from the latter.
    topics = list(camera_topics) + [build_module.STATE_ESTIMATION_TOPIC]

    with (
        open(args.output, "w", newline="") as output_file,
        open(args.log, "w", newline="") as log_file,
        open(args.mcap_path, "rb") as bag_file,
        torch.inference_mode(),
    ):
        output_writer = csv.writer(output_file)
        output_writer.writerow(["timestamp_ns", "x", "y", "yaw"])

        log_writer = csv.DictWriter(log_file, fieldnames=FusionLocalizer.TELEMETRY_FIELDS)
        log_writer.writeheader()

        reader = make_reader(bag_file, decoder_factories=[DecoderFactory()])

        for _, channel, _, msg in reader.iter_decoded_messages(topics=topics):
            topic = channel.topic

            if topic == build_module.STATE_ESTIMATION_TOPIC:
                telemetry.add(msg)
                continue

            if not hasattr(msg, "data") or len(msg.data) == 0:
                continue

            camera_idx = camera_topics[topic]
            stamp_ns = build_module.stamp_to_ns(msg.header.stamp)

            start_time = time.perf_counter()
            image = build_module.crop_and_resize(msg.data)
            buffers[camera_idx].append((stamp_ns, image))

            # Output is anchored to camera_fl.
            if camera_idx != 0 or stamp_ns < val_start_ns:
                continue
            if val_end_ns and stamp_ns > val_end_ns:
                break                       # end of the held-out block: stop reading the bag

            if previous_stamp is not None and stamp_ns <= previous_stamp:
                n_skipped_stamps += 1
                continue

            selected_images = []
            offsets = []
            missing_cameras = 0

            for camera_buffer in buffers:
                # Same selection rule as the dataset builder: newest image with stamp <= anchor.
                selected_stamp, selected_image = build_module.select_causal_image(camera_buffer, stamp_ns)
                if selected_image is None:
                    # Start-up fallback: never use a future-stamped secondary camera image.
                    selected_image = image
                    selected_stamp = stamp_ns
                    missing_cameras += 1
                selected_images.append(selected_image)
                offsets.append((stamp_ns - selected_stamp) * 1e-9)

            current_norm = normalize_stack(np.stack(selected_images, axis=0))
            current_image = torch.from_numpy(current_norm).unsqueeze(0).to(device)

            vo_delta = None
            q_var = None
            change_score = None
            dt = float("nan")

            if previous_image is None:
                _, embedding_t, _ = model.forward_single(current_image)
            else:
                dt = max((stamp_ns - previous_stamp) * 1e-9, 0.01)
                time_features = torch.tensor([[dt] + offsets], dtype=torch.float32, device=device)
                embedding_t, _, delta_t, log_var_t = model(previous_image, current_image, time_features)
                vo_delta = delta_t[0].cpu().numpy().astype(np.float64)
                q_var = np.exp(log_var_t[0].cpu().numpy()).astype(np.float64)
                change_score = frame_change_score(previous_norm, current_norm)

            embedding = embedding_t[0].cpu().numpy().astype(np.float32)

            # ---- which odometry drives this step? ------------------------------------------------------------
            tel = None
            if previous_stamp is not None and args.odom != "vo":
                tel = telemetry.integrate(previous_stamp, stamp_ns)
            odom_kind, motion, tel_speed, fell_back = select_odometry(args.odom, tel, vo_delta, health)
            if fell_back and not warned_odom:
                warned_odom = True
                print("[WARN] telemetry values out of range (speed / yaw rate); the learned VO is used for those frames.")
            if not health.ok and not warned_vo:
                warned_vo = True
                print(f"[INFO] the learned VO disagrees with the telemetry (median distance ratio "
                      f"{health.median_ratio:.2f}); the telemetry is kept, the VO is only a fallback.")
            n_tel_frames += odom_kind == "telemetry"

            if not localizer.started:
                n_cand = localizer.start(embedding, pose=init_pose)
                if init_pose is not None:
                    print("[START] using --init-pose")
                else:
                    print(f"[START] no initial pose is given: tracking {n_cand} candidate start positions for "
                          f"{params.init_min_frames}-{params.init_max_frames} frames")

            record = {name: float("nan") for name in FusionLocalizer.TELEMETRY_FIELDS}
            record.update(
                timestamp_ns=int(stamp_ns), dt=dt, max_camera_age_s=float(max(offsets)),
                missing_cameras=missing_cameras, tel_speed=tel_speed,
            )
            record.update(localizer.step(embedding, motion, q_var, change_score, odom_kind))

            if localizer.commit_frame is not None and not reported_commit:
                reported_commit = True
                print(f"[START] committed after {localizer.commit_frame} frames to x={record['x']:.2f} "
                      f"y={record['y']:.2f} yaw={math.degrees(record['yaw']):.1f}deg")

            previous_image = current_image
            previous_norm = current_norm
            previous_stamp = stamp_ns

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            timings_ms.append(elapsed_ms)

            n_frames += 1
            output_writer.writerow([int(stamp_ns), f"{record['x']:.6f}", f"{record['y']:.6f}", f"{record['yaw']:.6f}"])
            record.update(frame=n_frames, anchor_ms=elapsed_ms)
            log_writer.writerow(record)

            if n_frames == 1 or n_frames % 100 == 0:
                print(
                    f"[{n_frames:5d}] {record['mode']:11s} | odom={'tel' if odom_kind == 'telemetry' else 'vo '} | "
                    f"sim={record['top1_sim']:.3f} | gsim={record['global_sim']:.3f} | gate={int(record['gate_n']):4d} | "
                    f"hyp={int(record['n_hyp']):2d} | anchor={elapsed_ms:.1f}ms | "
                    f"x={record['x']:.2f}, y={record['y']:.2f}, yaw={math.degrees(record['yaw']):.1f}deg"
                )
                output_file.flush()
                log_file.flush()

    if n_frames == 0:
        raise RuntimeError("No predictions were produced. Check the bag and validation split.")

    timing_array = np.asarray(timings_ms)
    print("\n" + "=" * 70)
    print(f"[DONE] Wrote {n_frames} poses to {args.output}")
    print(f"[DONE] Telemetry: {args.log}")
    print(f"[INFO] Odometry: telemetry on {n_tel_frames}/{n_frames} frames, learned VO on the rest "
          f"(state_estimation samples read: {telemetry.n_added}, rejected: {telemetry.n_rejected})")
    print(f"[INFO] Retrieval updates: {localizer.n_updates}/{n_frames} | standstill frames: {localizer.n_still} | "
          f"global relocalizations: {localizer.n_reloc} | start committed at frame {localizer.commit_frame}")
    print(f"[INFO] Non-increasing anchor stamps skipped: {n_skipped_stamps}")
    print(f"[TIMING] Anchor processing mean={timing_array.mean():.2f}ms, p95={np.percentile(timing_array, 95):.2f}ms")
    print("[TIMING] Excludes bag decoding and secondary-camera preprocessing performed before the anchor.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap_path")
    parser.add_argument("--checkpoint", default="checkpoints/racing_model.pt")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--log", default="inference_telemetry.csv")
    parser.add_argument("--val-only", action="store_true",
                        help="process only the held-out validation block recorded in the checkpoint")
    parser.add_argument("--odom", choices=["auto", "telemetry", "vo"], default="auto",
                        help="motion source: auto = telemetry (speed + yaw rate) when present and plausible, else the "
                             "learned VO; telemetry = always; vo = never use telemetry")
    parser.add_argument("--init-pose", type=float, nargs=3, metavar=("X", "Y", "YAW"), default=None,
                        help="EXPERIMENTS ONLY. The competition gives no initial pose and the submission run must not "
                             "use one; by default the start position is found from the images.")
    add_fusion_args(parser)
    args = parser.parse_args()

    for name in ("q_x_floor", "q_scale", "q_tel_scale", "r_scale", "r_scale_tel", "gate_m", "gate_deg", "sim_tau"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative.")
    for name in ("r_scale", "r_scale_tel"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.gate_m is not None and args.gate_m <= 0:
        parser.error("--gate-m must be positive.")
    if args.gate_deg is not None and args.gate_deg <= 0:
        parser.error("--gate-deg must be positive.")

    run_causal_inference(args)


if __name__ == "__main__":
    main()
