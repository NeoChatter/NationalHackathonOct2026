"""
Numpy-only localisation filter (replaces GuardianSE2EKF and the mode logic in 3_causal_infer).

Physical ideas encoded here
  1. A car moves along its heading: lateral body displacement ~ 0, so only (dx_body, dyaw) from vision are used.
  2. Visual odometry has a SCALE error (a % of the distance), so process noise grows with dx.
  3. Place retrieval is accurate across the track (~ lap-to-lap line difference) and poor along it:
     the measurement covariance is anisotropic, rotated into the matched heading.
  4. Losing the track must never cause a jump: while lost, the search radius grows around the
     VO-predicted position, and a reset needs 5 consecutive, mutually consistent matches.
  5. The car is always on the track: its lateral offset from the nearest training path is clamped.

With a TrackMap (track_map.py) ideas 3-5 use the real track frame instead of the training path:
  3'. DB poses are known in Frenet (s, n): the search gate is also limited in arc length, and the anisotropic
      retrieval covariance is aligned with the track tangent.
  5'. The lateral clamp uses the real left / right borders (+ the pit / run-off corridor seen in training).
  6.  Soft heading prior: the heading stays near the racing-line heading at s (track tangent + learned offset).
Without a TrackMap the behaviour is unchanged.
"""
import math
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_CALIB = dict(
    odom_dx_std=0.30,        # m per frame      (overwritten by calibration from validation laps)
    odom_dyaw_std=0.01,      # rad per frame
    retr_along_std=6.0,      # m  retrieval error along the heading
    retr_lat_std=2.5,        # m  retrieval error across the heading
    retr_yaw_std=0.08,       # rad
    sim_thresh=0.5,          # cosine similarity below which a match is ignored
    lat_clamp=7.0,           # m  max lateral distance from the nearest training path
    sim_floor=0.80,          # below this cosine similarity a match is ignored completely
    # ---- track-frame options (only used when a TrackMap is passed) ----
    border_margin=0.5,       # m   keep the reference point this far inside the borders
    clamp_skip_m=15.0,       # m   further than this outside every corridor: lost / unmapped, do not clamp
    heading_prior=True,      #     soft prior: heading ~ racing-line heading at s
    heading_prior_floor_deg=2.0,     # minimum std of the prior
    heading_prior_inflate=10.0,      # variance inflation: the prior's error is correlated over ~1 s of frames
    heading_prior_gate_deg=30.0,     # larger deviation = the prior does not apply here (pit entry, spin)
)
CHI2_3DOF_99_9 = 16.27


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Tracker:
    def __init__(self, pose0, db_poses: np.ndarray, calib=None, track=None):
        """pose0 = (x, y, yaw); db_poses: (N,3) [x, y, yaw] of the training frames; track: optional TrackMap."""
        self.c = {**DEFAULT_CALIB, **(calib or {})}
        self.db_xy = np.asarray(db_poses[:, :2], dtype=np.float64)
        self.db_yaw = np.asarray(db_poses[:, 2], dtype=np.float64)
        self.tree = cKDTree(self.db_xy)
        self.track = track
        self.sn = (float('nan'), float('nan'))              # last (s, n) of the estimate (telemetry)
        self.n_clamped = 0
        self.n_heading_prior = 0
        if track is not None:
            self.db_s, self.db_n, self.db_k = track.project(self.db_xy, self.db_yaw)
        self.x = np.array([pose0[0], pose0[1], wrap(pose0[2])], dtype=np.float64)
        self.k = float(self.c.get('odom_dx_scale', 1.0))   # odometry scale, estimated ONLINE (VO under/over-reads speed)
        self.P = np.diag([0.3 ** 2, 0.3 ** 2, math.radians(1.0) ** 2, 0.15 ** 2])
        self.lost = 0
        self.chain = []
        self.dx_hist = deque(maxlen=3)
        self.v = 0.0     # m/s   (from VO, only used to coast through a dropped frame)
        self.w = 0.0     # rad/s

    # ------------------------------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return 'RELOCALIZING' if self.lost >= 10 else 'TRACKING'

    # ------------------------------------------------------------------------------------------
    def predict(self, dx_body: float, dyaw: float, dt: float):
        if dt <= 0.0:
            return
        self.dx_hist.append(float(dx_body))
        dx = self._calibrate_dx(float(np.median(self.dx_hist)))        # median of 3 kills single-frame spikes
        dx = float(np.clip(dx, 0.0, 60.0 * dt))
        dyaw = float(np.clip(dyaw, -1.5 * dt, 1.5 * dt))               # <= 1.5 rad/s
        self.v, self.w = dx / dt, dyaw / dt
        self._propagate(dx, dyaw)

    def _calibrate_dx(self, dx: float) -> float:
        """Speed-dependent VO bias (regression to the mean: reads too LOW when fast, too HIGH when slow).
        calibrate_odom.py measures predicted-vs-true forward step on the held-out laps and stores the curve."""
        xp, fp = self.c.get('dx_map_pred'), self.c.get('dx_map_true')
        if not xp or len(xp) < 2:
            return dx
        if dx > xp[-1]:
            return dx * (fp[-1] / max(xp[-1], 1e-6))
        return float(np.interp(dx, xp, fp))

    def coast(self, dt: float):
        """Dropped / stale camera set: keep predicting with the last speed and yaw rate."""
        if dt > 0.0:
            self._propagate(self.v * dt, self.w * dt)

    def _propagate(self, dx_vo: float, dyaw: float):
        x, y, yaw = self.x
        dx = self.k * dx_vo                                   # metres actually travelled
        ym = yaw + 0.5 * dyaw
        c, s = math.cos(ym), math.sin(ym)
        self.x = np.array([x + dx * c, y + dx * s, wrap(yaw + dyaw)])

        F = np.eye(4)
        F[0, 2], F[1, 2] = -dx * s, dx * c
        F[0, 3], F[1, 3] = dx_vo * c, dx_vo * s               # d(position)/d(scale)
        sig_dx = max(self.c['odom_dx_std'], 0.03 * dx) * max(self.k, 0.5)
        sig_w = max(self.c['odom_dyaw_std'], math.radians(0.3))
        sig_y = 0.05 + 0.01 * dx                              # small sideslip
        gx = np.array([c, s, 0.0, 0.0])
        gw = np.array([-0.5 * dx * s, 0.5 * dx * c, 1.0, 0.0])
        gy = np.array([-s, c, 0.0, 0.0])
        Q = np.outer(gx, gx) * sig_dx ** 2 + np.outer(gw, gw) * sig_w ** 2 + np.outer(gy, gy) * sig_y ** 2
        Q[3, 3] = 0.004 ** 2                                  # scale is a slow random walk
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------------------------------
    def gate_indices(self) -> np.ndarray:
        """Which database frames to search. Grows while lost; never the whole map in one jump."""
        if self.mode == 'TRACKING':
            r = min(35.0 + 10.0 * self.lost, 150.0)
            yaw_gate = math.radians(35.0 + 3.0 * self.lost)
        else:
            r = min(60.0 + 10.0 * self.lost, 400.0)
            yaw_gate = math.radians(90.0)
        cand = np.asarray(self.tree.query_ball_point(self.x[:2], r), dtype=np.int64)
        if len(cand) == 0:
            return cand
        dyaw = np.abs(np.arctan2(np.sin(self.db_yaw[cand] - self.x[2]), np.cos(self.db_yaw[cand] - self.x[2])))
        cand = cand[dyaw <= yaw_gate]
        if self.track is not None and len(cand):
            # same neighbourhood in arc length too: drops other parts of the track that are close in (x, y)
            s0, _, _ = self.track.project1(self.x[0], self.x[1], self.x[2])
            ds = np.abs(self.track.wrap_s(self.db_s[cand] - s0))
            cand = cand[ds <= 1.25 * r + 5.0]
        return cand

    # ------------------------------------------------------------------------------------------
    def update(self, cand_idx: np.ndarray, sims: np.ndarray) -> bool:
        """cand_idx: DB indices sorted by descending similarity; sims: matching cosine similarities."""
        thr, floor = self.c['sim_thresh'], self.c.get('sim_floor', 0.80)
        if len(cand_idx) == 0 or sims[0] < floor:             # hard floor only: below this a match is noise
            self.lost += 1
            return False
        # Soft weighting: a match below the calibrated threshold is not thrown away, it is trusted less
        # (R grows with the square of the shortfall). Unseen places (pit lane, other light) give weak,
        # not zero, information instead of freezing the filter.
        infl = min(1.0 + (max(0.0, thr - float(sims[0])) / 0.01) ** 2, 400.0)

        i0 = int(cand_idx[0])
        d = np.hypot(*(self.db_xy[cand_idx] - self.db_xy[i0]).T)
        dyw = np.abs(np.arctan2(np.sin(self.db_yaw[cand_idx] - self.db_yaw[i0]),
                                np.cos(self.db_yaw[cand_idx] - self.db_yaw[i0])))
        keep = (d <= 6.0) & (dyw <= math.radians(20.0))       # only blend neighbours of the best match
        w = np.exp((sims[keep] - sims[0]) * 20.0)
        w /= w.sum()
        zx = float(np.sum(w * self.db_xy[cand_idx[keep], 0]))
        zy = float(np.sum(w * self.db_xy[cand_idx[keep], 1]))
        zyaw = math.atan2(float(np.sum(w * np.sin(self.db_yaw[cand_idx[keep]]))),
                          float(np.sum(w * np.cos(self.db_yaw[cand_idx[keep]]))))
        z = np.array([zx, zy, zyaw])

        # along / across axes: the track tangent at the match if known, else the matched heading
        ang = float(self.track.yaw[self.db_k[i0]]) if self.track is not None else zyaw
        cz, sz = math.cos(ang), math.sin(ang)
        Rrot = np.array([[cz, -sz], [sz, cz]])
        R = np.zeros((3, 3))
        R[:2, :2] = Rrot @ np.diag([self.c['retr_along_std'] ** 2, self.c['retr_lat_std'] ** 2]) @ Rrot.T
        R[2, 2] = self.c['retr_yaw_std'] ** 2
        R *= infl

        if self.mode == 'TRACKING':
            if self._ekf_update(z, R):
                self.lost = 0
                self.chain.clear()
                return True
            self.lost += 1
            return False

        # RELOCALIZING: 5 consecutive STRONG matches, each within 8 m (~2 frames of motion) of the previous one
        if sims[0] < max(floor, thr - 0.05):
            self.chain = []
            self.lost += 1
            return False
        if self.chain and math.hypot(z[0] - self.chain[-1][0], z[1] - self.chain[-1][1]) > 8.0:
            self.chain = []
        self.chain.append(z)
        if len(self.chain) >= 5:
            self.x = z.copy()
            self.P = np.zeros((4, 4))
            self.P[:3, :3] = R + np.diag([1.0, 1.0, math.radians(2.0) ** 2])
            self.P[3, 3] = 0.05 ** 2
            self.dx_hist.clear()
            self.lost = 0
            self.chain = []
            return True
        self.lost += 1
        return False

    def _ekf_update(self, z: np.ndarray, R: np.ndarray) -> bool:
        y = np.array([z[0] - self.x[0], z[1] - self.x[1], wrap(z[2] - self.x[2])])
        S = self.P[:3, :3] + R
        try:
            d2 = float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            return False
        if d2 > CHI2_3DOF_99_9:
            return False
        K = self.P[:, :3] @ np.linalg.inv(S)                  # (4,3)
        dxk = K @ y
        self.x = self.x + dxk[:3]
        self.x[2] = wrap(self.x[2])
        self.k = float(np.clip(self.k + dxk[3], 0.6, 1.8))
        H = np.zeros((3, 4))
        H[:, :3] = np.eye(3)
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    # ------------------------------------------------------------------------------------------
    def pose_at(self, lag_s: float):
        """Pose extrapolated lag_s seconds forward (state is anchored at the camera_fl exposure time;
        the submitted timestamp is the newest image stamp used, which keeps the output strictly causal)."""
        if lag_s <= 0.0:
            return float(self.x[0]), float(self.x[1]), float(self.x[2])
        ds = self.k * self.v * lag_s
        ym = self.x[2] + 0.5 * self.w * lag_s
        return (float(self.x[0] + ds * math.cos(ym)), float(self.x[1] + ds * math.sin(ym)),
                float(wrap(self.x[2] + self.w * lag_s)))

    # ------------------------------------------------------------------------------------------
    def end_frame(self):
        """Track constraints after predict / update: heading prior + real-border clamp, or the old path clamp."""
        if self.track is None:
            self._clamp_to_training_path()
            return
        _, n, k = self.track.project1(self.x[0], self.x[1], self.x[2])
        if self.c.get('heading_prior', True):
            self._heading_prior(k, n)
        s, n, k = self.track.project1(self.x[0], self.x[1], self.x[2])
        n_new = self._constrain_lateral(k, n)
        if n_new != n:
            self.x[:2] += (n_new - n) * self.track.nrm[k]
            self.n_clamped += 1
        self.sn = (s, n_new)

    def _constrain_lateral(self, k: int, n: float) -> float:
        """Move n into the nearest allowed interval (main track between its borders, or the pit / run-off corridor)."""
        ivs = self.track.lateral_intervals(k, self.c['border_margin'])
        if not ivs or any(lo <= n <= hi for lo, hi in ivs):
            return n
        best = min((min(max(n, lo), hi) for lo, hi in ivs), key=lambda v: abs(v - n))
        if abs(best - n) > self.c['clamp_skip_m']:       # far outside everything: lost / unmapped, do not snap
            return n
        return float(best)

    def _heading_prior(self, k: int, n: float):
        """Soft EKF pseudo-measurement yaw = racing-line heading at s. Main track only; skipped when far off."""
        if not self.track.on_main(k, n):
            return
        ref, std = self.track.heading_ref(k)
        std = max(std, math.radians(self.c['heading_prior_floor_deg']))
        R = std * std * self.c['heading_prior_inflate']
        y = wrap(ref - self.x[2])
        if abs(y) > math.radians(self.c['heading_prior_gate_deg']):
            return
        S = self.P[2, 2] + R
        if y * y / S > 9.0:                                   # 3-sigma: disagrees with the filter, do not force it
            return
        K = self.P[:, 2] / S                                  # (4,)
        dxk = K * y
        self.x = self.x + dxk[:3]
        self.x[2] = wrap(self.x[2])
        self.k = float(np.clip(self.k + dxk[3], 0.6, 1.8))
        I_KH = np.eye(4)
        I_KH[:, 2] -= K
        self.P = I_KH @ self.P @ I_KH.T + R * np.outer(K, K)
        self.P = 0.5 * (self.P + self.P.T)
        self.n_heading_prior += 1

    def _clamp_to_training_path(self):
        """Corridor clamp (no track map): stay within lat_clamp metres (across) of the nearest training path."""
        dist, j = self.tree.query(self.x[:2])
        if dist > 40.0:
            return
        yj = self.db_yaw[j]
        n = np.array([-math.sin(yj), math.cos(yj)])
        lat = float((self.x[:2] - self.db_xy[j]) @ n)
        L = self.c['lat_clamp']
        excess = lat - float(np.clip(lat, -L, L))
        if excess != 0.0:
            self.x[:2] -= excess * n
