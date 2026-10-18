"""
Track-frame geometry for the localisation filter.

/constructor0/track_bounds is a visualization_msgs/MarkerArray with one marker per border line (left, right,
pit lane, ...). 1_build_dataset.py used to flatten every point into one array, which loses which point belongs to
which line. This module keeps them per marker and turns them into a track frame:

  training poses (one clean lap)  ->  closed centerline c(s), tangent heading psi(s)
  + border markers                ->  centerline re-centred between the real borders,
                                      left / right border distances w_left(s), w_right(s)
  + training poses again          ->  extra lateral corridor where training laps left the main track (pit lane,
                                      run-off), racing-line heading offset vs. the tangent and its lap-to-lap spread

Any pose (x, y) maps to Frenet coordinates (s, n):
  s = arc length along the centerline [m, 0..L), increasing in the driving direction
  n = signed lateral offset [m], + = left of the driving direction

Ground truth enters only through the TRAINING poses (the same frames as the place database). The borders come from
the bag's map topic. Everything is built offline and stored in the checkpoint; inference only needs numpy / scipy.
"""
import math

import numpy as np
from scipy.spatial import cKDTree

TRACK_BOUNDS_TOPIC = '/constructor0/track_bounds'
H5_GROUP = 'track_bounds'

# visualization_msgs/Marker constants
_LINE_STRIP, _LINE_LIST, _POINTS = 4, 5, 8
_NON_GEOMETRY_TYPES = {0, 9, 10, 11}      # ARROW, TEXT_VIEW_FACING, MESH_RESOURCE, TRIANGLE_LIST
_DELETE_ACTIONS = {2, 3}                  # DELETE, DELETEALL


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _odd(n) -> int:
    n = max(int(round(n)), 1)
    return n if n % 2 else n + 1


# ==============================================================================================
# markers: MCAP message -> per-marker point arrays -> HDF5
# ==============================================================================================
def _quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-9:                                   # an all-zero orientation in a marker means identity
        return np.eye(3)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def markers_from_msg(msg) -> list:
    """visualization_msgs/MarkerArray -> [dict(ns, id, type, frame_id, points (M,3))], marker pose applied."""
    out = []
    for m in getattr(msg, 'markers', None) or []:
        if int(getattr(m, 'action', 0)) in _DELETE_ACTIONS:
            continue
        mtype = int(getattr(m, 'type', -1))
        raw = getattr(m, 'points', None) or []
        if mtype in _NON_GEOMETRY_TYPES or len(raw) < 2:
            continue
        pts = np.array([[float(p.x), float(p.y), float(p.z)] for p in raw], dtype=np.float64)
        pose = getattr(m, 'pose', None)
        if pose is not None:
            o, t = pose.orientation, pose.position
            pts = pts @ _quat_to_rot(o.x, o.y, o.z, o.w).T + np.array([t.x, t.y, t.z], dtype=np.float64)
        header = getattr(m, 'header', None)
        out.append(dict(ns=str(getattr(m, 'ns', '') or ''), id=int(getattr(m, 'id', len(out))), type=mtype,
                        frame_id=str(getattr(header, 'frame_id', '') or ''), points=pts))
    return out


def read_track_bounds_mcap(mcap_path: str, topic: str = TRACK_BOUNDS_TOPIC) -> list:
    """Per-marker borders from the first non-empty track_bounds message of a bag (only reads that topic)."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in reader.iter_decoded_messages(topics=[topic]):
            markers = markers_from_msg(msg)
            if markers:
                return markers
    return []


def _as_str(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def save_markers_h5(h5f, markers: list, group: str = H5_GROUP):
    """One dataset per marker under /track_bounds/marker_XXX with ns / id / type / frame_id attributes."""
    if group in h5f:
        del h5f[group]
    g = h5f.create_group(group)
    g.attrs['topic'] = TRACK_BOUNDS_TOPIC
    for i, m in enumerate(markers):
        d = g.create_dataset(f'marker_{i:03d}', data=np.asarray(m['points'], dtype=np.float64))
        for key in ('ns', 'id', 'type', 'frame_id'):
            d.attrs[key] = m[key]


def load_markers_h5(h5f, group: str = H5_GROUP) -> list:
    """Per-marker borders if present; a legacy cache (flat 'track_borders') comes back as one POINTS marker."""
    if group in h5f:
        out = []
        for name in sorted(h5f[group].keys()):
            d = h5f[group][name]
            out.append(dict(ns=_as_str(d.attrs.get('ns', '')), id=int(d.attrs.get('id', 0)),
                            type=int(d.attrs.get('type', _POINTS)), frame_id=_as_str(d.attrs.get('frame_id', '')),
                            points=d[:].astype(np.float64)))
        return out
    if 'track_borders' in h5f and len(h5f['track_borders']):
        return [dict(ns='legacy_flat', id=0, type=_POINTS, frame_id='',
                     points=h5f['track_borders'][:].astype(np.float64))]
    return []


def flat_points(markers: list) -> np.ndarray:
    """All marker points in one (M,3) array: the old 'track_borders' layout, kept for compatibility."""
    return np.concatenate([m['points'] for m in markers], 0) if markers else np.zeros((0, 3))


def marker_xy(m: dict, step: float = 0.5) -> np.ndarray:
    """Marker -> dense (M,2) border samples. Line markers are densified so long segments leave no gaps."""
    p = np.asarray(m['points'], dtype=np.float64)[:, :2]
    if m['type'] == _LINE_STRIP and len(p) >= 2:
        a, b = p[:-1], p[1:]
    elif m['type'] == _LINE_LIST and len(p) >= 2:
        h = len(p) // 2
        a, b = p[0:2 * h:2], p[1:2 * h:2]
    else:
        return p
    seg = np.hypot(*(b - a).T)
    n = np.maximum(np.ceil(seg / step).astype(int), 1)
    rep = np.repeat(np.arange(len(a)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
    return np.vstack([a[rep] + (b - a)[rep] * t[:, None], b[-1:]])


# ==============================================================================================
# closed-curve helpers
# ==============================================================================================
def _tangent(xy: np.ndarray) -> np.ndarray:
    d = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def _normal(psi: np.ndarray) -> np.ndarray:
    return np.stack([-np.sin(psi), np.cos(psi)], axis=1)


def _smooth_circ(v: np.ndarray, w: int) -> np.ndarray:
    """Circular moving average along the first axis (window w, odd)."""
    w = _odd(w)
    if w <= 1 or len(v) <= w:
        return v.copy()
    pad, k = w // 2, np.ones(w) / w
    if v.ndim == 1:
        return np.convolve(np.r_[v[-pad:], v, v[:pad]], k, mode='valid')
    return np.stack([_smooth_circ(v[:, j], w) for j in range(v.shape[1])], axis=1)


def _fill_circ(v: np.ndarray, default: float) -> np.ndarray:
    """Fill NaNs by circular linear interpolation between the nearest valid neighbours."""
    ok = np.isfinite(v)
    if not ok.any():
        return np.full(len(v), float(default))
    K, i = len(v), np.arange(len(v))
    xi, vi = i[ok], v[ok]
    return np.interp(i, np.r_[xi - K, xi, xi + K], np.r_[vi, vi, vi])


def _resample_closed(pts: np.ndarray, ds: float):
    """Closed polyline -> equally spaced samples (spacing ~ds). Returns (xy, length)."""
    pts = np.asarray(pts, dtype=np.float64)
    step = np.hypot(*np.diff(pts, axis=0).T)
    pts = pts[np.r_[True, step > 1e-6]]
    P = np.vstack([pts, pts[:1]])
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    L = float(cum[-1])
    n = max(int(round(L / ds)), 8)
    s = np.linspace(0.0, L, n, endpoint=False)
    return np.stack([np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])], axis=1), L


def _reference_lap(xy: np.ndarray, yaw: np.ndarray, min_lap_len: float = 500.0, near_m: float = 8.0):
    """
    Pick one clean lap from the (time-ordered) training poses. Laps are split at passes through an anchor pose in
    the middle of the run; the lap that agrees best with all the others is kept, so a pit-lane lap or an off-track
    excursion does not become the reference.
    """
    N = len(xy)
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    laps = None
    for frac in (0.5, 0.35, 0.65, 0.2, 0.8):
        a = int(frac * (N - 1))
        d = np.hypot(*(xy - xy[a]).T)
        idx = np.flatnonzero((d < near_m) & (np.abs(wrap(yaw - yaw[a])) < math.radians(45.0)))
        if len(idx) == 0:
            continue
        passes = []
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            p = int(run[np.argmin(d[run])])
            if not passes or cum[p] - cum[passes[-1]] >= min_lap_len:
                passes.append(p)
        if len(passes) >= 2:
            laps = list(zip(passes[:-1], passes[1:]))
            break
    if laps is None:
        return None, {}
    if len(laps) == 1:
        a, b = laps[0]
        return xy[a:b], dict(n_laps=1, lap_len=float(cum[b] - cum[a]), lap_p99_dev=float('nan'))
    scores = []
    for i, (a, b) in enumerate(laps):
        others = np.concatenate([xy[c:e] for j, (c, e) in enumerate(laps) if j != i])
        dist, _ = cKDTree(others).query(xy[a:b])
        scores.append(float(np.percentile(dist, 99)))
    best = int(np.argmin(scores))
    a, b = laps[best]
    return xy[a:b], dict(n_laps=len(laps), lap_len=float(cum[b] - cum[a]), lap_p99_dev=scores[best])


def _walls(st_xy, st_yaw, ref_n, tree, B, max_half=25.0, min_gap=0.05, wins=(1.0, 2.5, 5.0)):
    """
    For every station: lateral offset (in station n coordinates) of the nearest border sample above and below the
    reference offset ref_n, using only border samples within an along-track window around the station (the window
    grows if a side has no sample). NaN where nothing is found within max_half.
    """
    K = len(st_xy)
    t = np.stack([np.cos(st_yaw), np.sin(st_yaw)], axis=1)
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
    centers = st_xy + nrm * ref_n[:, None]
    lists = tree.query_ball_point(centers, r=max_half + wins[-1])
    up, lo = np.full(K, np.nan), np.full(K, np.nan)
    for i, idx in enumerate(lists):
        if not idx:
            continue
        rel = B[idx] - centers[i]
        a, n = rel @ t[i], rel @ nrm[i]
        for w in wins:
            m = np.abs(a) <= w
            if np.isnan(up[i]):
                pos = n[m & (n > min_gap)]
                if len(pos):
                    up[i] = pos.min()
            if np.isnan(lo[i]):
                neg = n[m & (n < -min_gap)]
                if len(neg):
                    lo[i] = neg.max()
            if np.isfinite(up[i]) and np.isfinite(lo[i]):
                break
    up[up > max_half] = np.nan
    lo[lo < -max_half] = np.nan
    return ref_n + up, ref_n + lo


# ==============================================================================================
# track map
# ==============================================================================================
class TrackMap:
    """Closed centerline with per-station border widths, optional extra (pit / run-off) corridor and heading prior."""
    VERSION = 1

    def __init__(self, xy, w_left=None, w_right=None, pit_lo=None, pit_hi=None, head_off=None, head_std=None,
                 info=None, db_sn=None):
        self.xy = np.asarray(xy, dtype=np.float64)
        self.K = len(self.xy)
        seg = np.hypot(*np.diff(np.vstack([self.xy, self.xy[:1]]), axis=0).T)
        self.length = float(seg.sum())
        self.s = np.r_[0.0, np.cumsum(seg)[:-1]]
        self.yaw = _tangent(self.xy)                                   # track tangent heading psi(s)
        self.t = np.stack([np.cos(self.yaw), np.sin(self.yaw)], axis=1)
        self.nrm = _normal(self.yaw)

        def arr(v, default):
            return np.full(self.K, float(default)) if v is None else np.asarray(v, dtype=np.float64)

        self.w_left, self.w_right = arr(w_left, 7.0), arr(w_right, 7.0)
        self.pit_lo, self.pit_hi = arr(pit_lo, np.nan), arr(pit_hi, np.nan)
        self.head_off, self.head_std = arr(head_off, 0.0), arr(head_std, math.radians(5.0))
        self.info = dict(info or {})
        self.db_sn = None if db_sn is None else np.asarray(db_sn, dtype=np.float32)
        self.tree = cKDTree(self.xy)

    # ------------------------------------------------------------------------------------------
    def wrap_s(self, ds):
        """Signed arc-length difference folded into [-L/2, L/2)."""
        return (np.asarray(ds) + 0.5 * self.length) % self.length - 0.5 * self.length

    def project(self, xy, yaw=None, k_nn: int = 8):
        """(M,2) positions [+ headings] -> (s, n, station index). With a heading, stations whose tangent points the
        other way are skipped, so a pose never snaps onto an adjacent, opposite-direction part of the track."""
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        kq = min(k_nn, self.K)
        d, idx = self.tree.query(xy, k=kq)
        if kq == 1:
            d, idx = d[:, None], idx[:, None]
        if yaw is not None:
            yaw = np.broadcast_to(np.asarray(yaw, dtype=np.float64), (len(xy),))
            d = np.where(np.abs(wrap(self.yaw[idx] - yaw[:, None])) > 0.5 * math.pi, d + 1e6, d)
        k = idx[np.arange(len(xy)), np.argmin(d, axis=1)]
        rel = xy - self.xy[k]
        a = np.einsum('ij,ij->i', rel, self.t[k])
        n = np.einsum('ij,ij->i', rel, self.nrm[k])
        return (self.s[k] + a) % self.length, n, k

    def project1(self, x: float, y: float, yaw=None):
        s, n, k = self.project(np.array([[x, y]]), None if yaw is None else np.array([yaw]))
        return float(s[0]), float(n[0]), int(k[0])

    def to_xy(self, s, n):
        """Frenet (s, n) -> map (x, y)."""
        s = np.asarray(s, dtype=np.float64) % self.length
        k = np.clip(np.searchsorted(self.s, s, side='right') - 1, 0, self.K - 1)
        a = (s - self.s[k])[..., None]
        return self.xy[k] + a * self.t[k] + np.asarray(n, dtype=np.float64)[..., None] * self.nrm[k]

    # ------------------------------------------------------------------------------------------
    def on_main(self, k: int, n: float, tol: float = 1.0) -> bool:
        return -self.w_right[k] - tol <= n <= self.w_left[k] + tol

    def lateral_intervals(self, k: int, margin: float = 0.0):
        """Allowed n ranges at station k: the main track between its borders, plus the pit / run-off corridor."""
        out = []
        for lo, hi in ((-self.w_right[k], self.w_left[k]), (self.pit_lo[k], self.pit_hi[k])):
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            lo, hi = lo + margin, hi - margin
            if hi < lo:
                lo = hi = 0.5 * (lo + hi)
            out.append((float(lo), float(hi)))
        return out

    def heading_ref(self, k: int):
        """Expected heading at station k (tangent + mean racing-line offset) and its lap-to-lap spread [rad]."""
        return float(wrap(self.yaw[k] + self.head_off[k])), float(self.head_std[k])

    # ------------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = dict(version=self.VERSION, xy=self.xy, w_left=self.w_left, w_right=self.w_right, pit_lo=self.pit_lo,
                 pit_hi=self.pit_hi, head_off=self.head_off, head_std=self.head_std, info=self.info)
        if self.db_sn is not None:
            d['db_sn'] = self.db_sn
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'TrackMap':
        g = lambda key: None if d.get(key) is None else np.asarray(d[key])
        return cls(g('xy'), g('w_left'), g('w_right'), g('pit_lo'), g('pit_hi'), g('head_off'), g('head_std'),
                   info=d.get('info'), db_sn=g('db_sn'))

    def summary(self) -> str:
        i = self.info
        roles = [r['role'] for r in i.get('markers', [])]
        return (f'L={self.length:.0f} m, {self.K} stations, borders={"yes" if i.get("has_borders") else "NO (fallback width)"} '
                f'(both walls found at {100 * i.get("wall_coverage", 0):.0f}% of stations), '
                f'width L/R median {np.median(self.w_left):.1f}/{np.median(self.w_right):.1f} m, '
                f'pit/run-off stations {int(np.isfinite(self.pit_lo).sum())}, '
                f'markers left={roles.count("left")} right={roles.count("right")} other={roles.count("other")}, '
                f'heading spread median {math.degrees(float(np.median(self.head_std))):.1f} deg')


def build_track_map(db_poses, markers=None, ds: float = 1.0, fallback_half_width: float = 7.0,
                    max_half_width: float = 25.0, smooth_m: float = 15.0, recenter_iters: int = 2,
                    pit_half_width: float = 3.0, offtrack_tol: float = 1.0) -> TrackMap:
    """
    db_poses: (N,3) time-ordered training poses [x, y, yaw] (same frames as the place database).
    markers:  per-marker borders (load_markers_h5 / read_track_bounds_mcap). Without borders the corridor falls back
              to +-fallback_half_width around the reference lap (equivalent to the old lat_clamp).
    """
    poses = np.asarray(db_poses, dtype=np.float64)
    xy, yaw = poses[:, :2], poses[:, 2]

    # 1. reference lap -> smooth closed polyline
    lap, lap_info = _reference_lap(xy, yaw)
    if lap is None:
        raise ValueError('no closed lap found in the database poses (need at least one full training lap)')
    st, _ = _resample_closed(lap, ds)
    st, _ = _resample_closed(_smooth_circ(st, _odd(smooth_m / ds)), ds)

    # 2. re-centre between the real borders, then measure the widths
    markers = list(markers or [])
    dense = [marker_xy(m) for m in markers]
    B = np.concatenate(dense, 0) if dense else np.zeros((0, 2))
    has_borders = len(B) >= 10
    tree = cKDTree(B) if has_borders else None
    if has_borders:
        for _ in range(recenter_iters):
            psi = _tangent(st)
            up, lo = _walls(st, psi, np.zeros(len(st)), tree, B, max_half_width)
            mid = np.where(np.isfinite(up) & np.isfinite(lo), 0.5 * (up + lo), np.nan)
            shift = _smooth_circ(_fill_circ(mid, 0.0), _odd(smooth_m / ds))
            st, _ = _resample_closed(st + _normal(psi) * shift[:, None], ds)
        up, lo = _walls(st, _tangent(st), np.zeros(len(st)), tree, B, max_half_width)
        coverage = float((np.isfinite(up) & np.isfinite(lo)).mean())
        w_left, w_right = _fill_circ(up, fallback_half_width), _fill_circ(-lo, fallback_half_width)
    else:
        coverage = 0.0
        w_left = w_right = np.full(len(st), float(fallback_half_width))
    tm = TrackMap(st, w_left, w_right)
    K = tm.K

    # 3. database poses in the track frame
    s, n, k = tm.project(xy, yaw)
    on_main = (n <= tm.w_left[k] + offtrack_tol) & (n >= -tm.w_right[k] - offtrack_tol)

    # 4. heading prior: mean racing-line heading relative to the tangent + its lap-to-lap spread (main track only)
    off = wrap(yaw - tm.yaw[k])
    C, S, cnt = np.zeros(K), np.zeros(K), np.zeros(K)
    np.add.at(C, k[on_main], np.cos(off[on_main]))
    np.add.at(S, k[on_main], np.sin(off[on_main]))
    np.add.at(cnt, k[on_main], 1.0)
    w = _odd(5.0 / ds)
    Cs, Ss, cs = _smooth_circ(C, w), _smooth_circ(S, w), _smooth_circ(cnt, w)
    ok = cs * w >= 3.0
    R = np.hypot(Cs, Ss) / np.maximum(cs, 1e-9)                         # mean resultant length
    head_off = np.where(ok, np.arctan2(Ss, Cs), 0.0)
    head_std = np.where(ok, np.sqrt(-2.0 * np.log(np.clip(R, 1e-6, 1.0))), math.radians(5.0))

    # 5. extra corridor where training laps left the main track (pit lane, run-off), widened to the real borders
    pit_lo, pit_hi = np.full(K, np.inf), np.full(K, -np.inf)
    off_m = ~on_main & (np.abs(n) < 40.0)
    np.minimum.at(pit_lo, k[off_m], n[off_m])
    np.maximum.at(pit_hi, k[off_m], n[off_m])
    lo_d, hi_d = pit_lo.copy(), pit_hi.copy()
    for r in range(1, _odd(10.0 / ds) // 2 + 1):                        # dilate +-5 m along s: no gaps
        for sh in (r, -r):
            lo_d, hi_d = np.minimum(lo_d, np.roll(pit_lo, sh)), np.maximum(hi_d, np.roll(pit_hi, sh))
    has_pit = np.isfinite(lo_d) & np.isfinite(hi_d)
    p_lo, p_hi = np.full(K, np.nan), np.full(K, np.nan)
    p_lo[has_pit], p_hi[has_pit] = lo_d[has_pit] - pit_half_width, hi_d[has_pit] + pit_half_width
    if has_borders and has_pit.any():
        j = np.flatnonzero(has_pit)
        wall_up, _ = _walls(tm.xy[j], tm.yaw[j], hi_d[j], tree, B, max_half_width)
        _, wall_lo = _walls(tm.xy[j], tm.yaw[j], lo_d[j], tree, B, max_half_width)
        snap_up = np.isfinite(wall_up) & (wall_up - hi_d[j] <= 2.0 * pit_half_width)
        snap_lo = np.isfinite(wall_lo) & (lo_d[j] - wall_lo <= 2.0 * pit_half_width)
        p_hi[j[snap_up]] = wall_up[snap_up]
        p_lo[j[snap_lo]] = wall_lo[snap_lo]
    tm.pit_lo, tm.pit_hi = p_lo, p_hi
    tm.head_off, tm.head_std = head_off, head_std
    tm.db_sn = np.stack([s, n], axis=1).astype(np.float32)

    # 6. which marker is which (diagnostic only: the geometry above does not need the labels)
    roles = []
    for m, d in zip(markers, dense):
        if len(d) == 0:
            continue
        _, mn, mk = tm.project(d)
        near_l = float((np.abs(mn - tm.w_left[mk]) < 1.5).mean())
        near_r = float((np.abs(mn + tm.w_right[mk]) < 1.5).mean())
        role = 'left' if near_l > 0.6 else 'right' if near_r > 0.6 else 'other'
        roles.append(dict(ns=m['ns'], id=int(m['id']), type=int(m['type']), n_points=int(len(m['points'])),
                          role=role, median_n=float(np.median(mn))))

    tm.info = dict(lap_info, has_borders=bool(has_borders), wall_coverage=coverage, markers=roles,
                   db_on_main_pct=float(100.0 * on_main.mean()), ds=float(ds))
    return tm
