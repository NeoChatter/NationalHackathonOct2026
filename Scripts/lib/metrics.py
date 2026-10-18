"""
Scoring and error analysis. align() + score() follow tools/evaluate.py exactly (sorting, GT gap rule, linear
interpolation on the unwrapped yaw, body-frame lon / lat split, 1 s coverage bins), so numbers agree with the
official scorer. Everything else (slices, coverage, recovery) builds on the same aligned errors.
"""
import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MAX_GT_GAP_NS = 50_000_000
BIN_NS = 1_000_000_000
MIN_COVERAGE = 0.90


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def load_gt(path: str, min_ns: int = 0) -> pd.DataFrame:
    gt = pd.read_csv(path).sort_values('timestamp_ns').reset_index(drop=True)
    return gt[gt.timestamp_ns >= min_ns].reset_index(drop=True) if min_ns else gt


def align(gt: pd.DataFrame, sub: pd.DataFrame) -> dict:
    """Per evaluated submission row: errors + the row index into `sub` (to join telemetry)."""
    tg = gt.timestamp_ns.values.astype(np.int64)
    order = np.argsort(sub.timestamp_ns.values, kind='stable')
    ts = sub.timestamp_ns.values.astype(np.int64)[order]
    i = np.searchsorted(tg, ts)
    ok = (i > 0) & (i < len(tg))
    i = np.clip(i, 1, len(tg) - 1)
    ok &= (tg[i] - tg[i - 1]) <= MAX_GT_GAP_NS
    rows, ts, i = order[ok], ts[ok], i[ok]
    a = (ts - tg[i - 1]) / (tg[i] - tg[i - 1])
    yaw_u = np.unwrap(gt.yaw.values)
    gx = gt.x.values[i - 1] + a * (gt.x.values[i] - gt.x.values[i - 1])
    gy = gt.y.values[i - 1] + a * (gt.y.values[i] - gt.y.values[i - 1])
    gyaw = yaw_u[i - 1] + a * (yaw_u[i] - yaw_u[i - 1])
    dx, dy = sub.x.values[rows] - gx, sub.y.values[rows] - gy
    return dict(rows=rows, ts=ts, n_sub=len(sub), gt_t0=int(tg[0]), gt_t1=int(tg[-1]),
                gx=gx, gy=gy, gyaw=gyaw, err=np.hypot(dx, dy),
                lon=np.cos(gyaw) * dx + np.sin(gyaw) * dy, lat=-np.sin(gyaw) * dx + np.cos(gyaw) * dy,
                dyaw_deg=np.degrees(np.angle(np.exp(1j * (sub.yaw.values[rows] - gyaw)))))


def _rms(v) -> float:
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else float('nan')


def score(al: dict) -> dict:
    bins = np.arange(al['gt_t0'], al['gt_t1'], BIN_NS)
    covered = np.unique(np.searchsorted(bins, al['ts'], side='right') - 1)
    cov = len(covered) / max(len(bins), 1)
    e = al['err']
    return dict(n_sub=al['n_sub'], n_eval=len(e), coverage=float(cov), ranked=bool(cov >= MIN_COVERAGE),
                rmse=_rms(e), median=float(np.median(e)), p95=float(np.percentile(e, 95)),
                lat=_rms(al['lat']), lon=_rms(al['lon']), yaw_deg=_rms(al['dyaw_deg']),
                within_025=float(np.mean(e < 0.25) * 100), within_050=float(np.mean(e < 0.5) * 100),
                within_100=float(np.mean(e < 1.0) * 100))


def format_score(s: dict) -> str:
    lines = [f'poses submitted        {s["n_sub"]}   evaluated {s["n_eval"]}',
             f'coverage (1 s bins)    {s["coverage"] * 100:.1f} %   (>= {MIN_COVERAGE * 100:.0f} % to be ranked)',
             f'position RMSE  [m]     {s["rmse"]:.3f}   <- SCORE (lower is better)',
             f'position median/p95    {s["median"]:.3f} / {s["p95"]:.3f}',
             f'lateral RMSE   [m]     {s["lat"]:.3f}',
             f'longitudinal RMSE [m]  {s["lon"]:.3f}',
             f'yaw RMSE     [deg]     {s["yaw_deg"]:.3f}',
             f'within 0.25 m          {s["within_025"]:.1f} %',
             f'within 0.50 m          {s["within_050"]:.1f} %',
             f'within 1.00 m          {s["within_100"]:.1f} %']
    if not s['ranked']:
        lines.append('NOT RANKED: coverage below threshold')
    return '\n'.join(lines)


def slices(al: dict, tel: pd.DataFrame, block: int = 200) -> dict:
    """Share of the squared error by tracker mode and by block of frames (what decides the score)."""
    e2, lo2, la2 = al['err'] ** 2, al['lon'] ** 2, al['lat'] ** 2
    tot = float(e2.sum()) or 1.0
    frame = al['rows']

    def row(mask, label):
        return dict(label=label, frames=int(mask.sum()), pct_frames=float(100 * mask.mean()),
                    rmse=float(np.sqrt(e2[mask].mean())), lon=float(np.sqrt(lo2[mask].mean())),
                    lat=float(np.sqrt(la2[mask].mean())), share_pct=float(100 * e2[mask].sum() / tot))

    out = dict(by_mode=[], by_block=[])
    if tel is not None and 'mode' in tel:
        mode = tel['mode'].values[np.minimum(frame, len(tel) - 1)]
        for m in sorted(set(mode)):
            out['by_mode'].append(row(mode == m, m))
    for b in range(0, int(frame.max()) + 1 if len(frame) else 0, block):
        mask = (frame >= b) & (frame < b + block)
        if mask.any():
            out['by_block'].append(row(mask, f'frames {b:5d}-{b + block - 1:5d}'))
    return out


def recovery(err: np.ndarray, kidnap_flags: np.ndarray, thr: float = 3.0, hold: int = 10, rate_hz: float = 10.0):
    """Seconds after each kidnap until the error stays < thr for `hold` frames (inf = never)."""
    n = min(len(err), len(kidnap_flags))
    ok = (err[:n] < thr).astype(float)
    held = np.convolve(ok, np.ones(hold), mode='full')[hold - 1:n + hold - 1] >= hold - 1e-9
    out = []
    for i in np.flatnonzero(kidnap_flags[:n] == 1):
        j = np.flatnonzero(held[i + 1:])
        out.append((j[0] + 1) / rate_hz if len(j) else float('inf'))
    return np.asarray(out, dtype=np.float64)


def db_coverage(gx, gy, gyaw, db_poses: np.ndarray, track=None, compat_deg: float = 30.0):
    """Distance from each GT pose to the database: any heading / compatible heading, density, off main track."""
    xy = np.stack([gx, gy], 1)
    tree = cKDTree(db_poses[:, :2])
    near_any, _ = tree.query(xy)
    N = len(xy)
    near_head, db_10m = np.full(N, np.inf), np.zeros(N, dtype=int)
    for i, c in enumerate(tree.query_ball_point(xy, r=60.0)):
        if not c:
            continue
        c = np.asarray(c)
        okh = np.abs(wrap(db_poses[c, 2] - gyaw[i])) <= math.radians(compat_deg)
        if okh.any():
            d = np.hypot(*(db_poses[c[okh], :2] - xy[i]).T)
            near_head[i], db_10m[i] = d.min(), int((d <= 10.0).sum())
    off_main, s = np.zeros(N, dtype=bool), np.full(N, np.nan)
    if track is not None:
        s, n, k = track.project(xy, wrap(gyaw))
        off_main = ~((n <= track.w_left[k] + 1.0) & (n >= -track.w_right[k] - 1.0))
    return dict(near_any=near_any, near_head=near_head, db_10m=db_10m, off_main=off_main, s=s)
