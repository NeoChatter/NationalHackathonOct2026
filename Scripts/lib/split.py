"""
Train / validation split (from the rmse-2.740 branch, Scripts/2_train_model.py: choose_val_block / make_split).

  block (default) : n whole laps starting at val_start_frac of the run, with a guard band of val_guard_s seconds on
                    each side that is neither trained on nor put into the place database. On the organisers' training
                    recording this gives frames 4777-6975 (t = 478-698 s), the window of the rmse-2.740 reference.
  tail            : first 80 % train, last 20 % validation, no guard (the protocol of rmse-4.363 and earlier tags).
  full-train      : every frame trains and maps; no validation (the submission model).

The checkpoint stores what later stages need to recover the split exactly: val_start / val_end timestamps and the
timestamps of every database (= training) frame.
"""
import numpy as np

from .common import wrap


def choose_val_block(poses, n_laps=2, start_frac=0.40, fallback_frac=0.16):
    """
    A contiguous block of `n_laps` full laps starting at `start_frac` of the recording. It ends the n-th time the car
    comes back to its frame-k0 spot (within 12 m and ~30 deg), so it is a whole number of laps. Falls back to a fixed
    fraction of the recording if the car does not come back often enough.
    Returns (first_frame, end_frame_exclusive, description).
    """
    n = len(poses)
    k0 = int(start_frac * n)
    xy, yaw = poses[:, :2], poses[:, 3]
    near = (np.hypot(xy[:, 0] - xy[k0, 0], xy[:, 1] - xy[k0, 1]) < 12.0) & (np.abs(wrap(yaw - yaw[k0])) < 0.5)
    idx = np.flatnonzero(near[k0:]) + k0
    starts = idx[np.r_[True, np.diff(idx) > 50]] if len(idx) else idx      # one entry per pass through the spot
    if len(starts) > n_laps:
        return k0, int(starts[n_laps]), f'{n_laps} full laps (the car returns to its frame-{k0} position {n_laps}x)'
    return k0, min(n, k0 + int(fallback_frac * n)), f'fallback: {fallback_frac:.0%} of the recording (laps not detected)'


def make_split(poses, timestamps_ns, full_train=False, val_mode='block', val_laps=2, val_start_frac=0.40,
               val_guard_s=5.0):
    """Returns (train_indices, val_indices, description, info dict for the checkpoint)."""
    n = len(timestamps_ns)
    info = dict(mode='full' if full_train else val_mode, val_laps=int(val_laps), val_start_frac=float(val_start_frac),
                val_guard_s=float(val_guard_s if val_mode == 'block' else 0.0))
    if full_train:
        return np.arange(n), np.arange(0), 'full-train: all frames used for training, no validation', info
    if val_mode == 'tail':
        split = int(0.80 * n)
        info.update(first_frame=split, last_frame=n - 1)
        return (np.arange(split), np.arange(split, n),
                'tail: first 80% train, last 20% validation (rmse-4.363 protocol, includes the unmapped pit stop)', info)
    if val_mode != 'block':
        raise ValueError(f'unknown val_mode {val_mode!r} (block | tail)')

    a, b, how = choose_val_block(poses, n_laps=val_laps, start_frac=val_start_frac)
    t = timestamps_ns * 1e-9
    guard = (t >= t[a] - val_guard_s) & (t <= t[b - 1] + val_guard_s)   # also keeps neighbours out of the map
    info.update(first_frame=int(a), last_frame=int(b - 1))
    return np.flatnonzero(~guard), np.arange(a, b), (
        f'block: frames {a}-{b - 1} (t = {t[a] - t[0]:.0f}-{t[b - 1] - t[0]:.0f} s, {t[b - 1] - t[a]:.0f} s long, {how}); '
        f'{int(guard.sum() - (b - a))} guard frames ({val_guard_s:g} s each side) excluded from training'), info


def indices_from_checkpoint(ck: dict, timestamps_ns: np.ndarray):
    """(train_indices, val_indices) of a cache for the split a checkpoint was trained with.
    Training frames = the database frames (db_timestamps_ns); validation = [val_start, val_end] (0 = none)."""
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    vs = int(ck.get('val_start_timestamp_ns', 0) or 0)
    ve = int(ck.get('val_end_timestamp_ns', 0) or 0)
    if ck.get('db_timestamps_ns') is not None:
        db_ts = np.asarray(ck['db_timestamps_ns'], dtype=np.int64)
        train = np.flatnonzero(np.isin(ts, db_ts))
    else:                                                   # checkpoint without db timestamps: tail-style split
        train = np.flatnonzero(ts < vs) if vs > 0 else np.arange(len(ts))
    if vs > 0:
        val = np.flatnonzero((ts >= vs) & ((ts <= ve) if ve > 0 else True))
    else:
        val = np.zeros(0, dtype=np.int64)
    return train, val
