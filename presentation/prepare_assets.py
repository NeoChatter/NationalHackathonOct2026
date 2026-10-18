#!/usr/bin/env python3
"""
prepare_assets.py -- extract the two real photographs used in the jury deck from the project's own data.

Nothing here is synthetic: both images come from the provided recordings.

    python presentation/prepare_assets.py            # from the repository root

<<<<<<< HEAD
Writes to presentation/assets/ (input paths from config.toml, $RACING_CONFIG to use another config):
    night_fast.jpg   camera_fl, full resolution, training (night) session, car at ~39 m/s   ([paths].bag)
    day_sample.jpg   camera_fl, full resolution, daylight jury sample (illustration only)   ([paths].day_sample_bag)
    facts.json       numbers measured from the training cache (shown on slide 1)           ([paths].cache)
=======
Writes to presentation/assets/:
    night_fast.jpg   camera_fl, full resolution, training (night) session, car at ~39 m/s   (data/*.mcap)
    day_sample.jpg   camera_fl, full resolution, daylight jury sample                       (data/rosbag_*/*.mcap)
    facts.json       numbers measured from data/training_cache.h5 (shown on slide 1)
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510
"""
import io
import json
import os

import h5py
import numpy as np
from PIL import Image

<<<<<<< HEAD
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'Scripts'))
from lib import config  # noqa: E402  (paths come from config.toml: [paths].bag, day_sample_bag, cache)

_CFG = config.load()
OUT = os.path.join(ROOT, 'presentation', 'assets')
NIGHT_BAG = _CFG.path('bag')
DAY_BAG = _CFG.path('day_sample_bag')
CACHE = _CFG.path('cache')
=======
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'presentation', 'assets')
NIGHT_BAG = os.path.join(ROOT, 'data', 'c0_2026-10-13_22-53-42_camloc_0.mcap')
DAY_BAG = os.path.join(ROOT, 'data', 'rosbag_2026-10-13_16-08-46_cameras_lap',
                       'rosbag_2026-10-13_16-08-46_cameras_lap_0.mcap')
CACHE = os.path.join(ROOT, 'data', 'training_cache.h5')
>>>>>>> c1b8fc7e4529440db336b6be6f1d9b1054f58510
FL_TOPIC = '/constructor0/sensor/camera_fl/compressed_image'


def grab_frame(bag, start_ns=None, skip=0):
    """JPEG bytes of the first camera_fl frame at/after start_ns (log time), after skipping `skip` frames."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    with open(bag, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for n, (_, _, _, msg) in enumerate(
                reader.iter_decoded_messages(topics=[FL_TOPIC], start_time=start_ns)):
            if n >= skip:
                return bytes(msg.data)
    raise RuntimeError(f'no frame found in {bag}')


def save_jpeg(jpeg, path, width=1600):
    im = Image.open(io.BytesIO(jpeg)).convert('RGB')
    height = round(im.height * width / im.width)
    im.resize((width, height), Image.LANCZOS).save(path, quality=88)
    return list(im.size)


def main():
    os.makedirs(OUT, exist_ok=True)
    facts = {}

    with h5py.File(CACHE, 'r') as f:
        ts = f['timestamps_ns'][:]
        poses = f['poses'][:]

    dt = np.maximum(np.diff(ts) * 1e-9, 1e-3)
    speed = np.r_[np.nan, np.hypot(*np.diff(poses[:, :2], axis=0).T) / dt]
    speed[0] = speed[1]
    facts.update(
        n_frames=int(len(ts)),
        duration_s=float((ts[-1] - ts[0]) * 1e-9),
        speed_max=float(np.nanmax(speed)),
        speed_mean=float(np.nanmean(speed)),
    )

    # A fast moment (~38-40 m/s) for the night photo.
    candidates = np.flatnonzero((speed > 37.0) & (speed < 40.0))
    pick = int(candidates[len(candidates) // 3])
    facts['night_pick_speed'] = float(speed[pick])

    facts['night_size'] = save_jpeg(grab_frame(NIGHT_BAG, start_ns=int(ts[pick]) - 200_000_000),
                                    os.path.join(OUT, 'night_fast.jpg'))
    facts['day_size'] = save_jpeg(grab_frame(DAY_BAG, skip=330), os.path.join(OUT, 'day_sample.jpg'))

    with open(os.path.join(OUT, 'facts.json'), 'w') as fh:
        json.dump(facts, fh, indent=2)
    print(json.dumps(facts, indent=2))


if __name__ == '__main__':
    main()
