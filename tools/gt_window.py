#!/usr/bin/env python3
"""
Cut a ground-truth CSV down to the time span of a pose file, so the organisers' tools/evaluate.py can score a pose
file that covers only part of a recording (for example the held-out validation laps).

    python tools/extract_gt.py data/c0_2026-10-13_22-53-42_camloc_0.mcap results/gt.csv
    python tools/gt_window.py results/gt.csv results/poses/val_block_poses.csv results/gt_val_block.csv
    python tools/evaluate.py results/gt_val_block.csv results/poses/val_block_poses.csv

Why: tools/evaluate.py measures coverage over the whole ground-truth window. The held-out laps are 220 s of a
1194 s recording, so against the full ground truth it reports about 18 % coverage and "NOT RANKED". Against the cut
ground truth, coverage is measured over the laps that were predicted. Every error number is unchanged, because
evaluate.py interpolates the same ground-truth samples either way.

Keeps one ground-truth sample before the first pose and the first one at or after the last pose (evaluate.py needs
both for its interpolation). A full-run submission needs no cutting: score it against the full ground truth.
Standard library only; tools/evaluate.py itself is not modified.
"""
import bisect
import csv
import sys


def main(gt_path, pose_path, out_path):
    with open(pose_path, newline='') as f:
        stamps = [int(row['timestamp_ns']) for row in csv.DictReader(f)]
    if not stamps:
        sys.exit(f'no poses in {pose_path}')
    t_first, t_last = min(stamps), max(stamps)

    with open(gt_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        col = header.index('timestamp_ns')
        rows = sorted(reader, key=lambda r: int(r[col]))
    gt_stamps = [int(r[col]) for r in rows]

    lo = max(bisect.bisect_left(gt_stamps, t_first) - 1, 0)
    hi = min(bisect.bisect_left(gt_stamps, t_last), len(rows) - 1)
    if gt_stamps[lo] >= t_first or gt_stamps[hi] < t_last:
        print('[WARN] the ground truth does not fully cover the pose file; evaluate.py will skip the uncovered poses')

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows[lo:hi + 1])
    print(f'{hi - lo + 1} ground-truth poses ({(gt_stamps[hi] - gt_stamps[lo]) * 1e-9:.1f} s) -> {out_path}')
    print(f'score with: python tools/evaluate.py {out_path} {pose_path}')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('usage: python tools/gt_window.py <gt.csv> <poses.csv> <out_gt.csv>')
    main(*sys.argv[1:])
