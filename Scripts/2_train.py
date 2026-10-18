"""
STAGE 2 - train RacingLocNet and write a checkpoint that stages 3-6 can use directly.

  python Scripts/2_train.py                  # validation model: 2 whole laps held out -> [paths].val_checkpoint
  python Scripts/2_train.py --full-train     # submission model: every frame -> [paths].checkpoint

Recipe (rmse-2.740 branch): RacingLocNet (shared ResNet-18 over 4 cameras + camera gate -> 256-D place descriptor,
auxiliary pose head, early-fusion VO with its own uncertainty), soft spatial-kernel contrastive loss on paired batches
(every anchor gets a same-place partner from another pass), beta-NLL VO loss + relative forward-distance term,
PhotometricAugment, synthetic standstill pairs, pixel noise. Guarded 2-lap validation block (lib/split.py).
Added from the rmse-4.363 branch: CLAHE + robust normalisation (stored as ck['preproc']), structured lighting
effects, the measured retrieval calibration, the track map, the provenance chain.

Retrieval calibration ([calib].source): 'train_crosspass' measures the gated retrieval error of training frames
against the database with +-exclude_s around each query removed, so no validation lap is used and the full-train
model gets its own calibration. 'val' is the original measurement on the held-out laps (it leaks the scored laps).

Checkpoint: weights, model_config, place database (descriptors, [x, y, yaw], timestamps), track borders, split
(val_start / val_end timestamps), calib, track_map, preproc, augmentation settings, provenance. Saved weights-only
safe. Per-epoch held-out metrics go to <checkpoint>_history.csv (tools/plot_training.py). Next: 3_calibrate.py.
"""
import argparse
import csv
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # make `lib` importable from any CWD

from lib import config, provenance  # noqa: E402
from lib.augment import PhotometricAugment  # noqa: E402
from lib.data import (EVAL_KEYS, LocationPairBatchSampler, RacingDataset, embed_images, eval_retrieval,  # noqa: E402
                      evaluate_heldout)
from lib.models import RacingLocNet, pick_device, preproc_cfg, save_checkpoint, seed_everything  # noqa: E402
from lib.split import make_split  # noqa: E402
from lib.track_map import build_track_map, load_markers_h5  # noqa: E402


# ----------------------------------------------------------------------------------------------
# losses (V)
# ----------------------------------------------------------------------------------------------
def metric_contrastive_loss(emb: torch.Tensor, poses: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """Pulls embeddings of frames from different laps at the same physical place together, pushes distant ones apart
    (soft target from a spatial kernel exp(-d / 8 m) * exp(-dyaw / 0.35 rad))."""
    xy = poses[:, :2]
    yaw = poses[:, 4:5]
    dist_xy = torch.cdist(xy, xy, p=2)
    yaw_diff = torch.abs(torch.atan2(torch.sin(yaw - yaw.T), torch.cos(yaw - yaw.T)))
    spatial_sim = torch.exp(-dist_xy / 8.0) * torch.exp(-yaw_diff / 0.35)
    logits = (emb @ emb.T) / temp
    mask = ~torch.eye(len(emb), dtype=torch.bool, device=emb.device)
    logits = logits[mask].view(len(emb), -1)
    target_probs = F.softmax(spatial_sim[mask].view(len(emb), -1) * 5.0, dim=-1)
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def vo_loss(model, delta_pred, log_var, delta_gt, beta=0.5, rel_weight=1.0):
    """beta-NLL in normalised units (a proper variance, no [1, 2, 5] correction at inference) + a RELATIVE forward-
    distance term |dx_pred - dx_gt| / (|dx_gt| + 0.5) that counters the speed-dependent scale error."""
    raw_lv = log_var - model.log_var_offset
    z = (delta_pred - delta_gt) / model.delta_scale
    nll = 0.5 * torch.exp(-raw_lv) * z ** 2 + 0.5 * raw_lv
    loss_nll = (torch.exp(raw_lv.detach()) ** beta * nll).mean()
    rel = torch.abs(delta_pred[:, 0] - delta_gt[:, 0]) / (delta_gt[:, 0].abs() + 0.5)
    return loss_nll + rel_weight * rel.mean(), rel.mean()


HISTORY_FIELDS = ['epoch', 'lr', 'epoch_s', 'loss_vo', 'vo_rel_err', 'loss_contrastive', 'loss_aux'] + EVAL_KEYS


def write_history(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, float('nan')) for k in HISTORY_FIELDS})


# ----------------------------------------------------------------------------------------------
def train(args, cfg, record):
    seed_everything(args.seed, args.deterministic)
    device = pick_device(args.device)
    print(f'[TRAIN] device: {device} | seed {args.seed}')

    with h5py.File(args.data_h5, 'r') as h5f:
        timestamps_ns = h5f['timestamps_ns'][:]
        images = h5f['images'][:]
        dt_offsets = h5f['dt_offsets'][:]
        poses = h5f['poses'][:]
        track_borders = h5f['track_borders'][:] if 'track_borders' in h5f else np.zeros((0, 3))
        track_markers = load_markers_h5(h5f)
        causal_cache = bool(int(h5f.attrs.get('causal_sync', 0)))
        chain = [json.loads(h5f.attrs['provenance'])] if 'provenance' in h5f.attrs else []
    if not causal_cache:
        raise SystemExit('[ERROR] this cache was not built with the causal camera sync (attrs causal_sync != 1). '
                         'Rebuild it with Scripts/1_build_dataset.py: training on it would not match inference.')

    train_idx, val_idx, split_desc, split_info = make_split(
        poses, timestamps_ns, full_train=args.full_train, val_mode=args.val_mode, val_laps=args.val_laps,
        val_start_frac=args.val_start_frac, val_guard_s=args.val_guard_s)
    print(f'[SPLIT] total {len(timestamps_ns)} | train {len(train_idx)} | val {len(val_idx)}')
    print(f'[SPLIT] {split_desc}')

    preproc = preproc_cfg({'clahe': bool(args.clahe), 'norm': args.norm})
    augment = dict(photo_aug=float(args.photo_aug), photo_aug_p=float(args.photo_aug_p),
                   structured_p=float(args.structured_p))
    print(f'[CONFIG] preproc {preproc} | augment {augment} | vo_mode={args.vo_mode} pair_sampler={args.pair_sampler} '
          f'still_prob={args.still_prob:g} noise_std={args.noise_std:g} epochs={args.epochs} batch={args.batch_size} '
          f'lr={args.lr:g} pretrained={args.pretrained}')

    xy_mean = torch.tensor(poses[train_idx, :2].mean(axis=0), dtype=torch.float32, device=device)
    xy_std = torch.tensor(poses[train_idx, :2].std(axis=0) + 1e-5, dtype=torch.float32, device=device)

    photo_aug = PhotometricAugment(strength=args.photo_aug, p=args.photo_aug_p) if args.photo_aug > 0 else None
    train_ds = RacingDataset(images, dt_offsets, poses, timestamps_ns, train_idx, still_prob=args.still_prob,
                             noise_std=args.noise_std, photo_aug=photo_aug, structured_p=args.structured_p,
                             preproc=preproc)
    print(f'[DATA] {len(train_ds)} training pairs')
    loader_kwargs = dict(num_workers=args.workers, pin_memory=(device.type == 'cuda'),
                         persistent_workers=args.workers > 0)
    if args.pair_sampler:
        vi = train_ds.valid_indices
        sampler = LocationPairBatchSampler(poses[vi, :2], poses[vi, 3], timestamps_ns[vi] * 1e-9,
                                           batch_size=args.batch_size, pos_radius=args.pair_radius_m,
                                           pos_yaw_deg=args.pair_yaw_deg, min_gap_s=args.pair_min_gap_s,
                                           seed=args.seed, max_batches=args.max_batches)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  generator=torch.Generator().manual_seed(args.seed), **loader_kwargs)

    model = RacingLocNet(embed_dim=args.embed_dim, vo_mode=args.vo_mode, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.checkpoint) or '.', exist_ok=True)
    history_path = os.path.splitext(args.checkpoint)[0] + '_history.csv'
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.perf_counter()
        sums = dict(vo=0.0, rel=0.0, con=0.0, aux=0.0)
        n_b = 0
        for img_prev, img_curr, time_feat, delta_gt, pose_gt in train_loader:
            if args.max_batches and n_b >= args.max_batches:
                break
            img_prev = img_prev.to(device, non_blocking=True)
            img_curr = img_curr.to(device, non_blocking=True)
            time_feat = time_feat.to(device, non_blocking=True)
            delta_gt = delta_gt.to(device, non_blocking=True)
            pose_gt = pose_gt.to(device, non_blocking=True)

            optimizer.zero_grad()
            emb, aux_pose, delta_pred, log_var = model(img_prev, img_curr, time_feat)
            loss_vo, rel_err = vo_loss(model, delta_pred, log_var, delta_gt)
            loss_con = metric_contrastive_loss(emb, pose_gt)
            target_aux = torch.cat([(pose_gt[:, :2] - xy_mean) / xy_std, pose_gt[:, 2:4]], dim=-1)
            loss_aux = F.smooth_l1_loss(aux_pose, target_aux)
            loss = loss_vo + args.loss_w_contrastive * loss_con + args.loss_w_aux * loss_aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            sums['vo'] += loss_vo.item()
            sums['rel'] += rel_err.item()
            sums['con'] += loss_con.item()
            sums['aux'] += loss_aux.item()
            n_b += 1

        lr_now = optimizer.param_groups[0]['lr']
        scheduler.step()
        n_b = max(n_b, 1)
        epoch_s = time.perf_counter() - t_ep
        print(f'Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_s:.1f}s) | VO Loss: {sums["vo"] / n_b:.4f} '
              f'(rel dx err {sums["rel"] / n_b:.3f}) | Contrastive Loss: {sums["con"] / n_b:.4f} | '
              f'Aux Pose Loss: {sums["aux"] / n_b:.4f}')
        row = dict(epoch=epoch, lr=lr_now, epoch_s=epoch_s, loss_vo=sums['vo'] / n_b, vo_rel_err=sums['rel'] / n_b,
                   loss_contrastive=sums['con'] / n_b, loss_aux=sums['aux'] / n_b)
        if len(val_idx) > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            row.update(evaluate_heldout(model, images, dt_offsets, poses, timestamps_ns, train_idx, val_idx, device,
                                        preproc=preproc))
        history.append(row)
        write_history(history_path, history)       # rewritten every epoch, so a crashed run still leaves its curve

    # ---------------------------------------------------------------- place database
    print('[DB] embedding the training frames (place database)...')
    model.eval()
    db_emb = embed_images(model, images, train_idx, device, preproc=preproc)
    db_poses = poses[train_idx][:, [0, 1, 3]]

    # ---------------------------------------------------------------- retrieval calibration
    source = args.calib_source
    if source == 'val' and len(val_idx) == 0:
        print('[CALIB] no validation frames (full-train): using train_crosspass')
        source = 'train_crosspass'
    if source == 'val':
        rep, calib = eval_retrieval(model, images, poses, timestamps_ns, train_idx, val_idx, device, preproc,
                                    exclude_s=0.0, gate_m=args.calib_gate_m, db_emb=db_emb)
    else:
        q_idx = train_idx[::max(int(args.calib_query_stride), 1)]
        rep, calib = eval_retrieval(model, images, poses, timestamps_ns, train_idx, q_idx, device, preproc,
                                    exclude_s=args.calib_exclude_s, gate_m=args.calib_gate_m, db_emb=db_emb)
    calib['lat_clamp'] = 7.0
    print(f'[CALIB] source {source}: {rep}')
    print(f'[CALIB] {calib}')

    # ---------------------------------------------------------------- track map
    track_map = None
    try:
        tm = build_track_map(db_poses, track_markers, ds=args.track_ds, fallback_half_width=calib['lat_clamp'],
                             min_lap_len=args.min_lap_len_m)
        track_map = tm.to_dict()
        print(f'[TRACK] {tm.summary()}')
    except Exception as e:                                     # never lose a trained model over the track map
        print(f'[TRACK][WARN] track map not built ({e}); stage 3 rebuilds it')

    ck = {
        'format': 'combinedrelease-v1',
        'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'model_config': {'embed_dim': int(args.embed_dim), 'vo_mode': args.vo_mode, 'q_scaling_fixed': True},
        'train_args': {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in vars(args).items()},
        'xy_mean': xy_mean.cpu(), 'xy_std': xy_std.cpu(),
        'db_embeddings': db_emb.float(),
        'db_poses': torch.from_numpy(db_poses).float(),
        'db_timestamps_ns': torch.from_numpy(timestamps_ns[train_idx].astype(np.int64)),
        'track_borders': torch.from_numpy(np.asarray(track_borders, dtype=np.float32)),
        'val_start_timestamp_ns': int(timestamps_ns[val_idx[0]]) if len(val_idx) else 0,
        'val_end_timestamp_ns': int(timestamps_ns[val_idx[-1]]) if len(val_idx) else 0,
        'split': dict(split_info, description=split_desc, n_train=int(len(train_idx)), n_val=int(len(val_idx))),
        'calib': calib,
        'calib_info': dict(source=source, report=rep),
        'track_map': track_map,
        'preproc': preproc,
        'augment': augment,
        'provenance': chain,
    }
    provenance.append(ck, record)
    save_checkpoint(ck, args.checkpoint)
    print(f'[DONE] saved {args.checkpoint} ({len(db_emb)} database descriptors). Next: 3_calibrate.py')
    print(f'[DONE] per-epoch history: {history_path} (plot: python tools/plot_training.py {history_path})')
    if len(val_idx):
        print(f'[INFO] validation block: timestamp_ns {timestamps_ns[val_idx[0]]} .. {timestamps_ns[val_idx[-1]]} '
              f'(frames {val_idx[0]}-{val_idx[-1]})')


if __name__ == '__main__':
    cfg = config.bootstrap()
    provenance.configure(cfg)
    T, S, P, A, C = cfg['train'], cfg['split'], cfg['preproc'], cfg['augment'], cfg['calib']
    B = argparse.BooleanOptionalAction
    ap = config.parser(__doc__, cfg)
    ap.add_argument('--data-h5', default=cfg.path('cache'))
    ap.add_argument('--checkpoint', default=None,
                    help='output (default [paths].val_checkpoint, or [paths].checkpoint with --full-train)')
    ap.add_argument('--full-train', action='store_true', help='train / map on 100%% of the frames (submission model)')
    ap.add_argument('--seed', type=int, default=int(cfg.get('run', 'seed', 0)))
    ap.add_argument('--device', default=cfg.get('run', 'device', 'auto'))
    ap.add_argument('--deterministic', action=B, default=bool(cfg.get('run', 'deterministic', False)))
    ap.add_argument('--epochs', type=int, default=int(T.get('epochs', 20)))
    ap.add_argument('--batch-size', type=int, default=int(T.get('batch_size', 64)))
    ap.add_argument('--lr', type=float, default=float(T.get('lr', 3e-4)))
    ap.add_argument('--weight-decay', type=float, default=float(T.get('weight_decay', 1e-4)))
    ap.add_argument('--workers', type=int, default=int(T.get('workers', 4)))
    ap.add_argument('--eval-every', type=int, default=int(T.get('eval_every', 1)))
    ap.add_argument('--max-batches', type=int, default=int(T.get('max_batches_per_epoch', 0)),
                    help='cap batches per epoch (0 = full epochs; smoke tests only)')
    ap.add_argument('--pretrained', action=B, default=bool(T.get('pretrained', True)),
                    help='ImageNet ResNet-18 initialisation (downloads the weights once)')
    ap.add_argument('--embed-dim', type=int, default=int(T.get('embed_dim', 256)))
    ap.add_argument('--vo-mode', choices=['early', 'late'], default=T.get('vo_mode', 'early'))
    ap.add_argument('--still-prob', type=float, default=float(T.get('still_prob', 0.10)))
    ap.add_argument('--noise-std', type=float, default=float(T.get('noise_std', 0.04)))
    ap.add_argument('--pair-sampler', action=B, default=bool(T.get('pair_sampler', True)))
    ap.add_argument('--pair-radius-m', type=float, default=float(T.get('pair_radius_m', 3.0)))
    ap.add_argument('--pair-yaw-deg', type=float, default=float(T.get('pair_yaw_deg', 10.0)))
    ap.add_argument('--pair-min-gap-s', type=float, default=float(T.get('pair_min_gap_s', 15.0)))
    ap.add_argument('--loss-w-contrastive', type=float, default=float(T.get('loss_w_contrastive', 0.5)))
    ap.add_argument('--loss-w-aux', type=float, default=float(T.get('loss_w_aux', 0.5)))
    ap.add_argument('--val-mode', choices=['block', 'tail'], default=S.get('val_mode', 'block'))
    ap.add_argument('--val-laps', type=int, default=int(S.get('val_laps', 2)))
    ap.add_argument('--val-start-frac', type=float, default=float(S.get('val_start_frac', 0.40)))
    ap.add_argument('--val-guard-s', type=float, default=float(S.get('val_guard_s', 5.0)))
    ap.add_argument('--clahe', action=B, default=bool(P.get('clahe', True)))
    ap.add_argument('--norm', choices=['std', 'robust'], default=P.get('norm', 'robust'))
    ap.add_argument('--photo-aug', type=float, default=float(A.get('photo_aug', 1.0)))
    ap.add_argument('--photo-aug-p', type=float, default=float(A.get('photo_aug_p', 0.85)))
    ap.add_argument('--structured-p', type=float, default=float(A.get('structured_p', 0.5)))
    ap.add_argument('--calib-source', choices=['train_crosspass', 'val'], default=C.get('source', 'train_crosspass'))
    ap.add_argument('--calib-exclude-s', type=float, default=float(C.get('exclude_s', 20.0)))
    ap.add_argument('--calib-query-stride', type=int, default=int(C.get('query_stride', 5)))
    ap.add_argument('--calib-gate-m', type=float, default=float(C.get('gate_m', 35.0)))
    ap.add_argument('--track-ds', type=float, default=float(cfg.get('calibrate', 'ds', 1.0)))
    ap.add_argument('--min-lap-len-m', type=float, default=float(cfg.get('calibrate', 'min_lap_len_m', 500.0)))
    a = ap.parse_args()
    if a.batch_size % 2 and a.pair_sampler:
        ap.error('--batch-size must be even with the paired sampler')
    if a.checkpoint is None:
        a.checkpoint = cfg.path('checkpoint') if a.full_train else cfg.path('val_checkpoint')
    config.require_file(a.data_h5, 'cache', 'run Scripts/1_build_dataset.py first')
    rec = provenance.run_record('2_train', a, inputs={'data_h5': a.data_h5}, cfg=cfg)
    print(provenance.banner(rec))
    train(a, cfg, rec)
