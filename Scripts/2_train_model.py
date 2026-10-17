import argparse
import math
import os
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models


class RacingDataset(Dataset):
    """Provides consecutive multi-camera pairs (t-1, t) and GT body-frame motion + global pose labels."""

    def __init__(self, images, dt_offsets, poses, timestamps_ns, indices):
        self.images = images
        self.dt_offsets = dt_offsets
        self.poses = poses
        self.timestamps_ns = timestamps_ns
        # Valid indices `i` must have `i - 1` in the same split with dt < 0.5s
        valid = []
        for idx in indices:
            if idx > 0:
                dt = (timestamps_ns[idx] - timestamps_ns[idx - 1]) * 1e-9
                if 0.01 <= dt <= 0.5:
                    valid.append(idx)
        self.valid_indices = np.asarray(valid, dtype=np.int64)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, item):
        idx = self.valid_indices[item]
        img_prev = self.images[idx - 1].astype(np.float32) / 255.0  # (4, H, W)
        img_curr = self.images[idx].astype(np.float32) / 255.0      # (4, H, W)

        # Per-camera zero-mean unit-variance normalization (makes model invariant to night glare/brightness)
        img_prev = (img_prev - img_prev.mean(axis=(1, 2), keepdims=True)) / (img_prev.std(axis=(1, 2), keepdims=True) + 1e-5)
        img_curr = (img_curr - img_curr.mean(axis=(1, 2), keepdims=True)) / (img_curr.std(axis=(1, 2), keepdims=True) + 1e-5)

        dt = float((self.timestamps_ns[idx] - self.timestamps_ns[idx - 1]) * 1e-9)
        time_feat = np.concatenate([[dt], self.dt_offsets[idx]]).astype(np.float32)  # (5,)

        x1, y1, _, yaw1 = self.poses[idx - 1]
        x2, y2, _, yaw2 = self.poses[idx]

        # Compute exact ground-truth motion in base_link frame (dx_body, dy_body, dyaw)
        dx_w, dy_w = x2 - x1, y2 - y1
        c, s = math.cos(yaw1), math.sin(yaw1)
        dx_body = c * dx_w + s * dy_w
        dy_body = -s * dx_w + c * dy_w
        dyaw = math.atan2(math.sin(yaw2 - yaw1), math.cos(yaw2 - yaw1))

        delta_target = np.array([dx_body, dy_body, dyaw], dtype=np.float32)
        pose_target = np.array([x2, y2, math.sin(yaw2), math.cos(yaw2), yaw2], dtype=np.float32)

        return (
            torch.from_numpy(img_prev),
            torch.from_numpy(img_curr),
            torch.from_numpy(time_feat),
            torch.from_numpy(delta_target),
            torch.from_numpy(pose_target),
        )


class RacingLocNet(nn.Module):
    """
    Multi-Camera Neural Localizer & Visual Odometry Network:
    - Shared ResNet-18 feature extractor across the 4 cameras (fl, fr, cl, cr)
    - Learned Camera Attention Gate (down-weights motion-blurred side cameras at high speed)
    - Head 1 (Place Descriptor + Global Pose): 256-D L2-normalized embedding for <0.2ms GPU retrieval
    - Head 2 (Visual Ego-Motion + Uncertainty): Predicts (dx_body, dy_body, dyaw) + log_var from (t-1, t)
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Adapt first conv layer to accept 1-channel normalized grayscale image per camera
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # Outputs (B*4, 512, 1, 1)

        self.cam_emb = nn.Parameter(torch.randn(1, 4, 512) * 0.02)
        self.attn_gate = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))

        # Global Place Descriptor Head
        self.place_head = nn.Sequential(
            nn.Linear(512 * 4, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )
        # Auxiliary direct pose head (normalized coordinates) to structure the embedding space
        self.aux_pose_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 4),  # (x_norm, y_norm, sin_yaw, cos_yaw)
        )

        # Visual Ego-Motion (Odometry) Head: takes concat(feat_prev, feat_curr, feat_diff, time_feat)
        vo_in_dim = (512 * 4) * 3 + 5
        self.vo_head = nn.Sequential(
            nn.Linear(vo_in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 6),  # [dx_body, dy_body, dyaw, log_var_x, log_var_y, log_var_yaw]
        )

    def encode_multicam(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, H, W)
        B, C, H, W = x.shape
        feat = self.encoder(x.view(B * C, 1, H, W)).view(B, C, 512)
        feat = feat + self.cam_emb
        # Learned attention weights across the 4 cameras
        weights = torch.sigmoid(self.attn_gate(feat))  # (B, 4, 1)
        feat_weighted = (feat * weights).view(B, C * 512)
        return feat_weighted

    def forward_single(self, img_curr: torch.Tensor):
        feat_curr = self.encode_multicam(img_curr)
        raw_emb = self.place_head(feat_curr)
        emb = F.normalize(raw_emb, p=2, dim=-1)
        aux_pose = self.aux_pose_head(emb)
        return feat_curr, emb, aux_pose

    def forward(self, img_prev: torch.Tensor, img_curr: torch.Tensor, time_feat: torch.Tensor):
        feat_prev = self.encode_multicam(img_prev)
        feat_curr, emb_curr, aux_pose = self.forward_single(img_curr)

        vo_input = torch.cat([feat_prev, feat_curr, feat_curr - feat_prev, time_feat], dim=-1)
        vo_out = self.vo_head(vo_input)
        delta_pred = vo_out[:, :3]
        log_var = torch.clamp(vo_out[:, 3:], min=-8.0, max=4.0)
        return emb_curr, aux_pose, delta_pred, log_var


def metric_contrastive_loss(emb: torch.Tensor, poses: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """
    Supervised Metric Contrastive Loss: pulls embeddings of frames from different laps
    at the same physical track location (< 6m & < 20 deg) together, pushes distant frames apart.
    """
    xy = poses[:, :2]
    yaw = poses[:, 4:5]
    dist_xy = torch.cdist(xy, xy, p=2)
    yaw_diff = torch.abs(torch.atan2(torch.sin(yaw - yaw.T), torch.cos(yaw - yaw.T)))

    # Target similarity kernel exp(-d / sigma)
    spatial_sim = torch.exp(-dist_xy / 8.0) * torch.exp(-yaw_diff / 0.35)
    logits = (emb @ emb.T) / temp

    # Mask out self-diagonal
    mask = ~torch.eye(len(emb), dtype=torch.bool, device=emb.device)
    logits = logits[mask].view(len(emb), -1)
    target_probs = F.softmax(spatial_sim[mask].view(len(emb), -1) * 5.0, dim=-1)

    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[TRAIN] Using device: {device}')

    with h5py.File(args.data_h5, 'r') as h5f:
        timestamps_ns = h5f['timestamps_ns'][:]
        images = h5f['images'][:]
        dt_offsets = h5f['dt_offsets'][:]
        poses = h5f['poses'][:]
        track_borders = h5f['track_borders'][:] if 'track_borders' in h5f else np.zeros((0, 3))

    n_total = len(timestamps_ns)
    split_idx = n_total if args.full_train else int(0.80 * n_total)
    train_indices = np.arange(0, split_idx)
    val_indices = np.arange(split_idx, n_total)

    print(f'[SPLIT] Total frames: {n_total} | Train: {len(train_indices)} (first 80%) | Val: {len(val_indices)} (last 20% = ~2 laps)')

    # Coordinate normalization stats from training split
    xy_mean = torch.tensor(poses[train_indices, :2].mean(axis=0), dtype=torch.float32, device=device)
    xy_std = torch.tensor(poses[train_indices, :2].std(axis=0) + 1e-5, dtype=torch.float32, device=device)

    train_ds = RacingDataset(images, dt_offsets, poses, timestamps_ns, train_indices)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    model = RacingLocNet(embed_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.perf_counter()
        loss_vo_sum = loss_con_sum = loss_aux_sum = 0.0

        for img_prev, img_curr, time_feat, delta_gt, pose_gt in train_loader:
            img_prev = img_prev.to(device, non_blocking=True)
            img_curr = img_curr.to(device, non_blocking=True)
            time_feat = time_feat.to(device, non_blocking=True)
            delta_gt = delta_gt.to(device, non_blocking=True)
            pose_gt = pose_gt.to(device, non_blocking=True)

            optimizer.zero_grad()
            emb, aux_pose, delta_pred, log_var = model(img_prev, img_curr, time_feat)

            # 1. Heteroscedastic Visual Odometry Loss (weights yaw by 5x for heading accuracy)
            weights = torch.tensor([1.0, 2.0, 5.0], device=device)
            sq_err = (delta_pred - delta_gt) ** 2 * weights
            loss_vo = torch.mean(0.5 * torch.exp(-log_var) * sq_err + 0.5 * log_var)

            # 2. Supervised Metric Contrastive Place Loss
            loss_con = metric_contrastive_loss(emb, pose_gt)

            # 3. Auxiliary Global Pose Loss
            norm_xy = (pose_gt[:, :2] - xy_mean) / xy_std
            target_aux = torch.cat([norm_xy, pose_gt[:, 2:4]], dim=-1)
            loss_aux = F.smooth_l1_loss(aux_pose, target_aux)

            loss = loss_vo + 0.5 * loss_con + 0.5 * loss_aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            loss_vo_sum += loss_vo.item()
            loss_con_sum += loss_con.item()
            loss_aux_sum += loss_aux.item()

        scheduler.step()
        n_b = len(train_loader)
        print(
            f'Epoch [{epoch:02d}/{args.epochs:02d}] ({time.perf_counter()-t_ep:.1f}s) | '
            f'VO Loss: {loss_vo_sum/n_b:.4f} | Contrastive Loss: {loss_con_sum/n_b:.4f} | '
            f'Aux Pose Loss: {loss_aux_sum/n_b:.4f}'
        )

    # Build Neural Reference Map Database over all training frames
    print('[DB] Building GPU Neural Place Database from training laps...')
    model.eval()
    db_embeddings = []
    db_poses = poses[train_indices]  # (N_train, 4): [x, y, z, yaw]

    with torch.no_grad():
        for start in range(0, len(train_indices), 128):
            end = min(start + 128, len(train_indices))
            batch_idx = train_indices[start:end]
            imgs = images[batch_idx].astype(np.float32) / 255.0
            imgs = (imgs - imgs.mean(axis=(2, 3), keepdims=True)) / (imgs.std(axis=(2, 3), keepdims=True) + 1e-5)
            imgs_t = torch.from_numpy(imgs).to(device)
            _, emb, _ = model.forward_single(imgs_t)
            db_embeddings.append(emb.cpu())

    db_embeddings = torch.cat(db_embeddings, dim=0)

    os.makedirs(os.path.dirname(args.checkpoint) or '.', exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'xy_mean': xy_mean.cpu(),
            'xy_std': xy_std.cpu(),
            'db_embeddings': db_embeddings,
            'db_poses': torch.from_numpy(db_poses[:, [0, 1, 3]]).float(),  # [x, y, yaw]
            'track_borders': torch.from_numpy(track_borders).float(),
            'val_start_timestamp_ns': int(timestamps_ns[val_indices[0]]) if len(val_indices) > 0 else 0,
        },
        args.checkpoint,
    )
    print(f'[DONE] Saved trained model + {len(db_embeddings)} neural map descriptors to {args.checkpoint}')
    if len(val_indices) > 0:
        print(f'[INFO] Validation split starts at timestamp_ns = {timestamps_ns[val_indices[0]]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-h5', default='data/training_cache.h5')
    parser.add_argument('--checkpoint', default='checkpoints/racing_model.pt')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--full-train', action='store_true', help='Train on 100% of laps for final test submission')
    args = parser.parse_args()
    train(args)