import argparse
import csv
import math
import os
import random
import time
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision.models as models

# Natural size of a per-frame VO error in (dx [m], dy [m], dyaw [rad]). The early-fusion VO head predicts in these
# units, which keeps the three outputs (and their log-variances) equally well conditioned. The old head clamped
# log_var at -8, i.e. a yaw std of ~1 deg per frame, even though real yaw errors are ~0.2 deg.
DELTA_SCALE = (1.0, 0.1, 0.01)


# ----------------------------------------------------------------------------------------------------------
# Small helpers shared by training, evaluation and the other scripts
# ----------------------------------------------------------------------------------------------------------
def normalize_images(u8: np.ndarray) -> np.ndarray:
    """uint8 (..., H, W) -> float32, per-image zero mean / unit variance. Identical to the inference-time prep."""
    x = u8.astype(np.float32) / 255.0
    mean = x.mean(axis=(-2, -1), keepdims=True)
    std = x.std(axis=(-2, -1), keepdims=True)
    return (x - mean) / (std + 1e-5)


def wrap_np(a):
    return np.arctan2(np.sin(a), np.cos(a))


def gt_body_deltas(poses: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Ground-truth body-frame motion (dx, dy, dyaw) from frame idx-1 to frame idx. poses: [x, y, z, yaw]."""
    idx = np.asarray(idx, dtype=np.int64)
    x1, y1, yaw1 = poses[idx - 1, 0], poses[idx - 1, 1], poses[idx - 1, 3]
    x2, y2, yaw2 = poses[idx, 0], poses[idx, 1], poses[idx, 3]
    dxw, dyw = x2 - x1, y2 - y1
    c, s = np.cos(yaw1), np.sin(yaw1)
    return np.stack([c * dxw + s * dyw, -s * dxw + c * dyw, wrap_np(yaw2 - yaw1)], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------
# Appearance augmentation (night -> dusk / dawn / day), deliberately WITHOUT any geometric change
# ----------------------------------------------------------------------------------------------------------
def znorm(x: torch.Tensor) -> torch.Tensor:
    """(4, H, W) float -> per-camera zero mean / unit variance (same as normalize_images, in torch)."""
    mean = x.mean(dim=(1, 2), keepdim=True)
    std = ((x - mean) ** 2).mean(dim=(1, 2), keepdim=True).sqrt()
    return (x - mean) / (std + 1e-5)


class PhotometricAugment:
    """
    Changes only HOW the scene is lit, never WHERE things are: no shift, rotation, zoom or flip, because the pose
    is encoded in the image geometry (the camera mount is fixed) and moving pixels would corrupt the labels.

    The per-image zero-mean / unit-variance normalization already cancels a linear brightness / contrast change,
    so these are the NON-linear and spatially varying effects a dusk or dawn session adds on top of the night
    training data:
      gamma        nonlinear tone curve (dark regions lifted differently from bright ones)
      gain+clip    exposure change, bright areas saturate
      offset       raised black level / haze (the night sky is black, a dusk sky is not)
      sky glow     brighter band toward the top of the image (twilight sky, light falling off with height)
      shadow field smooth multiplicative unevenness (patches of sun / shade, lights on one side only)
      blur         lower sharpness (longer exposures, defocus)
    Per-camera jitter is added on top of the global values because each camera exposes on its own.
    The SAME draw is applied to the previous and the current frame, so the visual-odometry head still sees a
    photometrically consistent pair.

    strength scales every range (1.0 = defaults, 0 = off). p = probability a sample is augmented at all.
    """

    def __init__(self, strength=1.0, p=0.85):
        self.s = float(strength)
        self.p = float(p)

    def sample(self, n_cams=4):
        if self.s <= 0.0 or random.random() > self.p:
            return None
        s = self.s

        def log_uniform(lo, hi):
            return math.exp(random.uniform(s * math.log(lo), s * math.log(hi)))

        return dict(
            gamma=log_uniform(0.5, 2.2) * torch.exp(0.10 * s * torch.randn(n_cams, 1, 1)),
            gain=log_uniform(0.5, 3.0) * torch.exp(0.15 * s * torch.randn(n_cams, 1, 1)),
            offset=random.uniform(0.0, 0.20) * s if random.random() < 0.5 else 0.0,
            sky=random.uniform(0.0, 0.25) * s if random.random() < 0.5 else 0.0,
            field=(0.25 * s * torch.randn(n_cams, 1, 4, 8)) if random.random() < 0.5 else None,
            blur=random.choice([0, 0, 3, 5]),
        )

    @staticmethod
    def apply(x: torch.Tensor, prm) -> torch.Tensor:
        """x: (4, H, W) float in [0, 1]."""
        H, W = x.shape[-2:]
        x = x.clamp(0.0, 1.0) ** prm['gamma']
        x = x * prm['gain']
        if prm['field'] is not None:
            field = F.interpolate(prm['field'], size=(H, W), mode='bilinear', align_corners=False)
            x = x * torch.exp(field.squeeze(1))
        x = x + prm['offset']
        if prm['sky'] > 0.0:
            profile = torch.linspace(1.0, 0.0, H).view(1, H, 1) ** 1.5
            x = x + prm['sky'] * profile
        x = x.clamp(0.0, 1.0)
        k = prm['blur']
        if k > 1:
            x = F.avg_pool2d(x.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)[0]
        return x


# ----------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------
class RacingDataset(Dataset):
    """
    Consecutive multi-camera pairs (t-1, t) with GT body-frame motion and global pose labels.

    still_prob : probability that a sample is replaced by a synthetic STANDSTILL pair (prev == curr, target
                 motion exactly zero). The real recording has almost no stationary frames, so without this the
                 VO head never learns that "no image change" means "no motion" and reports ~9 m/s when parked.
    noise_std  : max std of Gaussian pixel noise added independently to both frames (in normalized units).
                 Makes the VO head robust to sensor / JPEG noise, which is all that differs between two frames
                 of a parked car.
    photo_aug  : PhotometricAugment or None (appearance changes only, see above).
    """

    def __init__(self, images, dt_offsets, poses, timestamps_ns, indices, still_prob=0.0, noise_std=0.0,
                 photo_aug=None):
        self.images = images
        self.dt_offsets = dt_offsets
        self.poses = poses
        self.timestamps_ns = timestamps_ns
        self.still_prob = float(still_prob)
        self.noise_std = float(noise_std)
        self.photo_aug = photo_aug

        # Valid indices `i` need `i - 1` IN THE SAME SPLIT (a held-out block must not leak into training pairs)
        # and dt in [0.01, 0.5] s.
        in_split = np.zeros(len(timestamps_ns), dtype=bool)
        in_split[np.asarray(indices, dtype=np.int64)] = True
        valid = []
        for idx in indices:
            if idx > 0 and in_split[idx - 1]:
                dt = (timestamps_ns[idx] - timestamps_ns[idx - 1]) * 1e-9
                if 0.01 <= dt <= 0.5:
                    valid.append(idx)
        self.valid_indices = np.asarray(valid, dtype=np.int64)
        self.delta_targets = (
            gt_body_deltas(poses, self.valid_indices) if len(valid) else np.zeros((0, 3), np.float32)
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, item):
        idx = int(self.valid_indices[item])
        img_curr = torch.from_numpy(self.images[idx]).float().div_(255.0)  # (4, H, W) in [0, 1]

        dt = float((self.timestamps_ns[idx] - self.timestamps_ns[idx - 1]) * 1e-9)
        time_feat = np.concatenate([[dt], self.dt_offsets[idx]]).astype(np.float32)  # (5,)

        if self.still_prob > 0.0 and random.random() < self.still_prob:
            img_prev = img_curr.clone()
            delta_target = np.zeros(3, dtype=np.float32)
        else:
            img_prev = torch.from_numpy(self.images[idx - 1]).float().div_(255.0)
            delta_target = self.delta_targets[item]

        # Appearance augmentation BEFORE normalization (normalization would cancel part of it), same draw for both
        # frames of the pair; independent draws for different samples, so an anchor and its same-place partner
        # (see LocationPairBatchSampler) usually look differently lit, which teaches lighting-invariant descriptors.
        if self.photo_aug is not None:
            prm = self.photo_aug.sample()
            if prm is not None:
                img_prev = self.photo_aug.apply(img_prev, prm)
                img_curr = self.photo_aug.apply(img_curr, prm)

        img_prev = znorm(img_prev)
        img_curr = znorm(img_curr)
        if self.noise_std > 0.0:
            sigma = random.uniform(0.0, self.noise_std)
            img_prev = img_prev + torch.randn_like(img_prev) * sigma
            img_curr = img_curr + torch.randn_like(img_curr) * sigma

        x2, y2, _, yaw2 = self.poses[idx]
        pose_target = np.array([x2, y2, math.sin(yaw2), math.cos(yaw2), yaw2], dtype=np.float32)

        return (
            img_prev,
            img_curr,
            torch.from_numpy(time_feat),
            torch.from_numpy(np.asarray(delta_target, dtype=np.float32)),
            torch.from_numpy(pose_target),
        )


class LocationPairBatchSampler(Sampler):
    """
    Every batch = B/2 random anchors + one partner per anchor taken from the SAME PLACE on a DIFFERENT pass
    (within `pos_radius` metres, `pos_yaw_deg` degrees, at least `min_gap_s` seconds apart).

    With plain random batches only a minority of anchors have any same-place partner, so the place-descriptor
    loss gets little of the signal it actually needs (resolving positions a few metres apart across laps).
    Anchors without any partner (e.g. a pit lane driven only once) get a random partner instead.
    """

    def __init__(self, xy, yaw, t_s, batch_size, pos_radius=3.0, pos_yaw_deg=10.0, min_gap_s=15.0, seed=0):
        assert batch_size % 2 == 0, 'batch size must be even for paired sampling'
        self.n = len(xy)
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.seed = seed
        self._epoch = 0

        cell = float(pos_radius)
        keys = np.floor(xy / cell).astype(np.int64)
        grid = defaultdict(list)
        for i, (kx, ky) in enumerate(keys):
            grid[(int(kx), int(ky))].append(i)

        yaw_tol = math.radians(pos_yaw_deg)
        self.positives = []
        for i in range(self.n):
            kx, ky = int(keys[i, 0]), int(keys[i, 1])
            cand = []
            for ax in (-1, 0, 1):
                for ay in (-1, 0, 1):
                    cand.extend(grid.get((kx + ax, ky + ay), ()))
            cand = np.asarray(cand, dtype=np.int64)
            dist = np.hypot(xy[cand, 0] - xy[i, 0], xy[cand, 1] - xy[i, 1])
            dyaw = np.abs(wrap_np(yaw[cand] - yaw[i]))
            gap = np.abs(t_s[cand] - t_s[i])
            ok = (dist <= pos_radius) & (dyaw <= yaw_tol) & (gap >= min_gap_s)
            self.positives.append(cand[ok])

        n_with = sum(1 for p in self.positives if len(p) > 0)
        print(f'[SAMPLER] {n_with}/{self.n} training items ({100.0 * n_with / max(self.n, 1):.1f}%) have a '
              f'same-place partner within {pos_radius:g} m / {pos_yaw_deg:g} deg on another pass')

    def __len__(self):
        return self.n // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        order = rng.permutation(self.n)
        for b in range(len(self)):
            anchors = order[b * self.half:(b + 1) * self.half]
            partners = []
            for i in anchors:
                pos = self.positives[int(i)]
                partners.append(int(pos[rng.integers(len(pos))]) if len(pos) else int(rng.integers(self.n)))
            yield [int(i) for i in anchors] + partners


# ----------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------
class PairMotionNet(nn.Module):
    """
    Early-fusion ego-motion estimator.

    The two frames are stacked on the channel axis ([prev, curr, gain*(curr-prev)]) BEFORE the first convolution,
    so the network sees pixel-level image motion directly. The old VO head only compared globally pooled
    descriptors of the two frames, which cannot measure displacement; it effectively memorized "what speed does
    the car have at this place", and fails anywhere it has not been (pit lane) or when parked.

    The final feature map is kept spatial (reduced to a few channels and flattened rather than averaged away),
    because WHERE in the image things move is what distinguishes forward motion, sideways motion and rotation.
    """

    def __init__(self, n_cams=4, pretrained=True, reduce_ch=32, pool_hw=(4, 8), time_dim=5, out_dim=6, diff_gain=3.0):
        super().__init__()
        self.diff_gain = diff_gain
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        old_conv = backbone.conv1
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            gray = old_conv.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
            conv1.weight.copy_(gray.repeat(1, 3, 1, 1) / 3.0)
        self.stem = nn.Sequential(conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.trunk = nn.Sequential(backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)
        self.reduce = nn.Sequential(nn.Conv2d(512, reduce_ch, kernel_size=1), nn.GELU(), nn.AdaptiveAvgPool2d(pool_hw))
        flat = n_cams * reduce_ch * pool_hw[0] * pool_hw[1]
        self.head = nn.Sequential(
            nn.Linear(flat + time_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, img_prev, img_curr, time_feat):
        B, C, H, W = img_curr.shape
        pair = torch.stack([img_prev, img_curr, self.diff_gain * (img_curr - img_prev)], dim=2)  # (B, C, 3, H, W)
        feat = self.reduce(self.trunk(self.stem(pair.reshape(B * C, 3, H, W))))
        feat = feat.reshape(B, -1)
        return self.head(torch.cat([feat, time_feat * 10.0], dim=-1))


class RacingLocNet(nn.Module):
    """
    Multi-Camera Neural Localizer & Visual Odometry Network:
    - Shared ResNet-18 feature extractor across the 4 cameras (fl, fr, cl, cr)
    - Learned Camera Attention Gate (down-weights motion-blurred side cameras at high speed)
    - Head 1 (Place Descriptor + Global Pose): 256-D L2-normalized embedding for GPU retrieval
    - Head 2 (Visual Ego-Motion + Uncertainty): (dx_body, dy_body, dyaw) + log_var from (t-1, t)

    vo_mode:
      'late'  : original head on pooled descriptors of both frames (kept so old checkpoints still load)
      'early' : PairMotionNet, early fusion of the two frames (default for new training)

    forward() always returns (embedding, aux_pose, delta_pred, log_var) with delta_pred in metres/radians and
    log_var = log of the PHYSICAL variance of delta_pred, in both modes.
    """

    def __init__(self, embed_dim: int = 256, vo_mode: str = 'late', pretrained: bool = True):
        super().__init__()
        assert vo_mode in ('late', 'early')
        self.vo_mode = vo_mode
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
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

        if vo_mode == 'late':
            # Visual Ego-Motion head: concat(feat_prev, feat_curr, feat_diff, time_feat)
            vo_in_dim = (512 * 4) * 3 + 5
            self.vo_head = nn.Sequential(
                nn.Linear(vo_in_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Linear(128, 6),  # [dx_body, dy_body, dyaw, log_var_x, log_var_y, log_var_yaw]
            )
            scale = torch.ones(3)
        else:
            self.motion = PairMotionNet(n_cams=4, pretrained=pretrained)
            scale = torch.tensor(DELTA_SCALE, dtype=torch.float32)
        # Not saved in the state dict, so old checkpoints stay loadable.
        self.register_buffer('delta_scale', scale, persistent=False)
        self.register_buffer('log_var_offset', 2.0 * torch.log(scale), persistent=False)

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
        feat_curr, emb_curr, aux_pose = self.forward_single(img_curr)

        if self.vo_mode == 'late':
            feat_prev = self.encode_multicam(img_prev)
            vo_out = self.vo_head(torch.cat([feat_prev, feat_curr, feat_curr - feat_prev, time_feat], dim=-1))
        else:
            vo_out = self.motion(img_prev, img_curr, time_feat)

        delta_pred = vo_out[:, :3] * self.delta_scale
        raw_log_var = torch.clamp(vo_out[:, 3:], min=-8.0, max=4.0)
        log_var = raw_log_var + self.log_var_offset
        return emb_curr, aux_pose, delta_pred, log_var


def model_config_from_checkpoint(ckpt: dict) -> dict:
    """Checkpoints written before this change have no config: they use the original 'late' VO head."""
    cfg = {'embed_dim': 256, 'vo_mode': 'late', 'q_scaling_fixed': False}
    cfg.update(ckpt.get('model_config', {}))
    return cfg


def build_model_from_checkpoint(ckpt: dict) -> 'RacingLocNet':
    """Construct the architecture a checkpoint was trained with (no ImageNet download) and load its weights."""
    cfg = model_config_from_checkpoint(ckpt)
    model = RacingLocNet(embed_dim=cfg['embed_dim'], vo_mode=cfg['vo_mode'], pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    return model


# ----------------------------------------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------------------------------------
def metric_contrastive_loss(emb: torch.Tensor, poses: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """
    Supervised Metric Contrastive Loss: pulls embeddings of frames from different laps
    at the same physical track location together, pushes distant frames apart.
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


def vo_loss(model, delta_pred, log_var, delta_gt, beta=0.5, rel_weight=1.0):
    """
    Heteroscedastic Gaussian NLL in normalized units, plus a RELATIVE forward-distance term.

    * NLL is computed on residuals divided by DELTA_SCALE with a proper (unweighted) variance, so exp(log_var) is a
      real variance and needs no [1, 2, 5] correction at inference time.
    * beta-NLL (Seitzer et al. 2022): weights each sample by var^beta (detached) so the network cannot "give up" on
      hard samples by inflating their variance, which is what drags predictions toward the mean speed.
    * The relative term |dx_pred - dx_gt| / (|dx_gt| + 0.5) makes a 0.3 m error at 1 m/step count much more than at
      4 m/step, countering the speed-dependent scale error (1.28 slow ... 0.80 fast) seen with the old head.
    """
    raw_lv = log_var - model.log_var_offset
    z = (delta_pred - delta_gt) / model.delta_scale
    nll = 0.5 * torch.exp(-raw_lv) * z ** 2 + 0.5 * raw_lv
    weight = torch.exp(raw_lv.detach()) ** beta
    loss_nll = (weight * nll).mean()
    rel = torch.abs(delta_pred[:, 0] - delta_gt[:, 0]) / (delta_gt[:, 0].abs() + 0.5)
    return loss_nll + rel_weight * rel.mean(), rel.mean()


# ----------------------------------------------------------------------------------------------------------
# Validation (held-out frames): VO accuracy by speed + retrieval quality, printed AND returned for the history log
# ----------------------------------------------------------------------------------------------------------
EVAL_KEYS = [
    'val_covered_frac', 'val_ret_median_m', 'val_ret_within2', 'val_ret_within5', 'val_sim_covered',
    'val_sim_uncovered', 'val_ret_median_aug_m', 'val_ret_within5_aug',
    'val_vo_rmse_dx', 'val_vo_rmse_dy', 'val_vo_rmse_dyaw_deg', 'val_vo_scale',
    'val_vo_scale_lt10', 'val_vo_scale_10_20', 'val_vo_scale_20_30', 'val_vo_scale_gt30', 'val_standstill_speed',
]
SPEED_BINS = [('lt10', 0.0, 10.0), ('10_20', 10.0, 20.0), ('20_30', 20.0, 30.0), ('gt30', 30.0, 1e9)]


@torch.no_grad()
def evaluate(model, images, dt_offsets, poses, timestamps_ns, train_indices, val_indices, device, ref_stride=2):
    """Returns a dict with every key in EVAL_KEYS (NaN when not measurable)."""
    model.eval()
    metrics = {k: float('nan') for k in EVAL_KEYS}

    def embed(idx):
        out = []
        for s in range(0, len(idx), 128):
            x = torch.from_numpy(normalize_images(images[idx[s:s + 128]])).to(device)
            out.append(model.forward_single(x)[1])
        return torch.cat(out, dim=0)

    def vo_predict(pair_idx):
        preds = []
        for s in range(0, len(pair_idx), 32):
            b = pair_idx[s:s + 32]
            dt_b = (timestamps_ns[b] - timestamps_ns[b - 1]) * 1e-9
            tf = np.concatenate([dt_b[:, None], dt_offsets[b]], axis=1).astype(np.float32)
            _, _, d, _ = model(
                torch.from_numpy(normalize_images(images[b - 1])).to(device),
                torch.from_numpy(normalize_images(images[b])).to(device),
                torch.from_numpy(tf).to(device),
            )
            preds.append(d.cpu().numpy())
        return np.concatenate(preds, axis=0).astype(np.float64)

    # ---- retrieval: held-out frames against the training map ---------------------------------------------
    ref_idx = train_indices[::ref_stride]
    ref_emb = embed(ref_idx)
    ref_xy = poses[ref_idx, :2]
    val_emb = embed(val_indices)
    top_sim, top_j = (val_emb @ ref_emb.T).max(dim=1)
    top_sim, top_j = top_sim.cpu().numpy(), top_j.cpu().numpy()
    ret_xy = ref_xy[top_j]
    val_xy = poses[val_indices, :2]
    err = np.hypot(ret_xy[:, 0] - val_xy[:, 0], ret_xy[:, 1] - val_xy[:, 1])
    d_map = torch.cdist(torch.from_numpy(val_xy), torch.from_numpy(poses[train_indices, :2])).min(dim=1).values.numpy()
    covered = d_map <= 5.0
    metrics['val_covered_frac'] = float(covered.mean())
    if covered.any():
        metrics.update(
            val_ret_median_m=float(np.median(err[covered])),
            val_ret_within2=float(np.mean(err[covered] <= 2)),
            val_ret_within5=float(np.mean(err[covered] <= 5)),
            val_sim_covered=float(np.median(top_sim[covered])),
        )
        if (~covered).any():
            metrics['val_sim_uncovered'] = float(np.median(top_sim[~covered]))
        print(f'    [VAL-RET] map covers {covered.mean():.1%} of val frames | on covered: median err '
              f'{metrics["val_ret_median_m"]:.2f} m, within 2 m {metrics["val_ret_within2"]:.1%}, '
              f'within 5 m {metrics["val_ret_within5"]:.1%} | top-1 sim median covered '
              f'{metrics["val_sim_covered"]:.3f}'
              + (f' vs uncovered {metrics["val_sim_uncovered"]:.3f}' if (~covered).any() else ''))

    # ---- lighting-shift probe: the SAME held-out frames, but with synthetic dusk-like lighting, against the
    # unchanged night map. This is the closest stand-in for the dusk / dawn test session we can measure.
    sub = np.arange(0, len(val_indices), 3)
    if covered[sub].any():
        aug = PhotometricAugment(strength=1.0, p=1.0)
        py_state, torch_state = random.getstate(), torch.get_rng_state()
        random.seed(1234)
        torch.manual_seed(1234)
        x_aug = [znorm(aug.apply(torch.from_numpy(images[val_indices[i]]).float().div_(255.0), aug.sample()))
                 for i in sub]
        random.setstate(py_state)
        torch.set_rng_state(torch_state)
        emb_aug = torch.cat([model.forward_single(torch.stack(x_aug[s:s + 128]).to(device))[1]
                             for s in range(0, len(x_aug), 128)], dim=0)
        j_aug = (emb_aug @ ref_emb.T).argmax(dim=1).cpu().numpy()
        ret_aug = ref_xy[j_aug]
        err_aug = np.hypot(ret_aug[:, 0] - val_xy[sub, 0], ret_aug[:, 1] - val_xy[sub, 1])
        cov_sub = covered[sub]
        metrics['val_ret_median_aug_m'] = float(np.median(err_aug[cov_sub]))
        metrics['val_ret_within5_aug'] = float(np.mean(err_aug[cov_sub] <= 5))
        print(f'    [VAL-RET-AUG] with synthetic dusk-like lighting on the query: median err '
              f'{metrics["val_ret_median_aug_m"]:.2f} m, within 5 m {metrics["val_ret_within5_aug"]:.1%} '
              f'(clean: {metrics["val_ret_median_m"]:.2f} m / {metrics["val_ret_within5"]:.1%})')

    # ---- VO ---------------------------------------------------------------------------------------------
    dt_all = (timestamps_ns[val_indices] - timestamps_ns[val_indices - 1]) * 1e-9
    pair_idx = val_indices[(val_indices > 0) & (dt_all >= 0.01) & (dt_all <= 0.5)]
    if len(pair_idx) == 0:
        model.train()
        return metrics
    pred = vo_predict(pair_idx)
    gt = gt_body_deltas(poses, pair_idx).astype(np.float64)
    dt = (timestamps_ns[pair_idx] - timestamps_ns[pair_idx - 1]) * 1e-9
    speed = np.abs(gt[:, 0]) / dt
    e = pred - gt
    metrics.update(
        val_vo_rmse_dx=float(np.sqrt((e[:, 0] ** 2).mean())),
        val_vo_rmse_dy=float(np.sqrt((e[:, 1] ** 2).mean())),
        val_vo_rmse_dyaw_deg=float(math.degrees(np.sqrt((e[:, 2] ** 2).mean()))),
        val_vo_scale=float(pred[:, 0].sum() / max(gt[:, 0].sum(), 1e-9)),
    )
    print(f'    [VAL-VO] step RMSE dx {metrics["val_vo_rmse_dx"]:.3f} m | dy {metrics["val_vo_rmse_dy"]:.3f} m | '
          f'dyaw {metrics["val_vo_rmse_dyaw_deg"]:.3f} deg | forward scale {metrics["val_vo_scale"]:.3f}')
    parts = []
    for name, lo, hi in SPEED_BINS:
        m = (speed >= lo) & (speed < hi)
        if m.sum() > 5 and gt[m, 0].sum() > 0:
            metrics[f'val_vo_scale_{name}'] = float(pred[m, 0].sum() / gt[m, 0].sum())
            parts.append(f'{lo:.0f}-{hi:.0f}: {metrics[f"val_vo_scale_{name}"]:.2f} (n={int(m.sum())})')
    print('    [VAL-VO] forward scale by GT speed [m/s] -> ' + ' | '.join(parts))
    still = speed < 0.3
    if still.sum() > 3:
        metrics['val_standstill_speed'] = float(np.mean(np.abs(pred[still, 0]) / dt[still]))
        print(f'    [VAL-VO] standstill probe: {int(still.sum())} frames with GT speed < 0.3 m/s -> mean predicted '
              f'speed {metrics["val_standstill_speed"]:.2f} m/s (should be ~0)')
    model.train()
    return metrics


def calibrate_still_threshold(images, poses, timestamps_ns, indices, max_moving=1500, max_parked=500, seed=0):
    """
    Measure, on the TRAINING data, how big the frame-to-frame picture change is when the car is parked (GT speed
    < 0.2 m/s) versus moving (> 1 m/s), using exactly the score the inference script uses (largest over the four
    cameras of the mean |difference| of the normalized images).

    Returns a standstill threshold, or 0.0 (= detector off) if the two cases are not clearly separable or the
    recording has too few parked frames. The threshold sits at 1.5x the 99th percentile of the parked scores and
    is only used when the 1st percentile of the moving scores is at least 2x that percentile, so a moving car
    is never mistaken for a parked one.
    """
    idx = indices[indices > 0]
    dt = (timestamps_ns[idx] - timestamps_ns[idx - 1]) * 1e-9
    ok = (dt >= 0.01) & (dt <= 0.5)
    speed = np.hypot(poses[idx, 0] - poses[idx - 1, 0], poses[idx, 1] - poses[idx - 1, 1]) / np.maximum(dt, 1e-3)
    parked = idx[ok & (speed < 0.2)]
    moving = idx[ok & (speed > 1.0)]
    if len(parked) < 10 or len(moving) < 10:
        print(f'[STILL] only {len(parked)} parked / {len(moving)} moving training pairs: standstill detector OFF')
        return 0.0
    rng = np.random.default_rng(seed)
    if len(parked) > max_parked:
        parked = rng.choice(parked, max_parked, replace=False)
    if len(moving) > max_moving:
        moving = rng.choice(moving, max_moving, replace=False)

    def score(i):
        a, b = normalize_images(images[i - 1]), normalize_images(images[i])
        return float(np.abs(b - a).mean(axis=(-2, -1)).max())

    parked_scores = np.array([score(int(i)) for i in parked])
    moving_scores = np.array([score(int(i)) for i in moving])
    p_hi, m_lo = float(np.percentile(parked_scores, 99)), float(np.percentile(moving_scores, 1))
    print(f'[STILL] parked pairs n={len(parked)}: 99th pct score {p_hi:.4f} | moving pairs n={len(moving)}: '
          f'1st pct score {m_lo:.4f}')
    if m_lo >= 2.0 * p_hi:
        thresh = 1.5 * p_hi
        print(f'[STILL] cleanly separable -> standstill threshold {thresh:.4f}')
        return thresh
    print('[STILL] parked and moving scores overlap: standstill detector OFF (the VO head handles standstill)')
    return 0.0


# ----------------------------------------------------------------------------------------------------------
# Train / validation split
# ----------------------------------------------------------------------------------------------------------
def choose_val_block(poses, n_laps=2, start_frac=0.40, fallback_frac=0.16):
    """
    A contiguous block of `n_laps` full laps starting at `start_frac` of the recording.

    The block starts where the car is at frame k0 and ends the n-th time it comes back to that same spot
    (within 12 m and ~30 deg of heading), so it is a whole number of laps. Falls back to a fixed fraction of the
    recording if the car does not come back often enough.
    Returns (first_frame, end_frame_exclusive, description).
    """
    n = len(poses)
    k0 = int(start_frac * n)
    xy, yaw = poses[:, :2], poses[:, 3]
    near = (np.hypot(xy[:, 0] - xy[k0, 0], xy[:, 1] - xy[k0, 1]) < 12.0) & (np.abs(wrap_np(yaw - yaw[k0])) < 0.5)
    idx = np.flatnonzero(near[k0:]) + k0
    starts = idx[np.r_[True, np.diff(idx) > 50]] if len(idx) else idx      # one entry per pass through the spot
    if len(starts) > n_laps:
        return k0, int(starts[n_laps]), f'{n_laps} full laps (the car returns to its frame-{k0} position {n_laps}x)'
    return k0, min(n, k0 + int(fallback_frac * n)), f'fallback: {fallback_frac:.0%} of the recording (laps not detected)'


def make_split(args, poses, timestamps_ns):
    """Returns (train_indices, val_indices, description)."""
    n = len(timestamps_ns)
    if args.full_train:
        return np.arange(n), np.arange(0), 'full-train: all frames used for training, no validation'
    if args.val_mode == 'tail':
        split = int(0.80 * n)
        return np.arange(split), np.arange(split, n), 'tail: first 80% train, last 20% validation (old protocol)'

    a, b, how = choose_val_block(poses, n_laps=args.val_laps, start_frac=args.val_start_frac)
    t = timestamps_ns * 1e-9
    guard = (t >= t[a] - args.val_guard_s) & (t <= t[b - 1] + args.val_guard_s)   # also keeps neighbours out of the map
    train_indices = np.flatnonzero(~guard)
    val_indices = np.arange(a, b)
    return train_indices, val_indices, (
        f'block: frames {a}-{b - 1} (t = {t[a] - t[0]:.0f}-{t[b - 1] - t[0]:.0f} s, {t[b - 1] - t[a]:.0f} s long, {how}); '
        f'{int(guard.sum() - len(val_indices))} guard frames ({args.val_guard_s:g} s each side) excluded from training')


HISTORY_FIELDS = ['epoch', 'lr', 'epoch_s', 'loss_vo', 'vo_rel_err', 'loss_contrastive', 'loss_aux'] + EVAL_KEYS


def write_history(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, float('nan')) for k in HISTORY_FIELDS})


# ----------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------
def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[TRAIN] Using device: {device}')

    with h5py.File(args.data_h5, 'r') as h5f:
        timestamps_ns = h5f['timestamps_ns'][:]
        images = h5f['images'][:]
        dt_offsets = h5f['dt_offsets'][:]
        poses = h5f['poses'][:]
        track_borders = h5f['track_borders'][:] if 'track_borders' in h5f else np.zeros((0, 3))
        causal_cache = bool(h5f.attrs.get('causal_sync', 0))
    if not causal_cache:
        print('[WARN] This cache was built before the causal-sync fix in 1_build_dataset.py. Its side-camera images '
              'can be stamped AFTER the anchor, unlike at inference time. Rebuild it for best results.')

    n_total = len(timestamps_ns)
    train_indices, val_indices, split_desc = make_split(args, poses, timestamps_ns)

    print(f'[SPLIT] Total frames: {n_total} | Train: {len(train_indices)} | Val: {len(val_indices)}')
    print(f'[SPLIT] {split_desc}')
    print(f'[CONFIG] vo_mode={args.vo_mode} pair_sampler={not args.no_pair_sampler} still_prob={args.still_prob:g} '
          f'noise_std={args.noise_std:g} photo_aug={args.photo_aug:g} (p={args.photo_aug_p:g}) '
          f'epochs={args.epochs} batch={args.batch_size} lr={args.lr:g}')

    # Coordinate normalization stats from training split
    xy_mean = torch.tensor(poses[train_indices, :2].mean(axis=0), dtype=torch.float32, device=device)
    xy_std = torch.tensor(poses[train_indices, :2].std(axis=0) + 1e-5, dtype=torch.float32, device=device)

    photo_aug = PhotometricAugment(strength=args.photo_aug, p=args.photo_aug_p) if args.photo_aug > 0 else None
    train_ds = RacingDataset(images, dt_offsets, poses, timestamps_ns, train_indices,
                             still_prob=args.still_prob, noise_std=args.noise_std, photo_aug=photo_aug)
    print(f'[DATA] {len(train_ds)} training pairs')
    loader_kwargs = dict(num_workers=args.workers, pin_memory=(device.type == 'cuda'),
                         persistent_workers=args.workers > 0)
    if args.no_pair_sampler:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    else:
        vi = train_ds.valid_indices
        sampler = LocationPairBatchSampler(
            poses[vi, :2], poses[vi, 3], timestamps_ns[vi] * 1e-9,
            batch_size=args.batch_size, seed=args.seed,
        )
        train_loader = DataLoader(train_ds, batch_sampler=sampler, **loader_kwargs)

    model = RacingLocNet(embed_dim=256, vo_mode=args.vo_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.checkpoint) or '.', exist_ok=True)
    history_path = os.path.splitext(args.checkpoint)[0] + '_history.csv'
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.perf_counter()
        loss_vo_sum = loss_rel_sum = loss_con_sum = loss_aux_sum = 0.0

        for img_prev, img_curr, time_feat, delta_gt, pose_gt in train_loader:
            img_prev = img_prev.to(device, non_blocking=True)
            img_curr = img_curr.to(device, non_blocking=True)
            time_feat = time_feat.to(device, non_blocking=True)
            delta_gt = delta_gt.to(device, non_blocking=True)
            pose_gt = pose_gt.to(device, non_blocking=True)

            optimizer.zero_grad()
            emb, aux_pose, delta_pred, log_var = model(img_prev, img_curr, time_feat)

            # 1. Visual odometry: beta-NLL in normalized units + relative forward-distance error
            loss_vo, rel_err = vo_loss(model, delta_pred, log_var, delta_gt)

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
            loss_rel_sum += rel_err.item()
            loss_con_sum += loss_con.item()
            loss_aux_sum += loss_aux.item()

        lr_now = optimizer.param_groups[0]['lr']
        scheduler.step()
        n_b = max(len(train_loader), 1)
        epoch_s = time.perf_counter() - t_ep
        print(
            f'Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_s:.1f}s) | '
            f'VO Loss: {loss_vo_sum/n_b:.4f} (rel dx err {loss_rel_sum/n_b:.3f}) | '
            f'Contrastive Loss: {loss_con_sum/n_b:.4f} | Aux Pose Loss: {loss_aux_sum/n_b:.4f}'
        )
        row = dict(epoch=epoch, lr=lr_now, epoch_s=epoch_s, loss_vo=loss_vo_sum / n_b, vo_rel_err=loss_rel_sum / n_b,
                   loss_contrastive=loss_con_sum / n_b, loss_aux=loss_aux_sum / n_b)

        if len(val_indices) > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            row.update(evaluate(model, images, dt_offsets, poses, timestamps_ns, train_indices, val_indices, device))
        history.append(row)
        write_history(history_path, history)       # rewritten every epoch, so a crashed run still leaves its curve

    # Build Neural Reference Map Database over all training frames
    print('[DB] Building GPU Neural Place Database from training laps...')
    model.eval()
    db_embeddings = []
    with torch.no_grad():
        for start in range(0, len(train_indices), 128):
            batch_idx = train_indices[start:start + 128]
            imgs_t = torch.from_numpy(normalize_images(images[batch_idx])).to(device)
            db_embeddings.append(model.forward_single(imgs_t)[1].cpu())
    db_embeddings = torch.cat(db_embeddings, dim=0)
    db_poses = poses[train_indices]  # (N_train, 4): [x, y, z, yaw]

    still_thresh = calibrate_still_threshold(images, poses, timestamps_ns, train_indices, seed=args.seed)

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'embed_dim': 256,
                'vo_mode': args.vo_mode,
                # New VO loss uses a proper variance: inference must NOT divide exp(log_var) by [1, 2, 5].
                'q_scaling_fixed': True,
                # Frame-change threshold below which the car is treated as parked (0 = detector off).
                'still_thresh': float(still_thresh),
            },
            'train_args': {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in vars(args).items()},
            'xy_mean': xy_mean.cpu(),
            'xy_std': xy_std.cpu(),
            'db_embeddings': db_embeddings,
            'db_poses': torch.from_numpy(db_poses[:, [0, 1, 3]]).float(),  # [x, y, yaw]
            'track_borders': torch.from_numpy(track_borders).float(),
            'val_start_timestamp_ns': int(timestamps_ns[val_indices[0]]) if len(val_indices) > 0 else 0,
            'val_end_timestamp_ns': int(timestamps_ns[val_indices[-1]]) if len(val_indices) > 0 else 0,
            # Capture time of every reference in db_embeddings / db_poses (the map is NOT a contiguous prefix any more).
            'db_timestamps_ns': torch.from_numpy(timestamps_ns[train_indices].astype(np.int64)),
        },
        args.checkpoint,
    )
    print(f'[DONE] Saved trained model + {len(db_embeddings)} neural map descriptors to {args.checkpoint}')
    print(f'[DONE] Per-epoch history: {history_path}   (plot it with: python tools/plot_training.py {history_path})')
    if len(val_indices) > 0:
        print(f'[INFO] Validation: timestamp_ns {timestamps_ns[val_indices[0]]} .. {timestamps_ns[val_indices[-1]]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-h5', default='data/training_cache.h5')
    parser.add_argument('--checkpoint', default='checkpoints/racing_model.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=1, help='measure held-out VO / retrieval metrics every N epochs')
    parser.add_argument('--val-mode', choices=['block', 'tail'], default='block',
                        help="'block' = hold out whole laps from the middle of the recording (default); "
                             "'tail' = old protocol, last 20%% (includes the unmapped pit stop)")
    parser.add_argument('--val-laps', type=int, default=2, help='block mode: number of laps held out')
    parser.add_argument('--val-start-frac', type=float, default=0.40,
                        help='block mode: where in the recording (0..1) the held-out laps begin')
    parser.add_argument('--val-guard-s', type=float, default=5.0,
                        help='block mode: seconds on each side of the block that are neither trained on nor mapped')
    parser.add_argument('--photo-aug', type=float, default=1.0,
                        help='strength of the appearance augmentation (gamma/exposure/haze/sky glow/shadows/blur); 0 = off')
    parser.add_argument('--photo-aug-p', type=float, default=0.85, help='probability a training sample is augmented')
    parser.add_argument('--vo-mode', choices=['early', 'late'], default='early',
                        help="'early' = early-fusion motion network (new); 'late' = original pooled-feature VO head")
    parser.add_argument('--still-prob', type=float, default=0.10,
                        help='fraction of samples replaced by a synthetic standstill pair (prev == curr, zero motion)')
    parser.add_argument('--noise-std', type=float, default=0.04, help='max std of pixel noise added to both frames')
    parser.add_argument('--no-pair-sampler', action='store_true',
                        help='use plain shuffled batches instead of same-place paired batches')
    parser.add_argument('--full-train', action='store_true', help='Train on 100%% of laps for final test submission')
    args = parser.parse_args()
    train(args)
