"""
Networks, network-input preprocessing, checkpoint I/O and test-time adaptation.

Network (from the rmse-2.740 branch, Scripts/2_train_model.py): RacingLocNet
  - shared ResNet-18 over the 4 cameras + learned camera gate -> 256-D L2-normalised place descriptor
  - auxiliary pose head (structures the embedding during training only)
  - PairMotionNet: early-fusion visual odometry with its own uncertainty (fallback odometry)

Preprocessing (from the rmse-4.363 branch, lib/models.py) is a property of the CHECKPOINT (ck['preproc']), so
training, calibration and inference can never disagree. Checkpoints without an entry (e.g. the rmse-2.740
racing_model_val.pt) get {'clahe': False, 'norm': 'std'} = per-camera mean/std, which is exactly what that branch did.

  clahe : local contrast equalisation. Robust to global contrast changes (haze, overcast); keeps local structure.
  norm  : 'std'    per-camera mean / std
          'robust' per-camera median / MAD, clipped. A large bright or dark region (sky, shadow) dominates the mean /
                   std and squashes the road; the median / MAD are insensitive to it.

Checkpoints are saved "weights-only safe" (tensors, numbers, strings, lists, dicts only: to_safe) and always loaded
with torch.load(weights_only=True), so loading a checkpoint never executes pickled code.
"""
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .common import IMG_H, IMG_W  # noqa: F401  (re-exported for the scripts)

# Natural size of a per-frame VO error in (dx [m], dy [m], dyaw [rad]); the VO head predicts in these units.
DELTA_SCALE = (1.0, 0.1, 0.01)
DEFAULT_PREPROC = dict(clahe=False, norm='std')


def preproc_cfg(preproc=None) -> dict:
    return {**DEFAULT_PREPROC, **(preproc or {})}


# ----------------------------------------------------------------------------------------------
# preprocessing (S)
# ----------------------------------------------------------------------------------------------
def normalize_cams(x: np.ndarray, mode: str = 'std') -> np.ndarray:
    """x: float32 in [0,1], shape (..., 4, H, W). Per-camera normalisation."""
    if mode == 'robust':
        med = np.median(x, axis=(-2, -1), keepdims=True)
        mad = np.median(np.abs(x - med), axis=(-2, -1), keepdims=True) * 1.4826
        return np.clip((x - med) / (mad + 1e-3), -4.0, 4.0).astype(np.float32)
    m = x.mean(axis=(-2, -1), keepdims=True)
    s = x.std(axis=(-2, -1), keepdims=True)
    return ((x - m) / (s + 1e-5)).astype(np.float32)


_CLAHE = None


def _clahe():
    global _CLAHE
    if _CLAHE is None:                                        # created lazily: one per process / DataLoader worker
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8))
    return _CLAHE


def clahe_u8(x: np.ndarray) -> np.ndarray:
    """Local contrast equalisation of every (H,W) image in a (..., H, W) uint8 array."""
    flat = x.reshape(-1, x.shape[-2], x.shape[-1])
    out = np.empty_like(flat)
    c = _clahe()
    for i in range(len(flat)):
        out[i] = c.apply(np.ascontiguousarray(flat[i]))
    return out.reshape(x.shape)


def preprocess(x: np.ndarray, preproc=None) -> np.ndarray:
    """(..., 4, H, W) uint8, or float in [0,1] (after augmentation) -> float32 network input."""
    cfg = preproc_cfg(preproc)
    if cfg['clahe']:
        u8 = x if x.dtype == np.uint8 else (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        x = clahe_u8(u8).astype(np.float32) / 255.0
    elif x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    return normalize_cams(np.asarray(x, dtype=np.float32), cfg['norm'])


# ----------------------------------------------------------------------------------------------
# networks (V)
# ----------------------------------------------------------------------------------------------
class PairMotionNet(nn.Module):
    """
    Early-fusion ego-motion estimator: the two frames are stacked on the channel axis ([prev, curr, gain*(curr-prev)])
    before the first convolution, so the network sees pixel-level image motion directly. The final feature map is kept
    spatial (reduced to a few channels and flattened), because WHERE in the image things move distinguishes forward
    motion, sideways motion and rotation.
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
        self.head = nn.Sequential(nn.Linear(flat + time_dim, 512), nn.LayerNorm(512), nn.GELU(),
                                  nn.Linear(512, 128), nn.GELU(), nn.Linear(128, out_dim))

    def forward(self, img_prev, img_curr, time_feat):
        B, C, H, W = img_curr.shape
        pair = torch.stack([img_prev, img_curr, self.diff_gain * (img_curr - img_prev)], dim=2)  # (B, C, 3, H, W)
        feat = self.reduce(self.trunk(self.stem(pair.reshape(B * C, 3, H, W))))
        return self.head(torch.cat([feat.reshape(B, -1), time_feat * 10.0], dim=-1))


class RacingLocNet(nn.Module):
    """
    Multi-camera place descriptor + visual odometry.
      forward_single(x)           -> (pooled features, embedding (B,256) L2-normalised, aux pose)
      forward(prev, curr, tfeat)  -> (embedding, aux pose, delta (dx, dy, dyaw) [m, m, rad], log_var (physical units))
    tfeat = [dt, age of each of the 4 camera images] in seconds (B,5).
    vo_mode 'early' = PairMotionNet (default); 'late' = the original pooled-feature VO head (old checkpoints load).
    """

    def __init__(self, embed_dim: int = 256, vo_mode: str = 'late', pretrained: bool = True):
        super().__init__()
        assert vo_mode in ('late', 'early')
        self.vo_mode = vo_mode
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # (B*4, 512, 1, 1)

        self.cam_emb = nn.Parameter(torch.randn(1, 4, 512) * 0.02)
        self.attn_gate = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))
        self.place_head = nn.Sequential(nn.Linear(512 * 4, 512), nn.BatchNorm1d(512), nn.GELU(),
                                        nn.Linear(512, embed_dim))
        self.aux_pose_head = nn.Sequential(nn.Linear(embed_dim, 128), nn.GELU(), nn.Linear(128, 4))

        if vo_mode == 'late':
            self.vo_head = nn.Sequential(nn.Linear((512 * 4) * 3 + 5, 512), nn.LayerNorm(512), nn.GELU(),
                                         nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 6))
            scale = torch.ones(3)
        else:
            self.motion = PairMotionNet(n_cams=4, pretrained=pretrained)
            scale = torch.tensor(DELTA_SCALE, dtype=torch.float32)
        self.register_buffer('delta_scale', scale, persistent=False)
        self.register_buffer('log_var_offset', 2.0 * torch.log(scale), persistent=False)

    def encode_multicam(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.encoder(x.reshape(B * C, 1, H, W)).reshape(B, C, 512) + self.cam_emb
        weights = torch.sigmoid(self.attn_gate(feat))           # (B, 4, 1)
        return (feat * weights).reshape(B, C * 512)

    def forward_single(self, img_curr: torch.Tensor):
        feat_curr = self.encode_multicam(img_curr)
        emb = F.normalize(self.place_head(feat_curr), p=2, dim=-1)
        return feat_curr, emb, self.aux_pose_head(emb)

    def forward(self, img_prev: torch.Tensor, img_curr: torch.Tensor, time_feat: torch.Tensor):
        feat_curr, emb_curr, aux_pose = self.forward_single(img_curr)
        if self.vo_mode == 'late':
            feat_prev = self.encode_multicam(img_prev)
            vo_out = self.vo_head(torch.cat([feat_prev, feat_curr, feat_curr - feat_prev, time_feat], dim=-1))
        else:
            vo_out = self.motion(img_prev, img_curr, time_feat)
        delta_pred = vo_out[:, :3] * self.delta_scale
        log_var = torch.clamp(vo_out[:, 3:], min=-8.0, max=4.0) + self.log_var_offset
        return emb_curr, aux_pose, delta_pred, log_var


def model_config_from_checkpoint(ck: dict) -> dict:
    cfg = {'embed_dim': 256, 'vo_mode': 'late'}
    cfg.update(ck.get('model_config', {}) or {})
    return cfg


def build_model_from_checkpoint(ck: dict, device) -> RacingLocNet:
    """The architecture a checkpoint was trained with (no ImageNet download), weights loaded, in eval mode."""
    cfg = model_config_from_checkpoint(ck)
    model = RacingLocNet(embed_dim=int(cfg['embed_dim']), vo_mode=cfg['vo_mode'], pretrained=False)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    return model.to(device).eval()


# ----------------------------------------------------------------------------------------------
# checkpoint I/O, seeds, device
# ----------------------------------------------------------------------------------------------
def to_safe(obj):
    """numpy arrays -> tensors, numpy scalars -> Python numbers (recursively): loadable with weights_only=True."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(obj))
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {(k if isinstance(k, (str, int)) else str(k)): to_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_safe(v) for v in obj]
    return obj


def save_checkpoint(ck: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(to_safe(ck), path)


def load_checkpoint(path: str, map_location='cpu') -> dict:
    if not path or not os.path.exists(path):
        raise SystemExit(f'[ERROR] checkpoint not found: {path!r}')
    return torch.load(path, map_location=map_location, weights_only=True)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(name: str = 'auto') -> torch.device:
    if name == 'cpu' or (name == 'auto' and not torch.cuda.is_available()):
        return torch.device('cpu')
    if name in ('auto', 'cuda') and not torch.cuda.is_available():
        raise SystemExit('[ERROR] device "cuda" requested but torch.cuda.is_available() is False')
    return torch.device('cuda' if name == 'auto' else name)


# ----------------------------------------------------------------------------------------------
# test-time adaptation (optional, [infer].adabn, off by default)
# ----------------------------------------------------------------------------------------------
def enable_test_time_bn(model: nn.Module, momentum: float = 0.01):
    """
    AdaBN-style causal adaptation: every BatchNorm2d keeps eval-mode behaviour but its running mean / variance follow
    the incoming frames (exponential average updated just before the current frame is normalised, so only past and
    current images are used). Experimental and unverified (Answers Q11): keep it off for submissions.
    """
    handles = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            def hook(mod, inp, mom=momentum):
                x = inp[0]
                with torch.no_grad():
                    mod.running_mean.mul_(1.0 - mom).add_(mom * x.mean(dim=(0, 2, 3)))
                    mod.running_var.mul_(1.0 - mom).add_(mom * x.var(dim=(0, 2, 3), unbiased=False))
            handles.append(m.register_forward_pre_hook(hook))
    return handles
