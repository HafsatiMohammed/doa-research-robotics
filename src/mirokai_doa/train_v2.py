#!/usr/bin/env python3
"""
Train DoA models (multi-source aware).
- Multi-source soft targets for CE (mixture of wrapped Gaussians).
- SRP-aligned primary angle for vMF/Δθ heads.
- Boolean VAD mask; robust when batches have zero speech.
- Mixed precision (AMP), gradient accumulation & clipping.
- EMA; validate with EMA weights.
- Multi-source recall@{1,2,3} within {5°,10°,15°}.
"""
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import argparse
import yaml
import math
from pathlib import Path
from typing import Dict, Any, Tuple
import math 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # no GUI, no Tk
import matplotlib.pyplot as plt
import torch.cuda.amp as amp

# repo-local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from mirokai_doa.doa_model import DoAEstimator
from mirokai_doa.train_utils import (
    build_model, build_optimizer, build_scheduler, ema_init, ema_update,
    angular_metrics_single, angular_metrics_multi,
    save_checkpoint, load_checkpoint, make_run_dir,
    set_seed, compute_loss, get_model_outputs_for_metrics, create_dataloader_from_config
)

# -------------------------------
# Args / Config
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser("Train DoA models (v2, multi-source aware)")
    p.add_argument('--cfg', type=str, required=False, help='Path to config YAML')
    p.add_argument('--model', type=str, choices=['scat', 'film', 'retin', 'basic'], required=True)

    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--save-root', type=str, default='models')

    # common overrides
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--device', type=str)

    # precomputed features
    p.add_argument('--use-precomputed', action='store_true')
    p.add_argument('--precomputed-path', type=str, default='feature_cache')

    # training quality/perf knobs
    p.add_argument('--accum-steps', type=int, default=1, help='gradient accumulation steps')
    p.add_argument('--clip-norm', type=float, default=None, help='max grad norm')
    p.add_argument('--workers', type=int, default=None, help='override num_workers for loaders')

    return p.parse_args()


def load_config(args) -> Dict[str, Any]:
    cfg = {
        'epochs': 100,
        'batch_size': 128,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'mixed_precision': True,
        'ema_decay': 0.999,
        'patience': 12,
        'grad_clip': 5.0,
        'optimizer': {'name': 'adamw', 'lr': 1e-3, 'weight_decay': 1e-4, 'betas': [0.9, 0.98], 'eps': 1e-8},
        'scheduler': {'name': 'onecycle', 'max_lr': 1e-3, 'pct_start': 0.1},
        'features': {'K': 72, 'max_frames': 6, 'vad_threshold': 0.6, 'vad_gt_masking': True},
        'dataloader': {'num_workers': 4, 'pin_memory': True},
        'use_precomputed': False,
        'precomputed_path': None,
    }
    if args.cfg:
        with open(args.cfg, 'r') as f:
            user = yaml.safe_load(f)
        cfg = {**cfg, **user}

    # CLI overrides
    if args.epochs is not None: cfg['epochs'] = args.epochs
    if args.batch_size is not None: cfg['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg['optimizer']['lr'] = 1e-3
        cfg['scheduler']['max_lr'] = 1e-3
    if args.device is not None: cfg['device'] = args.device
    if args.use_precomputed: cfg['use_precomputed'] = True
    if args.precomputed_path is not None: cfg['precomputed_path'] = args.precomputed_path
    if args.workers is not None:
        cfg.setdefault('dataloader', {})['num_workers'] = int(args.workers)
    return cfg


# -------------------------------
# Loss curriculum helpers
# -------------------------------



def find_logits_layer(model: nn.Module):
    """Try to locate the final classification layer."""
    preferred = ('head', 'classifier', 'fc', 'out', 'proj', 'logits', 'output')
    for name in preferred:
        if hasattr(model, name):
            mod = getattr(model, name)
            if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                return name, mod
    last = None
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            last = (name, mod)
    return last  # may be None if there isn't one

def total_grad_norm(model: nn.Module):
    g2 = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g2 += p.grad.detach().pow(2).sum().item()
    return g2 ** 0.5

def debug_grads(model: nn.Module):
    name_mod = find_logits_layer(model)
    tot = total_grad_norm(model)
    print(f"[grads] total L2: {tot:.4g}")
    if name_mod is not None:
        name, mod = name_mod
        wgrad = getattr(mod, "weight").grad
        bgrad = getattr(mod, "bias").grad if getattr(mod, "bias", None) is not None else None
        print(f"[grads] {name}.weight L2: {0.0 if wgrad is None else wgrad.norm().item():.4g}")
        if bgrad is not None:
            print(f"[grads] {name}.bias   L2: {0.0 if bgrad is None else bgrad.norm().item():.4g}")
    else:
        print("[grads] could not find a final linear/conv layer")




#### Needs to be mooved

# ---------- Helpers ----------
def _ensure_BTK(x: torch.Tensor):
    if x.dim() == 2:   # [B,K]
        return x.unsqueeze(1), True
    elif x.dim() == 3: # [B,T,K]
        return x, False
    else:
        raise ValueError(f"Expected [B,K] or [B,T,K], got {tuple(x.shape)}")

def _ensure_vad(vad_mask: torch.Tensor, B: int, T: int, device):
    if vad_mask is None:
        return torch.ones(B, T, device=device, dtype=torch.bool)
    if vad_mask.dim() == 1:
        vad_mask = vad_mask.unsqueeze(1)
    return (vad_mask > 0.5).to(torch.bool)

def _bin_centers_deg(K: int, device=None, dtype=None):
    step = 360.0 / K
    return (torch.arange(K, device=device, dtype=dtype) + 0.5) * step

def _circ_dist_deg(a_deg: torch.Tensor, b_deg: torch.Tensor):
    diff = (a_deg.unsqueeze(-1) - b_deg.unsqueeze(-2)).abs()
    return torch.minimum(diff, 360.0 - diff)

# ---------- Utility: sigmoid + GT count ----------
@torch.no_grad()
def sigmoid_scores_and_gtcount(
    logits: torch.Tensor,    # [B,T,K] or [B,K]
    gt_multi: torch.Tensor,  # [B,T,K] or [B,K] in {0,1}
):
    scores, sT = _ensure_BTK(logits)
    gt, _ = _ensure_BTK(gt_multi)
    B,T,K = scores.shape
    scores = torch.sigmoid(scores)              # multi-label scores
    gt_count = (gt > 0.5).sum(dim=-1).to(torch.int64)  # [B,T]
    if sT:
        scores = scores.squeeze(1)  # [B,K]
        gt_count = gt_count.squeeze(1)  # [B]
    return scores, gt_count

# ---------- Accuracy with dynamic k ----------
@torch.no_grad()
def accuracy_at_thr(
    logits: torch.Tensor,      # [B,T,K] or [B,K] raw logits
    gt_multi: torch.Tensor,    # [B,T,K] or [B,K] in {0,1}
    K: int,                    # e.g., 72
    thr_deg: float = 10.0,
    vad_mask: torch.Tensor = None,   # [B,T] or [B]
    k_mode: str = "gt",        # "gt" | "pred" | "fixed"
    topk: int = 3,             # used if k_mode="fixed"
    k_cap: int = 3,            # upper bound for dynamic k
    prob_thresh: float = None, # optional floor for keeping preds
    mass_thresh: float = None, # optional cumulative prob target for k_mode="pred"
    decode: str = "softmax",   # "softmax" (recommended) or "sigmoid"
):
    """
    Returns:
      {
        'any_hit_acc': fraction of voiced frames where ANY chosen preds
                       are within thr_deg of ANY GT,
        'all_gt_acc' : fraction where EVERY GT has some chosen pred within thr_deg,
        'used_frames': # voiced frames with GT >=1
      }

    k selection per voiced frame:
      - k_mode="gt":   k = clamp(#GT, 1, k_cap)
      - k_mode="pred": choose k by prob threshold and/or cumulative mass, then clamp to [1, k_cap]
      - k_mode="fixed":k = topk (clamped to K)
    Notes:
      - With decode="softmax" we match the multi-peak CE training; with "sigmoid" we match BCE.
      - Bin centers at (k+0.5)*360/K. Frames with no GT are skipped.
    """
    logits_BTK, _ = _ensure_BTK(logits)
    gt_BTK, _     = _ensure_BTK(gt_multi)
    B, T, KK = logits_BTK.shape
    assert KK == K, f"K mismatch: logits={KK}, expected {K}"

    vad      = _ensure_vad(vad_mask, B, T, logits_BTK.device)
    centers  = _bin_centers_deg(K, device=logits_BTK.device, dtype=logits_BTK.dtype)
    gt_bool  = gt_BTK > 0.5                      # [B,T,K]
    gt_count = gt_bool.sum(dim=-1)               # [B,T]

    # Scores for filtering; order can be taken from logits (monotonic).
    if decode == "softmax":
        scores = F.softmax(logits_BTK, dim=-1)
    elif decode == "sigmoid":
        scores = logits_BTK
    else:
        raise ValueError("decode must be 'softmax' or 'sigmoid'.")

    # Sort indices by descending logit (same order as scores)
    sorted_idx = logits_BTK.argsort(dim=-1, descending=True)  # [B,T,K]

    used_frames = 0
    any_hit_ok  = 0
    all_gt_ok   = 0

    for b in range(B):
        for t in range(T):
            if not vad[b, t]:
                continue
            gt_idx = torch.nonzero(gt_bool[b, t], as_tuple=False).flatten()  # [G]
            G = int(gt_idx.numel())
            if G == 0:
                continue  # skip non-target frames

            # ----- decide k and indices to use -----
            if k_mode == "gt":
                k_use = max(1, min(int(gt_count[b, t].item()), k_cap))
                p_idx = sorted_idx[b, t, :k_use]

            elif k_mode == "pred":
                # start from scores sorted high->low
                ranked = sorted_idx[b, t]               # [K]
                ranked_scores = scores[b, t, ranked]    # [K]

                # optional prob floor
                if prob_thresh is not None:
                    mask = ranked_scores >= prob_thresh
                    ranked = ranked[mask]
                    ranked_scores = ranked_scores[mask]

                if ranked_scores.numel() == 0:
                    used_frames += 1
                    continue

                # optional cumulative mass target (good for softmax decoding)
                if mass_thresh is not None:
                    csum = torch.cumsum(ranked_scores, dim=0)
                    k_use = int((csum < mass_thresh).sum().item()) + 1
                else:
                    # fallback: number that pass prob floor (or at least 1)
                    k_use = ranked_scores.numel()

                k_use = max(1, min(k_use, k_cap))
                p_idx = ranked[:k_use]

            elif k_mode == "fixed":
                k_use = max(1, min(topk, K))
                p_idx = sorted_idx[b, t, :k_use]

                if prob_thresh is not None:
                    keep = scores[b, t, p_idx] >= prob_thresh
                    p_idx = p_idx[keep]
                    if p_idx.numel() == 0:
                        used_frames += 1
                        continue
            else:
                raise ValueError(f"Unknown k_mode: {k_mode}")

            # ----- measure distances -----
            pred_deg = centers[p_idx]                 # [k']
            gt_deg   = centers[gt_idx]                # [G]
            d        = _circ_dist_deg(pred_deg, gt_deg)  # [k', G]

            any_hit = (d <= thr_deg).any().item()
            dmin_gt = d.min(dim=0).values             # [G]
            all_hit = (dmin_gt <= thr_deg).all().item()

            used_frames += 1
            any_hit_ok  += int(any_hit)
            all_gt_ok   += int(all_hit)

    if used_frames == 0:
        return {'any_hit_acc': float('nan'), 'all_gt_acc': float('nan'), 'used_frames': 0}
    return {
        'any_hit_acc': any_hit_ok / used_frames,
        'all_gt_acc':  all_gt_ok / used_frames,
        'used_frames': used_frames,
    }

###
def quantize_az_deg_torch(az: torch.Tensor, K: int = 72) -> torch.Tensor:
    # az: shape (B,), degrees
    res = 360.0 / K
    # Round to nearest bin index and wrap into [0, K)
    idx = torch.remainder(torch.round(az / res).to(torch.long), K)
    return idx  # shape (B,), dtype long


def cosine_decay(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 0: return end
    t = min(max(step / total_steps, 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))

def get_loss_config_for_epoch(epoch: int, total_epochs: int, K: int) -> Dict[str, Any]:
    split = int(0.4 * total_epochs)  # first 40% coarse, then fine
    sigma_deg = 12.0 if epoch < split else 7.5
    w_srp0, w_srp_end = 0.6, 0.2
    w_srp = cosine_decay(w_srp0, w_srp_end, epoch, max(total_epochs - 1, 1))
    return {
        "K": int(K),
        "sigma_deg": float(sigma_deg),
        "w_ce": 1.0,
        "w_vmf": 0.25,
        "w_srp": float(w_srp),
        "w_quiet": 0.2,
        "w_delta": 0.3,
        "w_tv": 0.1,
        "tau_srp": 1.0,
        "eps_srp": 0.02,
        "max_delta_deg": 15.0,
    }


# -------------------------------
# Data helpers
# -------------------------------
def make_loader_from_cfg(cfg: Dict[str, Any], split: str) -> DataLoader:
    if cfg.get('use_precomputed', False):
        from mirokai_doa.feature_dataloaders import PrecomputedWindowDataset
        ds = PrecomputedWindowDataset(cfg.get('precomputed_path') or 'feature_cache', split)
        return DataLoader(
            ds,
            batch_size=cfg['batch_size'],
            shuffle=(split == 'train'),
            num_workers=cfg['dataloader'].get('num_workers', 4),
            pin_memory=cfg['dataloader'].get('pin_memory', True),
            drop_last=(split == 'train'),
            persistent_workers=(cfg['dataloader'].get('num_workers', 0) > 0),
        )
    else:
        return create_dataloader_from_config(cfg, split=split)

def _normalize_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return:
      features [B, 12, T, F]
      vad      [B, T, F]
      srp      [B, T, K]
      srp_vad  [B, T, K]
      gt       [B, T, K]  (multi-hot across K)
    """
    if isinstance(batch, dict):
        features = batch['features']
        vad = batch['vad']
        srp = batch['srp_phat']
        srp_vad = batch['srp_phat_vad']
        gt = batch['ground_truth']
    else:
        features, vad, srp, srp_vad, gt = batch[:5]

    # collapse accidental [B,1,...]
    if features.dim() == 5 and features.size(1) == 1:
        features = features.squeeze(1); vad = vad.squeeze(1)
        srp = srp.squeeze(1); srp_vad = srp_vad.squeeze(1); gt = gt.squeeze(1)
    return features, vad, srp, srp_vad, gt

def _boolean_vad(vad: torch.Tensor, threshold: float) -> torch.Tensor:
    """vad: [B,T,F] or [B,T] -> [B,T] bool"""
    if vad.dim() == 3:
        v = vad[..., 0]
    else:
        v = vad
    return (v >= threshold)

@torch.no_grad()
def _srp_aligned_primary_theta(gt_multi: torch.Tensor, srp: torch.Tensor, K: int) -> torch.Tensor:
    """
    Choose a single 'primary' GT angle per frame as: GT bin closest to SRP argmax.
    gt_multi: [B,T,K] 0/1 (multi-hot); srp: [B,T,K]
    return: theta_primary [B,T] in radians ([-pi, pi))
    """
    B,T,K_ = gt_multi.shape; assert K_ == K
    device = gt_multi.device
    srp_idx = srp.argmax(dim=-1)  # [B,T]
    gt_bins = (gt_multi > 0)
    primary_idx = srp_idx.clone()
    for b in range(B):
        for t in range(T):
            active = torch.nonzero(gt_bins[b,t], as_tuple=False).flatten()
            if active.numel() == 0:
                continue
            diff = (active - srp_idx[b,t]).abs()
            diff = torch.minimum(diff, torch.tensor(K, device=device) - diff)
            primary_idx[b,t] = active[diff.argmin()]
    theta = torch.remainder(primary_idx.float() * 2.0 * math.pi / K, 2.0 * math.pi)
    return theta



@torch.no_grad()
def _multi_hot_to_soft(gt_multi: torch.Tensor, K: int, sigma_deg: float) -> torch.Tensor:
    """
    Convert multi-hot GT [B,T,K] into a soft target via circular Gaussian smoothing.
    Implemented as circular convolution using FFT (efficient).
    """
    B,T,K_ = gt_multi.shape; assert K_ == K
    device = gt_multi.device
    sigma = sigma_deg * math.pi / 180.0

    idx = torch.arange(K, device=device).float()
    rel = ((idx + K//2) % K) - K//2
    rel = rel * (2*math.pi / K)
    g = torch.exp(-0.5 * (rel / sigma)**2)
    g = g / g.sum()

    y = gt_multi.float().reshape(-1, K)   # [B*T,K]
    Yf = torch.fft.rfft(y, dim=-1)
    Gf = torch.fft.rfft(g, dim=-1)
    conv = torch.fft.irfft(Yf * Gf, n=K, dim=-1)
    conv = conv.clamp_min(1e-12)
    conv = conv / conv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return conv.view(B, T, K)

def build_targets(gt: torch.Tensor,
                  vad: torch.Tensor,
                  srp: torch.Tensor,
                  K: int,
                  sigma_deg: float,
                  vad_threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    From GT multi-hot (and mixture-level VAD), produce:
      theta_primary [B,T], vad_mask [B,T] (bool), y_soft [B,T,K]
    """
    vad_mask = _boolean_vad(vad, vad_threshold)              # [B,T]
    #gt_masked = gt.float() * vad_mask.unsqueeze(-1).float()  # zero-out when unvoiced

    #y_soft = _multi_hot_to_soft(gt_masked, K=K, sigma_deg=sigma_deg)
    #theta_primary = _srp_aligned_primary_theta(gt_masked, srp, K)
    return vad_mask, gt#_masked


# -------------------------------
# Train / Validate
# -------------------------------
def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    # Wrap to (-pi, pi]
    return torch.atan2(torch.sin(a), torch.cos(a))

@torch.no_grad()
def _quick_train_metrics(logits: torch.Tensor,
                         theta_primary: torch.Tensor,  # [B,T] radians
                         vad_mask: torch.Tensor,       # [B,T] bool/0-1
                         K: int) -> Dict[str, float]:
    """
    Cheap single-angle metrics computed on the current batch:
      - mae_speech (deg)
      - within_10_speech
      - within_15_speech
    """
    from mirokai_doa.utils_model import decode_angle_from_logits  # local import to avoid circulars
    theta_hat, _ = decode_angle_from_logits(logits)               # [B,T] in radians
    diff = _wrap_angle(theta_primary - theta_hat).abs()
    ae_deg = diff * (180.0 / math.pi)

    m = vad_mask.bool()
    if m.any():
        ae_s = ae_deg[m]
        mae_speech = float(ae_s.mean().item())
        w10 = float((ae_s <= 10.0).float().mean().item())
        w15 = float((ae_s <= 15.0).float().mean().item())
    else:
        mae_speech = float('nan'); w10 = float('nan'); w15 = float('nan')

    return {
        "mae_speech": mae_speech,
        "within_10_speech": w10,
        "within_15_speech": w15,
    }

def count_prior(logits: torch.Tensor, y_max: torch.Tensor):
    p = torch.sigmoid(logits.float())
    pred_cnt = p.sum(1)
    true_cnt = y_max.to(torch.float32).sum(1)
    return (pred_cnt - true_cnt).pow(2).mean()

CE_LOSS = nn.CrossEntropyLoss()

def _circ_bin_dist(a: torch.Tensor, b: torch.Tensor, K: int):
    """Circular distance in bin units (0..K-1)."""
    d = (a - b).abs()
    return torch.minimum(d, K - d)

def _angular_error_deg_from_logits(logits: torch.Tensor, target_bin: torch.Tensor):
    """
    Compute absolute angular error (degrees) by converting logits to a circular-mean angle,
    and comparing to the center angle of the target bin.
    logits: [B, T, C]; target_bin: [B, T] long in [0..C-1]
    returns: |angular error| in degrees, shape [B, T]
    """
    B, T, C = logits.shape
    probs = logits.softmax(dim=-1)

    device = logits.device
    dtype = probs.dtype
    k = torch.arange(C, device=device, dtype=dtype)        # [C]
    ang = 2 * math.pi * k / C                              # bin centers in rad

    x = (probs * torch.cos(ang)).sum(dim=-1)               # [B, T]
    y = (probs * torch.sin(ang)).sum(dim=-1)               # [B, T]
    pred_rad = torch.atan2(y, x) % (2 * math.pi)
    pred_deg = pred_rad * 180.0 / math.pi                  # [B, T]

    bin_size = 360.0 / C
    true_center_deg = ((target_bin.float() + 0.5) * bin_size) % 360.0

    # minimal circular difference in [-180, 180]
    err = ((pred_deg - true_center_deg + 180.0) % 360.0) - 180.0
    return err.abs()                                       # [B, T]
def quantize_az_deg(az, K=72) -> int: 
    res = 360.0 / K 
    return int(np.round(az / res)) % K

def train_one_epoch(
    model, loader, optim, sched, scaler, ema_buf, cfg, device, model_name,
    accum_steps=1, clip_norm=None, epoch=0, viz_every=10, viz_dir="check_train_plots"
):
    """
    Batch from `_normalize_batch` must be: feats [B,C,T,F], vad, srp, srp_vad, gt (one-hot [B,T,C]).
    Model takes input [B,T,F,C] and returns logits [B,T,C].
    """
    model.train()

    use_amp = False #bool(cfg.get("mixed_precision", True))
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    loss_sum = 0.0

    # ---- accuracy accumulators (expanded)
    correct1_sum = 0
    correct5_sum = 0
    within1bin_sum = 0
    mae_deg_sum = 0.0
    mse_deg_sum = 0.0
    acc5deg_sum = 0
    acc10deg_sum = 0

    frame_count = 0

    out_dir = Path(viz_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_vad_mask = bool(cfg.get("metrics", {}).get("use_vad_mask", False))
    vad_threshold = float(cfg.get("features", {}).get("vad_threshold", 0.6))
    sigma_deg = float(cfg.get("loss", {}).get("sigma_deg", 10.0))

    pbar = tqdm(loader, desc=f"train [{epoch}]", leave=False)

    for step, batch in enumerate(pbar):
        # ---- unpack & move
        # (keep your original data prep — unchanged)
        feats, vad, srp, srp_vad, gt = _normalize_batch(batch)
        feats = feats.to(device)  # [B,C,T,F]

        # match train: [B,C,T,F] -> [B,T,F,C]
        feats = feats.permute(0, 2, 3, 1).contiguous()

        # ---- forward & loss
        # print(vad)
        # print(vad.shape)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(feats)                         # [B,T,C]
            B, T, C = logits.shape
            # one-hot -> indices [B,T] (leave your target construction unchanged)
            # print(gt)
            az_k = quantize_az_deg_torch(gt, C)
            target = az_k.unsqueeze(1).expand(B, T).to(device)   # (B, T) non-contiguous
            # print(target)
            # print(logits.shape)
            loss = CE_LOSS(logits.permute(0, 2, 1), target) 
            used_frames_batch = target.numel()

        # ---- backward (accumulation + AMP)
        loss_to_backprop = loss / accum_steps
        loss_to_backprop.backward()

        optim.step()
        optim.zero_grad(set_to_none=True)

        # ---- metrics (ACCURACY: replaced/expanded)
        with torch.no_grad():
            # top-1 (exact bin)
            preds1 = logits.argmax(dim=-1)  # [B,T]
            # print(preds1.shape)
            # print(target.shape)
            correct1 = (preds1 == target).sum().item()

            # top-5 hit
            top5_idx = torch.topk(logits, k=min(5, C), dim=-1).indices  # [B,T,5]
            hit5 = (top5_idx == target.unsqueeze(-1)).any(dim=-1).sum().item()

            # within-1-bin (±1 bin with wrap-around)
            d1 = _circ_bin_dist(preds1, target, C)
            within1 = (d1 <= 1).sum().item()

            # angular error via circular mean (deg)
            err_deg = _angular_error_deg_from_logits(logits, target)    # [B,T]
            mae_batch = err_deg.sum().item()
            mse_batch = (err_deg ** 2).sum().item()
            acc5deg_batch = (err_deg <= 5.0).sum().item()
            acc10deg_batch = (err_deg <= 10.0).sum().item()

            # accumulate
            loss_sum       += loss.item() * used_frames_batch
            correct1_sum   += correct1
            correct5_sum   += hit5
            within1bin_sum += within1
            mae_deg_sum    += mae_batch
            mse_deg_sum    += mse_batch
            acc5deg_sum    += acc5deg_batch
            acc10deg_sum   += acc10deg_batch
            frame_count    += used_frames_batch

            # live averages
            avg_loss   = loss_sum / max(1, frame_count)
            acc1       = correct1_sum   / max(1, frame_count)
            acc5       = correct5_sum   / max(1, frame_count)
            acc_w1bin  = within1bin_sum / max(1, frame_count)
            mae_deg    = mae_deg_sum    / max(1, frame_count)
            acc_5deg   = acc5deg_sum    / max(1, frame_count)
            acc_10deg  = acc10deg_sum   / max(1, frame_count)

        # ---- tqdm postfix (live averages) — show new accuracy metrics
        lr = optim.param_groups[0]["lr"]
        pbar.set_postfix({
            "loss":   f"{avg_loss:.4f}",
            "acc@1":  f"{acc1:.3f}",
            "acc@5":  f"{acc5:.3f}",
            "±1bin":  f"{acc_w1bin:.3f}",
            "MAE°":   f"{mae_deg:.2f}",
            "acc@5°": f"{acc_5deg:.3f}",
            "lr":     f"{lr:.2e}",
            "frames": frame_count,
        })

        # ---- visualization every `viz_every` steps (unchanged)

        if step % viz_every == 0:
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)  # [B,T,C]
                B, T, C = logits.shape

                # -- pick 2 random (b,t) samples
                n_show = min(2, B * T)
                rand_flat = torch.randperm(B * T, device=logits.device)[:n_show].tolist()
                samples = [(i // T, i % T) for i in rand_flat]

                def pick_indices(p_row, mass_thresh=0.95, k_cap=3):
                    order = torch.argsort(p_row, descending=True)
                    csum  = torch.cumsum(p_row[order], dim=0)
                    k_use = int((csum < mass_thresh).sum().item()) + 1
                    k_use = max(1, min(k_use, k_cap))
                    return order[:k_use]

                # -- helper: get GT azimuth (deg) at (b,t) for many possible shapes
                def _gt_deg_at(gt_tensor, b_idx, t_idx, B, T):
                    """
                    Supports gt with shapes:
                    [B,T], [B,T,1], [T], [B], [N] (flat), or scalar.
                    Returns a Python float (degrees).
                    """
                    x = gt_tensor
                    # common cases first
                    if x.ndim == 3 and x.size(-1) == 1:      # [B,T,1]
                        v = x[b_idx, t_idx, 0]
                    elif x.ndim == 2:                         # [B,T]
                        v = x[b_idx, t_idx]
                    elif x.ndim == 1:                         # [T] or [B] or flat
                        if x.shape[0] == T:                   # per-frame, single batch
                            v = x[t_idx]
                        elif x.shape[0] == B:                 # per-batch, single frame
                            v = x[b_idx]
                        else:                                 # fallback: flat vector
                            flat_idx = min(b_idx * T + t_idx, x.numel() - 1)
                            v = x[flat_idx]
                    else:                                     # scalar or odd shape → squeeze
                        v = x.squeeze()

                    return float(v.detach().cpu().item())

                ncols = len(samples)
                fig, axs = plt.subplots(1, ncols, figsize=(6 * ncols, 5), constrained_layout=True)
                if ncols == 1:
                    axs = [axs]

                for ax, (b_idx, t_idx) in zip(axs, samples):
                    p_row = probs[b_idx, t_idx].detach().cpu()  # [C]

                    # ---- GT azimuth (deg) -> GT bin
                    gt_deg_scalar = _gt_deg_at(gt, b_idx, t_idx, B, T)
                    gt_bin = int(quantize_az_deg(gt_deg_scalar, C)) % C

                    # ---- predicted peak(s)
                    pred_idx = pick_indices(p_row, mass_thresh=0.95, k_cap=1).cpu().numpy()

                    # ---- plot distribution + markers
                    ax.plot(p_row.numpy(), label="softmax(logits)[b,t,:]")
                    ax.scatter(pred_idx, p_row[pred_idx].numpy(),
                            marker="*", s=120, label="pred peak(s)")

                    ax.scatter([gt_bin], [p_row[gt_bin].item()],
                            marker="o", s=80, facecolors='none', edgecolors='C3',
                            label=f"GT bin ({gt_deg_scalar:.1f}°)")
                    ax.axvline(gt_bin, linestyle="--", alpha=0.25)
                    # print(p_row)
                    # print(p_row.shape)
                    # print(p_row.max())
                    ax.set_ylim(0.0, p_row.max())
                    ax.set_xlim(-0.5, C - 0.5)
                    ax.set_title(f"epoch {epoch} step {step} · b={b_idx}, t={t_idx}")
                    ax.grid(True)
                    ax.legend(loc="upper right")

                fig.suptitle(f"{model_name} · epoch {epoch} · step {step}")
                fig.savefig(out_dir / f"{epoch}_{step}.png", dpi=300)
                plt.close(fig)

    # ---- epoch averages (keep original return contract)
    epoch_loss = loss_sum / max(1, frame_count)
    epoch_acc1 = correct1_sum / max(1, frame_count)

    return epoch_loss, {"acc@1": epoch_acc1, "frames": frame_count}


@torch.no_grad()
def evaluate(model, loader, device, model_name, K=None, ema_buf=None, cfg=None):
    """
    Validation to mirror your train loop:
      - Input feats: [B,C,T,F]  → permute to [B,T,F,C]
      - Model output: logits [B,T,C]
      - Loss: Cross-Entropy on logits (no softmax), same as train
      - Metrics: acc_any@5 (top-5 contains GT), acc_all@5 (same for one-hot)
    """
    was_training = model.training
    model.eval()

    # Optionally swap in EMA weights if a shadow dict is provided
    orig_state = None
    # if ema_buf is not None and isinstance(ema_buf, dict):
    #     orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    #     # overlay EMA params where present
    #     model.load_state_dict({**orig_state, **ema_buf}, strict=False)

    sum_loss = 0.0
    n_steps  = 0

    # ---- accuracy accumulators (expanded)
    sum_used = 0
    sum_acc1 = 0
    sum_acc5 = 0
    sum_within1 = 0
    mae_deg_sum = 0.0
    mse_deg_sum = 0.0
    acc5deg_sum = 0
    acc10deg_sum = 0

    # (kept for backward-compat keys)
    sum_any, sum_all = 0.0, 0.0

    pbar = tqdm(loader, desc="val", leave=False)
    for batch in pbar:
        feats, vad, srp, srp_vad, gt = _normalize_batch(batch)

        feats = feats.to(device)  # [B,C,T,F]

        # match train: [B,C,T,F] -> [B,T,F,C]
        feats = feats.permute(0, 2, 3, 1).contiguous()

        # forward
        logits = model(feats)                         # [B,T,C]

        B, T, C = logits.shape
        az_k = quantize_az_deg_torch(gt, C)
        target = az_k.unsqueeze(1).expand(B, T).to(device)   # (B, T) non-contiguous


        # same loss as train (logits, no softmax)

        val_loss =  CE_LOSS(logits.permute(0, 2, 1), target) 
 

        sum_loss += float(val_loss.item())
        n_steps  += 1

        # ---- metrics (ACCURACY: replaced/expanded)
        preds1 = logits.argmax(dim=-1)  # [B,T]
        acc1_hits = (preds1 == target).sum().item()

        top5 = torch.topk(logits, k=min(5, C), dim=-1).indices
        hits_any = (top5 == target.unsqueeze(-1)).any(dim=-1)   # [B,T] bool
        acc5_hits = hits_any.sum().item()

        d1 = _circ_bin_dist(preds1, target, C)
        within1_hits = (d1 <= 1).sum().item()

        err_deg = _angular_error_deg_from_logits(logits, target)
        mae_batch = err_deg.sum().item()
        mse_batch = (err_deg ** 2).sum().item()
        acc5deg_batch = (err_deg <= 5.0).sum().item()
        acc10deg_batch = (err_deg <= 10.0).sum().item()

        frames = target.numel()
        sum_used += frames
        sum_acc1 += acc1_hits
        sum_acc5 += acc5_hits
        sum_within1 += within1_hits
        mae_deg_sum += mae_batch
        mse_deg_sum += mse_batch
        acc5deg_sum += acc5deg_batch
        acc10deg_sum += acc10deg_batch

        # keep legacy keys (identical for one-hot)
        sum_any += acc5_hits
        sum_all += acc5_hits

        pbar.set_postfix({
            "val_loss": f"{(sum_loss / max(1, n_steps)):.4f}",
            "acc@1":    f"{(sum_acc1 / max(1, sum_used)):.3f}",
            "acc@5":    f"{(sum_acc5 / max(1, sum_used)):.3f}",
            "±1bin":    f"{(sum_within1 / max(1, sum_used)):.3f}",
            "MAE°":     f"{(mae_deg_sum / max(1, sum_used)):.2f}",
            "acc@5°":   f"{(acc5deg_sum / max(1, sum_used)):.3f}",
        })

    metrics = {
        "val_loss":   sum_loss / max(1, n_steps),
        # legacy keys preserved
        "acc_any@5": (sum_any / sum_used) if sum_used else float('nan'),
        "acc_all@5": (sum_all / sum_used) if sum_used else float('nan'),

        # new accuracy metrics
        "acc@1": (sum_acc1 / sum_used) if sum_used else float('nan'),
        "acc@5": (sum_acc5 / sum_used) if sum_used else float('nan'),
        "acc_within_1bin": (sum_within1 / sum_used) if sum_used else float('nan'),
        "mae_deg": (mae_deg_sum / sum_used) if sum_used else float('nan'),
        "rmse_deg": math.sqrt(mse_deg_sum / sum_used) if sum_used else float('nan'),
        "acc@5°": (acc5deg_sum / sum_used) if sum_used else float('nan'),
        "acc@10°": (acc10deg_sum / sum_used) if sum_used else float('nan'),
    }
    # keep key expected by other code paths (unchanged)
    metrics.setdefault("mae_speech", float('nan'))

    # restore original weights if EMA was applied
    if ema_buf is not None and orig_state is not None:
        model.load_state_dict(orig_state, strict=False)
    if was_training:
        model.train()
    return metrics

# -------------------------------
# Main
# -------------------------------
def main():
    import numpy as np
    import pandas as pd
    import torch

    args = parse_args()
    cfg = load_config(args)

    device = torch.device(cfg['device'])
    set_seed(cfg.get('seed', 56))
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Run dir + logging
    run_dir = make_run_dir(args.save_root, args.model, cfg)
    print(f"Run dir: {run_dir}")

    # Data
    print("Building loaders… (precomputed={} path={})"
          .format(cfg.get('use_precomputed', False), cfg.get('precomputed_path')))
    train_loader = make_loader_from_cfg(cfg, 'train')
    val_loader   = make_loader_from_cfg(cfg, 'val')
    steps_per_epoch = len(train_loader)
    print(f"Steps/epoch: {steps_per_epoch}, val_steps: {len(val_loader)}")

    # curriculum init
    K_data = cfg['features']['K']
    cfg['loss'] = get_loss_config_for_epoch(0, cfg['epochs'], K=K_data)

    # model / opt / sched
    # model = build_model(args.model, cfg).to(device)
    model = DoAEstimator(num_mics=12, num_classes=72).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # build_optimizer(model, cfg)
    scheduler = None #build_scheduler(optimizer, cfg, steps_per_epoch)
    ema_buf = ema_init(model) if cfg.get('ema_decay', 0) else None
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get('mixed_precision', True))

    # -----------------------
    # Early-stopping settings
    # -----------------------
    # Which validation metric to monitor
    best_key = cfg.get('early_stop', {}).get('monitor', 'mae_deg')  # e.g. 'mae_deg', 'acc@1', 'acc@5'
    min_delta = float(cfg.get('early_stop', {}).get('min_delta', 0.0))
    patience = int(cfg.get('early_stop', {}).get('patience', cfg.get('patience', 12)))

    # Infer mode if not provided: minimize for loss/mae/rmse/err, else maximize
    monitor_mode = cfg.get('early_stop', {}).get('mode', None)
    if monitor_mode not in ('min', 'max'):
        key_l = str(best_key).lower()
        monitor_mode = 'min' if any(s in key_l for s in ['loss', 'mae', 'rmse', 'err']) else 'max'

    best_metric = float('inf') if monitor_mode == 'min' else float('-inf')
    wait = 0

    hist = []
    start_epoch = 0

    # resume
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'ema_shadow' in ckpt and ckpt['ema_shadow'] is not None:
            ema_buf = {k: v.to(device) for k, v in ckpt['ema_shadow'].items()}
        else:
            if cfg.get('ema_decay', 0):
                ema_buf = ema_init(model)

        if 'optimizer' in ckpt and 'scheduler' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            #scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt and isinstance(ckpt['scaler'], dict):
            try:
                scaler.load_state_dict(ckpt['scaler'])
            except Exception:
                pass

        # restore monitor settings if present, keep defaults otherwise
        best_key = ckpt.get('best_key', best_key)
        monitor_mode = ckpt.get('monitor_mode', monitor_mode)
        best_metric = ckpt.get('best_metric',
                        ckpt.get('best_acc',
                        ckpt.get('best_acc_any@5',
                        ckpt.get('best_mae', best_metric))))
        hist = ckpt.get('metrics_history', hist)
        start_epoch = ckpt.get('epoch', -1) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best({best_key}, mode={monitor_mode})={best_metric:.4f}")

    # Train
    epochs = cfg['epochs']
    accum_steps = max(1, int(getattr(args, 'accum_steps', 1)))
    clip_norm = args.clip_norm if args.clip_norm is not None else cfg.get('grad_clip', None)

    prog = tqdm(range(start_epoch, epochs), desc="epochs", position=0)

    for epoch in prog:
        cfg['loss'] = get_loss_config_for_epoch(epoch, epochs, K=K_data)

        #---- one full training epoch (has its own pbar/plots inside)
        tr_loss, _ = train_one_epoch(
             model, train_loader, optimizer, scheduler, scaler,
             ema_buf, cfg, device, args.model,
             accum_steps=accum_steps, clip_norm=clip_norm, epoch=epoch,
             viz_every=500, viz_dir="check_train_plots"
         )


        # ---- validation (returns dict with mae_deg, acc@1, acc@5, acc_within_1bin, etc.)
        val_metrics = evaluate(
            model, val_loader, device, args.model, K=cfg['features']['K'],
            ema_buf=ema_buf, cfg=cfg
        )

        # ---- logging & progress
        row = {
            'epoch': epoch + 1,
            'train_loss': float(tr_loss),
            'lr': float(optimizer.param_groups[0]['lr']),
        }
        row.update({k: float(v) for k, v in val_metrics.items()})
        hist.append(row)

        disp_val = float(val_metrics.get(best_key, float('nan')))
        prog.set_postfix({
            "loss": f"{tr_loss:.4f}",
            best_key: f"{disp_val:.3f}" if not np.isnan(disp_val) else "nan",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

        # ---- save "last" checkpoint every epoch
        last_payload = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            #'scheduler': scheduler.state_dict(),
            #'scaler': scaler.state_dict() if scaler is not None else None,
            'ema_shadow': ema_buf,
            'best_key': best_key,
            'monitor_mode': monitor_mode,
            'best_metric': best_metric,
            'last_val_metrics': val_metrics,  # snapshot for this epoch
            'metrics_history': hist,
            'config': cfg,
            # backward-compat
            'best_acc': best_metric,
        }
        save_checkpoint(last_payload, run_dir / 'last.pt')

        # ---- early stopping / save "best"
        improved = False
        if not np.isnan(disp_val):
            if monitor_mode == 'min':
                improved = disp_val < (best_metric - min_delta)
            else:
                improved = disp_val > (best_metric + min_delta)

        if improved:
            best_metric = disp_val
            wait = 0

            # Prepare state dict to save (EMA overlay if provided)
            if ema_buf is not None:
                cur_state = model.state_dict()
                ema_state = {k: ema_buf.get(k, v) for k, v in cur_state.items()}
                model_state_to_save = ema_state
            else:
                model_state_to_save = model.state_dict()

            best_payload = {
                'epoch': epoch,
                'model_state_dict': model_state_to_save,
                'ema_shadow': ema_buf,
                'best_key': best_key,
                'monitor_mode': monitor_mode,
                'best_metric': best_metric,
                'best_val_metrics': val_metrics,  # snapshot for best epoch
                'config': cfg,
                'metrics_history': hist,
                # include optimizer/scheduler/scaler for reproducibility
                'optimizer': optimizer.state_dict(),
                #'scheduler': scheduler.state_dict(),
                #'scaler': scaler.state_dict() if scaler is not None else None,
                # backward-compat
                'best_acc': best_metric,
            }
            save_checkpoint(best_payload, run_dir / 'best.pt')
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no improvement in {best_key} for {patience} epochs; mode={monitor_mode}).")
                break

        # ---- persist CSV each epoch
        pd.DataFrame(hist).to_csv(run_dir / 'metrics.csv', index=False)

    print("\nTraining finished.")
    print(f"Best {best_key} ({monitor_mode}): {best_metric:.3f} -> run dir: {run_dir}")

if __name__ == '__main__':
    main()

