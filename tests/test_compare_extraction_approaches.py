#!/usr/bin/env python3
"""
Evaluate DoA models with multiple clip-level aggregation strategies.
Computes Acc@{5°,10°,15°,20°} + MAE/RMSE for each approach, using frames with VAD ≥ threshold.
"""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import yaml
import math
from tqdm import tqdm
from typing import Dict, Any, Tuple, List

from mirokai_doa.train_v2 import (
    make_loader_from_cfg,
    _normalize_batch,
    _angular_error_deg_from_logits,
    quantize_az_deg_torch,
)
from mirokai_doa.train_utils import build_model
from mirokai_doa.doa_model import DoAEstimator


# -----------------------------
# Utility: angles & circular ops
# -----------------------------
def _angles_deg(K: int, device=None):
    bin_size = 360.0 / K
    deg = (torch.arange(K, device=device, dtype=torch.float32) + 0.5) * bin_size
    rad = deg * math.pi / 180.0
    return deg, torch.cos(rad), torch.sin(rad)

def _softmax_temp(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    return torch.softmax(logits / tau, dim=-1)

def _circ_abs_err_deg(pred_deg: torch.Tensor, true_deg: torch.Tensor) -> torch.Tensor:
    # returns |wrapped difference| in degrees, both tensors [N]
    return torch.remainder(pred_deg - true_deg + 180.0, 360.0) - 180.0
    # caller should .abs() the result

def _clipwise_true_angle_from_bins(
    target_bins: torch.Tensor,  # [B, T]
    vad_mask: torch.Tensor,     # [B, T] bool
    K: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Circular-mean ground-truth angle per clip using VAD-active frames.
    Returns (true_deg[B], valid_mask[B])
    """
    device = target_bins.device
    bin_size = 360.0 / K
    # centers in degrees
    true_deg_frame = ((target_bins.float() + 0.5) * bin_size) % 360.0  # [B,T]
    rad = true_deg_frame * math.pi / 180.0
    w = vad_mask.float()
    wsum = w.sum(dim=1)  # [B]
    X = (w * torch.cos(rad)).sum(dim=1)  # [B]
    Y = (w * torch.sin(rad)).sum(dim=1)  # [B]
    valid = wsum > 0
    # fallback: if no active frames, set dummy angle 0 (won't be used if filtered by 'valid')
    pred_rad = torch.atan2(torch.where(valid, Y, torch.zeros_like(Y)),
                           torch.where(valid, X, torch.ones_like(X)))
    true_deg = (pred_rad * 180.0 / math.pi) % 360.0
    return true_deg, valid


# --------------------------------------
# Aggregation methods over time (clip level)
# --------------------------------------
@torch.no_grad()
def aggregate_over_time(
    logits: torch.Tensor,        # [B, T, K]
    vad_mask: torch.Tensor,      # [B, T] bool
    K: int,
    method: str,
    tau: float = 0.8,
    gamma: float = 1.5,
    smooth_k: int = 1
) -> torch.Tensor:
    """
    Returns clip-level prediction in degrees: pred_deg [B]
    """
    assert method in {"argmax_vote", "avglogits", "circmean", "hist"}
    device = logits.device
    B, T, C = logits.shape
    assert C == K
    deg, cos_th, sin_th = _angles_deg(K, device=device)

    # Masks & helpers
    m = vad_mask.float()  # [B,T]
    msum = m.sum(dim=1, keepdim=True).clamp_min(1e-8)

    if method == "argmax_vote":
        # vote using per-frame argmax bins
        per_bin = logits.argmax(dim=-1)  # [B,T]
        # one-hot counts, masked by VAD
        onehot = F.one_hot(per_bin, num_classes=K).float()  # [B,T,K]
        counts = (onehot * m.unsqueeze(-1)).sum(dim=1)      # [B,K]
        # fallback: average logits if no active frames
        no_frames = (msum.squeeze(1) < 1e-6)
        fallback = logits.mean(dim=1)                       # [B,K]
        # we pick mode over counts (or fallback argmax)
        mode_bins = counts.argmax(dim=-1)                   # [B]
        fallback_bins = fallback.argmax(dim=-1)
        pred_bins = torch.where(no_frames, fallback_bins, mode_bins)
        pred_deg = ((pred_bins.float() + 0.5) * (360.0 / K)) % 360.0
        return pred_deg

    if method == "avglogits":
        # sum (average) logits over time with VAD gating
        agg_logits = (logits * m.unsqueeze(-1)).sum(dim=1)  # [B,K]
        no_frames = (msum.squeeze(1) < 1e-6)
        agg_logits = torch.where(no_frames.unsqueeze(-1), logits.mean(dim=1), agg_logits)
        pred_bins = agg_logits.argmax(dim=-1)  # [B]
        pred_deg = ((pred_bins.float() + 0.5) * (360.0 / K)) % 360.0
        return pred_deg

    # probability-based methods
    probs = _softmax_temp(logits, tau=tau)  # [B,T,K]
    # per-frame circular mean vectors
    x = torch.einsum("btk,k->bt", probs, cos_th)  # [B,T]
    y = torch.einsum("btk,k->bt", probs, sin_th)  # [B,T]
    R_t = torch.clamp(torch.sqrt(x * x + y * y), 0, 1)      # [B,T]
    w = m * (R_t ** gamma)                                  # [B,T]
    wsum = w.sum(dim=1).clamp_min(1e-8)                     # [B]
    no_frames = (m.sum(dim=1) < 1e-6)

    if method == "circmean":
        X = (w * x).sum(dim=1)  # [B]
        Y = (w * y).sum(dim=1)  # [B]
        # fallback: unweighted mean if no active frames
        X = torch.where(no_frames, x.mean(dim=1), X)
        Y = torch.where(no_frames, y.mean(dim=1), Y)
        pred_rad = torch.atan2(Y, X)
        pred_deg = (pred_rad * 180.0 / math.pi) % 360.0
        return pred_deg

    if method == "hist":
        # weighted probability histogram across time
        agg = (w.unsqueeze(-1) * probs).sum(dim=1)  # [B,K]
        agg = torch.where(no_frames.unsqueeze(-1), probs.mean(dim=1), agg)
        if smooth_k and smooth_k > 0:
            # circular box filter with width (2*smooth_k+1)
            pad = torch.cat([agg[:, -smooth_k:], agg, agg[:, :smooth_k]], dim=1)  # [B, K+2s]
            kernel = torch.ones(1, 1, 2 * smooth_k + 1, device=device) / (2 * smooth_k + 1)
            agg = F.conv1d(pad.unsqueeze(1), kernel, padding=0).squeeze(1)       # [B,K]
        pred_bins = agg.argmax(dim=-1)  # [B]
        pred_deg = ((pred_bins.float() + 0.5) * (360.0 / K)) % 360.0
        return pred_deg

    raise ValueError(f"Unknown method {method}")


# --------------------------------------
# Evaluation across methods (clip-level)
# --------------------------------------
@torch.no_grad()
def evaluate_clipwise_methods(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    K: int = 72,
    tolerances: List[int] = [5, 10, 15, 20],
    vad_threshold: float = 0.5,
    methods: List[str] = ("argmax_vote", "avglogits", "circmean", "hist"),
    tau: float = 0.8,
    gamma: float = 1.5,
    smooth_k: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Returns dict of metrics per method.
    Metrics keys: total_clips, mae_deg, rmse_deg, acc@5°, acc@10°, ...
    """
    model.eval()
    # accumulators per method
    metrics = {
        m: {
            "total_clips": 0,
            "sum_abs": 0.0,
            "sum_sq": 0.0,
            **{f"acc@{tol}°_count": 0 for tol in tolerances},
        }
        for m in methods
    }

    pbar = tqdm(loader, desc="Testing (clipwise)", leave=True)
    for batch in pbar:
        feats, vad, srp, srp_vad, gt = _normalize_batch(batch)
        feats = feats.to(device)  # [B, C, T, F]
        feats = feats.permute(0, 2, 3, 1).contiguous()  # [B, T, F, C] -> model expects [B,T,F,C]
        logits = model(feats)  # [B, T, K]
        B, T, C = logits.shape
        assert C == K, f"Model output K={C} doesn't match config K={K}"

        # ground-truth bins per frame
        if gt.dim() == 3 and gt.shape[-1] == K:
            gt_bins = gt.argmax(dim=-1)  # [B,T]
        elif gt.dim() == 1:
            gt_bins = quantize_az_deg_torch(gt, K).unsqueeze(1).expand(B, T)  # [B,T]
        elif gt.dim() == 2:
            gt_bins = quantize_az_deg_torch(gt.flatten(), K).reshape(B, T)  # [B,T]
        else:
            raise ValueError(f"Unexpected GT shape: {gt.shape}")
        gt_bins = gt_bins.to(device)

        # VAD processing
        if vad.dim() == 3:
            vad_probs = vad[..., 0].to(device)  # [B,T]
        else:
            vad_probs = vad.to(device)          # [B,T]
        vad_mask = vad_probs >= vad_threshold   # [B,T] bool

        # clip-level GT angle (and valid mask)
        true_deg_clip, valid_clip = _clipwise_true_angle_from_bins(gt_bins, vad_mask, K)  # [B], [B]
        valid_idx = valid_clip.nonzero(as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            # nothing valid in this batch; continue
            continue

        # restrict tensors to valid clips only
        logits_v = logits[valid_idx]           # [Bv,T,K]
        vad_mask_v = vad_mask[valid_idx]       # [Bv,T]
        true_deg_v = true_deg_clip[valid_idx]  # [Bv]

        # evaluate each method
        preds_by_method = {}
        for m in methods:
            preds_by_method[m] = aggregate_over_time(
                logits_v, vad_mask_v, K, method=m, tau=tau, gamma=gamma, smooth_k=smooth_k
            )  # [Bv] degrees

        # accumulate metrics
        for m in methods:
            pred_deg = preds_by_method[m]
            err = _circ_abs_err_deg(pred_deg, true_deg_v).abs()  # [Bv]
            n = err.numel()
            metrics[m]["total_clips"] += int(n)
            metrics[m]["sum_abs"] += float(err.sum().item())
            metrics[m]["sum_sq"] += float((err * err).sum().item())
            for tol in tolerances:
                metrics[m][f"acc@{tol}°_count"] += int((err <= tol).sum().item())

        # progress text
        # show quick snapshot for circmean
        m = "circmean"
        nclips = metrics[m]["total_clips"]
        if nclips > 0:
            mae = metrics[m]["sum_abs"] / nclips
            acc10 = metrics[m]["acc@10°_count"] / nclips
            acc5  = metrics[m]["acc@5°_count"] / nclips
            pbar.set_postfix_str(f"VAD≥{vad_threshold:.1f} clips={nclips:,} | MAE°(circ)={mae:.2f} | Acc@5°={acc5:.3f} | Acc@10°={acc10:.3f}")

    # finalize to rates
    out = {}
    for m, d in metrics.items():
        n = max(1, d["total_clips"])
        out[m] = {
            "total_clips": d["total_clips"],
            "mae_deg": d["sum_abs"] / n,
            "rmse_deg": math.sqrt(d["sum_sq"] / n),
        }
        for tol in tolerances:
            out[m][f"acc@{tol}°"] = d[f"acc@{tol}°_count"] / n
    return out


# -----------------------------
# Model loading (unchanged)
# -----------------------------
def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device,
    model_name: str = None
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if model_name is None:
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        keys = list(state_dict.keys())
        if any('c1.net' in k or 'rnn.weight_ih_l0' in k or 'ff1.weight' in k for k in keys):
            model_name = 'doa'
        elif any('block1.0.weight' in k or 'mlp.0.weight' in k for k in keys):
            model_name = 'basic'
        elif any('srp_proto' in k or ('backbone' in k and 'cross' in str(keys)) for k in keys):
            model_name = 'scat'
        elif any('film' in k and 'blocks' in str(keys) for k in keys):
            model_name = 'film'
        elif any('cell' in k for k in keys):
            model_name = 'retin'
        else:
            raise ValueError("Could not detect model type from checkpoint. Please specify --model")
    print(f"Detected/Using model type: {model_name}")

    if model_name == 'doa':
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        num_mics = 12
        K = config.get('features', {}).get('K', 72)
        num_classes = K
        if 'c1.net.0.weight' in state_dict:
            num_mics = state_dict['c1.net.0.weight'].shape[1]
        if 'ff2.weight' in state_dict:
            num_classes = state_dict['ff2.weight'].shape[0]
        print(f"Building DoAEstimator with num_mics={num_mics}, num_classes={num_classes}")
        model = DoAEstimator(num_mics=num_mics, num_classes=num_classes)
    else:
        model = build_model(model_name, config)

    model = model.to(device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded model from {checkpoint_path}")
    return model


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Test DOA models with multiple clip-level aggregations")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--cfg', type=str, default=None, help='Path to config YAML (default: configs/train.yaml)')
    parser.add_argument('--model', type=str, choices=['scat', 'film', 'retin', 'basic', 'doa'],
                        default=None, help='Model type (auto-detected if not provided)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (default: cuda if available, else cpu)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size from config')
    parser.add_argument('--workers', type=int, default=None, help='Override num_workers from config')
    parser.add_argument('--precomputed-path', type=str, default=None, help='Path to precomputed features directory')
    parser.add_argument('--vad-threshold', type=float, default=0.5, help='VAD threshold (default: 0.5)')
    # Aggregation hyperparams
    parser.add_argument('--methods', type=str, default="argmax_vote,avglogits,circmean,hist",
                        help='Comma-separated methods to evaluate')
    parser.add_argument('--tau', type=float, default=0.8, help='Softmax temperature for prob methods')
    parser.add_argument('--gamma', type=float, default=1.5, help='Frame confidence exponent for prob methods')
    parser.add_argument('--smooth-k', type=int, default=1, help='Histogram circular smoothing half-width (bins)')
    args = parser.parse_args()

    # Load config
    if args.cfg:
        with open(args.cfg, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config_path = project_root / "configs" / "train.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {
                'features': {'K': 72, 'max_frames': 25, 'vad_threshold': 0.5},
                'batch_size': 64,
                'dataloader': {'num_workers': 4, 'pin_memory': True},
            }

    if 'batch_size' not in config:
        config['batch_size'] = 64
    if 'dataloader' not in config:
        config['dataloader'] = {}
    if 'num_workers' not in config['dataloader']:
        config['dataloader']['num_workers'] = 4
    if 'pin_memory' not in config['dataloader']:
        config['dataloader']['pin_memory'] = True

    # Precomputed features path (optional)
    if args.precomputed_path:
        precomputed_path = args.precomputed_path
    else:
        default_path = project_root / "feature_cache_stateofart"
        precomputed_path = str(default_path) if default_path.exists() else None
    if precomputed_path:
        config['use_precomputed'] = True
        config['precomputed_path'] = precomputed_path
        print(f"Using precomputed features from: {precomputed_path}")
    else:
        print("WARNING: No precomputed features path found. Using online feature computation.")

    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.workers is not None:
        config.setdefault('dataloader', {})['num_workers'] = args.workers

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # K
    K = config.get('features', {}).get('K', 72)
    print(f"Number of azimuth bins (K): {K}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device,
        model_name=args.model
    )

    # Dataloader
    print("\nCreating test dataloader...")
    test_loader = make_loader_from_cfg(config, split='test')
    print(f"Test batches: {len(test_loader)}")
    if len(test_loader) == 0:
        print("WARNING: Test dataloader is empty! Check your config and dataset paths.")
        return

    # Evaluate
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    tolerances = [5, 10, 15, 20]
    vad_threshold = args.vad_threshold  # default 0.5 per requirement
    print(f"\nEvaluating (clip-level) with VAD threshold: {vad_threshold}")
    results = evaluate_clipwise_methods(
        model=model,
        loader=test_loader,
        device=device,
        K=K,
        tolerances=tolerances,
        vad_threshold=vad_threshold,
        methods=methods,
        tau=args.tau,
        gamma=args.gamma,
        smooth_k=args.smooth_k,
    )

    # Pretty print per-method blocks
    print("\n" + "=" * 72)
    print("CLIP-LEVEL RESULTS (VAD-filtered, one prediction per clip)")
    print("=" * 72)
    for m in methods:
        r = results[m]
        print(f"\n[{m}]")
        print(f"  Clips evaluated: {r['total_clips']:,}")
        print(f"  MAE:  {r['mae_deg']:.2f}°")
        print(f"  RMSE: {r['rmse_deg']:.2f}°")
        for tol in tolerances:
            print(f"  Acc@{tol}°: {r[f'acc@{tol}°']:.3%}")

    # Comparison table
    print("\n" + "-" * 72)
    header = f"{'Method':<14} {'Clips':>8} {'MAE°':>8} {'RMSE°':>8} " + " ".join([f"Acc@{t}°".rjust(10) for t in tolerances])
    print(header)
    print("-" * 72)
    for m in methods:
        r = results[m]
        row = f"{m:<14} {r['total_clips']:>8} {r['mae_deg']:>8.2f} {r['rmse_deg']:>8.2f} " + " ".join([f"{r[f'acc@{t}°']*100:>9.2f}%" for t in tolerances])
        print(row)
    print("-" * 72)

    # Simple recommendation: best mean accuracy across the 4 thresholds
    scores = {m: np.mean([results[m][f"acc@{t}°"] for t in tolerances]) for m in methods}
    best = max(scores.items(), key=lambda kv: kv[1])
    print(f"\nRecommendation: **{best[0]}** (highest mean Acc across {{5°,10°,15°,20°}} = {best[1]*100:.2f}%)")
    if "circmean" in methods and best[0] != "circmean":
        print("Note: If scores are close, prefer 'circmean' for stability and proper circular handling.")

    print("=" * 72)


if __name__ == "__main__":
    main()
