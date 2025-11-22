#!/usr/bin/env python3
"""
Test DOA models on test dataset and compute accuracy metrics.
Computes accuracy at 5°, 10°, 15°, and 20° tolerances.
"""

import sys
from pathlib import Path
import torch.nn.functional as F

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
import torch.nn as nn
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
from mirokai_doa.models import SCATTiny, FiLMMixerSRP, ReTiNDoA


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    K: int = 72,
    tolerances: list = [5, 10, 15, 20],
    vad_threshold: float = 0.1,
    model_name: str = 'basic'
) -> Dict[str, float]:
    """
    Evaluate model on test set and compute accuracy metrics.
    
    Args:
        model: PyTorch model
        loader: Test dataloader
        device: Device to run inference on
        K: Number of azimuth bins
        tolerances: List of tolerance angles in degrees for accuracy computation
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Accumulators
    total_frames = 0
    sum_mae = 0.0
    sum_mse = 0.0
    
    # Accuracy counters for each tolerance
    acc_counts = {tol: 0 for tol in tolerances}
    
    # Additional metrics
    sum_acc1 = 0  # Exact bin match
    sum_acc5 = 0  # Top-5 contains GT
    
    pbar = tqdm(loader, desc="Testing", leave=True)
    for batch in pbar:
        feats, vad, srp, srp_vad, gt = _normalize_batch(batch)
        
        feats = feats.to(device)  # [B, C, T, F]
        srp = srp.to(device)        
        # Match train: [B, C, T, F] -> [B, T, F, C]
        feats = feats.permute(0, 2, 3, 1).contiguous()
        
        # Forward pass
        if model_name == "basic":
            logits = model(feats)
        elif model_name == "retin":

            logits,_,_ = model(feats, srp)
        else:  
                    
            logits= F.softmax(model(feats, srp), dim=-1)   
        
        B, T, C = logits.shape
        assert C == K, f"Model output K={C} doesn't match config K={K}"
        
        # Handle GT format: could be multi-hot [B, T, K] or angles [B] or [B, T]
        if gt.dim() == 3 and gt.shape[-1] == K:
            # Multi-hot format: [B, T, K] - use argmax to get primary source bin
            gt_bins = gt.argmax(dim=-1)  # [B, T]
        elif gt.dim() == 1:
            # Single angle per batch: [B] in degrees
            gt_bins = quantize_az_deg_torch(gt, K).unsqueeze(1).expand(B, T)  # [B, T]
        elif gt.dim() == 2:
            # Angles per frame: [B, T] in degrees
            gt_bins = quantize_az_deg_torch(gt.flatten(), K).reshape(B, T)  # [B, T]
        else:
            raise ValueError(f"Unexpected GT shape: {gt.shape}")
        
        target = gt_bins.to(device)  # [B, T]
        
        # Process VAD: vad can be [B, T, F] or [B, T]
        if vad.dim() == 3:
            # Take first frequency bin or mean across frequency
            vad_probs = vad[..., 0].to(device)  # [B, T]
        else:
            vad_probs = vad.to(device)  # [B, T]
        
        # Create VAD mask: only consider frames with VAD >= threshold
        vad_mask = vad_probs >= vad_threshold  # [B, T] bool
        
        # Get argmax prediction (discrete bin

        pred_bins = logits.argmax(dim=-1)  # [B, T]
        
        # Compute angular error in degrees using argmax (not circular mean)
        # Convert predicted bin to degrees
        bin_size = 360.0 / K
        pred_deg = ((pred_bins.float() + 0.5) * bin_size) % 360.0  # [B, T]
        true_deg = ((target.float() + 0.5) * bin_size) % 360.0     # [B, T]
        
        # Circular angular difference in degrees
        err_deg = ((pred_deg - true_deg + 180.0) % 360.0) - 180.0
        err_deg = err_deg.abs()  # [B, T]
        
        # Also compute circular mean error for MAE/RMSE (for comparison)
        err_deg_circmean = _angular_error_deg_from_logits(logits, target)  # [B, T]
        
        # Only accumulate errors for frames with active VAD
        active_frames = vad_mask.sum().item()
        total_frames += active_frames
        
        if active_frames > 0:
            # Mask errors to only active VAD frames
            err_deg_active = err_deg[vad_mask]  # Using argmax-based error
            err_deg_circmean_active = err_deg_circmean[vad_mask]  # Circular mean error
            
            # Use circular mean for MAE/RMSE (more accurate)
            sum_mae += err_deg_circmean_active.sum().item()
            sum_mse += (err_deg_circmean_active ** 2).sum().item()
            
            # Accuracy at different tolerances using argmax prediction
            for tol in tolerances:
                acc_counts[tol] += (err_deg_active <= tol).sum().item()
            
            # Exact bin match (top-1) - only on active VAD frames
            preds1 = logits.argmax(dim=-1)  # [B, T]
            preds1_active = preds1[vad_mask]
            target_active = target[vad_mask]
            sum_acc1 += (preds1_active == target_active).sum().item()
            
            # Top-5 contains GT - only on active VAD frames
            top5 = torch.topk(logits, k=min(5, K), dim=-1).indices  # [B, T, 5]
            hits_any = (top5 == target.unsqueeze(-1)).any(dim=-1)  # [B, T]
            sum_acc5 += hits_any[vad_mask].sum().item()
        
        # Update progress bar
        if total_frames > 0:
            current_mae = sum_mae / total_frames
            current_acc5 = acc_counts[5] / total_frames
            current_acc10 = acc_counts[10] / total_frames
        else:
            current_mae = 0.0
            current_acc5 = 0.0
            current_acc10 = 0.0
        # Format progress bar with VAD threshold
        pbar.set_postfix_str(
            f"VAD≥{vad_threshold:.1f}={total_frames:,} | "
            f"MAE°={current_mae:.2f} | "
            f"Acc@5°={current_acc5:.3f} | "
            f"Acc@10°={current_acc10:.3f}"
        )
    
    # Compute final metrics
    metrics = {
        "total_frames": total_frames,
        "mae_deg": sum_mae / max(1, total_frames),
        "rmse_deg": math.sqrt(sum_mse / max(1, total_frames)),
        "acc@1": sum_acc1 / max(1, total_frames),
        "acc@5": sum_acc5 / max(1, total_frames),
    }
    
    # Add accuracy at each tolerance
    for tol in tolerances:
        metrics[f"acc@{tol}°"] = acc_counts[tol] / max(1, total_frames)
    
    return metrics


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device,
    model_name: str = None
) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Device to load model on
        model_name: Model type ('scat', 'film', 'retin', 'basic', 'doa')
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Try to detect model type from checkpoint if not provided
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
    
    # Build model
    if model_name == 'basic':
        model = DoAEstimator(num_mics=12, num_classes=72).to(device)
    elif model_name == 'retin':
        model = ReTiNDoA(in_ch=12, K=72, C=128).to(device)
    elif model_name == 'film':
        model = FiLMMixerSRP( in_ch=12, K=72, C=128).to(device)
    elif model_name == 'scat':
        model = SCATTiny( in_ch=12, K=72, C=128).to(device)
    else:
        print('model not supported')
    
    model = model.to(device)
    print(model)
    
    # Get state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Check if RNN keys exist in checkpoint (for models with DoABackbone)
    has_rnn_keys = any(k.startswith('backbone.rnn.') for k in state_dict.keys())
    
    # If RNN keys exist, initialize RNN before loading weights
    # This ensures the RNN structure matches the checkpoint
    if has_rnn_keys and hasattr(model, 'backbone') and hasattr(model.backbone, 'rnn'):
        # Infer RNN input size from checkpoint weights
        # weight_ih_l0 has shape [hidden_size * 4, input_size] for LSTM
        rnn_key = 'backbone.rnn.weight_ih_l0'
        if rnn_key in state_dict:
            checkpoint_input_size = state_dict[rnn_key].shape[1]
            # Get RNN parameters from backbone
            backbone = model.backbone
            dirs = 2 if backbone.bidirectional else 1
            hidden = backbone.out_dim // dirs
            
            # Manually build RNN with the correct input size from checkpoint
            backbone.rnn = nn.LSTM(
                input_size=checkpoint_input_size,
                hidden_size=hidden,
                num_layers=backbone.num_layers,
                batch_first=True,
                bidirectional=backbone.bidirectional,
            ).to(device=device, dtype=torch.float32)
            
            print(f"Initialized RNN with input_size={checkpoint_input_size} from checkpoint")
    
    # Load all weights (including RNN if it was initialized)
    model.load_state_dict(state_dict, strict=False)
    
    print(f"Loaded model from {checkpoint_path}")
    if has_rnn_keys:
        print("RNN weights loaded from checkpoint")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Test DOA models on test dataset")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--cfg', type=str, default=None,
                       help='Path to config YAML (default: configs/train.yaml)')
    parser.add_argument('--model', type=str, choices=['scat', 'film', 'retin', 'basic', 'doa'],
                       default=None, help='Model type (auto-detected if not provided)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default: cuda if available, else cpu)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--workers', type=int, default=None,
                       help='Override num_workers from config')
    parser.add_argument('--precomputed-path', type=str, default=None,
                       help='Path to precomputed features directory (default: feature_cache_stateofart)')
    parser.add_argument('--vad-threshold', type=float, default=0.5,
                       help='VAD threshold for filtering frames (default: 0.5)')
    
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
            # Default config
            config = {
                'features': {'K': 72, 'max_frames': 25, 'vad_threshold': 0.5},
                'batch_size': 64,
                'dataloader': {'num_workers': 4, 'pin_memory': True},
            }
    
    # Ensure batch_size is set (default to 64)
    if 'batch_size' not in config:
        config['batch_size'] = 64
    
    # Ensure dataloader config exists
    if 'dataloader' not in config:
        config['dataloader'] = {}
    if 'num_workers' not in config['dataloader']:
        config['dataloader']['num_workers'] = 4
    if 'pin_memory' not in config['dataloader']:
        config['dataloader']['pin_memory'] = True
    
    # Set precomputed features path
    if args.precomputed_path:
        precomputed_path = args.precomputed_path
    else:
        # Default to feature_cache_stateofart if it exists
        default_path = project_root / "feature_cache_stateofart"
        if default_path.exists():
            precomputed_path = str(default_path)
        else:
            precomputed_path = None
    
    # Enable precomputed features if path is provided
    if precomputed_path:
        config['use_precomputed'] = True
        config['precomputed_path'] = precomputed_path
        print(f"Using precomputed features from: {precomputed_path}")
    else:
        print("WARNING: No precomputed features path found. Using online feature computation.")
    
    # Override config with CLI args
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.workers is not None:
        config.setdefault('dataloader', {})['num_workers'] = args.workers
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Get K from config
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
    
    # Create test dataloader
    print("\nCreating test dataloader...")
    test_loader = make_loader_from_cfg(config, split='test')
    print(f"Test batches: {len(test_loader)}")
    if len(test_loader) == 0:
        print("WARNING: Test dataloader is empty! Check your config and dataset paths.")
        return
    
    # Evaluate
    vad_threshold = 0.9 #args.vad_threshold
    print(f"\nEvaluating on test set (VAD threshold: {vad_threshold})...")
    metrics = evaluate_test_set(
        model=model,
        loader=test_loader,
        device=device,
        K=K,
        tolerances=[5, 10, 15, 20],
        vad_threshold=vad_threshold,
        model_name=args.model
    )
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS (VAD-filtered)")
    print("="*60)
    print(f"VAD threshold: {vad_threshold}")
    print(f"Total frames evaluated (VAD >= {vad_threshold}): {metrics['total_frames']:,}")
    print(f"\nMean Absolute Error (MAE): {metrics['mae_deg']:.2f}°")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse_deg']:.2f}°")
    print(f"\nExact bin match (Acc@1): {metrics['acc@1']:.3%}")
    print(f"Top-5 contains GT (Acc@5): {metrics['acc@5']:.3%}")
    print(f"\nAccuracy at tolerance thresholds:")
    for tol in [5, 10, 15, 20]:
        print(f"  Acc@{tol}°: {metrics[f'acc@{tol}°']:.3%}")
    print("="*60)


if __name__ == "__main__":
    main()

