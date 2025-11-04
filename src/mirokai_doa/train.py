#!/usr/bin/env python3
"""
Training script for DoA models.
Supports SCATTiny, FiLMMixerSRP, and ReTiNDoA models with comprehensive training features.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from mirokai_doa.train_utils import (
    build_model, build_optimizer, build_scheduler, ema_init, ema_update,
    angular_metrics, save_checkpoint, load_checkpoint, make_run_dir,
    set_seed, compute_loss, get_model_outputs_for_metrics, create_dataloader_from_config
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DoA models')
    
    # Config options
    parser.add_argument('--cfg', type=str, help='Path to config YAML file')
    parser.add_argument('--model', type=str, choices=['scat', 'film', 'retin'], 
                       required=True, help='Model type to train')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--save-root', type=str, default='models', 
                       help='Root directory to save models')
    
    # Override config options
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, help='Device (cpu/cuda)')
    parser.add_argument('--use-precomputed', action='store_true', 
                       help='Use precomputed features from disk')
    parser.add_argument('--precomputed-path', type=str, 
                       help='Path to precomputed features directory')
    
    return parser.parse_args()


def load_config(args) -> Dict[str, Any]:
    """Load and merge configuration."""
    # Load base config
    if args.cfg:
        with open(args.cfg, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'epochs': 100,
            'batch_size': 8,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'use_precomputed': False,  # Set to True to use precomputed features
            'precomputed_path': None,  # Path to precomputed features directory
            'optimizer': {
                'name': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'betas': [0.9, 0.98],
                'eps': 1e-8,
                'srp_lr_mult': 2.0
            },
            'scheduler': {
                'name': 'onecycle',
                'max_lr': 1e-3,
                'pct_start': 0.1
            },
            'loss': {
                'w_ce': 1.0,
                'w_vmf': 0.3,
                'w_srp': 0.6,
                'w_quiet': 0.2,
                'w_delta': 0.3
            },
            'patience': 10,
            'grad_clip': 5.0,
            'ema_decay': 0.999,
            'mixed_precision': True,
            'features': {
                'K': 72,
                'max_frames': 6,
                'vad_threshold': 0.6,
                'vad_gt_masking': True
            },
            'model': {
                'C': 128,
                'F_groups': 16,
                'M': 12,
                'heads': 2,
                'vmf_head': False
            },
            'dataset': {
                'epoch_size': 4000,
                'seed': 56
            },
            'dataloader': {
                'num_workers': 0,
                'pin_memory': False
            }
        }
    
    # Override with command line args
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['optimizer']['lr'] = args.lr
        config['scheduler']['max_lr'] = args.lr
    if args.device is not None:
        config['device'] = args.device
    if args.use_precomputed:
        config['use_precomputed'] = True
    if args.precomputed_path is not None:
        config['precomputed_path'] = args.precomputed_path
    
    return config


def train_epoch(model, dataloader, optimizer, scheduler, scaler, ema_shadow, 
               config, device, model_name):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Create progress bar
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        print('batch_idx: ', batch_idx)
        
        # Handle different batch formats (precomputed vs on-the-fly)
        if isinstance(batch, dict):
            # Precomputed features format: dict with keys
            features = batch['features']
            vad = batch['vad']
            srp_phat = batch['srp_phat']
            srp_phat_vad = batch['srp_phat_vad']
            ground_truth = batch['ground_truth']
        else:
            # On-the-fly features format: tuple
            if len(batch) == 5:
                features, vad, srp_phat, srp_phat_vad, ground_truth = batch
            else:
                # Take first 5 elements if batch has more
                features, vad, srp_phat, srp_phat_vad, ground_truth = batch[:5]
        
        # Handle tensor shapes - check if we need to squeeze
        if features.dim() == 5 and features.size(1) == 1:
            # The dataloader returns tensors with shape [num_sub_batches, 1, ...]
            # We need to reshape them to [num_sub_batches, ...] by squeezing the second dimension
            features = features.squeeze(1)  # [num_sub_batches, 12, T, F]
            vad = vad.squeeze(1)  # [num_sub_batches, T, F]
            srp_phat = srp_phat.squeeze(1)  # [num_sub_batches, T, K]
            srp_phat_vad = srp_phat_vad.squeeze(1)  # [num_sub_batches, T, K]
            ground_truth = ground_truth.squeeze(1)  # [num_sub_batches, T, K]
            
        features = features.to(device)
        vad = vad.to(device)
        srp_phat = srp_phat.to(device)
        srp_phat_vad = srp_phat_vad.to(device)
        ground_truth = ground_truth.to(device)
        
        # Process all sub-batches at once (much more efficient!)
        # features: [num_sub_batches, 12, T, F]
        # vad: [num_sub_batches, T, F] 
        # srp_phat: [num_sub_batches, T, K]
        # srp_phat_vad: [num_sub_batches, T, K]
        # ground_truth: [num_sub_batches, T, K]
        
        # Convert ground truth to angles (theta) for all sub-batches
        K = ground_truth.size(-1)
        theta_indices = ground_truth.argmax(dim=-1)  # [num_sub_batches, T]
        theta = (theta_indices.float() * 2 * np.pi / K) - np.pi  # [-π, π]
        
        # Use VAD from first frequency (same across frequencies)
        vad_mask = vad[:, :, 0]  # [num_sub_batches, T]
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision - process all sub-batches at once
        with amp.autocast(enabled=config.get('mixed_precision', True)):
            # Debug: Check if tensors are on CUDA
            if batch_idx == 0:  # Only print for first batch
                print(f"Features device: {features.device}, shape: {features.shape}")
                print(f"SRP-PHAT device: {srp_phat.device}, shape: {srp_phat.shape}")
                print(f"Model device: {next(model.parameters()).device}")
            
            model_outputs = model(features, srp_phat)  # Process all sub-batches together
            loss = compute_loss(model_outputs, theta, vad_mask, srp_phat, model_name, config)
            
            # Backward pass
            scaler.scale(loss).backward()
        
        # Gradient clipping
        if config.get('grad_clip', 0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Update EMA
        if ema_shadow is not None:
            ema_update(model, ema_shadow, config.get('ema_decay', 0.999))
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar with current loss
        avg_loss = total_loss / num_batches
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{avg_loss:.4f}'
        })
    
    return total_loss / num_batches


def validate_epoch(model, dataloader, device, model_name, K=72):
    """Validate for one epoch."""
    model.eval()
    all_metrics = []
    
    # Create progress bar
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            # Handle different batch formats (precomputed vs on-the-fly)
            if isinstance(batch, dict):
                # Precomputed features format: dict with keys
                features = batch['features']
                vad = batch['vad']
                srp_phat = batch['srp_phat']
                srp_phat_vad = batch['srp_phat_vad']
                ground_truth = batch['ground_truth']
            else:
                # On-the-fly features format: tuple
                if len(batch) == 5:
                    features, vad, srp_phat, srp_phat_vad, ground_truth = batch
                else:
                    # Take first 5 elements if batch has more
                    features, vad, srp_phat, srp_phat_vad, ground_truth = batch[:5]
            
            # Handle tensor shapes - check if we need to squeeze
            if features.dim() == 5 and features.size(1) == 1:
                # The dataloader returns tensors with shape [num_sub_batches, 1, ...]
                # We need to reshape them to [num_sub_batches, ...] by squeezing the second dimension
                features = features.squeeze(1)  # [num_sub_batches, 12, T, F]
                vad = vad.squeeze(1)  # [num_sub_batches, T, F]
                srp_phat = srp_phat.squeeze(1)  # [num_sub_batches, T, K]
                srp_phat_vad = srp_phat_vad.squeeze(1)  # [num_sub_batches, T, K]
                ground_truth = ground_truth.squeeze(1)  # [num_sub_batches, T, K]
                    
            features = features.to(device)
            vad = vad.to(device)
            srp_phat = srp_phat.to(device)
            srp_phat_vad = srp_phat_vad.to(device)
            ground_truth = ground_truth.to(device)
            
            # Process all sub-batches at once (much more efficient!)
            # Convert ground truth to angles for all sub-batches
            theta_indices = ground_truth.argmax(dim=-1)  # [num_sub_batches, T]
            theta = (theta_indices.float() * 2 * np.pi / K) - np.pi  # [-π, π]
            
            # Use VAD from first frequency
            vad_mask = vad[:, :, 0]  # [num_sub_batches, T]
            
            # Forward pass - process all sub-batches together
            model_outputs = model(features, srp_phat)
            logits = get_model_outputs_for_metrics(model_outputs, model_name)
            
            # Compute metrics for all sub-batches
            metrics = angular_metrics(logits, theta, vad_mask, K)
            all_metrics.append(metrics)
            
            # Update progress bar with current metrics
            current_mae = metrics.get('mae', float('nan'))
            current_acc_5 = metrics.get('acc_5deg', float('nan'))
            pbar.set_postfix({
                'MAE': f'{current_mae:.2f}°',
                'Acc@5°': f'{current_acc_5:.3f}'
            })
    
    # Average metrics across batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        avg_metrics[key] = np.mean(values) if values else float('nan')
    
    return avg_metrics


def main():
    """Main training function."""
    args = parse_args()
    config = load_config(args)
    
    # Set device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Set random seed: {seed}")
    
    # Set cudnn benchmark
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = config.get('allow_benchmark', False)
        print(f"CuDNN benchmark: {torch.backends.cudnn.benchmark}")
    
    # Create run directory
    run_dir = make_run_dir(args.save_root, args.model, config)
    print(f"Run directory: {run_dir}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_dataloader_from_config(config, split='train')
    val_loader = create_dataloader_from_config(config, split='val')
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Build model
    print(f"Building {args.model} model...")
    model = build_model(args.model, config)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    
    # Initialize EMA
    ema_shadow = None
    if config.get('ema_decay', 0) > 0:
        ema_shadow = ema_init(model)
        print(f"Initialized EMA with decay: {config['ema_decay']}")
    
    # Mixed precision scaler
    scaler = amp.GradScaler(enabled=config.get('mixed_precision', True))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_mae = float('inf')
    metrics_history = []
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        if ema_shadow is not None and 'ema_shadow' in checkpoint:
            ema_shadow = checkpoint['ema_shadow']
        start_epoch = checkpoint['epoch'] + 1
        best_mae = checkpoint.get('best_mae', float('inf'))
        metrics_history = checkpoint.get('metrics_history', [])
        print(f"Resumed from epoch {start_epoch}, best MAE: {best_mae:.6f}")
    
    # Training loop
    print(f"Starting training for {config['epochs']} epochs...")
    patience_counter = 0
    
    # Create epoch progress bar
    epoch_pbar = tqdm(range(start_epoch, config['epochs']), desc="Epochs", position=0)
    
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, 
            ema_shadow, config, device, args.model
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, args.model, config['features']['K'])
        
        # Log metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'lr': optimizer.param_groups[0]['lr'],
            **val_metrics
        }
        metrics_history.append(epoch_metrics)
        
        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix({
            'Loss': f'{train_loss:.4f}',
            'MAE': f'{val_metrics["mae_speech"]:.2f}°',
            'Acc@5°': f'{val_metrics["within_5_speech"]:.3f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # Print detailed metrics every 10 epochs or on the last epoch
        if (epoch + 1) % 10 == 0 or epoch == config['epochs'] - 1:
            print(f"\nEpoch {epoch+1} Metrics:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val MAE (speech): {val_metrics['mae_speech']:.6f}")
            print(f"  Val MAE (all): {val_metrics['mae_all']:.6f}")
            print(f"  Val Within 5° (speech): {val_metrics['within_5_speech']:.6f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Check for improvement
        current_mae = val_metrics['mae_speech']
        if not np.isnan(current_mae) and current_mae < best_mae - 1e-6:
            best_mae = current_mae
            patience_counter = 0
            
            # Save best model
            if ema_shadow is not None:
                # Save EMA weights
                model_state = {name: param.clone() for name, param in model.named_parameters()}
                for name, param in model.named_parameters():
                    if name in ema_shadow:
                        param.data.copy_(ema_shadow[name])
                best_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_mae': best_mae,
                    'config': config
                }
                # Restore original weights
                for name, param in model.named_parameters():
                    if name in model_state:
                        param.data.copy_(model_state[name])
            else:
                best_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_mae': best_mae,
                    'config': config
                }
            
            save_checkpoint(best_state, run_dir / 'best.pt')
            print(f"New best model saved! MAE: {best_mae:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
        
        # Save last checkpoint
        last_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'ema_shadow': ema_shadow,
            'best_mae': best_mae,
            'metrics_history': metrics_history,
            'config': config
        }
        save_checkpoint(last_state, run_dir / 'last.pt')
        
        # Save metrics CSV
        df = pd.DataFrame(metrics_history)
        df.to_csv(run_dir / 'metrics.csv', index=False)
        
        # Early stopping
        if patience_counter >= config.get('patience', 10):
            print(f"Early stopping after {patience_counter} epochs without improvement")
            break
    
    print(f"\nTraining completed!")
    print(f"Best MAE (speech): {best_mae:.6f}")
    print(f"Results saved to: {run_dir}")


if __name__ == '__main__':
    main()
