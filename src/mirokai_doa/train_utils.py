# mirokai_doa/train_utils.py
"""
Training utilities for DoA models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import hashlib
import yaml
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Union, List
import torch.nn.functional as F
from .model_film_test import FiLMMixerSRP #SCATTiny, , ReTiNDoA
from .model_basic import TFPoolClassifierNoCond
import math

from .losses import scat_loss, film_loss, retin_loss
from .utils_model import decode_angle_from_logits
from .mix_batcher import OnTheFlyMixtureDataset
from .feature_dataloaders import (
    create_online_feature_loader,
    create_precomputed_dataloader,
    STFTCfg, FeatCfg,
)

# ---------------- Model builders ----------------

def build_model(model_name: str, config: Dict[str, Any]) -> nn.Module:
    model_config = config.get('model', {})
    if model_name == "scat":
        return SCATTiny(
            in_ch=12,
            K=config['features']['K'],
            C=model_config.get('C', 128),
            F_groups=model_config.get('F_groups', 16),
            M=model_config.get('M', 12),
            heads=model_config.get('heads', 2),
            vmf_head=model_config.get('vmf_head', False)
        )
    elif model_name == "film":
        return FiLMMixerSRP(
            in_ch=12,
            K=config['features']['K'],
            C=model_config.get('C', 512),
            #F_groups=model_config.get('F_groups', 16),
            nblk=model_config.get('nblk', 4),
            vmf_head=model_config.get('vmf_head', False)
        )
    elif model_name == "retin":
        return ReTiNDoA(
            in_ch=12,
            K=config['features']['K'],
            C=model_config.get('C', 96),
            F_groups=model_config.get('F_groups', 16)
        )

    elif model_name == "basic":
        K=config['features']['K']
        return TFPoolClassifierNoCond(K=K)


    else:
        raise ValueError(f"Unknown model: {model_name}")

# ---------------- Optimizer / Scheduler ----------------

def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    opt = config.get('optimizer', {})
    lr = 1e-3 #float(opt.get('lr', 1e-3) if not isinstance(opt.get('lr', 1e-3), list) else opt.get('lr', [1e-3])[0])
    wd = float(opt.get('weight_decay', 1e-4) if not isinstance(opt.get('weight_decay', 1e-4), list) else opt.get('weight_decay', [1e-4])[0])
    srp_lr_mult = float(opt.get('srp_lr_mult', 2.0))

    no_decay, decay, srp_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if any(nd in name for nd in ['bias', 'norm', 'bn']):
            no_decay.append(p)
        elif any(tag in name for tag in ['srp', 'cls']):
            srp_params.append(p)
        else:
            decay.append(p)

    groups = [
        {'params': no_decay, 'weight_decay': 0.0, 'lr': lr},
        {'params': decay,    'weight_decay': wd,  'lr': lr},
        {'params': srp_params, 'weight_decay': wd, 'lr': lr * srp_lr_mult},
    ]
    name = opt.get('name', 'adamw').lower()
    eps = float(opt.get('eps', 1e-8))
    if name == 'adamw':
        return optim.AdamW(groups, betas=opt.get('betas', (0.9, 0.98)), eps=eps)
    elif name == 'adam':
        return optim.Adam(groups, betas=opt.get('betas', (0.9, 0.999)), eps=eps)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any], steps_per_epoch: int):
    sched = config.get('scheduler', {})
    name = sched.get('name', 'onecycle').lower()
    epochs = int(config.get('epochs', 100))
    total_steps = epochs * max(1, steps_per_epoch)

    if name == 'onecycle':
        max_lr = float(sched.get('max_lr', config.get('optimizer', {}).get('lr', 1e-3)
                                 if not isinstance(config.get('optimizer', {}).get('lr', 1e-3), list)
                                 else config.get('optimizer', {}).get('lr', [1e-3])[0]))
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=sched.get('pct_start', 0.1),
            anneal_strategy=sched.get('anneal_strategy', 'cos')
        )
    elif name == 'cosine_warmup':
        warmup_steps = int(total_steps * sched.get('warmup_ratio', 0.05))
        min_lr = float(sched.get('min_lr', 1e-5) if not isinstance(sched.get('min_lr', 1e-5), list) else sched.get('min_lr', [1e-5])[0])
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=total_steps - warmup_steps,
            T_mult=1,
            eta_min=min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")

# ---------------- EMA ----------------

def ema_init(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Initialize EMA shadow on the SAME device as each parameter."""
    shadow = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            shadow[name] = p.detach().clone().to(p.device)
    return shadow

@torch.no_grad()
def ema_update(model: nn.Module, shadow: Dict[str, torch.Tensor], decay: float = 0.999):
    """In-place EMA update; keeps tensors on param device."""
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name not in shadow:
            # create on the correct device if missing (e.g., model changed)
            shadow[name] = p.detach().clone().to(p.device)
        # move shadow to the param device if needed
        if shadow[name].device != p.device:
            shadow[name] = shadow[name].to(p.device)
        shadow[name].mul_(decay).add_(p.data, alpha=(1.0 - decay))
# ---------------- Metrics ----------------


@torch.no_grad()
def _circular_deg_dist_idx(i: torch.Tensor, j: torch.Tensor, K: int) -> torch.Tensor:
    """
    i, j: integer bin indices (any shape; will be broadcast)
    Return the minimum angular distance in degrees under 0..2π convention.
    """
    diff = torch.abs(i - j)
    step_deg = 360.0 / float(K)
    wrap = torch.minimum(diff, K - diff).to(torch.float32)
    return wrap * step_deg

@torch.no_grad()
def angular_metrics_single(
    logits: torch.Tensor,
    theta: torch.Tensor,        # [B,T] radians (primary/dominant)
    vad_mask: torch.Tensor,     # [B,T] bool OR float prob
    K: int = 72
) -> Dict[str, float]:
    """Single-angle metrics (dominant/primary source) on VAD=1 frames."""
    # decode
    theta_hat, _ = decode_angle_from_logits(logits)  # [B,T]
    # angular error (deg)
    diff = torch.atan2(torch.sin(theta - theta_hat), torch.cos(theta - theta_hat))
    ae_deg = diff.abs() * 180.0 / math.pi

    # robust VAD (accept bool or float probs)
    speech = vad_mask if vad_mask.dtype == torch.bool else (vad_mask >= 0.5)

    out: Dict[str, float] = {}
    if speech.any():
        ae_s = ae_deg[speech]
        out['mae_speech']   = float(ae_s.mean().item())
        out['median_speech'] = float(ae_s.median().item())
        for x in [5, 10, 15]:
            out[f'within_{x}_speech'] = float((ae_s <= x).float().mean().item())
    else:
        out['mae_speech'] = float('nan'); out['median_speech'] = float('nan')
        for x in [5,10,15]:
            out[f'within_{x}_speech'] = float('nan')

    # (debug) all-frames stats (includes non-speech)
    ae_all = ae_deg.flatten()
    out['mae_all']    = float(ae_all.mean().item())
    out['median_all'] = float(ae_all.median().item())
    for x in [5,10,15]:
        out[f'within_{x}_all'] = float((ae_all <= x).float().mean().item())
    return out

@torch.no_grad()
def angular_metrics_multi(
    logits: torch.Tensor,
    gt_multi: torch.Tensor,     # [B,T,K] 0/1 (already VAD-gated outside)
    vad_mask: torch.Tensor,     # [B,T] bool OR float prob (used only to skip frames)
    K: int,
    topk: int = 3,
    thresholds: List[int] = [5,10,15]
) -> Dict[str, float]:
    """
    Multi-source set recall@{1,2,3} within angle thresholds.
    For each voiced frame, take top-N predicted bins and count
    the fraction of GT sources matched within threshold.
    """
    B,T,K_ = gt_multi.shape
    assert K_ == K, f"K mismatch: gt={K_} vs {K}"

    # robust VAD (accept bool or float probs)
    speech = vad_mask if vad_mask.dtype == torch.bool else (vad_mask >= 0.5)

    probs = torch.softmax(logits, dim=-1)                # [B,T,K]
    topv, topi = probs.topk(k=min(topk, K), dim=-1)      # [B,T,topk]

    gt_bins = (gt_multi > 0).to(torch.int64)
    buckets: Dict[str, List[float]] = {f'multi/recall@{n}_{thr}deg': [] for n in [1,2,3] for thr in thresholds}

    for b in range(B):
        for t in range(T):
            if not speech[b, t]:
                continue
            gt_idx = torch.nonzero(gt_bins[b, t], as_tuple=False).flatten()  # [G]
            if gt_idx.numel() == 0:
                continue
            pred_idx = topi[b, t]  # [topk]
            for n in [1, 2, 3]:
                nn = min(n, pred_idx.numel())
                sel = pred_idx[:nn]  # [nn]
                # distance matrix [nn, G]
                dmat = _circular_deg_dist_idx(sel.unsqueeze(-1), gt_idx.unsqueeze(0), K)
                # best prediction for each GT (min over nn)
                dmin = dmat.min(dim=0).values  # [G]
                for thr in thresholds:
                    buckets[f'multi/recall@{n}_{thr}deg'].append(float((dmin <= thr).float().mean().item()))

    out = {k: (float(np.mean(v)) if len(v) else float('nan')) for k, v in buckets.items()}
    return out
# ---------------- Checkpointing ----------------

def save_checkpoint(state: Dict[str, Any], path: Union[str, Path]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location='cpu', weights_only=True)  # PyTorch ≥ 2.4
    except TypeError:
        return torch.load(path, map_location='cpu')

def make_run_dir(save_root: str, model_key: str, config: Dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config_str = yaml.dump(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    run_dir = Path(save_root) / model_key / f"{timestamp}-{config_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return run_dir

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- Data factory ----------------

def create_dataloader_from_config(config: Dict[str, Any], split: str = 'train') -> torch.utils.data.DataLoader:
    dataset_config   = config.get('dataset', {})
    features_config  = config.get('features', {})
    use_precomputed  = bool(config.get('use_precomputed', False))
    precomputed_path = config.get('precomputed_path', None)
    batch_size       = int(config.get('batch_size', 8))
    num_workers      = int(config.get('dataloader', {}).get('num_workers', 0))
    pin_memory       = bool(config.get('dataloader', {}).get('pin_memory', False))

    # If using precomputed shards, build the precomputed loader
    if use_precomputed and precomputed_path:
        return create_precomputed_dataloader(
            precomp_root=precomputed_path,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=(num_workers > 0),
        )

    # ---- Online (on-the-fly) path
    # mic geometry (meters): same as before
    mic_positions = np.array([
        [ 0.0277,  0.0000],
        [ 0.0000,  0.0277],
        [-0.0277,  0.0000],
        [ 0.0000, -0.0277],
    ], dtype=float)

    # base synthetic mixture dataset
    dataset = OnTheFlyMixtureDataset(
        rir_root=dataset_config.get('rir_root', '/path/to/rir_bank'),
        split=split,
        speech_root=dataset_config.get('speech_root', '/path/to/LibriSpeech'),
        local_noises_root=dataset_config.get('local_root', '/path/to/noise'),
        ambiences_root=dataset_config.get('amb_root', '/path/to/Ambiances'),
        epoch_size=dataset_config.get('epoch_size', 4000),
        base_seed=dataset_config.get('seed', 56),
    )

    # feature & window configs
    stft_cfg = STFTCfg(
        sr=features_config.get('sr', 16000),
        win_s=features_config.get('win_s', 0.032),
        hop_s=features_config.get('hop_s', 0.010),
        nfft=features_config.get('nfft', 512),
        center=bool(features_config.get('center', False)),
        window="hann",
        pad_mode="reflect",
    )
    feat_cfg = FeatCfg(
        K=features_config.get('K', 72),
        max_frames=features_config.get('max_frames', 64),
        vad_threshold=features_config.get('vad_threshold', 0.6),
        vad_gt_masking=features_config.get('vad_gt_masking', True),
        use_fp16=features_config.get('use_fp16', False),
    )

    # build online feature dataloader (windowed)
    loader = create_online_feature_loader(
        base_ds=dataset,
        mic_xy=mic_positions,
        stft_cfg=stft_cfg,
        feat_cfg=feat_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        windows_per_mix=int(config.get('windows_per_mix', 1)),
        window_strategy=str(config.get('window_strategy', 'random')),
        pin_memory=pin_memory,
        drop_last=True,
        verbose_iter=False,
    )
    return loader

# ---------------- Loss wrapper (multi-source aware) ----------------


def safe_bce_from_probs(
    probs: torch.Tensor,      # [B,K], already sigmoid(...)
    y: torch.Tensor,          # [B,K] in {0,1}
    *,
    row_balance: bool = True,     # weight positives ~ (#neg/#pos) per row
    pos_boost: float = 1.0,       # >1 => extra weight to positives
    fn_alpha: float = 0.0,        # >0 => extra penalty for false negatives
    reduction: str = "mean",      # "mean" or "sum"
    eps: float = 1e-7
) -> torch.Tensor:
    # do BCE in fp32, outside autocast
    with torch.amp.autocast(device_type='cuda', enabled=False):
        p = probs.float().clamp(eps, 1 - eps)
        t = y.float()

        # per-element BCE (on probabilities)
        per_elem = -(t * torch.log(p) + (1. - t) * torch.log1p(-p))  # [B,K]

        if not row_balance:
            if reduction == "sum":
                return per_elem.sum()
            return per_elem.mean()

        keep = (t.sum(1) > 0)
        if not keep.any():
            return (probs * 0.).sum()  # grad-safe 0

        p = p[keep]; t = t[keep]; per_elem = per_elem[keep]
        K = t.size(1)

        pos_cnt = t.sum(1, keepdim=True).clamp(min=1.0)          # [b,1]
        pos_w   = ((K - pos_cnt) / pos_cnt) * pos_boost          # [b,1]

        if fn_alpha > 0:
            pos_w = pos_w * (1.0 - p) * fn_alpha + pos_w         # FN emphasis

        w = torch.where(t > 0, pos_w, torch.ones_like(t))        # [b,K]

        if reduction == "sum":
            return (per_elem * w).sum()
        # mean: normalize per row then average rows
        per_row = (per_elem * w).sum(1) / (w.sum(1) + eps)
        return per_row.mean()
def multipeak_ce_with_rank(
    logits: torch.Tensor,   # [B, 72] raw scores from a *linear* head (no sigmoid)
    y_max: torch.Tensor,    # [B, 72] in {0,1}, 1–3 ones per row
    *,
    temperature: float = 0.7,     # sharpen CE; 0.6–0.9 usually fine
    label_smoothing: float = 0.02,
    rank_margin: float = 0.3,     # soft hinge margin for pos>neg
    alpha_rank: float = 0.5,      # 0.3–1.0: weight of the ranking term
    eps: float = 1e-8,
):
    y = y_max.to(logits.dtype)
    keep = (y.sum(1) > 0)
    if not keep.any():
        # grad-safe zero (returns a scalar with grad)
        return (logits * 0.).sum()

    l = logits[keep] / temperature         # [b,72]
    t = y[keep]
    K = t.size(1)

    # ---------- (1) Multi-peak soft target CE ----------
    pos_cnt = t.sum(1, keepdim=True).clamp_min(1.0)   # [b,1]
    tgt = t / pos_cnt                                 # uniform over active bins
    if label_smoothing > 0:
        tgt = (1.0 - label_smoothing) * tgt + label_smoothing / K

    logp = F.log_softmax(l, dim=1)
    ce = -(tgt * logp).sum(1).mean()

    # ---------- (2) Smooth pairwise ranking ----------
    # Encourages every positive logit to exceed every negative by 'rank_margin'
    rank_loss = l.new_zeros(())
    b = l.size(0)
    pos_mask = t.bool()
    neg_mask = ~pos_mask
    for i in range(b):
        if pos_mask[i].any() and neg_mask[i].any():
            lp = l[i, pos_mask[i]].unsqueeze(1)   # [P,1]
            ln = l[i, neg_mask[i]].unsqueeze(0)   # [1,N]
            # soft hinge: softplus(margin - (lp - ln))
            rank_loss = rank_loss + F.softplus(rank_margin - (lp - ln)).mean()
    rank_loss = rank_loss / b

    return ce + alpha_rank * rank_loss


def bce_softf1_count(
    logits: torch.Tensor,   # [B, 72] raw
    y: torch.Tensor,        # [B, 72] in {0,1}
    *,
    beta: float = 2.0,          # >1 => recall emphasis (harder on FN)
    alpha_f1: float = 0.5,      # weight of soft-F1 term
    alpha_count: float = 0.05,  # weight of count prior
    pos_boost: float = 1.0,     # optional global boost on positives
    clamp_posw: tuple = (1.0, 8.0),  # clamp for per-row pos weights
    eps: float = 1e-8,
):
    y = y.to(logits.dtype)
    keep = (y.sum(1) > 0)
    if not keep.any():
        # gradient-safe zero
        return (logits * 0.).sum()

    l = logits[keep]
    t = y[keep]
    B, K = t.shape

    # --- 1) BCEWithLogits + per-row positive weighting (clamped)
    pos_cnt = t.sum(1, keepdim=True).clamp(min=1.0)        # [b,1]
    # weight ~ (#neg / #pos) but clamped for stability
    w_pos = ((K - pos_cnt) / pos_cnt).clamp(*clamp_posw) * pos_boost
    w = torch.where(t > 0, w_pos, torch.ones_like(t))
    bce = F.binary_cross_entropy_with_logits(l, t, weight=w)

    # --- 2) Soft F1 / F-beta (recall-focused, differentiable)
    p = torch.sigmoid(l)
    tp = (p * t).sum(1)
    fp = (p * (1 - t)).sum(1)
    fn = ((1 - p) * t).sum(1)
    beta2 = beta * beta
    fbeta = (1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp + eps)
    soft_f1_loss = 1.0 - fbeta.mean()

    # --- 3) Count prior (keeps predicted #ones near truth per row)
    pred_cnt = p.sum(1)                # in [0, K]
    true_cnt = t.sum(1)                # in {1,2,3}
    count_loss = F.mse_loss(pred_cnt, true_cnt)

    return bce + alpha_f1 * soft_f1_loss + alpha_count * count_loss
#################################

def compute_loss(model_outputs: Any,
                 global_step : int,
                 vad_mask: torch.Tensor,            # [B,T] bool/0-1
                 srp: torch.Tensor,                   # [B,T,K]
                 model_name: str,
                 config: Dict[str, Any],
                 target_dist: Optional[torch.Tensor] = None  # [B,T,K] soft target for CE
                 ) -> torch.Tensor:
    """
    Route to the right loss. If target_dist is given, CE uses it (multi-source).
    """
    loss_cfg = config.get('loss', {})
    K = int(loss_cfg.get('K', 72))
    sigma_deg = float(loss_cfg.get('sigma_deg', 7.5))
    tau_srp = float(loss_cfg.get('tau_srp', 1.0))
    eps_srp = float(loss_cfg.get('eps_srp', 0.02))
    w_tv = float(loss_cfg.get('w_tv', 0.0))

    # If y_soft provided, CE will consume it; otherwise CE builds from theta.
    theta_or_target = target_dist #if target_dist is not None else theta_or_primary

    if model_name == "scat":
        return scat_loss(
            model_outputs, theta_or_target, vad_mask, srp,
            K=K, sigma_deg=sigma_deg,
            w_ce=loss_cfg.get('w_ce', 1.0),
            w_vmf=loss_cfg.get('w_vmf', 0.3),
            w_srp=loss_cfg.get('w_srp', 0.6),
            w_quiet=loss_cfg.get('w_quiet', 0.2),
            w_tv=w_tv, tau_srp=tau_srp, eps_srp=eps_srp
        )

    elif model_name == "film":


        #theta_or_target = torch.mean(theta_or_target, 1)
        #print(theta_or_target.shape)
        #print(model_outputs.shape)
        return crit(
            model_outputs, theta_or_target)  
        # return film_loss(
        #     model_outputs, theta_or_target, vad_mask, srp,
        #     K=K, sigma_deg=sigma_deg,
        #     w_ce=loss_cfg.get('w_ce', 1.0),
        #     w_vmf=loss_cfg.get('w_vmf', 0.2),
        #     w_srp=loss_cfg.get('w_srp', 0.5),
        #     w_quiet=loss_cfg.get('w_quiet', 0.2),
        #     w_tv=w_tv, tau_srp=tau_srp, eps_srp=eps_srp
        # )

    elif model_name == "retin":
        if isinstance(model_outputs, tuple):
            logits, delta, _ = model_outputs
        else:
            raise ValueError("ReTiN model should return (logits, delta, hT) tuple")
        return retin_loss(
            logits, delta, theta_or_target, vad_mask, srp,
            K=K, sigma_deg=sigma_deg,
            w_ce=loss_cfg.get('w_ce', 1.0),
            w_delta=loss_cfg.get('w_delta', 0.3),
            w_srp=loss_cfg.get('w_srp', 0.5),
            w_quiet=loss_cfg.get('w_quiet', 0.2),
            w_tv=w_tv, tau_srp=tau_srp, eps_srp=eps_srp,
            max_delta_deg=loss_cfg.get('max_delta_deg', 15.0)
        )
    


    elif model_name == "basic":

        #  import matplotlib.pyplot as plt
        #  plt.plot(theta_or_target[0,:].detach().cpu().numpy())
        #  plt.show()
        #print(theta_or_target)
        loss = safe_bce_from_probs(
        model_outputs, target_dist)



        # loss = asymmetric_row_balanced_bce(
        #         model_outputs, (target_dist > 0).float(),
        #         gamma_pos=0.0,
        #         gamma_neg=4.0,
        #         neg_margin=0.05,
        #         pos_boost=1.2,     # small global push for positives
        #         fn_alpha=1.0       # extra FN penalty; try 0.5–2.0
        #     )
        # loss =   bce_rows_rowbalanced(
        #         logits=model_outputs,
        #         y=target_dist,
        #         pos_boost=2,   # try 1.2–2.0
        #         fn_alpha=2     # try 0.5–2.0; set 0 to disable
        #     ) 


        #print(loss_per_class)          
        return  loss 
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_model_outputs_for_metrics(model_outputs: Any, model_name: str) -> torch.Tensor:
    """Extract logits from model outputs for metrics computation."""
    if model_name in ["scat", "film", "basic"]:
        if isinstance(model_outputs, tuple):
            logits, _ = model_outputs
        else:
            logits = model_outputs
        return logits
    elif model_name == "retin":
        if isinstance(model_outputs, tuple):
            logits, _, _ = model_outputs
        else:
            raise ValueError("ReTiN model should return (logits, delta, hT) tuple")
        return logits
    else:
        raise ValueError(f"Unknown model: {model_name}")

