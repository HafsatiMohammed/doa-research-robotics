from .utils_model import *
import numpy as np
import math
import torch
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional

def _zero_like(x: torch.Tensor) -> torch.Tensor:
    return x.new_zeros(())

def _safe_voiced_mean(x: torch.Tensor, vad_mask: torch.Tensor) -> torch.Tensor:
    """Mean on frames where vad_mask==True; return exact 0 if none."""
    m = vad_mask.bool()
    denom = m.sum()
    if denom.item() == 0:
        return _zero_like(x)
    return x[m].mean()

def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap to (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))

# --------------------- SRP confidence & loss ---------------------

@torch.no_grad()
def srp_confidence(srp: torch.Tensor, a: float = 0.7, b: float = 2.0) -> torch.Tensor:
    """
    Frame-wise SRP reliability in (0,1), combining PSR and discrete curvature.
    srp: [B,T,K] nonnegative
    return: [B,T,1]
    """
    q = srp / (srp.sum(dim=-1, keepdim=True) + 1e-8)
    vmax, _ = q.max(dim=-1, keepdim=True)
    vmean = (q.sum(-1, keepdim=True) - vmax) / (q.size(-1) - 1)
    vstd = (q - vmean).clamp_min(0).std(dim=-1, keepdim=True) + 1e-8
    psr = (vmax - vmean) / vstd

    q_rollp = torch.roll(q, 1, dims=-1)
    q_rolln = torch.roll(q, -1, dims=-1)
    curv = (q_rollp - 2 * q + q_rolln).abs().mean(dim=-1, keepdim=True)

    conf_psr = torch.sigmoid(a * (psr - b))
    curv_norm = (curv / (curv.median() + 1e-8)).clamp(0, 1)
    return 0.5 * conf_psr + 0.5 * curv_norm

def srp_ce_loss(logits: torch.Tensor,
                srp: torch.Tensor,
                vad_mask: torch.Tensor,
                tau_model: float = 1.0,
                tau_srp: float = 1.0,
                eps: float = 0.02,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Cross-entropy to SRP soft targets (tempered + epsilon-smoothed).
    logits: [B,T,K], srp: [B,T,K], vad_mask: [B,T]
    """
    B, T, K = logits.shape
    logp = F.log_softmax(logits / tau_model, dim=-1)
    q = srp / (srp.sum(-1, keepdim=True) + 1e-8)
    q = F.softmax(torch.log(q + 1e-12) / tau_srp, dim=-1)
    q = (1 - eps) * q + eps / K

    ce = -(q * logp).sum(dim=-1)  # [B,T]
    if weight is not None:
        if weight.dim() == 3 and weight.size(-1) == 1:
            weight = weight.squeeze(-1)
        ce = ce * weight

    return _safe_voiced_mean(ce, vad_mask)

# --------------------- temporal regularizer ---------------------

def temporal_tv(theta_hat: torch.Tensor,
                vad_mask: torch.Tensor,
                deg_limit_per_s: float = 90.0,
                fps: float = 100.0) -> torch.Tensor:
    """
    Penalize inter-frame angular changes exceeding a physical speed prior.
    theta_hat: [B,T] in radians
    vad_mask: [B,T] bool/0-1
    """
    lim = math.radians(deg_limit_per_s / fps)
    d = _wrap_angle(torch.roll(theta_hat, -1, dims=-1) - theta_hat).abs()
    m = vad_mask.bool() & torch.roll(vad_mask.bool(), -1, dims=-1)
    if m.sum().item() == 0:
        return _zero_like(theta_hat)
    excess = (d - lim).clamp_min(0.0)
    return excess[m].mean()

# --------------------- circular CE (supports soft targets) ---------------------

def ce_circular_loss(
    logits: torch.Tensor,                  # [B,T,K]
    theta_or_target: torch.Tensor,         # [B,T] (angles) OR [B,T,K] (soft target)
    vad_mask: torch.Tensor,                # [B,T] (bool/float)
    K: int,
    sigma_deg: float = None
) -> torch.Tensor:
    """
    If theta_or_target.ndim==2: build a wrapped Gaussian target around theta (old behavior, now 0..2π grid).
    If theta_or_target.ndim==3: treat as a soft target distribution y_soft (new behavior).
    """
    logp = torch.log_softmax(logits, dim=-1)              # [B,T,K]
    m = vad_mask.bool().unsqueeze(-1)                     # [B,T,1]

    sigma = (sigma_deg or 10.0) * math.pi / 180.0
    k = torch.arange(K, device=logits.device).float()
    grid = (k * (2.0 * math.pi / K)).view(1, 1, K)  # [1,1,K] in [0,2π)

    # theta_or_target can be [B,T] (angles) or [B,T,K] (soft target)
    if theta_or_target.dim() == 3:
        y_soft = theta_or_target.clamp_min(1e-8)     # already normalized
    else:
        theta = torch.remainder(theta_or_target, 2.0 * math.pi)   # [B,T] in [0,2π)
        # circular wrapped distance via atan2(sin Δ, cos Δ)
        d = torch.atan2(torch.sin(grid - theta.unsqueeze(-1)),
                        torch.cos(grid - theta.unsqueeze(-1)))
        y_soft = torch.exp(-0.5 * (d / sigma) ** 2)
        y_soft = y_soft / y_soft.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    nll = -(y_soft * logp)
    if m.sum().item() == 0:
        return logits.new_zeros(())
    return nll[m.expand_as(nll)].mean()


# ---------------- core losses ----------------

def scat_loss(outputs,
              theta_or_target, vad_mask, srp,
              K=72, sigma_deg=7.5,
              w_ce=1.0, w_vmf=0.3, w_srp=0.6, w_quiet=0.2,
              w_tv: float = 0.0,
              tau_srp: float = 1.0,
              eps_srp: float = 0.02):
    if isinstance(outputs, tuple):
        logits, (mu, kappa) = outputs
    else:
        logits = outputs; mu = kappa = None

    assert logits.size(-1) == K, f"logits K={logits.size(-1)} != {K}"

    # Base CE (accepts either angles or soft target [B,T,K])
    L_ce = ce_circular_loss(logits, theta_or_target, vad_mask, K=K, sigma_deg=sigma_deg)

    # vMF head (optional; only meaningful if angles provided)
    if (mu is not None) and (kappa is not None) and (theta_or_target.dim() == 2):
        L_vmf = von_mises_nll(theta_or_target, mu, kappa.clamp_min(1e-3))
    else:
        L_vmf = _zero_like(logits)

    # SRP teacher (tempered CE) with confidence weighting
    conf = srp_confidence(srp)                                    # [B,T,1]
    L_srp = srp_ce_loss(logits, srp, vad_mask, tau_model=1.0, tau_srp=tau_srp, eps=eps_srp, weight=conf)

    # Quiet entropy (encourage high-entropy on unvoiced)
    L_quiet = quiet_entropy_loss(logits, vad_mask, tau=0.6)

    # Optional temporal smoothness
    theta_hat, _ = decode_angle_from_logits(logits.detach())      # [B,T]
    L_tv = temporal_tv(theta_hat, vad_mask) if w_tv > 0 else _zero_like(logits)

    total = w_ce * L_ce + w_vmf * L_vmf + w_srp * L_srp + w_quiet * L_quiet + w_tv * L_tv
    return total

def film_loss(outputs,
              theta_or_target, vad_mask, srp,
              K=72, sigma_deg=7.5,
              w_ce=1.0, w_vmf=0.2, w_srp=0.5, w_quiet=0.2,
              w_tv: float = 0.0,
              tau_srp: float = 1.0,
              eps_srp: float = 0.02):
    if isinstance(outputs, tuple):
        logits, (mu, kappa) = outputs
    else:
        logits = outputs; mu = kappa = None

    assert logits.size(-1) == K, f"logits K={logits.size(-1)} != {K}"

    #print(theta_or_target.shape)
    #print(theta_or_target)

    L_ce = ce_circular_loss(logits, theta_or_target, vad_mask, K=K, sigma_deg=sigma_deg)
    
    L_quiet = quiet_entropy_loss(logits, vad_mask, tau=0.6)
    L_srp = srp_ce_loss(logits, srp, vad_mask, tau_model=1.0, tau_srp=tau_srp, eps=eps_srp,
                        weight=srp_confidence(srp))

    if (mu is not None) and (kappa is not None) and (theta_or_target.dim() == 2):
        L_vmf = von_mises_nll(theta_or_target, mu, kappa.clamp_min(1e-3))
    else:
        L_vmf = _zero_like(logits)

    theta_hat, _ = decode_angle_from_logits(logits.detach())
    L_tv = temporal_tv(theta_hat, vad_mask) if w_tv > 0 else _zero_like(logits)

    return w_ce * L_ce + w_vmf * L_vmf +  L_srp + w_quiet * L_quiet + w_tv * L_tv

def retin_loss(logits,
               delta,
               theta_or_target, vad_mask, srp,
               K=72, sigma_deg=7.5,
               w_ce=1.0, w_delta=0.3, w_srp=0.5, w_quiet=0.2,
               w_tv: float = 0.0,
               tau_srp: float = 1.0,
               eps_srp: float = 0.02,
               max_delta_deg: float = 15.0):
    assert logits.size(-1) == K, f"logits K={logits.size(-1)} != {K}"

    L_ce = ce_circular_loss(logits, theta_or_target, vad_mask, K=K, sigma_deg=sigma_deg)
    L_quiet = quiet_entropy_loss(logits, vad_mask, tau=0.6)
    L_srp = srp_ce_loss(logits, srp, vad_mask, tau_model=1.0, tau_srp=tau_srp, eps=eps_srp,
                        weight=srp_confidence(srp))

    # Δθ refiner: bound via tanh and use smooth periodic loss 1 - cos(err)
    theta_hat, _ = decode_angle_from_logits(logits.detach())      # [B,T]
    if theta_or_target.dim() == 2:
        theta_primary = theta_or_target
    else:
        # If soft targets used, anchor residual to current estimate
        theta_primary = theta_hat.detach()

    residual = _wrap_angle(theta_primary - theta_hat)             # [B,T]
    delta_b = math.radians(max_delta_deg) * torch.tanh(delta).squeeze(-1)  # [B,T]
    err = _wrap_angle(residual - delta_b)
    L_delta = _safe_voiced_mean(1.0 - torch.cos(err), vad_mask)

    L_tv = temporal_tv(theta_hat, vad_mask) if w_tv > 0 else _zero_like(logits)

    return w_ce * L_ce + w_quiet * L_quiet + w_srp * L_srp + w_delta * L_delta + w_tv * L_tv

