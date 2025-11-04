import torch, torch.nn as nn, torch.nn.functional as F
from torch.special import i0e  # numerically stable I0
import math 




# ----- von Mises negative log-likelihood (aux regression) -----
def von_mises_nll(theta_gt, mu, kappa):
    """mu: [B,T,2] unit (cos,sin), kappa: [B,T,1]>=0"""
    t = torch.stack([torch.cos(theta_gt), torch.sin(theta_gt)], dim=-1)  # [B,T,2]
    cos_delta = (t * mu).sum(dim=-1, keepdim=True)
    # log I0(kappa) = log(i0e(kappa)) + kappa (stabilized)
    return (-kappa * cos_delta + (torch.log(i0e(kappa)) + kappa)).mean()



# ----- SRP distillation (curvature-weighted) -----
def kld_srp(logits, srp, vad, weight=None):
    """
    KL(q||p) where q=SRP posterior (normalized), p=model softmax.
    logits: [B,T,K], srp: [B,T,K], vad: [B,T]
    weight: optional reliability weight, accepted shapes: [B,T] or [B,T,1]
    returns scalar loss
    """
    q = (srp / (srp.sum(dim=-1, keepdim=True) + 1e-8)).clamp_min(1e-8)   # [B,T,K]
    p = F.softmax(logits, dim=-1).clamp_min(1e-8)                         # [B,T,K]
    kl = (q * (q.log() - p.log())).sum(dim=-1)                            # [B,T]

    w = vad.float()                                                       # [B,T]
    if weight is not None:
        # squeeze trailing singleton if present
        if weight.dim() == 3 and weight.size(-1) == 1:
            weight = weight.squeeze(-1)                                   # [B,T]
        # if someone passed [T,B], fix it
        if weight.shape == (w.shape[1], w.shape[0]):
            weight = weight.transpose(0, 1)                                # [B,T]
        w = w * weight                                                    # [B,T]

    denom = w.sum().clamp_min(1e-8)
    return (kl * w).sum() / denom


# ----- Quiet loss: penalize confident predictions when VAD=0 -----
def quiet_entropy_loss(logits, vad, tau=0.6):
    """Encourage high entropy (low confidence) when VAD=0."""
    p = F.softmax(logits, dim=-1).clamp_min(1e-8)
    H = -(p * p.log()).sum(dim=-1)             # [B,T]
    Hn = H / (torch.log(torch.tensor(p.size(-1), device=logits.device)))
    w = (1.0 - vad.float())
    pen = torch.relu(tau - Hn)                 # want entropy >= tau
    return (pen * w).sum() / (w.sum() + 1e-8 + (w.sum()==0).float())

# ----- Pair-consistency (optional, low weight) -----
def pair_consistency_loss(theta_pred, ipd_pairs_lowF, d_meters, fs=16000, freqs=None, w=None):
    """
    Enforce that predicted azimuth implies TDOA consistent with observed low-F IPD.
    ipd_pairs_lowF: list of tensors per pair, each [B,T,F_low] phase in radians
    d_meters: list of pair spacings (m) in same order
    freqs: [F_low] frequencies (Hz). If None, assumes linear up to 4k.
    theta_pred: [B,T] radians (decoded).
    """
    if freqs is None:
        F_low = ipd_pairs_lowF[0].size(-1)
        freqs = torch.linspace(50, 4000, F_low, device=theta_pred.device)
    c = 343.0
    loss = 0.0; count = 0
    for ipd, d in zip(ipd_pairs_lowF, d_meters):
        tdoa = (d * torch.cos(theta_pred)) / c  # [B,T]
        exp_ipd = (2*torch.pi*tdoa.unsqueeze(-1)*freqs)       # [B,T,F]
        # wrap error to [-pi,pi]
        err = torch.atan2(torch.sin(ipd - exp_ipd), torch.cos(ipd - exp_ipd)).abs()
        l = err.mean()
        loss = loss + l; count += 1
    return loss / max(count,1)

# ----- decode and confidence -----



def _wrap_angle(x: torch.Tensor) -> torch.Tensor:
    # wrap to (-pi, pi]
    return (x + math.pi) % (2 * math.pi) - math.pi

@torch.no_grad()
def decode_angle_from_logits(
    logits: torch.Tensor,   # [B,T,K] or [B,K] raw logits
    K: int,
    method: str = "argmax", # "argmax", "circ_mean", or "topm_mean"
    temp: float = 0.75,     # sharpening for "circ_mean"
    topm: int = 3           # top-M for "topm_mean"
):
    """
    Returns:
      theta_hat: [B,T] radians in [0, 2π)
      conf:      [B,T] confidence in [0,1]
    Notes:
      - Bin centers at (k + 0.5) * 2π/K (fixes 0.5-bin bias)
      - "argmax": pick center of top bin by logits
      - "circ_mean": circular mean of softmax(logits/temp)
      - "topm_mean": circular mean over top-M (by sigmoid) with normalized weights
      - Confidence: resultant length R for mean-based methods; max prob for argmax
    """
    squeeze_T = False
    if logits.dim() == 2:           # [B,K] -> [B,1,K]
        logits = logits.unsqueeze(1)
        squeeze_T = True

    B, T, KK = logits.shape
    assert KK == K, f"K mismatch: logits={KK} vs K={K}"

    step = (2.0 * math.pi) / K
    # Bin centers
    k = torch.arange(K, device=logits.device, dtype=logits.dtype) + 0.5
    thetas = step * k  # [K], centers in [0, 2π)

    if method == "argmax":
        idx = logits.argmax(dim=-1)               # [B,T]
        theta_hat = thetas[idx]                   # [B,T]
        # confidence from softmax(logits/temp)
        p = F.softmax(logits / temp, dim=-1)      # [B,T,K]
        conf = p.max(dim=-1).values               # [B,T]

    elif method == "circ_mean":
        p = F.softmax(logits / temp, dim=-1)      # [B,T,K]
        c = (p * torch.cos(thetas)).sum(dim=-1)   # [B,T]
        s = (p * torch.sin(thetas)).sum(dim=-1)   # [B,T]
        theta_hat = torch.atan2(s, c).remainder(2.0 * math.pi)
        # mean resultant length (0..1)
        R = torch.sqrt(c*c + s*s).clamp(0, 1)
        conf = R

    elif method == "topm_mean":
        q = torch.sigmoid(logits)                 # [B,T,K], multi-label scores
        m = min(topm, K)
        topv, topi = q.topk(k=m, dim=-1)          # [B,T,m]
        # normalize weights over top-M
        w = topv / (topv.sum(dim=-1, keepdim=True) + 1e-8)
        th_sel = thetas.view(1,1,-1).expand(B,T,K).gather(-1, topi)  # [B,T,m]
        c = (w * torch.cos(th_sel)).sum(dim=-1)
        s = (w * torch.sin(th_sel)).sum(dim=-1)
        theta_hat = torch.atan2(s, c).remainder(2.0 * math.pi)
        # confidence = resultant length using top-M weights
        R = torch.sqrt(c*c + s*s).clamp(0, 1)
        conf = R
    else:
        raise ValueError(f"Unknown method: {method}")

    if squeeze_T:
        theta_hat = theta_hat.squeeze(1)
        conf = conf.squeeze(1)
    return theta_hat, conf



def angle_to_bin(theta: torch.Tensor, K: int) -> torch.Tensor:
    """theta in radians (any range) -> bin index in [0..K-1] under 0..2π convention."""
    import math, torch
    step = (2.0 * math.pi) / K
    return torch.remainder(torch.round((theta % (2.0 * math.pi)) / step), K).long()

def bin_to_angle(idx: torch.Tensor, K: int, device=None, dtype=None) -> torch.Tensor:
    """bin index [0..K-1] -> center angle in [0, 2π)."""
    import math, torch
    step = (2.0 * math.pi) / K
    return (idx.to(dtype or torch.float32) * step).to(device)  # in [0, 2π)