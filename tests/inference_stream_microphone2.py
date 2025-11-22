#!/usr/bin/env python3
"""
Real-time DOA (histogram-based, speech-aware) with IMPULSE-READY burst mode:
- Onset detector (spectral flux z-score) + level jump + mic coherence
- Burst mode focuses on last frames or single best frame, with energy weighting and sharper softmax
- Event gate with hysteresis, hold, and refractory to reduce false positives
- Visualization: lines toward current DOA(s) only (Green=speech, Orange=non-speech)
- Up to 3 sources per class

Recommended impulse-friendly CLI:
  --window-ms 160 --hop-ms 80 \
  --onset-z-on 2.2 --onset-z-off 1.2 --coh-min 0.55 \
  --burst-on-db 3.0 --burst-hold-ms 300 \
  --burst-tail-frames 3 --burst-peak-frame-only \
  --burst-tau 0.65 --burst-gamma 1.0 --burst-smooth-k 0 \
  --burst-min-peak-height 0.03 --burst-min-window-mass 0.04
"""

import sys
from pathlib import Path

# Add src directory to Python path so mirokai_doa can be imported
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import math
import time
import queue
import argparse
from typing import Optional, Dict, List, Iterable, Tuple

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pyaudio

# ===== Try to import your InferenceModel =====
try:
    from tests.test_inference import InferenceModel
except ImportError:
    try:
        from .test_inference import InferenceModel
    except ImportError:
        tests_dir = Path(__file__).parent
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))
        from test_inference import InferenceModel


# -------------------------
# Small math helpers
# -------------------------
def _angles_deg(K: int, device=None):
    bin_size = 360.0 / K
    deg = (torch.arange(K, device=device, dtype=torch.float32) + 0.5) * bin_size
    rad = deg * math.pi / 180.0
    return deg, torch.cos(rad), torch.sin(rad), bin_size

def _softmax_temp(logits: torch.Tensor, tau: float = 0.8) -> torch.Tensor:
    return torch.softmax(logits / tau, dim=-1)

def _circular_window_sum(row: torch.Tensor, idx: int, half_w: int) -> float:
    K = row.numel()
    if half_w <= 0:
        return float(row[idx].item())
    acc = 0.0
    for d in range(-half_w, half_w + 1):
        acc += float(row[(idx + d) % K].item())
    return acc

def _parabolic_peak_refine(row: torch.Tensor, k: int) -> float:
    km1, kp1 = (k - 1) % row.numel(), (k + 1) % row.numel()
    y1, y2, y3 = row[km1].item(), row[k].item(), row[kp1].item()
    denom = (y1 - 2 * y2 + y3)
    if denom == 0:
        return 0.0
    delta = 0.5 * (y1 - y3) / denom
    return float(max(min(delta, 0.5), -0.5))

def _min_circ_separation_bins(a: int, chosen: List[int], K: int) -> int:
    if not chosen:
        return K
    dmin = K
    for j in chosen:
        d = abs(a - j)
        d = min(d, K - d)
        dmin = min(dmin, d)
    return dmin


# -------------------------
# SAFE GETTERS
# -------------------------
def _get_first(d: Dict, keys: Iterable[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _to_float_scalar(x) -> float:
    if x is None:
        return 0.0
    try:
        import numpy as _np
        if isinstance(x, (list, tuple)):
            return float(_np.mean(x))
        if isinstance(x, _np.ndarray):
            return float(x.mean())
        return float(x)
    except Exception:
        return 0.0


# -------------------------
# Audio helpers
# -------------------------
def byte_to_float(data: bytes) -> np.ndarray:
    samples = np.frombuffer(data, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0

def _chunk_to_floatarray(data: bytes, channels: int) -> np.ndarray:
    float_data = byte_to_float(data)
    return float_data.reshape(-1, channels).T

def rms_dbfs(x: np.ndarray, eps: float = 1e-9) -> float:
    val = np.sqrt((x * x).mean())
    return 20.0 * np.log10(max(val, eps))

def frame_rms_energy(audio_buffer: np.ndarray, T: int) -> torch.Tensor:
    """
    Split audio_buffer (C,N) into T equal segments; return per-frame RMS (normalized).
    """
    C, N = audio_buffer.shape
    if T <= 0:
        return torch.ones(1)
    edges = np.linspace(0, N, T + 1, dtype=int)
    e = []
    for i in range(T):
        seg = audio_buffer[:, edges[i]:edges[i+1]]
        if seg.size == 0:
            e.append(0.0)
        else:
            rms = np.sqrt((seg * seg).mean())
            e.append(rms)
    e = np.asarray(e, dtype=np.float32)
    e = e / max(e.mean(), 1e-6)
    return torch.from_numpy(e)

def spectral_flux_per_frame(audio_buffer: np.ndarray, T: int) -> np.ndarray:
    """
    Compute per-frame spectral flux across T segments from mono mix.
    Flux_t = sum(ReLU(|X_t| - |X_{t-1}|)) / (sum(|X_t|)+eps)
    """
    C, N = audio_buffer.shape
    if T <= 1:
        return np.zeros((T,), dtype=np.float32)
    mono = audio_buffer.mean(axis=0)  # average channels
    edges = np.linspace(0, N, T + 1, dtype=int)
    mags = []
    for i in range(T):
        seg = mono[edges[i]:edges[i+1]]
        if seg.size == 0:
            mags.append(np.zeros(1, dtype=np.float32))
            continue
        # rFFT magnitude
        win = np.hanning(len(seg)) if len(seg) > 8 else np.ones_like(seg)
        S = np.fft.rfft(seg * win, n=len(seg))
        mags.append(np.abs(S).astype(np.float32))
    # flux
    flux = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        a = mags[t-1]
        b = mags[t]
        L = min(len(a), len(b))
        if L == 0:
            flux[t] = 0.0
            continue
        diff = b[:L] - a[:L]
        pos = np.maximum(diff, 0.0)
        denom = np.sum(b[:L]) + 1e-6
        flux[t] = float(np.sum(pos) / denom)
    return flux


# -------------------------
# Onset: spectral-flux z-score + coherence
# -------------------------
class OnsetDetector:
    """
    Keeps EMA mean/var of spectral flux; returns z-score per window.
    Also computes inter-mic coherence on the last segment.
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.mu = 0.0
        self.var = 1.0  # avoid div-by-zero
        self.inited = False

    def update_flux(self, flux_recent: float) -> float:
        if not self.inited:
            self.mu = flux_recent
            self.var = 1e-3 + abs(flux_recent)
            self.inited = True
        # Welford-like EMA update for mean/var
        delta = flux_recent - self.mu
        self.mu += self.alpha * delta
        self.var = (1 - self.alpha) * self.var + self.alpha * delta * delta
        sigma = max(np.sqrt(self.var), 1e-6)
        z = (flux_recent - self.mu) / sigma
        return float(z)

    @staticmethod
    def last_segment_coherence(audio_buffer: np.ndarray, T: int, pairs: List[Tuple[int,int]] = [(0,1),(0,2),(0,3)]) -> float:
        """
        Pearson correlation on the LAST segment between channel pairs; return max |r|.
        """
        C, N = audio_buffer.shape
        if T < 1:
            return 0.0
        edges = np.linspace(0, N, T + 1, dtype=int)
        s0, s1 = int(edges[-2]), int(edges[-1])
        seg = audio_buffer[:, s0:s1]  # (C, L)
        if seg.shape[1] < 16:
            return 0.0
        rmax = 0.0
        for (i,j) in pairs:
            xi = seg[i] - seg[i].mean()
            xj = seg[j] - seg[j].mean()
            denom = (np.linalg.norm(xi) * np.linalg.norm(xj) + 1e-9)
            r = float(np.dot(xi, xj) / denom)
            rmax = max(rmax, abs(r))
        return rmax


# -------------------------
# Histogram DOA detector
# -------------------------
class HistDOADetector:
    def __init__(
        self,
        K: int = 72,
        vad_threshold: float = 0.5,
        tau: float = 0.8,
        gamma: float = 1.5,
        smooth_k: int = 1,
        window_bins: int = 1,
        min_peak_height: float = 0.10,
        min_window_mass: float = 0.24,
        min_sep_deg: float = 20.0,
        min_active_ratio: float = 0.20,
        max_sources: int = 3,
        device: str = "cpu",
        # impulse helpers
        recency_decay: Optional[float] = None,
        tail_frames: Optional[int] = None,
        energy_beta: float = 0.0,
        peak_frame_only: bool = False,
    ):
        self.K = int(K)
        self.vad_threshold = float(vad_threshold)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.smooth_k = int(smooth_k)
        self.window_bins = int(window_bins)
        self.min_peak_height = float(min_peak_height)
        self.min_window_mass = float(min_window_mass)
        self.min_sep_deg = float(min_sep_deg)
        self.min_active_ratio = float(min_active_ratio)
        self.max_sources = int(max_sources)
        self.device = torch.device(device)

        self.recency_decay = recency_decay
        self.tail_frames = None if (tail_frames is None or tail_frames <= 0) else int(tail_frames)
        self.energy_beta = float(energy_beta)
        self.peak_frame_only = bool(peak_frame_only)

        self._deg, self._cos, self._sin, self._bin_size = _angles_deg(self.K, device=self.device)


    @torch.no_grad()
    def detect_from_preaggregated(
        self,
        hist_speech: Optional[np.ndarray],
        hist_nonspeech: Optional[np.ndarray],
        hist_overall: Optional[np.ndarray],
        vad_mean: float = 0.0,
    ) -> Dict[str, any]:
        """
        Fallback when only pre-aggregated hist arrays are available.
        - If only a single 'hist_overall' is provided, route it to speech or non-speech
          based on vad_mean vs self.vad_threshold.
        Returns the same structure as detect(): speech/nonspeech dicts with peaks/R_clip/etc.
        """
        device, K = self.device, self.K

        def _prep(hn: Optional[np.ndarray]) -> Optional[torch.Tensor]:
            if hn is None:
                return None
            h = torch.from_numpy(hn).float().to(device)
            h = h / h.sum().clamp_min(1e-8)
            if self.smooth_k > 0:
                s = self.smooth_k
                pad = torch.cat([h[-s:], h, h[:s]], dim=0).view(1, 1, -1)
                kernel = torch.ones(1, 1, 2 * s + 1, device=device) / (2 * s + 1)
                h = F.conv1d(pad, kernel, padding=0).view(-1)
            return h

        sp_hist_t = _prep(hist_speech)
        ns_hist_t = _prep(hist_nonspeech)

        # If neither speech nor nonspeech provided but overall is, route by vad_mean
        if sp_hist_t is None and ns_hist_t is None and hist_overall is not None:
            h = _prep(hist_overall)
            if h is not None:
                if vad_mean >= self.vad_threshold:
                    sp_hist_t = h
                else:
                    ns_hist_t = h

        def _R_clip(h: Optional[torch.Tensor]) -> float:
            if h is None:
                return 0.0
            X = torch.dot(h, self._cos)
            Y = torch.dot(h, self._sin)
            return float(torch.sqrt(X * X + Y * Y).item())

        # Active ratios (1.0 if we actually got a hist for that class)
        sp_active = 1.0 if sp_hist_t is not None else 0.0
        ns_active = 1.0 if ns_hist_t is not None else 0.0

        # Peak picking
        sp_peaks = self._pick_peaks(sp_hist_t) if sp_hist_t is not None and sp_active >= self.min_active_ratio else []
        ns_peaks = self._pick_peaks(ns_hist_t) if ns_hist_t is not None and ns_active >= self.min_active_ratio else []

        bins_deg = (np.arange(K) + 0.5) * (360.0 / K)
        return {
            "speech": {
                "peaks": sp_peaks,
                "active_ratio": sp_active,
                "R_clip": _R_clip(sp_hist_t),
                "hist": None if sp_hist_t is None else sp_hist_t.detach().cpu().numpy(),
                "bins_deg": bins_deg,
            },
            "nonspeech": {
                "peaks": ns_peaks,
                "active_ratio": ns_active,
                "R_clip": _R_clip(ns_hist_t),
                "hist": None if ns_hist_t is None else ns_hist_t.detach().cpu().numpy(),
                "bins_deg": bins_deg,
            },
            "has_event": bool(sp_peaks or ns_peaks),
        }    

    @torch.no_grad()
    def _aggregate_histogram(self, logits: torch.Tensor, mask: torch.Tensor,
                             energies: Optional[torch.Tensor]):
        probs = _softmax_temp(logits.to(self.device), tau=self.tau)  # [T,K]
        T = probs.shape[0]
        m = mask.float().to(self.device)

        # Optional tail clipping
        if self.tail_frames is not None and self.tail_frames < T:
            start = T - self.tail_frames
            probs = probs[start:]
            m = m[start:]
            if energies is not None:
                energies = energies[start:]
            T = probs.shape[0]

        # Peak-frame-only path
        if self.peak_frame_only:
            x = torch.matmul(probs, self._cos)
            y = torch.matmul(probs, self._sin)
            R_t = torch.clamp(torch.sqrt(x * x + y * y), 0, 1)
            w = m.clone()
            if energies is not None and self.energy_beta > 0.0:
                e_norm = energies / energies.mean().clamp_min(1e-6)
                w = w * (e_norm ** self.energy_beta)
            score = R_t * w
            if score.sum() <= 0:
                hist = probs.mean(dim=0)
            else:
                t_star = int(torch.argmax(score).item())
                hist = probs[t_star]
            hist = hist / hist.sum().clamp_min(1e-8)
            X = torch.dot(hist, self._cos); Y = torch.dot(hist, self._sin)
            R_clip = float(torch.sqrt(X * X + Y * Y).item())
            active_ratio = float(m.mean().item())
            return hist, active_ratio, R_clip

        # Weighted histogram
        x = torch.matmul(probs, self._cos)
        y = torch.matmul(probs, self._sin)
        R_t = torch.clamp(torch.sqrt(x * x + y * y), 0, 1)
        w = m * (R_t ** self.gamma)

        if self.recency_decay is not None:
            t = torch.arange(T, device=w.device, dtype=torch.float32)
            decay = (self.recency_decay ** (T - 1 - t)).clamp_min(1e-6)
            w = w * decay

        if energies is not None and self.energy_beta > 0.0:
            e = energies.to(w.device).float()
            e_norm = e / e.mean().clamp_min(1e-6)
            w = w * (e_norm ** self.energy_beta)

        if w.sum() <= 0:
            w = torch.ones_like(w) * 1e-6

        hist = torch.matmul(w, probs)
        hist = hist / hist.sum().clamp_min(1e-8)

        if self.smooth_k > 0:
            s = self.smooth_k
            pad = torch.cat([hist[-s:], hist, hist[:s]], dim=0).view(1, 1, -1)
            kernel = torch.ones(1, 1, 2 * s + 1, device=self.device) / (2 * s + 1)
            hist = F.conv1d(pad, kernel, padding=0).view(-1)

        X = torch.dot(hist, self._cos)
        Y = torch.dot(hist, self._sin)
        R_clip = float(torch.sqrt(X * X + Y * Y).item())
        active_ratio = float(m.mean().item())
        return hist, active_ratio, R_clip

    @torch.no_grad()
    def _pick_peaks(self, hist: torch.Tensor) -> List[Dict[str, float]]:
        K = self.K
        bin_size = self._bin_size
        left = torch.roll(hist, 1, 0)
        right = torch.roll(hist, -1, 0)
        cand_idxs = ((hist > left) & (hist > right)).nonzero(as_tuple=False).flatten().tolist()
        cand_idxs.sort(key=lambda i: float(hist[i].item()), reverse=True)
        chosen, out = [], []
        min_sep_bins = max(1, int(round(self.min_sep_deg / bin_size)))
        for idx in cand_idxs:
            if _min_circ_separation_bins(idx, chosen, K) < min_sep_bins:
                continue
            if float(hist[idx].item()) < self.min_peak_height:
                continue
            mass = _circular_window_sum(hist, idx, self.window_bins)
            if mass < self.min_window_mass:
                continue
            delta = _parabolic_peak_refine(hist, idx)
            angle_deg = ((idx + 0.5 + delta) * bin_size) % 360.0
            out.append({"azimuth_deg": angle_deg, "score": float(mass)})
            chosen.append(idx)
            if len(out) >= self.max_sources:
                break
        return out

    @torch.no_grad()
    def detect(self, logits: torch.Tensor, vad_probs: torch.Tensor,
               energies: Optional[torch.Tensor] = None) -> Dict[str, any]:
        if isinstance(logits, np.ndarray): logits = torch.from_numpy(logits).float()
        if isinstance(vad_probs, np.ndarray): vad_probs = torch.from_numpy(vad_probs).float()
        logits, vad_probs = logits.to(self.device), vad_probs.to(self.device)
        speech_mask = (vad_probs >= self.vad_threshold)
        nonspeech_mask = ~speech_mask

        sp_hist, sp_active, sp_R = self._aggregate_histogram(logits, speech_mask, energies)
        ns_hist, ns_active, ns_R = self._aggregate_histogram(logits, nonspeech_mask, energies)

        sp_peaks = self._pick_peaks(sp_hist) if sp_active >= self.min_active_ratio else []
        ns_peaks = self._pick_peaks(ns_hist) if ns_active >= self.min_active_ratio else []
        bins_deg = (np.arange(self.K) + 0.5) * (360.0 / self.K)
        return {
            "speech": {"peaks": sp_peaks, "active_ratio": sp_active, "R_clip": sp_R,
                       "hist": sp_hist.detach().cpu().numpy(), "bins_deg": bins_deg},
            "nonspeech": {"peaks": ns_peaks, "active_ratio": ns_active, "R_clip": ns_R,
                          "hist": ns_hist.detach().cpu().numpy(), "bins_deg": bins_deg},
            "has_event": bool(sp_peaks or ns_peaks),
        }


# -------------------------
# Visualization
# -------------------------
class CurrentLineVisualizer:
    def __init__(self, title: str = "Current DOA (speech=green, non-speech=orange)"):
        self.fig = plt.figure(figsize=(7.5, 7.5))
        self.ax = self.fig.add_subplot(111, projection='polar')
        self._setup_axes(title)
        plt.ion()
        plt.show(block=False)

    def _setup_axes(self, title: str):
        ax = self.ax
        ax.clear()
        ax.set_title(title, fontsize=13, fontweight='bold', pad=16)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetalim(0, 2*np.pi)
        ax.set_ylim(0, 1.05)
        ax.set_yticklabels([])
        ax.add_patch(Circle((0, 0), 1.0, fill=False, color='gray', linestyle='--', linewidth=1, alpha=0.5))
        ax.grid(alpha=0.2)

    def update(self, speech_peaks: List[Dict], nonspeech_peaks: List[Dict]):
        self._setup_axes("Current DOA (speech=green, non-speech=orange)")

        def _draw(peaks: List[Dict], color: str):
            for pk in peaks[:3]:
                az = float(pk["azimuth_deg"])
                sc = float(pk.get("score", 0.2))
                lw = 2.0 + 5.0 * float(np.clip(sc, 0.0, 0.6))
                theta = np.deg2rad(az)
                self.ax.plot([theta, theta], [0.0, 1.0], color=color, linewidth=lw, solid_capstyle='round')
                self.ax.text(theta, 1.02, f"{az:.0f}°", ha='center', va='bottom', fontsize=10,
                             color=color, fontweight='bold')

        _draw(speech_peaks, "tab:green")
        _draw(nonspeech_peaks, "orange")

        self.ax.plot([], [], color="tab:green", linewidth=4, label="speech")
        self.ax.plot([], [], color="orange", linewidth=4, label="non-speech")
        self.ax.legend(loc="upper right", framealpha=0.85, fontsize=9)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# -------------------------
# Per-frame extractor
# -------------------------
def _extract_per_frame(results: Dict):
    logits = _get_first(results, ["logits", "per_frame_logits", "frame_logits"])
    vad    = _get_first(results, ["vad_probs", "vad", "per_frame_vad", "vad_frame"])
    if logits is None or vad is None:
        return None, None
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    if isinstance(vad, np.ndarray):
        vad = torch.from_numpy(vad).float()
    if logits.ndim == 3 and logits.size(0) == 1:
        logits = logits.squeeze(0)
    if vad.ndim > 1:
        vad = vad.squeeze(-1)
    return logits, vad


# -------------------------
# Gate + Burst
# -------------------------
class LevelChangeGate:
    def __init__(
        self,
        delta_on_db: float = 2.5,
        delta_off_db: float = 1.0,
        level_min_dbfs: float = -60.0,
        ema_alpha: float = 0.05,
        vad_threshold: float = 0.5,
        min_R_clip: float = 0.18,
        hold_ms: int = 300,
        refractory_ms: int = 120
    ):
        self.delta_on_db = float(delta_on_db)
        self.delta_off_db = float(delta_off_db)
        self.level_min_dbfs = float(level_min_dbfs)
        self.ema_alpha = float(ema_alpha)
        self.vad_threshold = float(vad_threshold)
        self.min_R_clip = float(min_R_clip)
        self.hold_s = float(hold_ms) / 1000.0
        self.refractory_s = float(refractory_ms) / 1000.0
        self.bg_dbfs = None
        self.active = False
        self.last_change_time = 0.0

    def update(self, level_dbfs: float, now_s: float, vad_mean: float,
               peaks_count: int, R_clip_max: float):
        if self.bg_dbfs is None:
            self.bg_dbfs = level_dbfs
        diff_db = level_dbfs - self.bg_dbfs

        want_open = (
            (now_s - self.last_change_time) >= self.refractory_s and
            ((level_dbfs > self.level_min_dbfs and diff_db >= self.delta_on_db) or
             (vad_mean >= self.vad_threshold) or
             (peaks_count > 0 and R_clip_max >= self.min_R_clip))
        )

        if not self.active:
            if want_open:
                self.active = True
                self.last_change_time = now_s
        else:
            if (now_s - self.last_change_time) >= self.hold_s:
                want_close = (
                    (diff_db <= self.delta_off_db) and
                    (vad_mean < self.vad_threshold) and
                    (peaks_count == 0 or R_clip_max < self.min_R_clip)
                )
                if want_close:
                    self.active = False
                    self.last_change_time = now_s

        # EMA update after decision
        self.bg_dbfs = (1.0 - self.ema_alpha) * self.bg_dbfs + self.ema_alpha * level_dbfs
        return self.active, diff_db


class BurstTrigger:
    """
    Fires when BOTH:
      - flux_z >= onset_z_on
      - level diff >= burst_on_db
    Also requires coherence >= coh_min to set/keep active.
    Holds for hold_ms and respects refractory.
    """
    def __init__(self, onset_z_on: float = 2.2, onset_z_off: float = 1.2,
                 burst_on_db: float = 3.0, hold_ms: int = 300,
                 coh_min: float = 0.55, refractory_ms: int = 120):
        self.onset_z_on = float(onset_z_on)
        self.onset_z_off = float(onset_z_off)
        self.burst_on_db = float(burst_on_db)
        self.hold_s = hold_ms / 1000.0
        self.coh_min = float(coh_min)
        self.refractory_s = refractory_ms / 1000.0
        self.active_until = 0.0
        self.last_change_time = -1e9

    def update(self, flux_z: float, diff_db: float, coherence: float, now_s: float) -> bool:
        # Activate
        if (now_s - self.last_change_time) >= self.refractory_s:
            if (flux_z >= self.onset_z_on) and (diff_db >= self.burst_on_db) and (coherence >= self.coh_min):
                self.active_until = max(self.active_until, now_s + self.hold_s)
                self.last_change_time = now_s
        # Keep active while within hold, unless strong off condition
        active = (now_s < self.active_until)
        if active and (flux_z <= self.onset_z_off or coherence < self.coh_min):
            # allow early release if evidence disappears quickly
            self.active_until = now_s
            active = False
        return active


# -------------------------
# Streaming loop
# -------------------------
def stream_inference_from_microphone(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device_index: Optional[int] = None,
    sample_rate: int = 16000,
    window_ms: int = 200,
    hop_ms: int = 100,
    max_sources: int = 3,
    chunk_size: int = 5600,
    # normal hist params
    K: int = 72,
    vad_threshold: float = 0.5,
    tau: float = 0.8,
    gamma: float = 1.5,
    smooth_k: int = 1,
    window_bins: int = 1,
    min_peak_height: float = 0.10,
    min_window_mass: float = 0.24,
    min_sep_deg: float = 20.0,
    min_active_ratio: float = 0.20,
    # event-gate
    level_delta_on_db: float = 2.5,
    level_delta_off_db: float = 1.0,
    level_min_dbfs: float = -60.0,
    level_ema_alpha: float = 0.05,
    event_hold_ms: int = 280,
    min_R_clip: float = 0.20,
    event_refractory_ms: int = 120,
    # onset z-score + coherence
    onset_alpha: float = 0.05,
    onset_z_on: float = 2.2,
    onset_z_off: float = 1.2,
    coh_min: float = 0.55,
    # burst (impulse) tuning
    burst_on_db: float = 3.0,
    burst_hold_ms: int = 300,
    burst_recency_decay: Optional[float] = 0.88,
    burst_tau: float = 0.65,
    burst_gamma: float = 1.0,
    burst_smooth_k: int = 0,
    burst_window_bins: int = 0,
    burst_min_peak_height: float = 0.03,
    burst_min_window_mass: float = 0.04,
    burst_min_sep_deg: float = 12.0,
    burst_min_active_ratio: float = 0.05,
    burst_tail_frames: int = 3,
    burst_energy_beta: float = 1.0,
    burst_peak_frame_only: bool = True
):
    # Load config
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "train.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if "features" in config and "K" in config["features"]:
        K = int(config["features"]["K"])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InferenceModel(checkpoint_path=checkpoint_path, config=config, device=device)

    # Detectors
    det_normal = HistDOADetector(
        K=K, vad_threshold=vad_threshold, tau=tau, gamma=gamma, smooth_k=smooth_k,
        window_bins=window_bins, min_peak_height=min_peak_height, min_window_mass=min_window_mass,
        min_sep_deg=min_sep_deg, min_active_ratio=min_active_ratio, max_sources=max_sources,
        device=device, recency_decay=None, tail_frames=None, energy_beta=0.0, peak_frame_only=False
    )

    det_burst = HistDOADetector(
        K=K, vad_threshold=vad_threshold, tau=burst_tau, gamma=burst_gamma, smooth_k=burst_smooth_k,
        window_bins=burst_window_bins, min_peak_height=burst_min_peak_height, min_window_mass=burst_min_window_mass,
        min_sep_deg=burst_min_sep_deg, min_active_ratio=burst_min_active_ratio, max_sources=max_sources,
        device=device, recency_decay=burst_recency_decay, tail_frames=burst_tail_frames,
        energy_beta=burst_energy_beta, peak_frame_only=burst_peak_frame_only
    )

    gate = LevelChangeGate(
        delta_on_db=level_delta_on_db,
        delta_off_db=level_delta_off_db,
        level_min_dbfs=level_min_dbfs,
        ema_alpha=level_ema_alpha,
        vad_threshold=vad_threshold,
        min_R_clip=min_R_clip,
        hold_ms=event_hold_ms,
        refractory_ms=event_refractory_ms
    )
    onset = OnsetDetector(alpha=onset_alpha)
    burst = BurstTrigger(onset_z_on=onset_z_on, onset_z_off=onset_z_off,
                         burst_on_db=burst_on_db, hold_ms=burst_hold_ms,
                         coh_min=coh_min, refractory_ms=event_refractory_ms)

    visualizer = CurrentLineVisualizer()

    window_samples = int(sample_rate * window_ms / 1000)
    hop_samples = int(sample_rate * hop_ms / 1000)

    p = pyaudio.PyAudio()
    # Auto-detect ReSpeaker if device_index not specified
    if device_index is None:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                name = info['name'].lower()
                if 'respeaker' in name or 'seeed' in name or '2886' in name:
                    device_index = i
                    break
    if device_index is None:
        print("\n[Audio] Could not auto-detect Respeaker. Use --device-index or --list-devices.\n")
        # fall through to open default input
    CHANNELS = 6
    RAW_CHANNELS = [1, 4, 3, 2]   # your requested order
    FORMAT = pyaudio.paInt16

    audio_buffer = np.zeros((4, window_samples), dtype=np.float32)
    buffer_fill = 0
    start_time = time.time()

    audio_queue = queue.Queue()
    stream_closed = False

    def _fill_buffer(in_data, frame_count, time_info, status_flags):
        if not stream_closed:
            audio_queue.put(in_data)
        return None, pyaudio.paContinue

    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size,
            stream_callback=_fill_buffer
        )
    except Exception as e:
        print(f"\n[Audio] Could not open input device (index {device_index}).")
        print("        Use --list-devices to find a valid device, then pass --device-index N.")
        print(f"        Error: {e}\n")
        p.terminate()
        return

    stream.start_stream()

    try:
        while True:
            try:
                data = audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            chunk_all = _chunk_to_floatarray(data, CHANNELS)  # (6, N)
            audio_chunk = chunk_all[RAW_CHANNELS, :]          # (4, N)
            n = audio_chunk.shape[1]

            if buffer_fill + n <= window_samples:
                audio_buffer[:, buffer_fill:buffer_fill + n] = audio_chunk
                buffer_fill += n
                continue

            remaining = window_samples - buffer_fill
            if remaining > 0:
                audio_buffer[:, buffer_fill:] = audio_chunk[:, :remaining]
                buffer_fill = window_samples

            # Inference on full window
            t0 = time.perf_counter()
            results = model.inference(
                audio_buffer,
                return_histogram=True,
                return_per_frame=True,
                detect_multi_source=False
            )
            t_model = (time.perf_counter() - t0) * 1000.0

            logits, vad_pf = _extract_per_frame(results)
            level = rms_dbfs(audio_buffer)
            if logits is not None and vad_pf is not None:
                T = int(logits.shape[0])
                energies = frame_rms_energy(audio_buffer, T)      # [T]
                flux = spectral_flux_per_frame(audio_buffer, T)   # [T]
                flux_recent = float(max(flux[-1], flux[-2] if T >= 2 else 0.0))
                flux_z = onset.update_flux(flux_recent)
                coh = OnsetDetector.last_segment_coherence(audio_buffer, T)
                vad_mean = float(vad_pf.mean().item())
                # quick provisional pass for gate inputs
                det_tmp = det_normal.detect(logits, vad_pf, energies.to(device))
                Rmax = max(det_tmp["speech"]["R_clip"], det_tmp["nonspeech"]["R_clip"])
                peaks_count = len(det_tmp["speech"]["peaks"]) + len(det_tmp["nonspeech"]["peaks"])
            else:
                energies = None
                flux_z, coh, vad_mean = 0.0, 0.0, _to_float_scalar(_get_first(results, ["vad_mean","vad_probability","vad"]))
                det_tmp = det_normal.detect_from_preaggregated(
                    hist_speech=_get_first(results, ["speech_histogram", "histogram_speech"]),
                    hist_nonspeech=_get_first(results, ["nonspeech_histogram", "histogram_nonspeech"]),
                    hist_overall=_get_first(results, ["histogram","hist"]),
                    vad_mean=vad_mean
                )
                Rmax = max(det_tmp["speech"]["R_clip"], det_tmp["nonspeech"]["R_clip"])
                peaks_count = len(det_tmp["speech"]["peaks"]) + len(det_tmp["nonspeech"]["peaks"])

            now = time.time() - start_time

            # Update gates
            gate_open, diff_db = gate.update(level_dbfs=level, now_s=now, vad_mean=vad_mean,
                                             peaks_count=peaks_count, R_clip_max=Rmax)
            burst_active = burst.update(flux_z=flux_z, diff_db=diff_db, coherence=coh, now_s=now)

            # Choose detector and compute final detection
            use_det = det_burst if burst_active else det_normal
            if logits is not None and vad_pf is not None:
                t1 = time.perf_counter()
                det = use_det.detect(logits, vad_pf, energies.to(device))
                t_hist = (time.perf_counter() - t1) * 1000.0
            else:
                det = use_det.detect_from_preaggregated(
                    hist_speech=_get_first(results, ["speech_histogram", "histogram_speech"]),
                    hist_nonspeech=_get_first(results, ["nonspeech_histogram", "histogram_nonspeech"]),
                    hist_overall=_get_first(results, ["histogram", "hist"]),
                    vad_mean=vad_mean
                )
                t_hist = 0.0

            # Visualize (only if gate open)
            if gate_open:
                visualizer.update(det["speech"]["peaks"], det["nonspeech"]["peaks"])
                gate_str = "OPEN "
            else:
                visualizer.update([], [])
                gate_str = "CLOSED"

            print(f"[{now:6.2f}s] LVL={level:6.1f} dBFS diff={diff_db:+4.1f} | "
                  f"FLUXz={flux_z:4.2f} COH={coh:4.2f} | "
                  f"GATE={gate_str} BURST={'YES' if burst_active else 'no '} | "
                  f"MODEL={t_model:5.1f}ms HIST+PEAKS={t_hist:5.1f}ms | "
                  f"Sp(R={det['speech']['R_clip']:.2f}, n={len(det['speech']['peaks'])}) "
                  f"NS(R={det['nonspeech']['R_clip']:.2f}, n={len(det['nonspeech']['peaks'])})")

            # Slide buffer by hop
            audio_buffer[:, :-hop_samples] = audio_buffer[:, hop_samples:]
            buffer_fill = window_samples - hop_samples
            if n > remaining:
                carry = min(n - remaining, hop_samples)
                if carry > 0:
                    audio_buffer[:, buffer_fill:buffer_fill + carry] = audio_chunk[:, remaining:remaining + carry]
                    buffer_fill += carry

    except KeyboardInterrupt:
        pass
    finally:
        stream_closed = True
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()
        plt.close('all')


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Real-time DOA (hist) with impulse-optimized burst mode and reduced false positives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Suggested impulse-friendly settings:
  --window-ms 160 --hop-ms 80 \
  --onset-z-on 2.2 --onset-z-off 1.2 --coh-min 0.55 \
  --burst-on-db 3.0 --burst-hold-ms 300 \
  --burst-tail-frames 3 --burst-peak-frame-only \
  --burst-tau 0.65 --burst-gamma 1.0 --burst-smooth-k 0 \
  --burst-min-peak-height 0.03 --burst-min-window-mass 0.04
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--device-index', type=int, default=None)
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--window-ms', type=int, default=350)
    parser.add_argument('--hop-ms', type=int, default=100)
    parser.add_argument('--max-sources', type=int, default=3)
    parser.add_argument('--chunk-size', type=int, default=5600)
    parser.add_argument('--list-devices', action='store_true')

    # normal hist params
    parser.add_argument('--K', type=int, default=72)
    parser.add_argument('--vad-threshold', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument('--smooth-k', type=int, default=1)
    parser.add_argument('--window-bins', type=int, default=1)
    parser.add_argument('--min-peak-height', type=float, default=0.1)
    parser.add_argument('--min-window-mass', type=float, default=0.1)
    parser.add_argument('--min-sep-deg', type=float, default=15)
    parser.add_argument('--min-active-ratio', type=float, default=0.01)

    # event gate
    parser.add_argument('--level-delta-on-db', type=float, default=2.5)
    parser.add_argument('--level-delta-off-db', type=float, default=1.0)
    parser.add_argument('--level-min-dbfs', type=float, default=-60.0)
    parser.add_argument('--level-ema-alpha', type=float, default=0.05)
    parser.add_argument('--event-hold-ms', type=int, default=280)
    parser.add_argument('--event-refractory-ms', type=int, default=120)
    parser.add_argument('--min-R-clip', type=float, default=0.20)

    # onset/coherence
    parser.add_argument('--onset-alpha', type=float, default=0.05)
    parser.add_argument('--onset-z-on', type=float, default=1.8)
    parser.add_argument('--onset-z-off', type=float, default=1.2)
    parser.add_argument('--coh-min', type=float, default=0.55)

    # burst
    parser.add_argument('--burst-on-db', type=float, default=2.0)
    parser.add_argument('--burst-hold-ms', type=int, default=300)
    parser.add_argument('--burst-recency-decay', type=float, default=0.88)
    parser.add_argument('--burst-tau', type=float, default=0.65)
    parser.add_argument('--burst-gamma', type=float, default=1.0)
    parser.add_argument('--burst-smooth-k', type=int, default=0)
    parser.add_argument('--burst-window-bins', type=int, default=0)
    parser.add_argument('--burst-min-peak-height', type=float, default=0.03)
    parser.add_argument('--burst-min-window-mass', type=float, default=0.04)
    parser.add_argument('--burst-min-sep-deg', type=float, default=12.0)
    parser.add_argument('--burst-min-active-ratio', type=float, default=0.05)
    parser.add_argument('--burst-tail-frames', type=int, default=3)
    parser.add_argument('--burst-energy-beta', type=float, default=1.0)
    parser.add_argument('--burst-peak-frame-only', action='store_true')

    args = parser.parse_args()

    if args.list_devices:
        p = pyaudio.PyAudio()
        print("\nAvailable audio input devices:")
        print("-" * 80)
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"Device {i}: {info['name']}")
                print(f"  Channels: {info['maxInputChannels']}, Sample Rate: {info['defaultSampleRate']:.0f} Hz\n")
        p.terminate()
        return

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        parser.error(f"Checkpoint not found: {ckpt}")

    stream_inference_from_microphone(
        checkpoint_path=str(ckpt),
        config_path=args.config,
        device_index=args.device_index,
        sample_rate=args.sample_rate,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        max_sources=args.max_sources,
        chunk_size=args.chunk_size,
        K=args.K,
        vad_threshold=args.vad_threshold,
        tau=args.tau,
        gamma=args.gamma,
        smooth_k=args.smooth_k,
        window_bins=args.window_bins,
        min_peak_height=args.min_peak_height,
        min_window_mass=args.min_window_mass,
        min_sep_deg=args.min_sep_deg,
        min_active_ratio=args.min_active_ratio,
        level_delta_on_db=args.level_delta_on_db,
        level_delta_off_db=args.level_delta_off_db,
        level_min_dbfs=args.level_min_dbfs,
        level_ema_alpha=args.level_ema_alpha,
        event_hold_ms=args.event_hold_ms,
        event_refractory_ms=args.event_refractory_ms,
        min_R_clip=args.min_R_clip,
        onset_alpha=args.onset_alpha,
        onset_z_on=args.onset_z_on,
        onset_z_off=args.onset_z_off,
        coh_min=args.coh_min,
        burst_on_db=args.burst_on_db,
        burst_hold_ms=args.burst_hold_ms,
        burst_recency_decay=args.burst_recency_decay,
        burst_tau=args.burst_tau,
        burst_gamma=args.burst_gamma,
        burst_smooth_k=args.burst_smooth_k,
        burst_window_bins=args.burst_window_bins,
        burst_min_peak_height=args.burst_min_peak_height,
        burst_min_window_mass=args.burst_min_window_mass,
        burst_min_sep_deg=args.burst_min_sep_deg,
        burst_min_active_ratio=args.burst_min_active_ratio,
        burst_tail_frames=args.burst_tail_frames,
        burst_energy_beta=args.burst_energy_beta,
        burst_peak_frame_only=args.burst_peak_frame_only,
    )


if __name__ == "__main__":
    main()
