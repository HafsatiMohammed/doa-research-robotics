# mix_batcher.py
from __future__ import annotations
import os, json, glob, math, random, threading, collections, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import yaml
from pathlib import Path
from collections import Counter
import soundfile as sf

try:
    import torchaudio
    HAVE_TORCHAUDIO = True
except Exception:
    HAVE_TORCHAUDIO = False

# ---- local deps (same as your project) ----
try:
    from . import features
    from . import srp
    from . import vad
except ImportError:
    import features
    import srp
    import vad


# ----------------- helpers -----------------

def load_config(config_path: str = "configs/train.yaml") -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p, "r") as f: return yaml.safe_load(f)
    print(f"[mix_batcher] Warning: {config_path} not found, using defaults.")
    return {}

def quantize_az_deg(az, K=72) -> int:
    res = 360.0 / K
    return int(np.round(az / res)) % K

def _next_pow2(n: int) -> int: return 1 << (n - 1).bit_length()

def _to_mono(x: np.ndarray) -> np.ndarray: return x if x.ndim == 1 else x.mean(axis=1)

def _rms(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp((x**2).mean(dim=-1), min=eps))

def _set_rms(x: torch.Tensor, target: float, eps: float = 1e-12) -> torch.Tensor:
    cur = _rms(x, eps=eps)[(...,) + (None,)]
    return x * (target / torch.clamp(cur, min=1e-6))

def _resample_if_needed(wave: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out: return wave
    if HAVE_TORCHAUDIO:
        return torchaudio.functional.resample(wave, sr_in, sr_out)
    T_in = wave.shape[-1]
    T_out = int(round(T_in * sr_out / sr_in))
    t = torch.linspace(0, T_in - 1, T_out, device=wave.device)
    t0 = torch.clamp(t.floor().long(), 0, T_in - 1)
    t1 = torch.clamp(t0 + 1, 0, T_in - 1)
    w = t - t0.float()
    return (1 - w) * wave[..., t0] + w * wave[..., t1]


# ------------- small caches ----------------

class WaveCache:
    def __init__(self, max_items: int = 256):
        self.max_items = max_items
        self.d: collections.OrderedDict[str, Tuple[np.ndarray, int]] = collections.OrderedDict()
        self.lock = threading.Lock()

    def __getstate__(self):
        st = self.__dict__.copy(); st["lock"] = None; return st
    def __setstate__(self, st):
        self.__dict__.update(st); self.lock = threading.Lock()

    def get(self, path: str) -> Tuple[np.ndarray, int]:
        with self.lock:
            if path in self.d:
                v = self.d.pop(path); self.d[path] = v; return v
        wav, sr = sf.read(path, always_2d=True)
        wav = wav.mean(axis=1).astype(np.float32)
        with self.lock:
            self.d[path] = (wav, sr)
            if len(self.d) > self.max_items: self.d.popitem(last=False)
        return wav, sr

_global_wave_cache = None
def _get_global_wave_cache(max_items: int = 256) -> WaveCache:
    global _global_wave_cache
    if _global_wave_cache is None: _global_wave_cache = WaveCache(max_items=max_items)
    return _global_wave_cache


# ---------------- RIR bank -----------------

@dataclass(frozen=True)
class RIRPose:
    room_id: str
    pose_id: str
    rt60_s: float
    az_deg: float
    distance_m: Optional[float]
    src_xyz: Optional[Tuple[float, float, float]]
    wavs_4: List[str]
    meta_path: str

class RIRBank:
    def __init__(self, root: str, split: str):
        self.root = root; self.split = split
        self.poses: List[RIRPose] = []; self.rooms: Dict[str, Dict[str, Any]] = {}
        self._scan()

    def _scan(self):
        split_dir = os.path.join(self.root, self.split)
        for rd in sorted(glob.glob(os.path.join(split_dir, "room-*"))):
            rm_fp = os.path.join(rd, "room_meta.json")
            if not os.path.isfile(rm_fp): continue
            with open(rm_fp, "r") as f: R = json.load(f)
            room_id = R["room"]["id"]; rt60_s = float(R["room"]["rt60_s"])
            self.rooms[room_id] = R

            prod = glob.glob(os.path.join(rd, "array-*", "rt60-*", "srcpose-*", "meta.json"))
            if prod:
                for pm in prod:
                    pose_dir = os.path.dirname(pm)
                    wavs = sorted(glob.glob(os.path.join(pose_dir, "*_mic-*.wav")))
                    if len(wavs) != 4: continue
                    with open(pm, "r") as f: P = json.load(f)
                    src = P.get("src_pose", P)
                    az = float(src.get("azimuth_deg", P.get("az_deg", np.nan)))
                    dist = float(src.get("distance_m", P.get("distance_m", np.nan)))
                    sxyz = src.get("src_xyz_room_m", P.get("src_xyz_room_m", None))
                    pose_id = os.path.basename(pose_dir)
                    self.poses.append(RIRPose(room_id, pose_id, rt60_s, az, dist, tuple(sxyz) if sxyz else None, wavs, pm))
                continue

            demo = glob.glob(os.path.join(rd, "_poses", "srcpose-*", "meta.json"))
            for pm in demo:
                pose_dir = os.path.dirname(pm)
                wavs = sorted(glob.glob(os.path.join(pose_dir, "rir_demo_*_mic-*.wav")))
                if len(wavs) != 4: continue
                with open(pm, "r") as f: P = json.load(f)
                az = float(P.get("az_deg", np.nan)); dist = float(P.get("distance_m", np.nan))
                sxyz = P.get("src_xyz_room_m", None); pose_id = os.path.basename(pose_dir)
                self.poses.append(RIRPose(room_id, pose_id, rt60_s, az, dist, tuple(sxyz) if sxyz else None, wavs, pm))

        if not self.poses:
            raise RuntimeError(f"No RIR poses found under {self.root}/{self.split}")

    def poses_in_room(self, room_id: str) -> List[RIRPose]:
        return [p for p in self.poses if p.room_id == room_id]

    def load_rirs_4(self, pose: RIRPose) -> torch.Tensor:
        chans = []
        for w in sorted(pose.wavs_4):
            x, sr = sf.read(w)
            x = x[:, 0] if x.ndim > 1 else x
            chans.append(torch.from_numpy(x.astype(np.float32)))
        return torch.stack(chans, dim=0)


# -------- FFT conv / aug (unchanged) ------

def fft_conv_4ch_mono(x: torch.Tensor, rirs_4: torch.Tensor) -> torch.Tensor:
    T = x.numel(); Lr = rirs_4.shape[-1]
    n_fft = _next_pow2(T + Lr - 1)
    X = torch.fft.rfft(x, n_fft); Y = []
    for m in range(4):
        H = torch.fft.rfft(rirs_4[m], n_fft)
        y = torch.fft.irfft(X * H, n_fft)[:T]; Y.append(y)
    return torch.stack(Y, dim=0)

def fractional_delay_fd(y: torch.Tensor, tau_sec: float, sr: int) -> torch.Tensor:
    if abs(tau_sec) < 1e-7: return y
    T = y.shape[-1]; n_fft = _next_pow2(T)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr).to(y.device)
    phase = torch.exp(-1j * 2.0 * math.pi * freqs * tau_sec)
    Y = torch.fft.rfft(y, n_fft); Y = Y * phase[None, :]
    return torch.fft.irfft(Y, n_fft)[..., :T]

@dataclass
class AugmentCfg:
    gain_db_sigma: float = 1.0
    phase_deg_sigma: float = 3.0
    frac_delay_sigma_us: float = 30
    eq_prob: float = 0.5

def apply_per_mic_aug(y: torch.Tensor, sr: int, rng: random.Random, cfg: AugmentCfg) -> torch.Tensor:
    g_db = torch.tensor([rng.gauss(0.0, cfg.gain_db_sigma) for _ in range(4)], dtype=torch.float32, device=y.device)
    y = y * (10.0 ** (g_db / 20.0)).view(4, 1)
    ph_rad = torch.tensor([rng.gauss(0.0, cfg.phase_deg_sigma) for _ in range(4)], dtype=torch.float32, device=y.device) * (math.pi/180)
    n_fft = _next_pow2(y.shape[-1])
    Y = torch.fft.rfft(y, n_fft) * torch.exp(1j * ph_rad)[:, None]
    y = torch.fft.irfft(Y, n_fft)[..., : y.shape[-1]]
    taus = [rng.gauss(0.0, cfg.frac_delay_sigma_us) * 1e-6 for _ in range(4)]
    for m in range(4): y[m] = fractional_delay_fd(y[m:m+1], taus[m], sr)[0]
    if rng.random() < cfg.eq_prob:
        T = y.shape[-1]; n_fft = _next_pow2(T); freqs = torch.fft.rfftfreq(n_fft, d=1.0/sr).to(y.device)
        low_db = rng.uniform(-1.5, 1.5); mid_db = rng.uniform(-1.0, 1.0); high_db = rng.uniform(-1.5, 1.5)
        env = torch.ones_like(freqs); env[freqs <= 300] *= 10**(low_db/20); env[(freqs>300)&(freqs<4000)] *= 10**(mid_db/20); env[freqs>=4000] *= 10**(high_db/20)
        Y = torch.fft.rfft(y, n_fft) * env[None, :]; y = torch.fft.irfft(Y, n_fft)[..., :T]
    return y


# ----------------- audio idx ----------------

def _scan_wavs(root: str) -> List[str]:
    if not os.path.isdir(root): raise RuntimeError(f"Audio root not found: {root}")
    exts = (".wav", ".flac", ".ogg"); paths = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if not d.startswith(".")]
        for fn in fns:
            if fn.startswith("."): continue
            if os.path.splitext(fn)[1].lower() in exts:
                paths.append(os.path.join(dp, fn))
    if not paths: raise RuntimeError(f"No audio under {root}")
    return sorted(paths)

class AudioIndex:
    def __init__(self, root: str, cache: WaveCache):
        self.root = root; self.paths = _scan_wavs(root)
        self.cache_items = getattr(cache, "max_items", 256) if cache is not None else 256
        self.cache = None
    def __getstate__(self): st = self.__dict__.copy(); st["cache"]=None; return st
    def __setstate__(self, st): self.__dict__.update(st); self.cache=None
    def sample_path(self, rng: random.Random) -> str: return rng.choice(self.paths)
    def get_segment(self, path: str, sr_target: int, T: int, rng: random.Random) -> torch.Tensor:
        if not isinstance(self.cache, WaveCache):
            self.cache = _get_global_wave_cache(self.cache_items)
        wav, sr = self.cache.get(path)
        x = torch.from_numpy(_to_mono(wav)).float()
        if sr != sr_target: x = _resample_if_needed(x.unsqueeze(0), sr, sr_target).squeeze(0)
        if x.numel() < T:
            rep = (T + x.numel() - 1) // x.numel(); x = x.repeat(rep)[:T]
        else:
            st = rng.randrange(0, x.numel() - T + 1); x = x[st:st+T]
        return x


# -------- epoch plan / class ratios --------

class StrictCountsPlanner:
    """Keeps your 1/2/3-speaker priors; samples SNR ~ N(10,4²) truncated to [0,20]."""
    def __init__(self, epoch_size: int, rng: random.Random):
        self.epoch_size = epoch_size; self.rng = rng
    def _trunc_gauss(self, mu=10.0, sigma=4.0, lo=0.0, hi=20.0):
        for _ in range(64):
            x = self.rng.gauss(mu, sigma)
            if lo <= x <= hi: return x
        return max(lo, min(hi, x))
    def build_plan(self) -> Dict[str, List[Any]]:
        N = self.epoch_size
        counts_ratio = {1: 0.50, 2: 0.35, 3: 0.15}
        counts = {k: int(round(v * N)) for k, v in counts_ratio.items()}
        diff = N - sum(counts.values()); counts[1] += diff
        n_speech_plan = []
        for k, c in counts.items(): n_speech_plan.extend([k]*c)
        self.rng.shuffle(n_speech_plan)
        snr_plan = [self._trunc_gauss() for _ in range(N)]
        return {"n_speech": n_speech_plan, "snr_db": snr_plan}


# ------------- on-the-fly dataset ----------

@dataclass
class MixConfig:
    sr: int = 16000
    dur_s: float = 4.0
    target_rms: float = 0.05
    min_sep_deg: float = 15.0
    radial_diversity: bool = True
    ambience_K_poses: int = 6
    include_ambience: bool = True
    include_local_events: bool = False  # <<< disabled per your request
    local_event_len_s_range = (0.2, 1.5)
    clip_guard_db: float = 0.5
    K_az: int = 72                     # for even-az binning per mixture

class OnTheFlyMixtureDataset(torch.utils.data.Dataset):
    """
    RETURNS:
      mixture: torch.Tensor [4, T]
      meta: dict with fields:
        - 'azimuths_deg': List[float]          # one value (n_speech == 1)
        - 'elevations_deg': List[float]        # one value (n_speech == 1)
        - 'dry_sources_fp16': List[np.ndarray] # each [T] float16; pre-gain, speech-only
        - 'snr_db', 'rt60_s', 'n_speech', 'room_id'
        - 'speech_files', 'rir_pose_ids', 'rir_meta'
        - 'diffuse_poses': List[str]           # pose_ids used for late-reverb noise
        - 'late_start_ms': float               # cutoff used to define "late" part of SRIR
    Notes:
      * Exactly ONE speech source per mixture (n_speech == 1).
      * Diffuse noise is created by taking a single ambience segment and convolving it
        with the LATE reverb (after 'late_start_ms') of THREE random SRIRs from the
        same room, then summing those three 4ch results together.
    """

    def __init__(self, rir_root: str, split: str,
                 speech_root: str, local_noises_root: str, ambiences_root: str,
                 epoch_size: int, base_seed: int = 1234,
                 cfg: 'MixConfig' = None, aug: 'AugmentCfg' = None,
                 wave_cache_items: int = 512):
        super().__init__()
        if cfg is None: cfg = MixConfig()
        if aug is None: aug = AugmentCfg()

        self.rir = RIRBank(rir_root, split)
        self.speech = AudioIndex(speech_root, WaveCache(wave_cache_items))
        self.localn = AudioIndex(local_noises_root, self.speech.cache)
        self.amb = AudioIndex(ambiences_root, self.speech.cache)

        self.cfg = cfg
        self.aug_cfg = aug
        self.epoch_size = epoch_size
        self.base_seed = int(base_seed)
        self._epoch = 0

        # We still build a plan to get SNR etc., but we will force n_speech=1
        self._plan = StrictCountsPlanner(epoch_size, random.Random(self._seed_for("plan", 0))).build_plan()

        # Optional config knobs (no changes needed in MixConfig)
        # Where "late" starts in SRIR (ms), and short fade-in (ms) to prevent clicks.
        self._late_start_ms = float(getattr(self.cfg, "late_start_ms", 80.0))
        self._late_ramp_ms  = float(getattr(self.cfg, "late_ramp_ms", 10.0))
        # How many late SRIRs to sum for diffuse noise
        self._n_late_srir   = int(getattr(self.cfg, "n_late_srir", 3))

    def _seed_for(self, *tags) -> int:
        h = 0x9E3779B97F4A7C15
        for t in tags:
            v = int(t) if isinstance(t, (int, np.integer)) else int.from_bytes(str(t).encode(), "little", signed=False)
            h ^= v + 0x9E3779B97F4A7C15 + ((h << 6) & ((1<<64)-1)) + (h >> 2)
            h &= (1<<64)-1
        return h & 0x7FFFFFFF

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        rng = random.Random(self._seed_for("plan", epoch))
        self._plan = StrictCountsPlanner(self.epoch_size, rng).build_plan()

    def __len__(self):
        return self.epoch_size

    # ----- even-azimuth selection within a room -----
    def _choose_poses_even_az(self, room_id: str, n: int, rng: random.Random) -> List['RIRPose']:
        cand = self.rir.poses_in_room(room_id)
        if len(cand) < n:
            return rng.sample(self.rir.poses_in_room(room_id), min(n, len(cand)))
        K = int(self.cfg.K_az)
        offset = rng.randrange(0, K)
        centers = [((b + offset) % K) * (360.0 / K) for b in range(0, K, max(1, K // n))]
        chosen: List[RIRPose] = []
        used = set()
        for cdeg in centers:
            best = None; best_d = 1e9
            for p in cand:
                if p.pose_id in used: continue
                d = abs(p.az_deg - cdeg); d = min(d, 360.0 - d)
                if d < best_d:
                    ok = True
                    for z in chosen:
                        dz = abs(z.az_deg - p.az_deg); dz = min(dz, 360.0 - dz)
                        if dz < self.cfg.min_sep_deg: ok = False; break
                    if ok:
                        best_d = d; best = p
            if best is not None:
                chosen.append(best); used.add(best.pose_id)
            if len(chosen) >= n: break
        if len(chosen) < n:
            remain = [p for p in cand if p.pose_id not in used]
            rng.shuffle(remain)
            for p in remain:
                ok = True
                for z in chosen:
                    dz = abs(z.az_deg - p.az_deg); dz = min(dz, 360.0 - dz)
                    if dz < self.cfg.min_sep_deg: ok = False; break
                if ok:
                    chosen.append(p)
                if len(chosen) >= n: break
        return chosen[:n]

    # ----- keep only the LATE part of a 4ch SRIR, with a short fade-in ramp -----
    def _late_only(self, rirs_4: torch.Tensor, sr: int, start_ms: float, ramp_ms: float) -> torch.Tensor:
        """
        rirs_4: torch.Tensor [4, L]
        Returns: torch.Tensor [4, L] with early portion zeroed and a ramp on the boundary.
        """
        assert rirs_4.dim() == 2 and rirs_4.size(0) == 4, "Expected [4, L] SRIR"
        L = rirs_4.size(1)
        start_samp = int(round((start_ms / 1000.0) * sr))
        ramp = int(round((ramp_ms / 1000.0) * sr))
        out = rirs_4.clone()
        if start_samp >= L:
            # everything is early -> zero
            return out.zero_()

        # Zero early part up to start_samp
        if start_samp > 0:
            out[:, :start_samp] = 0.0

        # Apply a short fade-in ramp to avoid a hard step
        if ramp > 0 and start_samp < L:
            r_end = min(L, start_samp + ramp)
            n = r_end - start_samp
            if n > 0:
                w = torch.linspace(0.0, 1.0, steps=n, device=out.device, dtype=out.dtype)
                out[:, start_samp:r_end] *= w

        return out

    def __getitem__(self, idx: int):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker else 0
        seed = self._seed_for(self.base_seed, self._epoch, idx, wid)
        rng = random.Random(seed)

        # (1) Force exactly ONE speech source per mixture
        n_sp = 1

        # keep SNR from plan (or fallback)
        snr_db = float(self._plan["snr_db"][idx]) if "snr_db" in self._plan else 0.0

        # sample one room; then pick a single speech pose in that room
        by_room: Dict[str, List['RIRPose']] = {}
        for p in self.rir.poses: by_room.setdefault(p.room_id, []).append(p)
        room_id = rng.choice(list(by_room.keys()))
        room_meta = self.rir.rooms[room_id]
        rt60_s = float(room_meta["room"]["rt60_s"])

        T = int(round(self.cfg.dur_s * self.cfg.sr))
        device = torch.device("cpu")

        # choose ONE pose (even-az method still works with n=1)
        speech_pose = self._choose_poses_even_az(room_id, 1, rng)[0]

        speech_stems = []
        az_list: List[float] = []
        src_xyz_list: List[tuple] = []
        speech_ids = []
        dry_keep_fp16: List[np.ndarray] = []

        # --- speech source (single) ---
        wav_path = self.speech.sample_path(rng)
        speech_ids.append(os.path.relpath(wav_path, self.speech.root))
        dry = self.speech.get_segment(wav_path, self.cfg.sr, T, rng)  # [T]
        dry = _set_rms(dry, self.cfg.target_rms)                      # pre-gain, for metadata copy
        dry_keep_fp16.append(dry.cpu().numpy().astype(np.float16))    # keep dry source (FP16)

        rirs_4 = self.rir.load_rirs_4(speech_pose).to(device)         # [4, L]
        y4 = fft_conv_4ch_mono(dry.to(device), rirs_4)                # convolve
        y4 = apply_per_mic_aug(y4, self.cfg.sr, rng, self.aug_cfg)
        speech_stems.append(y4)
        #print(speech_pose)
        az_list.append(float(speech_pose.az_deg))
        # elevation may not exist on all datasets; default to 0.0 if absent
        src_xyz_list.append(speech_pose.src_xyz)

        # --- mix speech stems (only one) ---
        speech_mix = torch.zeros(4, T, dtype=torch.float32, device=device)
        for y4 in speech_stems:
            # crop or pad to T, in case conv changed length (depending on your fft_conv)
            if y4.size(1) != T:
                if y4.size(1) > T:
                    speech_mix += y4[:, :T]
                else:
                    pad = torch.zeros(4, T - y4.size(1), dtype=y4.dtype, device=y4.device)
                    speech_mix += torch.cat([y4, pad], dim=1)
            else:
                speech_mix += y4

        # --- diffuse noise from LATE reverb sum of K random SRIRs ---
        noise_mix = torch.zeros_like(speech_mix)
        diffuse_pose_ids: List[str] = []
        if getattr(self.cfg, "include_ambience", True):
            # Excitation: one ambience segment
            amb_path = self.amb.sample_path(rng)
            amb = self.amb.get_segment(amb_path, self.cfg.sr, T, rng).to(device)  # [T]

            # Choose K random poses (distinct) in the same room
            poses_in_room = self.rir.poses_in_room(room_id)
            # Make sure we have something to choose from
            k = min(self._n_late_srir, max(1, len(poses_in_room)))
            late_poses = rng.sample(poses_in_room, k)

            # Convolve ambience with the "late-only" SRIRs and sum
            for p in late_poses:
                rirs_4_full = self.rir.load_rirs_4(p).to(device)
                rirs_4_late = self._late_only(rirs_4_full, self.cfg.sr, self._late_start_ms, self._late_ramp_ms)
                y4n = fft_conv_4ch_mono(amb, rirs_4_late)  # [4, T?]
                # shape-safe add
                if y4n.size(1) != T:
                    if y4n.size(1) > T:
                        noise_mix += y4n[:, :T]
                    else:
                        pad = torch.zeros(4, T - y4n.size(1), dtype=y4n.dtype, device=y4n.device)
                        noise_mix += torch.cat([y4n, pad], dim=1)
                else:
                    noise_mix += y4n
                diffuse_pose_ids.append(p.pose_id)

        # local events (kept disabled as before)
        if getattr(self.cfg, "include_local_events", False):
            n_local = 0  # explicitly keep 0 unless re-enabled
            # --- re-enable by filling this block as you had before ---
        else:
            n_local = 0

        # --- SNR scaling ---
        Es = (speech_mix**2).mean().item()
        En = (noise_mix**2).mean().item()
        k = 0.0 if En < 1e-12 else math.sqrt(Es / (En * (10.0 ** (snr_db / 10.0))))
        mixture = speech_mix + noise_mix * k

        # --- limiter headroom ---
        peak = mixture.abs().amax().item()
        limit = 10 ** (-self.cfg.clip_guard_db / 20)
        if peak > limit:
            mixture = mixture * (limit / peak)
        meta = {
            "n_speech": n_sp,
            "azimuths_deg": az_list,
            "src_xyz": src_xyz_list,                 # <-- NEW
            "snr_db": float(snr_db),
            "room_id": room_id,
            "rt60_s": rt60_s,
            "speech_files": speech_ids,
            "rir_pose_ids": [speech_pose.pose_id],
            "rir_meta": [speech_pose.meta_path],
            "dry_sources_fp16": dry_keep_fp16,         # keep dry sources for per-source VAD
            "diffuse_poses": diffuse_pose_ids,         # <-- NEW (trace diffuse SRIRs)
            "late_start_ms": float(self._late_start_ms), # <-- NEW (document cutoff)
        }
        return mixture, meta


# ------------- DataLoader glue -------------

def worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    base_seed = info.seed
    random.seed(base_seed); np.random.seed(base_seed % (1<<32))

def collate_mix(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]):
    xs = [b[0] for b in batch]; metas = [b[1] for b in batch]
    return torch.stack(xs, dim=0), metas


# --------- feature computation (kept for compat; prefer feature_dataloaders) ----

def compute_batch_features_fast(*args, **kwargs):
    raise NotImplementedError("Use feature_dataloaders.create_online_feature_loader / precompute_feature_windows instead.")
