from __future__ import annotations

import os, json, math, time, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from functools import lru_cache

# ---- your modules (unchanged science) ----
from .mix_batcher import (
    OnTheFlyMixtureDataset, quantize_az_deg, worker_init_fn, collate_mix
)
from . import features as featmod       # stft_multi, compute_mag_phase_cos_sin
from . import srp as srpmod             # srp_phat_..., srp_weighted_...
from . import vad as vadmod             # VoiceActivityDetector


@lru_cache(maxsize=1)
def _get_vad_cached(sr: int):
    # Construct once per process; safe to call from workers.
    return vadmod.VoiceActivityDetector(threshold=0.2, frame_rate=sr)


# ======================= config dataclasses =======================

@dataclass
class STFTCfg:
    sr: int = 16000
    win_s: float = 0.032
    hop_s: float = 0.010
    nfft: int = 512
    center: bool = False
    window: str = "hann"
    pad_mode: str = "reflect"

@dataclass
class FeatCfg:
    K: int = 72                     # az bins
    max_frames: int = 6             # window length (STFT frames)
    vad_threshold: float = 0.6
    vad_gt_masking: bool = True     # kept for compatibility (unused for GT now)
    use_fp16: bool = False


# ======================= helpers (prints & math) =======================

def _np_dtype(fp16: bool):
    return np.float16 if fp16 else np.float32

def _pt_dtype(fp16: bool):
    return torch.float16 if fp16 else torch.float32

def _pick_window_starts(n_frames: int, T: int, n_windows: int, strategy: str = "random") -> List[int]:
    if n_frames <= T:
        return [0] * n_windows
    if strategy == "sequential":
        stride = max(1, (n_frames - T) // max(1, (n_windows - 1)))
        return [min(i * stride, n_frames - T) for i in range(n_windows)]
    return [random.randint(0, n_frames - T) for _ in range(n_windows)]

def _compute_stft_feats_vad(x4: np.ndarray,
                            stft: STFTCfg,
                            vad_detector: vadmod.VoiceActivityDetector,
                            verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    x4: [4, T] float32
    Returns:
      X        : [T_frames, 4, F] complex
      freqs    : [F]
      feats    : [T_frames, 12, F] (mag + cos + sin per channel)
      vad_probs: [T_frames] (mixture VAD)
    """
    t0 = time.time()
    X, freqs, times = featmod.stft_multi(
        x4.T, fs=stft.sr,
        win_s=stft.win_s, hop_s=stft.hop_s,
        nfft=stft.nfft, window=stft.window,
        center=stft.center, pad_mode=stft.pad_mode
    )  # [T, 4, F]
    feats = featmod.compute_mag_phase_cos_sin(X, dtype=np.float32)  # [T,12,F]
    t1 = time.time()

    vad_probs = vad_detector.vad_probs_same_frame_count(
        x4[0], stft.sr,
        hop_ms=stft.hop_s * 1000.0,
        base_win_ms=stft.win_s * 1000.0,
        n_frames=X.shape[0],
        pad_mode="constant"
    ).astype(np.float32)  # [T]
    t2 = time.time()

    if verbose:
        print(f"[stft] T={X.shape[0]}, F={X.shape[2]}  | STFT+feats={t1-t0:.3f}s  VAD={t2-t1:.3f}s", flush=True)

    return X, freqs, feats, vad_probs

def _srp_maps_for_window(X_win: np.ndarray,
                         freqs: np.ndarray,
                         mic_xy: np.ndarray,
                         vad_win: np.ndarray,
                         K: int) -> Tuple[np.ndarray, np.ndarray]:
    az_res_deg = 360.0 / K
    _, srp_vec     = srpmod.srp_phat_azimuth_map_360_from_stft(X_win,  freqs, mic_xy, az_res_deg=az_res_deg)
    _, srp_vad_vec = srpmod.srp_weighted_VAD_phat_azimuth_map_360_from_stft(X_win, freqs, mic_xy, vad_win, az_res_deg=az_res_deg)
    return srp_vec.astype(np.float32), srp_vad_vec.astype(np.float32)

def _pack_windows(feats: np.ndarray, vad_probs: np.ndarray,
                  srp_vec: np.ndarray, srp_vad_vec: np.ndarray,
                  T: int, F: int, K: int, dtype: Any
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xf = feats.transpose(1, 0, 2).astype(dtype, copy=False)       # [12,T,F]
    V  = np.tile(vad_probs[:, None], (1, F)).astype(dtype, copy=False)     # [T,F]
    S  = np.tile(srp_vec[None, :], (T, 1)).astype(dtype, copy=False)       # [T,K]
    Sv = np.tile(srp_vad_vec[None, :], (T, 1)).astype(dtype, copy=False)   # [T,K]
    return Xf, V, S, Sv


# ======================= ONLINE (windowed) =======================

class OnTheFlyWindowedFeatureDataset(Dataset):
    """
    Wraps your OnTheFlyMixtureDataset and returns W fixed-length windows of features per item.
    Heavy work (STFT/feats/SRP/VAD) happens inside the worker processes.

    IMPORTANT CHANGE:
      - Ground truth is now built from *per-source VAD* (multi-hot across K).
      - Mixture VAD is exported but not used to zero the GT.
    """
    def __init__(self,
                 base_ds: OnTheFlyMixtureDataset,
                 mic_xy: np.ndarray,
                 stft_cfg: STFTCfg,
                 feat_cfg: FeatCfg,
                 windows_per_mix: int = 1,
                 window_strategy: str = "random",
                 verbose: bool = False):
        super().__init__()
        self.base = base_ds
        self.mic_xy = np.asarray(mic_xy, dtype=float)
        self.stft = stft_cfg
        self.featcfg = feat_cfg
        self.W = int(max(1, windows_per_mix))
        self.window_strategy = window_strategy
        self.verbose = verbose

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mix, meta = self.base[idx]                 # mix: [4, T_samples] torch
        print(f'here are the meta data {meta}')
        x4 = mix.cpu().numpy().astype(np.float32)
        vad_detector = _get_vad_cached(self.stft.sr)

        # STFT / feats / mixture VAD
        X, freqs, feats, vad_probs = _compute_stft_feats_vad(
            x4, self.stft, vad_detector, verbose=False if not self.verbose else True
        )
        T_total, _, F = X.shape
        T = self.featcfg.max_frames
        K = self.featcfg.K
        starts = _pick_window_starts(T_total, T, self.W, self.window_strategy)
        dtype_np = _np_dtype(self.featcfg.use_fp16)

        # Per-source VADs (on dry stems) aligned to STFT length
        src_vads: List[np.ndarray] = []
        for dry_fp16 in meta.get("dry_sources_fp16", []):
            dry = np.asarray(dry_fp16, dtype=np.float32)
            v = vad_detector.vad_probs_same_frame_count(
                dry, self.stft.sr,
                hop_ms=self.stft.hop_s * 1000.0,
                base_win_ms=self.stft.win_s * 1000.0,
                n_frames=X.shape[0]
            ).astype(dtype_np)
            src_vads.append(v)
        az_bins = [quantize_az_deg(az, K) for az in meta.get("azimuths_deg", [])]

        feats_list, vad_list, srp_list, srp_vad_list, gt_list = [], [], [], [], []
        for s0 in starts:
            s1 = s0 + T
            X_win        = X[s0:s1]                                  # [T,4,F]
            feats_win    = feats[s0:s1]                              # [T,12,F]
            vad_win      = vad_probs[s0:s1]                          # [T]
            srp_vec, srp_vad_vec = _srp_maps_for_window(X_win, freqs, self.mic_xy, vad_win, K)

            feat_w, vad_w, srp_w, srp_vad_w = _pack_windows(
                feats_win, vad_win, srp_vec, srp_vad_vec, T, F, K, dtype_np
            )
            #print(feats_w)
            
		
            # --- Multi-source GT from per-source VAD (no mixture gating) ---
            gt_w = np.zeros((T, K), dtype=dtype_np)
            if az_bins and src_vads:
                for k_bin, vsrc_full in zip(az_bins, src_vads):
                    vsrc = vsrc_full[s0:s1]  # [T]
                    active = (vsrc >= self.featcfg.vad_threshold).astype(dtype_np)
                    gt_w[:, k_bin] += active
                gt_w = np.clip(gt_w, 0, 1)

            feats_list.append(torch.from_numpy(feat_w))
            vad_list.append(torch.from_numpy(vad_w))
            srp_list.append(torch.from_numpy(srp_w))
            srp_vad_list.append(torch.from_numpy(srp_vad_w))
            gt_list.append(torch.from_numpy(gt_w))

        return {
            "features": torch.stack(feats_list, dim=0),        # [W, 12, T, F]
            "vad": torch.stack(vad_list, dim=0),               # [W, T, F]
            "srp_phat": torch.stack(srp_list, dim=0),          # [W, T, K]
            "srp_phat_vad": torch.stack(srp_vad_list, dim=0),  # [W, T, K]
            "ground_truth": torch.stack(gt_list, dim=0),       # [W, T, K]
        }

def _collate_windows(batch: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
    feats = torch.cat([b["features"]     for b in batch], dim=0)
    vad   = torch.cat([b["vad"]          for b in batch], dim=0)
    srp   = torch.cat([b["srp_phat"]     for b in batch], dim=0)
    srpv  = torch.cat([b["srp_phat_vad"] for b in batch], dim=0)
    gt    = torch.cat([b["ground_truth"] for b in batch], dim=0)
    return feats, vad, srp, srpv, gt

def create_online_feature_loader(
    base_ds: OnTheFlyMixtureDataset,
    mic_xy: np.ndarray,
    stft_cfg: STFTCfg,
    feat_cfg: FeatCfg,
    batch_size: int = 64,
    num_workers: int = 8,
    windows_per_mix: int = 1,
    window_strategy: str = "random",
    pin_memory: bool = True,
    drop_last: bool = True,
    verbose_iter: bool = True,
):
    ds = OnTheFlyWindowedFeatureDataset(
        base_ds, mic_xy, stft_cfg, feat_cfg,
        windows_per_mix=windows_per_mix, window_strategy=window_strategy, verbose=False
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_windows,
        worker_init_fn=worker_init_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=False,
        prefetch_factor=(2 if num_workers > 0 else None),
    )

    if verbose_iter:
        print(f"[loader] online windows: batch_size={batch_size}, workers={num_workers}, W/mix={windows_per_mix}", flush=True)

    return loader


# ======================= PRECOMPUTE (window shards) =======================

class ShardCache:
    def __init__(self, max_items=2):
        self.d = {}
        self.max = max_items
        self.order = []
    def get(self, path):
        if path in self.d:
            self.order.remove(path); self.order.append(path)
            return self.d[path]
        obj = torch.load(path, map_location="cpu")
        self.d[path] = obj
        self.order.append(path)
        if len(self.order) > self.max:
            old = self.order.pop(0)
            self.d.pop(old, None)
        return obj

class PrecomputedWindowDataset(Dataset):
    """Reads feature windows from torch shards saved as dict of tensors."""
    def __init__(self, root: str, split: str):
        super().__init__()
        self.root  = Path(root) / split
        with open(self.root / "manifest.json", "r") as f:
            M = json.load(f)
        self.shards = M["shards"]           # [{'path': 'shard-00000.pt', 'size': 2048}, ...]
        print(self.shards)
        self.total  = M["total"]
        print(self.total)    
        for s in self.shards:
            s["path"] = str(self.root / s["path"])
        self.index = []
        for si, s in enumerate(self.shards):
            for i in range(s["size"]):
                self.index.append((si, i))
        self.cache = ShardCache(max_items=2)

    def __len__(self): return self.total

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        #print(idx)
        si, li = self.index[idx]
        #print(si,li)
        shard = self.cache.get(self.shards[si]["path"])
        return {
            "features":     shard["features"][li],
            "vad":          shard["vad"][li],
            "srp_phat":     shard["srp_phat"][li],
            "srp_phat_vad": shard["srp_phat_vad"][li],
            "ground_truth": shard["ground_truth"][li],
        }

def _flush_shard(buf: Dict[str, List[torch.Tensor]], out_dir: Path, shard_id: int):
    blob = {k: torch.cat(v, dim=0) for k, v in buf.items()}
    fname = f"shard-{shard_id:05d}.pt"
    torch.save(blob, out_dir / fname)
    return {"path": fname, "size": int(blob["features"].shape[0])}

def precompute_feature_windows(
    out_root: str,
    split: str,
    base_ds: OnTheFlyMixtureDataset,
    mic_xy: np.ndarray,
    stft_cfg: STFTCfg,
    feat_cfg: FeatCfg,
    windows_per_mix: int = 2,
    shard_size: int = 2048,
    num_workers: int = 8,
    verbose: bool = True
):
    """
    Walk mixtures once, compute window features, and write sharded .pt files + manifest.json

    IMPORTANT CHANGE:
      - GT is multi-hot from per-source VAD (same as online).
      - Mixture VAD is exported but not used to gate GT.
    """
    out_dir = Path(out_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    mix_loader = DataLoader(
        base_ds, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=collate_mix, worker_init_fn=worker_init_fn, drop_last=False,
        persistent_workers=(num_workers > 0), prefetch_factor=(2 if num_workers>0 else None)
    )

    shards_meta: List[Dict[str, Any]] = []
    shard_id = 0
    total_windows = 0
    buf = {k: [] for k in ["features","vad","srp_phat","srp_phat_vad","ground_truth"]}

    if verbose:
        print(f"[precompute] start: split={split}, epoch_size={len(base_ds)}, W/mix={windows_per_mix}, shard_size={shard_size}")
    print(f'len de base ds-------------{len(base_ds)}-------------------')
    pbar = tqdm(total=len(base_ds), desc=f"precompute:{split}", unit="mix")
    for (mix_batch, meta_batch) in mix_loader:
        mix = mix_batch[0].numpy().astype(np.float32)          # [4,T]
        meta = meta_batch[0]
        vad_detector = _get_vad_cached(stft_cfg.sr)

        # STFT / feats / mixture VAD
        X, freqs, feats, vad_probs = _compute_stft_feats_vad(mix, stft_cfg, vad_detector, verbose=False)
        T_total, _, F = X.shape
        T = feat_cfg.max_frames
        K = feat_cfg.K
        starts = _pick_window_starts(T_total, T, windows_per_mix, "random")
        dtype_np = _np_dtype(feat_cfg.use_fp16)

        # Per-source VADs aligned to STFT length
        src_vads: List[np.ndarray] = []
        for dry_fp16 in meta.get("dry_sources_fp16", []):
            dry = np.asarray(dry_fp16, dtype=np.float32)
            v = vad_detector.vad_probs_same_frame_count(
                dry, stft_cfg.sr,
                hop_ms=stft_cfg.hop_s * 1000.0,
                base_win_ms=stft_cfg.win_s * 1000.0,
                n_frames=X.shape[0]
            ).astype(dtype_np)
            src_vads.append(v)
        az_bins = meta.get("azimuths_deg", []) #[quantize_az_deg(az, K) for az in meta.get("azimuths_deg", [])]
        #print(f'starts-------------{starts}-------------------')
        # accumulate W windows
        for s0 in starts:
            s1 = s0 + T
            X_win        = X[s0:s1]
            feats_win    = feats[s0:s1]
            vad_win      = vad_probs[s0:s1]
            srp_vec, srp_vad_vec = _srp_maps_for_window(X_win, freqs, mic_xy, vad_win, K)

            feat_w, vad_w, srp_w, srp_vad_w = _pack_windows(
                feats_win, vad_win, srp_vec, srp_vad_vec, T, F, K, dtype_np
            )
            #print('------- I\'m here ----------')

            # #--- Multi-source GT from per-source VAD (no mixture gating) ---
            # gt_w = np.zeros((T, K), dtype=dtype_np)
            # if az_bins and src_vads:
            #     for k_bin, vsrc_full in zip(az_bins, src_vads):
            #         vsrc = vsrc_full[s0:s1]
            #         active = (vsrc >= feat_cfg.vad_threshold).astype(dtype_np)
            #         gt_w[:, k_bin] += active
            #     gt_w = np.clip(gt_w, 0, 1)


            #gt_w = np.zeros((T, K), dtype=dtype_np)
            if az_bins:
                # az_bins can be a single int or a list of ints (one per source)
                bins = np.atleast_1d(az_bins).astype(int)
                gt_w =  np.array(az_bins[0])  # set ALL frames in those bins to 1


                # print(torch.from_numpy(srp_w).shape)
                # print(torch.from_numpy(gt_w).shape)
                # srp = torch.from_numpy(srp_w)
                # gt  = torch.from_numpy(gt_w)
                # print(gt)
                # #gt  = (gt > 0.5).astype(np.uint8)  # keep it 0/1

                # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))

                # axes[0].imshow(srp, cmap='gray', aspect='auto')
                # axes[0].set_title(f"srp_w {getattr(srp, 'shape', None)}")
                # axes[0].axis('off')

                # axes[1].imshow(gt, cmap='gray', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
                # axes[1].set_title(f"gt_w {getattr(gt, 'shape', None)} (binary)")
                # axes[1].axis('off')

                # plt.tight_layout()
                # plt.show()


                #print('------- I\'m concatnaiting ----------')
                buf["features"].append(torch.from_numpy(feat_w)[None])
                buf["vad"].append(torch.from_numpy(vad_w)[None])
                buf["srp_phat"].append(torch.from_numpy(srp_w)[None])
                buf["srp_phat_vad"].append(torch.from_numpy(srp_vad_w)[None])
                buf["ground_truth"].append(torch.from_numpy(gt_w)[None])
                total_windows += 1
                #print(torch.from_numpy(vad_w)[None].shape)
                #import matplotlib.pyplot as plt
                #plt.plot(torch.from_numpy(vad_w)[None][0,:,0].detach().cpu().numpy())
                #plt.show()
                # shard flush check on every append keeps shards tight
                if len(buf["features"]) >= shard_size:
                    shard_meta = _flush_shard(buf, out_dir, shard_id)
                    shards_meta.append(shard_meta)
                    shard_id += 1
                    buf = {k: [] for k in ["features","vad","srp_phat","srp_phat_vad","ground_truth"]}

        pbar.update(1)

    if buf["features"]:
        shard_meta = _flush_shard(buf, out_dir, shard_id)
        shards_meta.append(shard_meta)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump({"total": int(total_windows), "shards": shards_meta}, f, indent=2)

    pbar.close()
    if verbose:
        print(f"[precompute] done: windows={total_windows}, shards={len(shards_meta)}, out={out_dir}")


# ======================= convenience loader for precomputed =======================

def create_precomputed_dataloader(
    precomp_root: str,
    split: str,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    persistent_workers: Optional[bool] = None,
):
    """
    Helper to mirror your train_utils usage but backed by PrecomputedWindowDataset.
    """
    ds = PrecomputedWindowDataset(precomp_root, split)
    if persistent_workers is None:
        persistent_workers = (num_workers > 0)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )
    return loader


# ======================= CLI TESTS (prints + tqdm) =======================

def _print_batch_shapes(tag: str, batch):
    feat, vad, srp, srpv, gt = batch
    print(f"\n[{tag}] shapes:")
    print(f"  features:       {tuple(feat.shape)}  (B, 12, T, F)")
    print(f"  vad:            {tuple(vad.shape)}   (B, T, F)")
    print(f"  srp_phat:       {tuple(srp.shape)}   (B, T, K)")
    print(f"  srp_phat_vad:   {tuple(srpv.shape)}  (B, T, K)")
    print(f"  ground_truth:   {tuple(gt.shape)}    (B, T, K)", flush=True)

def _load_yaml(p: str) -> Dict[str, Any]:
    import yaml
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _mic_xy_from_cfgs(train_cfg: Dict[str, Any], constraint_cfg: Dict[str, Any]) -> np.ndarray:
    # prefer train.yaml microphone.positions if present, else constraint array.mics (mm->m)
    mic = train_cfg.get("microphone", {}).get("positions")
    if mic:
        return np.asarray(mic, dtype=float)
    mics_mm = constraint_cfg.get("array", {}).get("mics", [])
    return np.asarray([[m["x_mm"]/1000.0, m["y_mm"]/1000.0] for m in mics_mm], dtype=float)

def _build_base_ds(train_cfg: Dict[str, Any], split: str, epoch_size: int) -> OnTheFlyMixtureDataset:
    ds_cfg = train_cfg.get("dataset", {})
    return OnTheFlyMixtureDataset(
        rir_root=ds_cfg["rir_root"],
        split=split,
        speech_root=ds_cfg["speech_root"],
        local_noises_root=ds_cfg.get("local_root", ds_cfg.get("noise_root")),
        ambiences_root=ds_cfg["amb_root"],
        epoch_size=epoch_size,
        base_seed=ds_cfg.get("seed", 56),
    )


# ======================= main =======================


if __name__ == "__main__":
    import argparse, os
    from tqdm import tqdm
    import torch
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser("Feature dataloaders (online & precomputed) with prints+tqdm")
    ap.add_argument("--mode", required=True, choices=["online-test", "precompute", "precomputed-test"])
    ap.add_argument("--cfg", default="configs/train_full.yaml")
    ap.add_argument("--constraint", default="configs/constraint.yaml")
    ap.add_argument("--split", default="train", choices=["train","val","test"])

    # generation / precompute sizing
    ap.add_argument("--epoch_size", type=int, default=2048, help="number of mixtures to synthesize (online) or enumerate (precompute)")
    ap.add_argument("--windows_per_mix", type=int, default=1, help="how many T-length windows to cut from each mixture")
    ap.add_argument("--window_strategy", default="random", choices=["random","sequential"], help="window picking strategy for online-test")

    # dataloader
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--use_fp16", action="store_true")

    # precompute output
    ap.add_argument("--precomp_root", default="feature_cache")
    ap.add_argument("--shard_size", type=int, default=2048)

    # printing / sanity
    ap.add_argument("--print_batches", type=int, default=3, help="how many batches to print in test modes")
    args = ap.parse_args()

    # tame thread oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # ---- load configs & mic geometry
    train_cfg = _load_yaml(args.cfg)
    cons_cfg  = _load_yaml(args.constraint)
    mic_xy = _mic_xy_from_cfgs(train_cfg, cons_cfg)

    # ---- feature & STFT configs
    feat = train_cfg.get("features", {})
    stft_cfg = STFTCfg(
        sr=feat.get("sr", 16000),
        win_s=feat.get("win_s", 0.032),
        hop_s=feat.get("hop_s", 0.010),
        nfft=feat.get("nfft", 512),
        center=bool(feat.get("center", False)),
        window="hann",
        pad_mode="reflect",
    )
    feat_cfg = FeatCfg(
        K=feat.get("K", 72),
        max_frames=feat.get("max_frames", 64),
        vad_threshold=feat.get("vad_threshold", 0.6),
        vad_gt_masking=feat.get("vad_gt_masking", True),
        use_fp16=args.use_fp16
    )

    if args.mode == "online-test":
        # on-the-fly generation (no disk writes)
        base_ds = _build_base_ds(train_cfg, args.split, args.epoch_size)
        base_ds.set_epoch(0)

        loader = create_online_feature_loader(
            base_ds, mic_xy, stft_cfg, feat_cfg,
            batch_size=args.batch_size, num_workers=args.num_workers,
            windows_per_mix=args.windows_per_mix,
            window_strategy=args.window_strategy,
            verbose_iter=True
        )

        print("[online] iterating…", flush=True)
        for bi, batch in enumerate(tqdm(loader, total=args.print_batches, desc="online-batches")):
            _print_batch_shapes("ONLINE", batch)
            if bi + 1 >= args.print_batches:
                break

    elif args.mode == "precompute":
        # synthesize + write shards under precomp_root/<split>/
        base_ds = _build_base_ds(train_cfg, args.split, args.epoch_size)
        base_ds.set_epoch(0)

        precompute_feature_windows(
            out_root=args.precomp_root, split=args.split,
            base_ds=base_ds, mic_xy=mic_xy,
            stft_cfg=stft_cfg, feat_cfg=feat_cfg,
            windows_per_mix=args.windows_per_mix,
            shard_size=args.shard_size,
            num_workers=args.num_workers,
            verbose=True
        )
        print(f"[precompute] done → {os.path.join(args.precomp_root, args.split)}", flush=True)

    elif args.mode == "precomputed-test":
        # iterate the saved shards to check shapes & integrity
        ds = PrecomputedWindowDataset(args.precomp_root, args.split)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(args.num_workers > 0)
        )
        print("[precomputed] iterating…", flush=True)
        for bi, batch in enumerate(tqdm(loader, total=args.print_batches, desc="precomputed-batches")):
            _print_batch_shapes(
                "PRECOMPUTED",
                (batch["features"], batch["vad"], batch["srp_phat"], batch["srp_phat_vad"], batch["ground_truth"])
            )
            if bi + 1 >= args.print_batches:
                break
