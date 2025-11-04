#!/usr/bin/env python3
# precompute_features.py
"""
Precompute DoA features into sharded files for train/val/test.

It relies on your existing mix_batcher.py to:
  - build the OnTheFlyMixtureDataset
  - compute fast features (compute_batch_features_fast)

Output layout:
  <out_root>/
    train/
      shard-000000.pt
      shard-000001.pt
      ...
      manifest.jsonl
    val/
      ...
    test/
      ...

Each shard-*.pt contains a dict:
  {
    "features":     List[Tensor[S_i, 1, 12, T, F]],
    "vad":          List[Tensor[S_i, 1, T, F]],
    "srp_phat":     List[Tensor[S_i, 1, T, K]],
    "srp_phat_vad": List[Tensor[S_i, 1, T, K]],
    "ground_truth": List[Tensor[S_i, 1, T, K]],
    "meta":         List[Dict],  (stub by default)
    "count": int,
    "cfg": {...}
  }

Usage example:
  python precompute_features.py \
    --rir_root /data/rir_bank \
    --speech_root /data/LibriSpeech \
    --local_root /data/noise \
    --amb_root /data/Ambiances \
    --train_size 100000 --val_size 4000 --test_size 4000 \
    --batch_size 16 --num_workers 4 \
    --out_root ./features_v1 \
    --sr 16000 --win_s 0.032 --hop_s 0.010 --nfft 512 \
    --K 72 --max_frames 6 --vad_threshold 0.6
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

# Import your pipeline pieces
# Ensure mix_batcher.py is importable (same folder or PYTHONPATH).
from .mix_batcher import (
    OnTheFlyMixtureDataset,
    create_feature_dataloader_fast,  # uses compute_batch_features_fast
)


# ---------------------- Writer ----------------------

class FeatureShardWriter:
    """
    Collects fixed-shape feature slices into shard files for fast training.
    Each shard is a torch.save() of a dict with lists of tensors and metas.

    File layout:
      <out_root>/<split>/shard-000000.pt
      <out_root>/<split>/shard-000001.pt
      ...
      <out_root>/<split>/manifest.jsonl
    """
    def __init__(self, out_dir: str, split: str, shard_size: int = 1024, cfg_header: Dict[str, Any] | None = None):
        self.split_dir = os.path.join(out_dir, split)
        os.makedirs(self.split_dir, exist_ok=True)
        self.shard_size = int(shard_size)
        self.cfg_header = cfg_header or {}
        self._reset_buffer()
        self.shard_idx = 0
        self.manifest_path = os.path.join(self.split_dir, "manifest.jsonl")
        # header line (optional)
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps({
                "type": "header",
                "created_utc": datetime.utcnow().isoformat() + "Z",
                "split": split,
                "cfg": self.cfg_header
            }) + "\n")

    def _reset_buffer(self):
        self.buf = {
            "features": [],
            "vad": [],
            "srp_phat": [],
            "srp_phat_vad": [],
            "ground_truth": [],
            "meta": []
        }

    def _flush(self):
        if len(self.buf["features"]) == 0:
            return
        shard_name = f"shard-{self.shard_idx:06d}.pt"
        shard_path = os.path.join(self.split_dir, shard_name)
        payload = {
            "features": self.buf["features"],
            "vad": self.buf["vad"],
            "srp_phat": self.buf["srp_phat"],
            "srp_phat_vad": self.buf["srp_phat_vad"],
            "ground_truth": self.buf["ground_truth"],
            "meta": self.buf["meta"],
            "count": len(self.buf["features"]),
            "cfg": self.cfg_header,
        }
        torch.save(payload, shard_path)

        with open(self.manifest_path, "a") as f:
            f.write(json.dumps({
                "type": "shard",
                "file": shard_name,
                "count": payload["count"]
            }) + "\n")

        self.shard_idx += 1
        self._reset_buffer()

    def add_batch(self, features: torch.Tensor, vad: torch.Tensor, srp: torch.Tensor, srp_vad: torch.Tensor, gt: torch.Tensor, metas: List[Dict[str, Any]]):
        """
        features/vad/srp/srp_vad/gt shape: [S, 1, ...]
        metas: python list aligned with S (same count). If you don't have rich metas, pass stubs.
        """
        assert features.shape[0] == vad.shape[0] == srp.shape[0] == srp_vad.shape[0] == gt.shape[0], "S mismatch"
        S = features.shape[0]
        # move to cpu to persist
        features = features.detach().cpu()
        vad = vad.detach().cpu()
        srp = srp.detach().cpu()
        srp_vad = srp_vad.detach().cpu()
        gt = gt.detach().cpu()
        print(features)
        print(vad)
        print(srp)
        print(srp_vad)
        print(gt)

        for i in range(S):
            self.buf["features"].append(features[i])
            self.buf["vad"].append(vad[i])
            self.buf["srp_phat"].append(srp[i])
            self.buf["srp_phat_vad"].append(srp_vad[i])
            self.buf["ground_truth"].append(gt[i])
            self.buf["meta"].append(metas[i])
            if len(self.buf["features"]) >= self.shard_size:
                self._flush()

    def close(self):
        self._flush()


# ---------------------- Precompute ----------------------

def default_mic_xy() -> np.ndarray:
    # Seeed Mic Array v2 4ch default used in your codebase
    return np.array([
        [ 0.0277,  0.0   ],  # Mic 0 (A): 0°
        [ 0.0,     0.0277],  # Mic 1 (B): 90°
        [-0.0277,  0.0   ],  # Mic 2 (C): 180°
        [ 0.0,    -0.0277],  # Mic 3 (D): 270°
    ], dtype=float)


@dataclass
class SplitPlan:
    name: str
    epoch_size: int


def precompute_split(
    split: SplitPlan,
    out_root: str,
    rir_root: str,
    speech_root: str,
    local_root: str,
    amb_root: str,
    batch_size: int,
    num_workers: int,
    *,
    mic_xy: np.ndarray,
    seed: int,
    sr: int,
    win_s: float,
    hop_s: float,
    nfft: int,
    K: int,
    max_frames: int,
    vad_threshold: float,
    device: str,
    shard_size: int,
    use_fp16: bool,
) -> Dict[str, int]:
    """
    Generate features for one split and save to shards in out_root/split.name.
    Returns simple counters (batches, slices).
    """
    # Build dataset
    ds = OnTheFlyMixtureDataset(
        rir_root=rir_root,
        split=split.name,
        speech_root=speech_root,
        local_noises_root=local_root,
        ambiences_root=amb_root,
        epoch_size=split.epoch_size,
        base_seed=seed,
    )
    ds.set_epoch(0)

    # Fast feature loader: yields (features, vad, srp, srp_vad, gt)
    loader = create_feature_dataloader_fast(
        dataset=ds,
        mic_xy=mic_xy,
        batch_size=batch_size,
        num_workers=num_workers,
        sr=sr,
        win_s=win_s,
        hop_s=hop_s,
        nfft=nfft,
        K=K,
        vad_threshold=vad_threshold,
        device=device,
        max_frames=max_frames,
        vad_gt_masking=True,
        use_fp16=use_fp16,
        verbose=False,
    )

    cfg_header = {
        "sr": sr, "win_s": win_s, "hop_s": hop_s, "nfft": nfft,
        "K": K, "max_frames": max_frames, "vad_threshold": vad_threshold,
        "dtype": "fp16" if use_fp16 else "fp32",
        "device_saved": "cpu",
        "mic_xy": np.asarray(mic_xy, dtype=float).tolist(),
        "generator_split": split.name,
        "epoch_size": split.epoch_size,
        "batch_size": batch_size
    }
    writer = FeatureShardWriter(out_root, split.name, shard_size=shard_size, cfg_header=cfg_header)

    num_batches = len(loader)
    total_slices = 0
    pbar = tqdm(total=num_batches, desc=f"[{split.name}] precompute", unit="batch")

    for (features, vad, srp, srp_vad, gt) in loader:
        S = features.shape[0]
        total_slices += S
        # minimal aligned metas (stub). If you want rich per-slice metas,
        # modify compute_batch_features_fast to return metas_per_slice.
        metas_stub = [{"split": split.name}] * S
        writer.add_batch(features, vad, srp, srp_vad, gt, metas_stub)
        pbar.update(1)

    pbar.close()
    writer.close()
    return {"batches": num_batches, "slices": total_slices}


def precompute_all(
    out_root: str,
    rir_root: str,
    speech_root: str,
    local_root: str,
    amb_root: str,
    train_size: int,
    val_size: int,
    test_size: int,
    batch_size: int,
    num_workers: int,
    *,
    mic_xy: np.ndarray | None = None,
    seed: int = 56,
    sr: int = 16000,
    win_s: float = 0.032,
    hop_s: float = 0.010,
    nfft: int = 512,
    K: int = 72,
    max_frames: int = 6,
    vad_threshold: float = 0.6,
    device: str = "cpu",
    shard_size: int = 1024,
    use_fp16: bool = False,
):
    os.makedirs(out_root, exist_ok=True)
    if mic_xy is None:
        mic_xy = default_mic_xy()

    splits = [
        SplitPlan("train", train_size),
        SplitPlan("val",   val_size),
        SplitPlan("test",  test_size),
    ]

    all_stats = {}
    for sp in splits:
        stats = precompute_split(
            split=sp,
            out_root=out_root,
            rir_root=rir_root,
            speech_root=speech_root,
            local_root=local_root,
            amb_root=amb_root,
            batch_size=batch_size,
            num_workers=num_workers,
            mic_xy=mic_xy,
            seed=seed,
            sr=sr, win_s=win_s, hop_s=hop_s, nfft=nfft,
            K=K, max_frames=max_frames, vad_threshold=vad_threshold,
            device=device, shard_size=shard_size, use_fp16=use_fp16,
        )
        all_stats[sp.name] = stats

    print("\n=== Precompute summary ===")
    for name, st in all_stats.items():
        print(f"  {name:5s}: batches={st['batches']}, slices={st['slices']}")
    print(f"Output root: {out_root}")


# ---------------------- Reader demo (optional) ----------------------

class ShardedFeatureDataset(torch.utils.data.Dataset):
    """
    Reads precomputed shards. Keeps one shard cached in memory.
    """
    def __init__(self, split_dir: str):
        manifest = os.path.join(split_dir, "manifest.jsonl")
        if not os.path.isfile(manifest):
            raise FileNotFoundError(f"manifest.jsonl not found in {split_dir}")
        self.split_dir = split_dir
        self.shards = []
        with open(manifest, "r") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("type") == "shard":
                    self.shards.append((obj["file"], int(obj["count"])))
        # Build global index
        self.index = []
        for si, (_, count) in enumerate(self.shards):
            for i in range(count):
                self.index.append((si, i))
        self._cache = None
        self._cache_si = None

    def __len__(self):
        return len(self.index)

    def _load_shard(self, si):
        if self._cache_si == si and self._cache is not None:
            return self._cache
        fname, _ = self.shards[si]
        path = os.path.join(self.split_dir, fname)
        payload = torch.load(path, map_location="cpu")
        self._cache = payload
        self._cache_si = si
        return payload

    def __getitem__(self, idx):
        si, i = self.index[idx]
        shard = self._load_shard(si)
        x = {
            "features": shard["features"][i],       # [1,12,T,F]
            "vad": shard["vad"][i],                 # [1,T,F]
            "srp_phat": shard["srp_phat"][i],       # [1,T,K]
            "srp_phat_vad": shard["srp_phat_vad"][i],
            "ground_truth": shard["ground_truth"][i],
            "meta": shard["meta"][i],
        }
        return x


def reader_demo(out_root: str, split: str, batch_size: int = 64, num_workers: int = 2):
    split_dir = os.path.join(out_root, split)
    ds = ShardedFeatureDataset(split_dir)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    pbar = tqdm(total=len(loader), desc=f"[{split}] read_demo", unit="batch")
    for batch in loader:
        # Simulate a cheap training step
        _ = batch["features"].shape
        pbar.update(1)
    pbar.close()
    print(f"Read {len(ds)} records from {split_dir}.")


# ---------------------- CLI ----------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Precompute DoA features to shards (train/val/test)")
    ap.add_argument("--rir_root", required=False, default="/path/to/rir_bank")
    ap.add_argument("--speech_root", required=False, default="/path/to/LibriSpeech")
    ap.add_argument("--local_root", required=False, default="/path/to/noise")
    ap.add_argument("--amb_root", required=False, default="/path/to/Ambiances")

    ap.add_argument("--out_root", required=False, default="./features_v1")
    ap.add_argument("--train_size", type=int, default=100000)
    ap.add_argument("--val_size",   type=int, default=4000)
    ap.add_argument("--test_size",  type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--shard_size", type=int, default=1024)

    ap.add_argument("--seed", type=int, default=56)

    # Feature params
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--win_s", type=float, default=0.032)
    ap.add_argument("--hop_s", type=float, default=0.010)
    ap.add_argument("--nfft", type=int, default=512)
    ap.add_argument("--K", type=int, default=72)
    ap.add_argument("--max_frames", type=int, default=6)
    ap.add_argument("--vad_threshold", type=float, default=0.6)
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--device", default="cpu")  # features are saved on CPU

    ap.add_argument("--mode", choices=["precompute", "read_demo"], default="precompute")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train", help="used in read_demo")

    args = ap.parse_args()

    if args.mode == "precompute":
        precompute_all(
            out_root=args.out_root,
            rir_root=args.rir_root,
            speech_root=args.speech_root,
            local_root=args.local_root,
            amb_root=args.amb_root,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mic_xy=default_mic_xy(),
            seed=args.seed,
            sr=args.sr,
            win_s=args.win_s,
            hop_s=args.hop_s,
            nfft=args.nfft,
            K=args.K,
            max_frames=args.max_frames,
            vad_threshold=args.vad_threshold,
            device=args.device,
            shard_size=args.shard_size,
            use_fp16=(args.dtype == "fp16"),
        )
    else:
        reader_demo(args.out_root, args.split)


if __name__ == "__main__":
    main()
