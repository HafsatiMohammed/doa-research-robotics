"""
Lightweight analyzer for precomputed windows with tqdm + low memory.

Example:
  python -m src.mirokai_doa.analyze_dataset \
    --precomp_root feature_cache \
    --split train \
    --batch_size 32 \
    --max_batches 200 \
    --vad_thr 0.6 \
    --out_dir analyses/train
"""

import os, math, json, argparse
import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .feature_dataloaders import create_precomputed_dataloader


def _circular_deg_dist_idx(i, j, K: int) -> torch.Tensor:
    di = i.unsqueeze(-1).to(torch.int64)
    dj = j.unsqueeze(0).to(torch.int64)
    d = (di - dj).abs()
    d = torch.minimum(d, K - d)
    return d.to(torch.float32) * (360.0 / K)


def analyze_precomputed(precomp_root: str,
                        split: str = "train",
                        batch_size: int = 256,
                        max_batches: int = 200,
                        vad_thr: float = 0.6,
                        out_dir: str = "analyses/train",
                        n_visual_samples: int = 100):
    os.makedirs(out_dir, exist_ok=True)

    # Keep workers=0, pin_memory=False to reduce RSS
    loader = create_precomputed_dataloader(
        precomp_root=precomp_root,
        split=split,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )

    # Running tallies (low memory)
    az_hist = None
    nsrc_hist = np.zeros(4, dtype=np.int64)  # frames with 0,1,2,>=3 GT sources
    tp = fp = fn = tn = 0
    n_frames_total = 0
    n_frames_voiced_mix = 0
    n_frames_voiced_gt  = 0

    # Online histogram for SRP confidence (no giant arrays)
    conf_edges = np.linspace(0, 1, 41)
    conf_counts = np.zeros(conf_edges.size - 1, dtype=np.int64)

    srp_top1_within = {5: 0, 10: 0, 15: 0}
    srp_top1_den    = 0

    # tiny buffers for a couple of visualizations
    visual = []

    it = tqdm(loader, total=max_batches, desc=f"{split} scan", leave=False)
    for bi, batch in enumerate(it):
        if bi >= max_batches:
            break
        
        # Drop the heavy blob ASAP (we don't analyze it)
        batch.pop("features", None)

        vad = batch["vad"]             # [B,T,F] float
        srp = batch["srp_phat"]        # [B,T,K]
        gt  = batch["ground_truth"]    # [B,T,K]

        # All CPU tensors here; keep them as tensors for vectorized ops
        B, T, F = vad.shape
        K = srp.size(-1)
        if az_hist is None:
            az_hist = np.zeros(K, dtype=np.int64)

        mix_vad = (vad[..., 0] >= vad_thr)    # [B,T] bool
        gt_any  = (gt.sum(dim=-1) > 0)        # [B,T] bool

        # VAD confusion and counts
        n_frames_total     += int(B * T)
        n_frames_voiced_mix += int(mix_vad.sum().item())
        n_frames_voiced_gt  += int(gt_any.sum().item())

        tp += int(((mix_vad) & (gt_any)).sum().item())
        fp += int(((mix_vad) & (~gt_any)).sum().item())
        fn += int(((~mix_vad) & (gt_any)).sum().item())
        tn += int(((~mix_vad) & (~gt_any)).sum().item())

        # sources per frame
        nsrc = gt.sum(dim=-1).clamp(max=3).to(torch.int64)  # [B,T], 3=3+
        for kbin in (0,1,2,3):
            nsrc_hist[kbin] += int((nsrc == kbin).sum().item())

        # azimuth coverage
        az_hist += gt.sum(dim=(0,1)).cpu().numpy().astype(np.int64)

        # SRP confidence (normalized entropy) → online histogram
        ps = srp / (srp.sum(dim=-1, keepdim=True) + 1e-9)  # [B,T,K]
        H = -(ps.clamp_min(1e-9) * ps.clamp_min(1e-9).log()).sum(dim=-1)   # [B,T]
        conf = 1.0 - H / math.log(K)                                       # [B,T]
        cvals = conf.reshape(-1).cpu().numpy()
        c_hist, _ = np.histogram(cvals, bins=conf_edges)
        conf_counts += c_hist

        # SRP top-1 vs GT (within thresholds)
        top1 = ps.argmax(dim=-1)  # [B,T]
        # only on GT-voiced frames
        for b in range(B):
            voiced_idx = torch.nonzero(gt_any[b], as_tuple=False).flatten()
            if voiced_idx.numel() == 0:
                continue
            for t in voiced_idx.tolist():
                gt_idx = torch.nonzero(gt[b, t] > 0.5, as_tuple=False).flatten()
                if gt_idx.numel() == 0:
                    continue
                dmin = _circular_deg_dist_idx(top1[b, t], gt_idx, K).min().item()
                for thr in (5,10,15):
                    if dmin <= thr:
                        srp_top1_within[thr] += 1
                srp_top1_den += 1

        # Keep a couple visuals (very small slices)
        if len(visual) < n_visual_samples:
            # Take only the first item and to(CPU) numpy
            srp0 = (ps[0].cpu().numpy()).astype(np.float32)  # [T,K]
            gt0  = (gt[0].cpu().numpy()).astype(np.float32)  # [T,K]
            visual.append((srp0, gt0))

        # free references quickly
        del vad, srp, gt, ps, H, conf, cvals, c_hist, top1, nsrc, mix_vad, gt_any

    # summarize
    vad_total = tp + fp + fn + tn
    vad_acc = (tp + tn) / max(1, vad_total)
    vad_prec = tp / max(1, tp + fp)
    vad_rec  = tp / max(1, tp + fn)
    vad_f1   = (2 * vad_prec * vad_rec) / max(1e-9, (vad_prec + vad_rec))

    srp_within = {str(k): float(srp_top1_within[k] / max(1, srp_top1_den)) for k in (5,10,15)}
    conf_stats = {}
    # approx stats from histogram midpoints
    mids = 0.5 * (conf_edges[:-1] + conf_edges[1:])
    totalc = conf_counts.sum()
    if totalc > 0:
        mean = float((mids * conf_counts).sum() / totalc)
    else:
        mean = float("nan")
    conf_stats["hist_edges"] = conf_edges.tolist()
    conf_stats["hist_counts"] = [int(x) for x in conf_counts.tolist()]
    conf_stats["mean"] = mean
    conf_stats["n"] = int(totalc)

    summary = {
        "frames_total": int(n_frames_total),
        "frames_voiced_by_mixtureVAD": int(n_frames_voiced_mix),
        "frames_voiced_by_GT": int(n_frames_voiced_gt),
        "vad_confusion": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
        "vad_metrics": {"acc": float(vad_acc), "prec": float(vad_prec), "rec": float(vad_rec), "f1": float(vad_f1)},
        "nsrc_hist": {"0": int(nsrc_hist[0]), "1": int(nsrc_hist[1]), "2": int(nsrc_hist[2]), "3_or_more": int(nsrc_hist[3])},
        "azimuth_hist_minmax": {"min": int(az_hist.min()), "max": int(az_hist.max())} if az_hist is not None else None,
        "srp_top1_within_deg": srp_within,
        "srp_confidence_hist": conf_stats,
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # plots (tiny)
    if az_hist is not None:
        plt.figure(figsize=(10,3))
        plt.bar(np.arange(az_hist.size), az_hist, width=0.9)
        plt.title("Azimuth GT bin coverage (counts per K bin)")
        plt.xlabel("bin (0..K-1)"); plt.ylabel("count")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "azimuth_hist.png")); plt.close()

    plt.figure(figsize=(4,3))
    labels = ["0","1","2","3+"]
    vals = nsrc_hist / max(1, nsrc_hist.sum())
    plt.bar(labels, vals)
    plt.title("#sources per frame (fraction)")
    plt.ylim(0,1); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nsrc_hist.png")); plt.close()

    if conf_counts.sum() > 0:
        plt.figure(figsize=(4,3))
        centers = 0.5 * (conf_edges[:-1] + conf_edges[1:])
        plt.bar(centers, conf_counts / conf_counts.sum(), width=centers[1]-centers[0])
        plt.title("SRP normalized confidence (1 - H/logK)")
        plt.xlabel("confidence"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "srp_conf_hist.png")); plt.close()

    for si, (srp0, gt0) in enumerate(visual):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
        im0 = axes[0].imshow(srp0.T, aspect="auto", origin="lower")
        axes[0].set_title("SRP p(az|x)"); axes[0].set_xlabel("time"); axes[0].set_ylabel("az bin")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(gt0.T, aspect="auto", origin="lower")
        axes[1].set_title("GT (multi-hot)"); axes[1].set_xlabel("time")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
        fig.suptitle(f"Sample {si}")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"srp_vs_gt_sample{si}.png"))
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precomp_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_batches", type=int, default=200)
    ap.add_argument("--vad_thr", type=float, default=0.6)
    ap.add_argument("--out_dir", type=str, default="analyses/train")
    args = ap.parse_args()

    analyze_precomputed(
        precomp_root=args.precomp_root,
        split=args.split,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        vad_thr=args.vad_thr,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
