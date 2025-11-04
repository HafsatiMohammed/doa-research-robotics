#!/usr/bin/env bash
set -euo pipefail

# ---- config (edit if needed) ----
CFG="configs/train.yaml"
CONSTRAINT="configs/constraint.yaml"
PRECOMP_ROOT="feature_cache_stateofart"

# Mixtures (4 s each): ~30h/5h/5h
TRAIN_MIXES=30000
VAL_MIXES=9500
TEST_MIXES=9500

NUM_WORKERS=8
WINDOWS_PER_MIX=16
SHARD_SIZE=100

# ---- run ----
echo "[train] precompute..."
python -m src.mirokai_doa.feature_dataloaders \
  --mode precompute \
  --cfg "$CFG" \
  --constraint "$CONSTRAINT" \
  --split train \
  --epoch_size "$TRAIN_MIXES" \
  --num_workers "$NUM_WORKERS" \
  --windows_per_mix "$WINDOWS_PER_MIX" \
  --precomp_root "$PRECOMP_ROOT" \
  --shard_size "$SHARD_SIZE"

echo "[val] precompute..."
python -m src.mirokai_doa.feature_dataloaders \
  --mode precompute \
  --cfg "$CFG" \
  --constraint "$CONSTRAINT" \
  --split val \
  --epoch_size "$VAL_MIXES" \
  --num_workers "$NUM_WORKERS" \
  --windows_per_mix "$WINDOWS_PER_MIX" \
  --precomp_root "$PRECOMP_ROOT" \
  --shard_size "$SHARD_SIZE"

echo "[test] precompute..."
python -m src.mirokai_doa.feature_dataloaders \
  --mode precompute \
  --cfg "$CFG" \
  --constraint "$CONSTRAINT" \
  --split test \
  --epoch_size "$TEST_MIXES" \
  --num_workers "$NUM_WORKERS" \
  --windows_per_mix "$WINDOWS_PER_MIX" \
  --precomp_root "$PRECOMP_ROOT" \
  --shard_size "$SHARD_SIZE"

echo "Done."
