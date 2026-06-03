#!/usr/bin/env bash
# scripts/train_all_v3.sh — train mf/bp/cc v3 (esm3b) concurrently, one GPU each.
# Winning architecture from the MF sweep (both / d_hidden=512 / go_emb comb) + LR scheduler.
# Usage: bash scripts/train_all_v3.sh
set -euo pipefail
ROOT="$(pwd)"
SHARE_TMP="${SWEEP_TMPDIR:-/raid/team/datn/.tmp_sweep}"
mkdir -p "$SHARE_TMP"
CONFIGS=(configs/mf_v3_esm3b.yaml configs/bp_v3_esm3b.yaml configs/cc_v3_esm3b.yaml)
gpu=0
for cfg in "${CONFIGS[@]}"; do
  base=$(basename "$cfg" .yaml)
  sess="tr_$base"
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" \
    "cd '$ROOT'; ulimit -n 1048576; export TMPDIR='$SHARE_TMP'; CUDA_VISIBLE_DEVICES=$gpu python main.py --config $cfg 2>&1 | tee logs/${base}.train.run.log"
  echo "[TRAIN] $base -> GPU $gpu  (session $sess)"
  gpu=$((gpu + 1))
done
echo "[TRAIN] launched mf/bp/cc on GPU 0/1/2. monitor: tmux ls ; tail -f logs/*_v3_esm3b.train.run.log"
