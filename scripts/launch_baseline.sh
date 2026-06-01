#!/usr/bin/env bash
# scripts/launch_baseline.sh — one detached tmux session per branch, one GPU each.
# Usage: bash scripts/launch_baseline.sh
set -euo pipefail
cd /raid/team/datn

declare -A GPU=( [mf]=0 [bp]=1 [cc]=2 )
for b in mf bp cc; do
  g=${GPU[$b]}
  sess="train_$b"
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" \
    "CUDA_VISIBLE_DEVICES=$g python main.py --config configs/${b}_v3.yaml 2>&1 | tee logs/${b}_v3_h200.log"
  echo "[LAUNCH] $sess on GPU $g -> logs/${b}_v3_h200.log"
done
echo "[LAUNCH] attach with: tmux attach -t train_mf   (detach: Ctrl-b d)"
echo "[LAUNCH] watch GPUs:  watch -n1 nvidia-smi"
