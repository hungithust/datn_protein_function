#!/usr/bin/env bash
# scripts/launch_sweep.sh — run all sweep configs, one GPU each (0-7).
# Usage: bash scripts/launch_sweep.sh configs/sweep
set -euo pipefail
# Run from the current working directory (repo root). Do not hardcode a path.
SWEEP_DIR="${1:-configs/sweep}"
# file_system mp sharing (set in main.py) writes shared tensors under $TMPDIR.
# The container's /dev/shm is capped at 16G — too small for 8 cells x 8 workers
# of ~650MB ESM-2 batches. Point TMPDIR at large /raid disk instead (writable
# per hackathon rules), avoiding shm exhaustion without touching the container.
SHARE_TMP="${SWEEP_TMPDIR:-/raid/team/datn/.tmp_sweep}"
mkdir -p "$SHARE_TMP"
mapfile -t CFGS < <(ls "$SWEEP_DIR"/*.yaml | sort)
n=${#CFGS[@]}
echo "[SWEEP] $n configs over $(nvidia-smi -L | wc -l) GPUs"
gpu=0
for cfg in "${CFGS[@]}"; do
  base=$(basename "$cfg" .yaml)
  sess="sw_$base"
  tmux kill-session -t "$sess" 2>/dev/null || true
  # ulimit -n high + file_system mp sharing (set in main.py) + TMPDIR on tmpfs:
  # lets DataLoader workers run concurrently across 8 cells without exhausting
  # shm file descriptors (Bus error) or the open-file limit (Too many open files).
  tmux new-session -d -s "$sess" \
    "ulimit -n 1048576; export TMPDIR='$SHARE_TMP'; CUDA_VISIBLE_DEVICES=$gpu python main.py --config $cfg 2>&1 | tee logs/${base}.run.log"
  echo "[SWEEP] $base -> GPU $gpu"
  gpu=$(( (gpu + 1) % 8 ))
done
echo "[SWEEP] launched. monitor: watch -n1 nvidia-smi ; tmux ls"
