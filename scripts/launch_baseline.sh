#!/usr/bin/env bash
# scripts/launch_baseline.sh — one detached tmux session per branch, one GPU each.
# Run INSIDE the jupyterlab container:
#     docker exec -it jupyterlab bash
#     bash /workspace/datn/scripts/launch_baseline.sh
set -euo pipefail
REPO_DIR=/workspace/datn
cd "$REPO_DIR"
mkdir -p logs

declare -A GPU=( [mf]=0 [bp]=1 [cc]=2 )
for b in mf bp cc; do
  g=${GPU[$b]}
  sess="train_$b"
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" \
    ". $REPO_DIR/.venv/bin/activate; cd $REPO_DIR; CUDA_VISIBLE_DEVICES=$g python main.py --config configs/${b}_v3.yaml 2>&1 | tee logs/${b}_v3_h200.log"
  echo "[LAUNCH] $sess on GPU $g -> logs/${b}_v3_h200.log"
done
echo "[LAUNCH] attach with: tmux attach -t train_mf   (detach: Ctrl-b d)"
echo "[LAUNCH] watch GPUs:  watch -n1 nvidia-smi"
