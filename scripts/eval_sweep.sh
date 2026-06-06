#!/usr/bin/env bash
# scripts/eval_sweep.sh — eval every v3 config in a dir on a split (uses each best.pt).
# Usage: bash scripts/eval_sweep.sh <dir> [split] [extra main.py flags...]
#   e.g. bash scripts/eval_sweep.sh configs/sweep_reg test_LT_95
set -euo pipefail
ROOT="$(pwd)"
DIR="${1:?usage: eval_sweep.sh <dir> [split] [extra...]}"
SPLIT="${2:-test}"
shift 2 2>/dev/null || shift $#
EXTRA=("$@")
for cfg in "$DIR"/*.yaml; do
  base=$(basename "$cfg" .yaml)
  echo "===== EVAL $base ($SPLIT) ${EXTRA[*]:-} ====="
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=0 python main.py --config "$cfg" \
      --eval-only --test-split "$SPLIT" "${EXTRA[@]}" 2>&1 | tee "logs/${base}.eval_${SPLIT}.log" )
done
echo "[EVAL] done. metrics JSON: results/*_reg_*.eval_${SPLIT}.json"
