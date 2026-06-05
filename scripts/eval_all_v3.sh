#!/usr/bin/env bash
# scripts/eval_all_v3.sh — eval mf/bp/cc v3 on a test split using each best.pt.
# Usage: bash scripts/eval_all_v3.sh [split] [extra main.py flags...]
#   e.g. bash scripts/eval_all_v3.sh test_LT_95 --tune-alpha
set -euo pipefail
ROOT="$(pwd)"
SPLIT="${1:-test}"
shift || true
EXTRA=("$@")
CONFIGS=(configs/mf_v3_esm3b.yaml configs/bp_v3_esm3b.yaml configs/cc_v3_esm3b.yaml)
for cfg in "${CONFIGS[@]}"; do
  base=$(basename "$cfg" .yaml)
  echo "===== EVAL $base ($SPLIT) ${EXTRA[*]:-} ====="
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=0 python main.py --config "$cfg" \
      --eval-only --test-split "$SPLIT" "${EXTRA[@]}" 2>&1 | tee "logs/${base}.eval_${SPLIT}.log" )
done
echo "[EVAL] done. metrics JSON: results/*_v3_esm3b.eval_${SPLIT}.json"
