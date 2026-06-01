#!/usr/bin/env bash
# scripts/pull_kaggle_data.sh — download precomputed AMPR datasets.
# Run INSIDE the jupyterlab container after server_setup.sh, from your checkout:
#     docker exec -it jupyterlab bash
#     cd /workspace/datn && bash scripts/pull_kaggle_data.sh
# (paths are derived from this script's location — no hardcoded base dir)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KAGGLE_CONFIG_DIR="$REPO_DIR/.kaggle"
# shellcheck disable=SC1091
. "$REPO_DIR/.venv/bin/activate"

DATA="$REPO_DIR/data"
mkdir -p "$DATA/embeddings" "$DATA/contact_maps" "$DATA/pdbch" "$DATA/_dl"
cd "$DATA/_dl"

# whole-dataset download + unzip — for small multi-file datasets.
pull () {  # $1 = dataset slug, $2 = subdir under _dl
  echo "[PULL] $1 (full dataset)"
  kaggle datasets download -d "$1" -p "$2" --unzip
}

# single-file download — for the big ones. Streams one file directly (no
# server-side whole-archive prep that stalls on 50GB datasets) and skips any
# junk files in the dataset. Unzips explicitly so progress is visible.
# Idempotent: skips download if the final file already exists.
pull_file () {  # $1 = slug, $2 = filename, $3 = subdir under _dl
  echo "[PULL] $1 :: $2"
  mkdir -p "$3"
  if [ -f "$3/$2" ]; then echo "[PULL] $2 already present — skip"; return; fi
  kaggle datasets download -d "$1" -f "$2" -p "$3"
  if [ -f "$3/$2.zip" ]; then
    echo "[PULL] unzip $2.zip"
    unzip -o "$3/$2.zip" -d "$3"
    rm -f "$3/$2.zip"
  fi
}

pull_file hungnguyenviet04/ampr-phase3-embeddings esm2_residue.h5 emb1
pull      hungnguyenviet04/ampr-phase3-embeddings-2                emb2
pull_file hungnguyenviet04/cmap-all               cmap_all.h5     cmap
pull      hungnguyenviet04/ampr-pdbch-phase0                      pdbch

echo "[PULL] place files at config-expected paths"
ln -sf "$DATA/_dl/emb1/esm2_residue.h5"        "$DATA/embeddings/esm2_residue.h5"
ln -sf "$DATA/_dl/emb2/ppi_deepgo.npy"         "$DATA/embeddings/ppi_deepgo.npy"
ln -sf "$DATA/_dl/emb2/ppi_deepgo_mask.npy"    "$DATA/embeddings/ppi_deepgo_mask.npy"
ln -sf "$DATA/_dl/cmap/cmap_all.h5"            "$DATA/contact_maps/cmap_all.h5"
for f in labels_mf labels_bp labels_cc dag_matrix_mf dag_matrix_bp dag_matrix_cc; do
  ln -sf "$DATA/_dl/pdbch/$f.npy"  "$DATA/pdbch/$f.npy"
done
ln -sf "$DATA/_dl/pdbch/splits.json"         "$DATA/pdbch/splits.json"
ln -sf "$DATA/_dl/pdbch/protein_order.json"  "$DATA/pdbch/protein_order.json"
for b in mf bp cc; do
  ln -sf "$DATA/_dl/pdbch/go_emb_$b.npy" "$DATA/embeddings/go_emb_$b.npy"
done
echo "[PULL] done — review any missing symlink targets above"
ls -lL "$DATA/embeddings" "$DATA/contact_maps" "$DATA/pdbch"
