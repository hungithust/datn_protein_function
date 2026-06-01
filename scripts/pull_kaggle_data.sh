#!/usr/bin/env bash
# scripts/pull_kaggle_data.sh — download precomputed AMPR datasets to /raid.
set -euo pipefail
DATA=/raid/team/datn/data
mkdir -p "$DATA/embeddings" "$DATA/contact_maps" "$DATA/pdbch" "$DATA/_dl"
cd "$DATA/_dl"

pull () {  # $1 = dataset slug, $2 = subdir under _dl
  echo "[PULL] $1"
  kaggle datasets download -d "$1" -p "$2" --unzip
}

pull hungnguyenviet04/ampr-phase3-embeddings   emb1
pull hungnguyenviet04/ampr-phase3-embeddings-2 emb2
pull hungnguyenviet04/cmap-all                 cmap
pull hungnguyenviet04/ampr-pdbch-phase0        pdbch

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
