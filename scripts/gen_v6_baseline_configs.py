#!/usr/bin/env python
"""Generate 650M PDB-30K-only baseline configs (NO SWISS-MODEL pretrain).

Purpose: isolate the data-expansion effect from the 3B->650M backbone change.
These configs use the *identical* finetune recipe as the v6sm finetune
(650M, d_model 1280, 60 epochs, lr 3e-4, ASL+DAG, DIAMOND ensemble) but are
trained from scratch on 30K PDB — i.e. run WITHOUT --init-from. Comparing
AMPR-B (220K->30K) against this baseline at the same 650M backbone tells us
whether the SWISS-MODEL pretrain helped, hurt, or was neutral.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.gen_v6_configs import BRANCHES, SEEDS, _common_model, _training


def generate(out_dir, pdb_emb="data/embeddings/esm2_650m_pdb.h5"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for short, (branch, n_terms) in BRANCHES.items():
        for seed in SEEDS:
            ck = f"checkpoints/{short}_v6_pdb30base_s{seed}/"
            cfg = {
                "branch": branch, "n_terms": n_terms, "seed": seed,
                "data": {
                    "protein_order": "data/pdbch/protein_order.json",
                    "splits":        "data/pdbch/splits.json",
                    "labels":        f"data/pdbch/labels_{short}.npy",
                    "dag_matrix":    f"data/pdbch/dag_matrix_{short}.npy",
                    "esm2_h5":       pdb_emb,
                    "ppi_emb":       "data/embeddings/ppi_deepgo.npy",
                    "ppi_mask":      "data/embeddings/ppi_deepgo_mask.npy",
                    "cmap_h5":       "data/contact_maps/cmap_all.h5",
                    "go_emb":        f"data/embeddings/go_emb_{short}_v2.npy",
                    "diamond_tsv":   f"data/diamond/diamond_results_{short}.tsv",
                },
                "model": _common_model(),
                # SAME recipe as v6sm finetune (fair comparison), minus pretrain init.
                "training": _training(epochs=60, lr=3.0e-4, seed=seed),
                "inference": {
                    "use_dag_propagation": True, "use_diamond_ensemble": True,
                    "diamond_alpha": 0.6, "threshold_path": f"{ck}threshold.json",
                },
                "output": {
                    "checkpoint_dir": ck,
                    "log_file": f"logs/{short}_v6_pdb30base_s{seed}.log",
                    "results_file": f"results/{short}_v6_pdb30base_s{seed}_predictions.tsv",
                },
            }
            p = out / f"{short}_v6_pdb30base_s{seed}.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            paths.append(str(p))
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="configs")
    ap.add_argument("--pdb-emb", default="data/embeddings/esm2_650m_pdb.h5")
    args = ap.parse_args()
    paths = generate(args.out, args.pdb_emb)
    print(f"[gen_v6_baseline] wrote {len(paths)} configs")


if __name__ == "__main__":
    main()
