#!/usr/bin/env python
"""Generate v6 SWISS-MODEL pretrain + PDB finetune configs.

Pretrain: 3 configs (mf/bp/cc), seed 42, train on SWISS-MODEL artifacts.
Finetune: 9 configs (3 branches × seeds {42,123,2024}), train on PDB at 650M,
          initialized from the matching pretrain checkpoint (via --init-from).
"""
import argparse
from pathlib import Path

import yaml

BRANCHES = {"mf": ("MF", 489), "bp": ("BP", 1943), "cc": ("CC", 320)}
SEEDS = [42, 123, 2024]


def _common_model():
    return {
        "version": "v3",
        "structure_modality": "gnn",
        "seq": {"d_model": 1280, "n_transformer_layers": 2, "n_heads": 8, "dropout": 0.4},
        "gnn": {"node_dim": 256, "n_layers": 3, "cmap_threshold": 10.0},
        "ppi": {"in_dim": 256, "hidden": 512},
        "fusion": {"d_model": 512, "n_layers": 2, "n_heads": 8},
        "classifier": "both",
        "d_hidden": 512,
    }


def _training(epochs, lr, seed):
    return {
        "epochs": epochs, "batch_size": 64, "lr": lr,
        "lr_scheduler": "plateau", "lr_factor": 0.5, "lr_patience": 5, "lr_min": 1.0e-5,
        "weight_decay": 1.0e-2, "loss_type": "asl",
        "asl_gamma_neg": 4, "asl_gamma_pos": 0, "asl_clip": 0.05,
        "lambda_dag": 0.5, "seed": seed, "device": "auto",
        "max_seq_len": 1000, "num_workers": 16,
    }


def generate(out_dir, sm_dir, pdb_emb, sm_emb, sm_cmap):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    for short, (branch, n_terms) in BRANCHES.items():
        # --- pretrain (SWISS-MODEL) ---
        ck = f"checkpoints/{short}_v6sm_pretrain/"
        cfg = {
            "branch": branch, "n_terms": n_terms, "seed": 42,
            "data": {
                "protein_order": f"{sm_dir}/protein_order_sm.json",
                "splits":        f"{sm_dir}/splits_sm.json",
                "labels":        f"{sm_dir}/labels_{short}_sm.npy",
                "dag_matrix":    f"data/pdbch/dag_matrix_{short}.npy",
                "esm2_h5":       sm_emb,
                "ppi_emb":       f"{sm_dir}/ppi_zero_sm.npy",
                "ppi_mask":      f"{sm_dir}/ppi_mask_sm.npy",
                "cmap_h5":       sm_cmap,
                "go_emb":        f"data/embeddings/go_emb_{short}_v2.npy",
            },
            "model": _common_model(),
            # warm-continue pretrain (launcher --init-from old best.pt): lower LR to
            # refine from the ~0.58 checkpoint instead of disrupting it.
            "training": _training(epochs=30, lr=5.0e-4, seed=42),
            "inference": {"use_dag_propagation": True},
            "output": {
                "checkpoint_dir": ck,
                "log_file": f"logs/{short}_v6sm_pretrain.log",
                "results_file": f"results/{short}_v6sm_pretrain_predictions.tsv",
            },
        }
        p = out / f"{short}_v6sm_pretrain.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        paths.append(str(p))

        # --- finetune (PDB), one per seed ---
        for seed in SEEDS:
            ck = f"checkpoints/{short}_v6sm_finetune_s{seed}/"
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
                # 60 epochs + lr 3e-4: prior 30ep/1e-4 under-trained (val Fmax still
                # climbing monotonically at epoch 30, no plateau).
                "training": _training(epochs=60, lr=3.0e-4, seed=seed),
                "inference": {
                    "use_dag_propagation": True, "use_diamond_ensemble": True,
                    "diamond_alpha": 0.6,
                    "threshold_path": f"{ck}threshold.json",
                },
                "output": {
                    "checkpoint_dir": ck,
                    "log_file": f"logs/{short}_v6sm_finetune_s{seed}.log",
                    "results_file": f"results/{short}_v6sm_finetune_s{seed}_predictions.tsv",
                },
            }
            p = out / f"{short}_v6sm_finetune_s{seed}.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            paths.append(str(p))

    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="configs")
    ap.add_argument("--sm-dir", default="data/swissmodel_art")
    ap.add_argument("--pdb-emb", default="data/embeddings/esm2_650m_pdb.h5")
    ap.add_argument("--sm-emb", default="data/embeddings/esm2_650m_sm.h5")
    ap.add_argument("--sm-cmap", default="data/swissmodel_art/cmap_all_sm.h5")
    args = ap.parse_args()
    paths = generate(args.out, args.sm_dir, args.pdb_emb, args.sm_emb, args.sm_cmap)
    print(f"[gen_v6] wrote {len(paths)} configs")


if __name__ == "__main__":
    main()
