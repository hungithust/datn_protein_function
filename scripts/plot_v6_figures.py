#!/usr/bin/env python
"""Plot the identity-curve figure (Fmax vs sequence-identity bin) for the thesis.

Reads the +DIAMOND ensemble Fmax from results/*.ensemble_*.json for the two v6
recipes and overlays the DeepFRI / HEAL baselines (HEAL supp Tables S3.2). One
panel per ontology. PR curves need raw probs (not stored in the metric JSONs) —
add --dump-probs to ensemble_eval if you want those; this script does the
identity curve, which is the central results figure.

Usage:
  python scripts/plot_v6_figures.py [--results "results/*.ensemble_*.json"] [--out docs/figures]
"""
import argparse
import glob as _glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINS = ["test_LT_30", "test_LT_40", "test_LT_50", "test_LT_70", "test_LT_95"]
XLAB = ["<30", "<40", "<50", "<70", "<95"]
PAT = re.compile(r"^(?P<branch>mf|bp|cc)_(?P<recipe>.+?)_s\d+_predictions\.ensemble_(?P<split>.+)\.json$")

# HEAL supplementary Table S3.2 (Fmax, PDBch, LT bins) — Gu et al. btad410 (2023)
BASELINES = {
    "MF": {"DeepFRI": [0.544, 0.552, 0.575, 0.604, 0.626], "HEAL": [0.698, 0.702, 0.719, 0.735, 0.749]},
    "BP": {"DeepFRI": [0.502, 0.510, 0.517, 0.533, 0.540], "HEAL": [0.582, 0.578, 0.582, 0.592, 0.594]},
    "CC": {"DeepFRI": [0.605, 0.606, 0.606, 0.605, 0.612], "HEAL": [0.684, 0.682, 0.684, 0.686, 0.687]},
}
RECIPE_LABEL = {"v6_pdb30base": "AMPR 650M-PDB-30K", "v6sm_finetune": "AMPR-B 220K→30K"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/*.ensemble_*.json")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()

    # data[branch][recipe][split] = ensemble Fmax
    data = {}
    for fp in sorted(_glob.glob(args.results)):
        m = PAT.match(Path(fp).name)
        if not m:
            continue
        d = json.loads(Path(fp).read_text())
        blk = d.get("ensemble") or d.get("dag")
        if not blk:
            continue
        br = m["branch"].upper()
        data.setdefault(br, {}).setdefault(m["recipe"], {})[m["split"]] = blk["fmax"]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for br in ("MF", "BP", "CC"):
        plt.figure(figsize=(5, 4))
        for model, ys in BASELINES.get(br, {}).items():
            plt.plot(XLAB, ys, marker="s", linestyle="--", label=model)
        for recipe, label in RECIPE_LABEL.items():
            d = data.get(br, {}).get(recipe)
            if not d:
                continue
            ys = [d.get(b) for b in BINS]
            if any(v is None for v in ys):
                continue
            plt.plot(XLAB, ys, marker="o", label=label)
        plt.xlabel("Sequence identity to train (%)")
        plt.ylabel("Fmax")
        plt.title(f"{br} — identity curve (PDBch, +DIAMOND ens)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        p = outdir / f"identity_curve_{br.lower()}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"[plot] wrote {p}")


if __name__ == "__main__":
    main()
