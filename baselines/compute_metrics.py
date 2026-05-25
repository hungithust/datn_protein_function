#!/usr/bin/env python
"""Compute Fmax/AUPR/Smin per identity bin for a predictions .npz file."""

import argparse
import json
import logging
from pathlib import Path

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import numpy as np

from ampr.evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="[metric] %(message)s")
log = logging.getLogger(__name__)


def metrics_per_bin(y_true, y_pred, prot_ids, splits, term_ic):
    out = {}
    for bin_key in ["test_LT_30", "test_LT_40", "test_LT_50", "test_LT_70",
                    "test_LT_95", "test"]:
        bin_ids = set(splits[bin_key])
        mask = np.asarray([p in bin_ids for p in prot_ids])
        if mask.sum() == 0:
            continue
        m = compute_all_metrics(y_true[mask], y_pred[mask], term_ic)
        out[bin_key] = {
            "fmax": m["fmax"],
            "auprc_micro": m["auprc_micro"],
            "auprc_macro": m["auprc_macro"],
            "smin": m["smin"],
            "n_proteins": m["n_proteins"],
        }
        log.info("%s: n=%d Fmax=%.4f AUPR=%.4f Smin=%.4f",
                 bin_key, m["n_proteins"], m["fmax"], m["auprc_micro"], m["smin"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("data/pdbch/splits.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.predictions, allow_pickle=True)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    prot_ids = data["prot_ids"].tolist()

    labels = np.load(args.labels)
    freq = labels.mean(axis=0).clip(1e-7, 1.0)
    term_ic = (-np.log2(freq)).astype(np.float32)

    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)

    metrics = metrics_per_bin(y_true, y_pred, prot_ids, splits, term_ic)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
