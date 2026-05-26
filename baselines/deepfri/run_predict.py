#!/usr/bin/env python
"""Run DeepFRI pretrained on AMPR's test set, save predictions in our format.

Saves to results/baselines/deepfri/predictions_{ont}.npz with keys:
    y_pred:   (N_test, n_terms_for_ont)  float32 — sigmoid probabilities
    y_true:   (N_test, n_terms_for_ont)  float32 — ground truth from labels_{ont}.npy
    prot_ids: (N_test,)                  array of strings
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="[deepfri_pred] %(message)s")
log = logging.getLogger(__name__)

CMAP_THRESHOLD = 10.0
AA_INDEX = {a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWYBOUXZ-.")}


def seq2onehot(seq: str) -> np.ndarray:
    L = len(seq)
    M = np.zeros((L, 26), dtype=np.float32)
    for i, a in enumerate(seq):
        M[i, AA_INDEX.get(a, AA_INDEX["X"])] = 1.0
    return M


def predict_one(model, cmap: np.ndarray, seq: str) -> np.ndarray:
    A = (cmap < CMAP_THRESHOLD).astype(np.float32)
    S = seq2onehot(seq)
    A = A[None, ...]  # (1, L, L)
    S = S[None, ...]  # (1, L, 26)
    out = model.predict([A, S], verbose=0)
    return out.reshape(-1)


def load_deepfri_model(weights_path: Path):
    """Lazy TF import so non-TF environments don't break the test runner."""
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_root / "DeepFRI"))
    from deepfrier.layers import GraphConv, MultiGraphConv, SumPooling
    custom = {
        "GraphConv": GraphConv,
        "MultiGraphConv": MultiGraphConv,
        "SumPooling": SumPooling,
        # CuDNNLSTM was removed in Keras 3 / TF 2.16+ — alias to standard LSTM
        "CuDNNLSTM": tf.keras.layers.LSTM,
    }
    return load_model(str(weights_path), custom_objects=custom, compile=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", choices=["mf", "bp", "cc"], required=True)
    parser.add_argument("--weights", type=Path, required=True,
                        help="Path to DeepFRI-MERGED_*.hdf5 for this ontology")
    parser.add_argument("--cmap-h5", type=Path, default=Path("data/pdbch/contact_maps_test.h5"))
    parser.add_argument("--splits", type=Path, default=Path("data/pdbch/splits.json"))
    parser.add_argument("--protein-order", type=Path, default=Path("data/pdbch/protein_order.json"))
    parser.add_argument("--labels", type=Path, required=True,
                        help="data/pdbch/labels_{ont}.npy")
    parser.add_argument("--go-terms", type=Path, required=True,
                        help="data/pdbch/go_terms_{ont}.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    log.info("loading DeepFRI weights %s", args.weights)
    model = load_deepfri_model(args.weights)

    # Load model's GO term order from companion json (DeepFRI saves model_params.json)
    params_path = args.weights.with_name(args.weights.stem + "_model_params.json")
    with open(params_path, "r", encoding="utf-8") as f:
        model_goterms = json.load(f)["goterms"]

    with open(args.go_terms, "r", encoding="utf-8") as f:
        our_goterms = json.load(f)
    # Column reorder: model output column → our column index
    reorder = []
    for t in our_goterms:
        if t in model_goterms:
            reorder.append(model_goterms.index(t))
        else:
            reorder.append(-1)  # term not in DeepFRI's set — will be zeroed
    reorder = np.asarray(reorder)

    with open(args.splits, "r", encoding="utf-8") as f:
        test_ids = json.load(f)["test"]
    with open(args.protein_order, "r", encoding="utf-8") as f:
        prot_order = json.load(f)
    row_of = {p: i for i, p in enumerate(prot_order)}
    labels_full = np.load(args.labels)

    y_true = np.zeros((len(test_ids), len(our_goterms)), dtype=np.float32)
    y_pred = np.zeros((len(test_ids), len(our_goterms)), dtype=np.float32)

    n_missing_cmap = 0
    with h5py.File(args.cmap_h5, "r") as cmaps:
        for i, pid in enumerate(test_ids):
            if pid in row_of:
                y_true[i] = labels_full[row_of[pid]]
            if pid not in cmaps:
                n_missing_cmap += 1
                continue
            cmap = cmaps[pid][...]
            seq = cmaps[pid].attrs["sequence"]
            try:
                pred = predict_one(model, cmap, seq)
                # Map model's output columns → our column order
                for j, k in enumerate(reorder):
                    if k >= 0 and k < len(pred):
                        y_pred[i, j] = pred[k]
            except Exception as e:
                log.warning("%s: %s", pid, e)
            if (i + 1) % 200 == 0:
                log.info("progress %d/%d", i + 1, len(test_ids))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, y_true=y_true, y_pred=y_pred,
                        prot_ids=np.asarray(test_ids))
    log.info("saved %s (%d missing cmaps)", args.out, n_missing_cmap)


if __name__ == "__main__":
    main()
