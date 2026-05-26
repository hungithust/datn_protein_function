"""Parse DeepFRI's PDB_GO TFRecord files.

Real schema (per inspection of PDB_GO_*.tfrecords):
    prot_id          bytes_list  len=1
    L                int64_list  len=1
    seq_1hot         float_list  len=L*26
    ca_dist_matrix   float_list  len=L*L
    cb_dist_matrix   float_list  len=L*L  (used as distance matrix)
    mf_labels        int64_list  len=489   (full one-hot vector)
    bp_labels        int64_list  len=1943
    cc_labels        int64_list  len=320
"""

from pathlib import Path
from typing import Iterator
import numpy as np


def iter_tfrecord(path: Path, dist_key: str = "cb_dist_matrix") -> Iterator[dict]:
    """Yield dicts: {prot_id, L, seq_1hot (L,26), cmap (L,L), labels {mf,bp,cc}}."""
    import tensorflow as tf

    raw_dataset = tf.data.TFRecordDataset(str(path))
    for raw in raw_dataset:
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        f = ex.features.feature
        L = int(f["L"].int64_list.value[0])
        prot_id = f["prot_id"].bytes_list.value[0].decode()
        seq_1hot = np.asarray(f["seq_1hot"].float_list.value, dtype=np.float32).reshape(L, 26)
        cmap = np.asarray(f[dist_key].float_list.value, dtype=np.float32).reshape(L, L)
        labels = {
            ont: np.asarray(f[f"{ont}_labels"].int64_list.value, dtype=np.int64)
            for ont in ("mf", "bp", "cc")
        }
        yield {
            "prot_id": prot_id,
            "L": L,
            "seq_1hot": seq_1hot,
            "cmap": cmap,
            "labels": labels,
        }
