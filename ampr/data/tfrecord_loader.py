"""Parse DeepFRI's PDB_GO TFRecord files.

Each record contains: prot_id, sequence one-hot, contact map, per-ontology labels.
"""

from pathlib import Path
from typing import Iterator
import numpy as np


def iter_tfrecord(path: Path) -> Iterator[dict]:
    """Yield dicts: {prot_id, cmap (L,L), seq_1hot (L,26), labels {mf,bp,cc}}."""
    import tensorflow as tf

    raw_dataset = tf.data.TFRecordDataset(str(path))
    for raw in raw_dataset:
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        f = ex.features.feature
        L = int(f["L"].int64_list.value[0])
        prot_id = f["prot_id"].bytes_list.value[0].decode()
        seq_1hot = np.frombuffer(f["seq_1hot"].bytes_list.value[0], dtype=np.float32).reshape(L, 26)
        cmap = np.frombuffer(f["cmap"].bytes_list.value[0], dtype=np.float32).reshape(L, L)
        labels = {}
        for ont in ("mf", "bp", "cc"):
            n = int(f[f"{ont}_n"].int64_list.value[0])
            arr = np.frombuffer(f[f"{ont}_labels"].bytes_list.value[0], dtype=np.int64)
            assert arr.size == n, f"{prot_id} {ont}: declared {n}, got {arr.size}"
            labels[ont] = arr
        yield {
            "prot_id": prot_id,
            "L": L,
            "seq_1hot": seq_1hot,
            "cmap": cmap,
            "labels": labels,
        }
