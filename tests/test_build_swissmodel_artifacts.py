"""Tests for scripts/build_swissmodel_artifacts.py using a synthetic TFRecord."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write(path, pid, L, mf, bp, cc):
    seq_1hot = np.eye(26, dtype=np.float32)[[0] * L]  # all 'A'
    cmap = np.zeros((L, L), dtype=np.float32)
    def _f(v): return tf.train.Feature(float_list=tf.train.FloatList(value=v))
    def _i(v): return tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    def _b(v): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    feat = {
        "L": _i([L]), "prot_id": _b(pid.encode()),
        "seq_1hot": _f(seq_1hot.reshape(-1).tolist()),
        "ca_dist_matrix": _f(cmap.reshape(-1).tolist()),
        "cb_dist_matrix": _f(cmap.reshape(-1).tolist()),
        "mf_labels": _i(mf), "bp_labels": _i(bp), "cc_labels": _i(cc),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feat))
    with tf.io.TFRecordWriter(str(path)) as w:
        w.write(ex.SerializeToString())


def test_build_artifacts(tmp_path):
    from scripts.build_swissmodel_artifacts import build_artifacts
    train = tmp_path / "SM_train_0.tfrecords"
    valid = tmp_path / "SM_valid_0.tfrecords"
    _write(train, "P1-A", 3, [1, 0] + [0] * 487, [0] * 1943, [1] + [0] * 319)
    _write(valid, "P2-A", 4, [0, 1] + [0] * 487, [1] + [0] * 1942, [0] * 320)
    out = tmp_path / "out"

    build_artifacts(str(train.parent / "SM_train_*.tfrecords"),
                    str(valid.parent / "SM_valid_*.tfrecords"),
                    str(out), n_terms={"mf": 489, "bp": 1943, "cc": 320})

    order = json.loads((out / "protein_order_sm.json").read_text())
    assert order == ["P1-A", "P2-A"]
    splits = json.loads((out / "splits_sm.json").read_text())
    assert splits["train"] == ["P1-A"] and splits["valid"] == ["P2-A"]
    mf = np.load(out / "labels_mf_sm.npy")
    assert mf.shape == (2, 489) and mf[0, 0] == 1.0 and mf[1, 1] == 1.0
    mask = np.load(out / "ppi_mask_sm.npy")
    ppi = np.load(out / "ppi_zero_sm.npy")
    assert mask.shape == (2,) and mask.sum() == 0
    assert ppi.shape == (2, 256) and ppi.sum() == 0
    fasta = (out / "sequences_sm.fasta").read_text()
    assert ">P1-A" in fasta and "AAA" in fasta
