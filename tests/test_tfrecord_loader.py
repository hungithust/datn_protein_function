"""Tests for ampr/data/tfrecord_loader.py. Uses synthetic TFRecord (no real data)."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


def _write_synth_tfrecord(path):
    L = 5
    cmap = np.random.RandomState(0).rand(L, L).astype(np.float32)
    cmap = (cmap + cmap.T) / 2
    np.fill_diagonal(cmap, 0)
    seq_1hot = np.eye(26, dtype=np.float32)[:L]
    mf_lab = np.array([1, 0, 1], dtype=np.int64)
    bp_lab = np.array([0, 1], dtype=np.int64)
    cc_lab = np.array([1], dtype=np.int64)

    def _bytes(v):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    def _int64(vs):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=vs))

    feature = {
        "L": _int64([L]),
        "prot_id": _bytes(b"1AAA-A"),
        "seq_1hot": _bytes(seq_1hot.tobytes()),
        "cmap": _bytes(cmap.tobytes()),
        "mf_labels": _bytes(mf_lab.tobytes()),
        "mf_n": _int64([3]),
        "bp_labels": _bytes(bp_lab.tobytes()),
        "bp_n": _int64([2]),
        "cc_labels": _bytes(cc_lab.tobytes()),
        "cc_n": _int64([1]),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feature))
    with tf.io.TFRecordWriter(str(path)) as w:
        w.write(ex.SerializeToString())
    return L, cmap, mf_lab, bp_lab, cc_lab


def test_iterates_one_record(tmp_path):
    from ampr.data.tfrecord_loader import iter_tfrecord
    L, expected_cmap, mf_lab, bp_lab, cc_lab = _write_synth_tfrecord(tmp_path / "x.tfrecords")
    records = list(iter_tfrecord(tmp_path / "x.tfrecords"))
    assert len(records) == 1
    r = records[0]
    assert r["prot_id"] == "1AAA-A"
    assert r["cmap"].shape == (L, L)
    np.testing.assert_allclose(r["cmap"], expected_cmap)
    np.testing.assert_array_equal(r["labels"]["mf"], mf_lab)
    np.testing.assert_array_equal(r["labels"]["bp"], bp_lab)
    np.testing.assert_array_equal(r["labels"]["cc"], cc_lab)
