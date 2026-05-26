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
    mf_lab = np.array([1, 0, 1, 0, 1], dtype=np.int64)
    bp_lab = np.array([0, 1, 0], dtype=np.int64)
    cc_lab = np.array([1, 1], dtype=np.int64)

    def _float(vs):
        return tf.train.Feature(float_list=tf.train.FloatList(value=vs))
    def _int64(vs):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=vs))
    def _bytes(v):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

    feature = {
        "L": _int64([L]),
        "prot_id": _bytes(b"1AAA-A"),
        "seq_1hot": _float(seq_1hot.reshape(-1).tolist()),
        "ca_dist_matrix": _float(cmap.reshape(-1).tolist()),
        "cb_dist_matrix": _float(cmap.reshape(-1).tolist()),
        "mf_labels": _int64(mf_lab.tolist()),
        "bp_labels": _int64(bp_lab.tolist()),
        "cc_labels": _int64(cc_lab.tolist()),
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
    assert r["seq_1hot"].shape == (L, 26)
    np.testing.assert_allclose(r["cmap"], expected_cmap)
    np.testing.assert_array_equal(r["labels"]["mf"], mf_lab)
    np.testing.assert_array_equal(r["labels"]["bp"], bp_lab)
    np.testing.assert_array_equal(r["labels"]["cc"], cc_lab)
