"""Tests for ampr/data/contact_map_h5.py."""

from pathlib import Path
import h5py
import numpy as np
import pytest


@pytest.fixture
def tiny_h5_set(tmp_path):
    train = tmp_path / "train.h5"
    valid = tmp_path / "valid.h5"
    test = tmp_path / "test.h5"
    with h5py.File(train, "w") as f:
        f.create_dataset("1AAA-A", data=np.ones((3, 3), dtype=np.float32))
    with h5py.File(valid, "w") as f:
        f.create_dataset("1BBB-A", data=2*np.ones((4, 4), dtype=np.float32))
    with h5py.File(test, "w") as f:
        f.create_dataset("1CCC-A", data=3*np.ones((2, 2), dtype=np.float32))
    return {"train": train, "valid": valid, "test": test}


def test_lookup_finds_protein_in_any_file(tiny_h5_set):
    from ampr.data.contact_map_h5 import ContactMapStore
    store = ContactMapStore(tiny_h5_set)
    assert store["1AAA-A"].shape == (3, 3)
    assert store["1BBB-A"].shape == (4, 4)
    assert store["1CCC-A"].shape == (2, 2)


def test_missing_protein_raises(tiny_h5_set):
    from ampr.data.contact_map_h5 import ContactMapStore
    store = ContactMapStore(tiny_h5_set)
    with pytest.raises(KeyError):
        store["1ZZZ-A"]


def test_get_sequence_returns_stored_attr(tmp_path):
    """ContactMapStore.get_sequence() returns the sequence attr saved by tfrecord_to_h5."""
    from ampr.data.contact_map_h5 import ContactMapStore
    h5_path = tmp_path / "train.h5"
    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset("1AAA-A", data=np.ones((3, 3), dtype=np.float32))
        ds.attrs["sequence"] = "ACD"
    store = ContactMapStore({"train": h5_path})
    assert store.get_sequence("1AAA-A") == "ACD"


def test_contains_works(tiny_h5_set):
    from ampr.data.contact_map_h5 import ContactMapStore
    store = ContactMapStore(tiny_h5_set)
    assert "1AAA-A" in store
    assert "1ZZZ-A" not in store
