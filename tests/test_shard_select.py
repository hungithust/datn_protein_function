from scripts.precompute_esm2_residue import select_shard


def test_shard_partitions_disjoint_and_complete():
    ids = [f"p{i}" for i in range(10)]
    shards = [select_shard(ids, s, 3) for s in range(3)]
    # disjoint
    seen = sum(shards, [])
    assert sorted(seen) == sorted(ids)
    # roughly balanced
    assert all(2 <= len(s) <= 4 for s in shards)


def test_shard_single_returns_all():
    ids = ["a", "b", "c"]
    assert select_shard(ids, 0, 1) == ids
