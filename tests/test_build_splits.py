"""Tests for scripts/build_splits_from_deepfri.py"""

import json
import subprocess
import sys
from pathlib import Path


def run_build_splits(data_dir: Path) -> dict:
    """Invoke the script and return the parsed splits.json."""
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "build_splits_from_deepfri.py"
    subprocess.run(
        [sys.executable, str(script), "--data-dir", str(data_dir)],
        check=True,
    )
    with open(data_dir / "splits.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_train_valid_test_lists_match_source_files(tiny_pdbch):
    splits = run_build_splits(tiny_pdbch)
    assert splits["train"] == ["1AAA-A", "1BBB-A"]
    assert splits["valid"] == ["1CCC-A"]
    # Test list preserves test.csv row order
    assert splits["test"] == ["1CCC-A", "1AAA-A"]


def test_identity_bins_are_cumulative(tiny_pdbch):
    splits = run_build_splits(tiny_pdbch)
    # 1AAA-A has <30%=1,<40%=1,<50%=1,<70%=1,<95%=1 → appears in all bins
    # 1CCC-A has <30%=0,<40%=1,<50%=1,<70%=1,<95%=1 → appears in LT_40..LT_95 only
    assert splits["test_LT_30"] == ["1AAA-A"]
    assert sorted(splits["test_LT_40"]) == sorted(["1AAA-A", "1CCC-A"])
    assert sorted(splits["test_LT_50"]) == sorted(["1AAA-A", "1CCC-A"])
    assert sorted(splits["test_LT_70"]) == sorted(["1AAA-A", "1CCC-A"])
    assert sorted(splits["test_LT_95"]) == sorted(["1AAA-A", "1CCC-A"])


def test_splits_json_has_no_unexpected_keys(tiny_pdbch):
    splits = run_build_splits(tiny_pdbch)
    expected_keys = {
        "train", "valid", "test",
        "test_LT_30", "test_LT_40", "test_LT_50", "test_LT_70", "test_LT_95",
    }
    assert set(splits.keys()) == expected_keys
