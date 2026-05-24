#!/usr/bin/env python
"""Generate splits.json from DeepFRI's nrPDB-GO_2019.06.18_{train,valid}.txt + _test.csv.

Reads from <data-dir>/{train.txt,valid.txt,test.csv} and writes
<data-dir>/splits.json with cumulative identity bins matching the HEAL paper:
    test_LT_X = every test protein whose <X%> column == 1.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[build_splits] %(message)s")
log = logging.getLogger(__name__)

IDENTITY_COLUMNS = ["<30%", "<40%", "<50%", "<70%", "<95%"]


def read_id_list(path: Path) -> list[str]:
    """One protein ID per line, blanks stripped."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_test_csv(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Return (ordered test IDs, {bin_name: [IDs]}).

    bin_name is 'test_LT_30', etc. Cumulative: an ID belongs to test_LT_X
    iff its <X%> column == "1".
    """
    test_ids: list[str] = []
    bins: dict[str, list[str]] = {f"test_LT_{c.strip('<%')}": [] for c in IDENTITY_COLUMNS}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["PDB-chain"].strip()
            test_ids.append(pid)
            for col in IDENTITY_COLUMNS:
                if row[col].strip() == "1":
                    bin_key = f"test_LT_{col.strip('<%')}"
                    bins[bin_key].append(pid)
    return test_ids, bins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing train.txt, valid.txt, test.csv")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    train = read_id_list(data_dir / "train.txt")
    valid = read_id_list(data_dir / "valid.txt")
    test, bins = read_test_csv(data_dir / "test.csv")

    splits = {"train": train, "valid": valid, "test": test, **bins}

    out_path = data_dir / "splits.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    log.info("train=%d valid=%d test=%d", len(train), len(valid), len(test))
    for k in sorted(bins):
        log.info("%s=%d", k, len(bins[k]))
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
