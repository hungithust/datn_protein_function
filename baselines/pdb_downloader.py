#!/usr/bin/env python
"""Download mmCIF files from RCSB for a list of PDB-chain IDs.

Input:  data/pdbch/splits.json key 'test' (list of "PDBID-CHAIN" strings)
Output: data/pdbch/test_pdb_files/{PDBID}.cif.gz
        data/pdbch/test_pdb_files/_failed.txt  (IDs that failed to download)

Resumable: skips files already on disk.
"""

import argparse
import gzip
import json
import logging
import time
from pathlib import Path

import urllib.request
import urllib.error

logging.basicConfig(level=logging.INFO, format="[pdb_dl] %(message)s")
log = logging.getLogger(__name__)

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"


def download_one(pdb_id: str, out_path: Path, retries: int = 2) -> bool:
    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    for attempt in range(retries + 1):
        try:
            urllib.request.urlretrieve(url, str(out_path))
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False  # obsoleted — don't retry
            if attempt == retries:
                log.warning("HTTP %d for %s after %d retries", e.code, pdb_id, retries)
                return False
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == retries:
                log.warning("error for %s: %s", pdb_id, e)
                return False
            time.sleep(2 ** attempt)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("data/pdbch/splits.json"))
    parser.add_argument("--split-key", default="test")
    parser.add_argument("--out-dir", type=Path, default=Path("data/pdbch/test_pdb_files"))
    args = parser.parse_args()

    with open(args.splits, "r", encoding="utf-8") as f:
        ids = json.load(f)[args.split_key]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # PDB IDs only (drop -CHAIN suffix); one file per PDB regardless of chain count
    pdb_ids = sorted({pid.split("-")[0].upper() for pid in ids})
    log.info("downloading %d unique PDB structures for %d chain IDs",
             len(pdb_ids), len(ids))

    failed = []
    for i, pdb_id in enumerate(pdb_ids, 1):
        out = args.out_dir / f"{pdb_id}.cif.gz"
        if out.exists():
            continue
        ok = download_one(pdb_id, out)
        if not ok:
            failed.append(pdb_id)
        if i % 100 == 0:
            log.info("progress %d/%d (failed so far: %d)", i, len(pdb_ids), len(failed))

    (args.out_dir / "_failed.txt").write_text("\n".join(failed), encoding="utf-8")
    log.info("done — %d failed (see _failed.txt)", len(failed))


if __name__ == "__main__":
    main()
