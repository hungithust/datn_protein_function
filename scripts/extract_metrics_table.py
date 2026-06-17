#!/usr/bin/env python
"""Extract a tidy multi-metric table from results/*.ensemble_*.json.

Each ensemble JSON (written by scripts/ensemble_eval.py) holds the full
compute_all_metrics dict for the raw / dag / ensemble blocks. This flattens them
into one markdown table + CSV for the thesis appendix (B.1/B.2).

Usage:
  python scripts/extract_metrics_table.py [--glob "results/*.ensemble_*.json"] [--out docs/metrics_v6_full.md]
"""
import argparse
import glob as _glob
import json
import re
from pathlib import Path

METRICS = ["fmax", "smin", "smin_raw", "auprc_micro", "auprc_macro",
           "micro_auroc", "coverage"]

# e.g. mf_v6_pdb30base_s42_predictions.ensemble_test_LT_30.json
PAT = re.compile(r"^(?P<branch>mf|bp|cc)_(?P<recipe>.+?)_s\d+_predictions\.ensemble_(?P<split>.+)\.json$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/*.ensemble_*.json")
    ap.add_argument("--out", default="docs/metrics_v6_full.md")
    args = ap.parse_args()

    rows = []
    for fp in sorted(_glob.glob(args.glob)):
        m = PAT.match(Path(fp).name)
        if not m:
            continue
        d = json.loads(Path(fp).read_text())
        branch, recipe, split = m["branch"].upper(), m["recipe"], m["split"]
        for block in ("raw", "dag", "ensemble"):
            if block not in d:
                continue
            b = d[block]
            rows.append((recipe, branch, split, block,
                         *[b.get(k, float("nan")) for k in METRICS]))

    if not rows:
        print(f"[extract] no JSON matched {args.glob}")
        return

    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    header = ["recipe", "branch", "bin", "block", "Fmax", "Smin", "Smin_raw",
              "AUPRC_micro", "AUPRC_macro", "AUROC_micro", "Coverage"]

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    csv = [",".join(header)]
    for r in rows:
        cells = [r[0], r[1], r[2], r[3], *[fmt(x) for x in r[4:]]]
        md.append("| " + " | ".join(cells) + " |")
        csv.append(",".join(cells))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    Path(out).with_suffix(".csv").write_text("\n".join(csv) + "\n")
    print(f"[extract] {len(rows)} rows -> {out} (+ .csv)")
    print("\n".join(md[:8]) + ("\n..." if len(md) > 8 else ""))


if __name__ == "__main__":
    main()
