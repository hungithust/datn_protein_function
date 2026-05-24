#!/usr/bin/env python
"""Generate results/baselines/comparison_table.md from all metric JSONs."""

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/baselines/comparison_table.md"))
    args = parser.parse_args()

    heal = json.load(open("data/deepfri_baseline.json", "r", encoding="utf-8"))

    lines = ["# Baseline Comparison Table\n",
             "Fmax across identity bins. AMPR numbers added once Phase 2 trains.\n"]

    for ont in ["mf", "bp", "cc"]:
        ONT = ont.upper()
        lines.append(f"\n## {ONT}\n")
        lines.append("| Bin | DeepFRI (HEAL paper) | DeepFRI (reproduced) | Diamond | AMPR |")
        lines.append("|---|---|---|---|---|")
        dfri = load_metrics(Path(f"results/baselines/deepfri/metrics_{ont}.json"))
        diam = load_metrics(Path(f"results/baselines/diamond/metrics_{ont}.json"))
        for bin_short, bin_key in [("<30%", "test_LT_30"), ("<40%", "test_LT_40"),
                                    ("<50%", "test_LT_50"), ("<70%", "test_LT_70"),
                                    ("<95%", "test_LT_95")]:
            lt_key = bin_key.replace("test_LT_", "LT_")
            paper = heal.get(ONT, {}).get(lt_key, {}).get("fmax")
            paper_s = f"{paper:.3f}" if paper else "—"
            d = dfri.get(bin_key, {}).get("fmax")
            d_s = f"{d:.3f}" if d is not None else "—"
            dia = diam.get(bin_key, {}).get("fmax")
            dia_s = f"{dia:.3f}" if dia is not None else "—"
            lines.append(f"| {bin_short} | {paper_s} | {d_s} | {dia_s} | TBD |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
