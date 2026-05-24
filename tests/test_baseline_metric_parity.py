"""The Phase 1 acceptance gate: do our reproduced DeepFRI numbers match HEAL paper?

Skipped if DeepFRI predictions don't exist yet.
HEAL Table S3.2 reports (averaged across 10 bootstrap samples):
    MF<30%=0.544, BP<30%=0.502, CC<30%=0.605
We accept ±0.025 (single-run vs 10-bootstrap noise).
"""

import json
from pathlib import Path

import pytest

METRICS_DIR = Path(__file__).resolve().parents[1] / "results" / "baselines" / "deepfri"

HEAL_BASELINE = {
    "mf": {"test_LT_30": 0.544, "test_LT_40": 0.552, "test_LT_50": 0.575,
           "test_LT_70": 0.604, "test_LT_95": 0.626},
    "bp": {"test_LT_30": 0.502, "test_LT_40": 0.510, "test_LT_50": 0.517,
           "test_LT_70": 0.533, "test_LT_95": 0.540},
    "cc": {"test_LT_30": 0.605, "test_LT_40": 0.606, "test_LT_50": 0.606,
           "test_LT_70": 0.605, "test_LT_95": 0.612},
}

TOLERANCE = 0.025


def _have_metrics(ont: str) -> bool:
    return (METRICS_DIR / f"metrics_{ont}.json").exists()


@pytest.mark.parametrize("ont", ["mf", "bp", "cc"])
def test_deepfri_reproduces_heal_paper(ont):
    if not _have_metrics(ont):
        pytest.skip(f"metrics_{ont}.json not generated yet — run Task 3+4 first")
    with open(METRICS_DIR / f"metrics_{ont}.json", "r", encoding="utf-8") as f:
        ours = json.load(f)
    for bin_key, paper_fmax in HEAL_BASELINE[ont].items():
        if bin_key not in ours:
            continue
        diff = abs(ours[bin_key]["fmax"] - paper_fmax)
        assert diff < TOLERANCE, \
            (f"{ont.upper()} {bin_key}: reproduced Fmax={ours[bin_key]['fmax']:.3f} "
             f"vs HEAL paper {paper_fmax:.3f} (diff {diff:.3f} > tol {TOLERANCE})")
