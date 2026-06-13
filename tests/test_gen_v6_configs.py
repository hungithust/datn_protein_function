"""Test scripts/gen_v6_configs.py emits valid pretrain + finetune configs."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_generates_configs(tmp_path):
    from scripts.gen_v6_configs import generate
    paths = generate(out_dir=str(tmp_path), sm_dir="data/swissmodel_art",
                     pdb_emb="data/embeddings/esm2_650m_pdb.h5",
                     sm_emb="data/embeddings/esm2_650m_sm.h5",
                     sm_cmap="data/swissmodel_art/cmap_all_sm.h5")
    # 3 pretrain + 9 finetune
    assert len(paths) == 12
    pre = yaml.safe_load(Path(tmp_path / "mf_v6sm_pretrain.yaml").read_text())
    assert pre["branch"] == "MF" and pre["n_terms"] == 489
    assert pre["model"]["seq"]["d_model"] == 1280            # 650M, not 2560
    assert pre["data"]["esm2_h5"].endswith("esm2_650m_sm.h5")
    assert pre["data"]["splits"].endswith("splits_sm.json")
    ft = yaml.safe_load(Path(tmp_path / "bp_v6sm_finetune_s123.yaml").read_text())
    assert ft["branch"] == "BP" and ft["n_terms"] == 1943
    assert ft["seed"] == 123
    assert ft["data"]["esm2_h5"].endswith("esm2_650m_pdb.h5")
    assert ft["data"]["splits"].endswith("pdbch/splits.json")
    assert "s123" in ft["output"]["checkpoint_dir"]
