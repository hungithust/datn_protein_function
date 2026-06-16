"""Test scripts/gen_v6_baseline_configs.py emits the 650M PDB-30K baseline configs."""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_generates_baseline(tmp_path):
    from scripts.gen_v6_baseline_configs import generate
    paths = generate(out_dir=str(tmp_path), pdb_emb="data/embeddings/esm2_650m_pdb.h5")
    assert len(paths) == 9  # 3 branches x 3 seeds
    c = yaml.safe_load(Path(tmp_path / "mf_v6_pdb30base_s42.yaml").read_text())
    assert c["branch"] == "MF" and c["seed"] == 42
    assert c["n_terms"] == 489
    assert c["model"]["seq"]["d_model"] == 1280
    assert c["data"]["esm2_h5"].endswith("esm2_650m_pdb.h5")
    assert c["data"]["splits"].endswith("pdbch/splits.json")
    assert "pdb30base" in c["output"]["checkpoint_dir"]
    # identical finetune recipe as v6sm (so the only difference is the pretrain init)
    assert c["training"]["epochs"] == 60 and c["training"]["lr"] == 3.0e-4
