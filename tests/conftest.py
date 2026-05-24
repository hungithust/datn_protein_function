"""Shared fixtures for Phase 0 data-build tests.

Each fixture writes tiny, hand-crafted text files into a tmp_path and yields
the directory. All real Phase 0 scripts accept --data-dir <path>, so tests
can run them against the fixture directory without touching real data.
"""

from pathlib import Path
import pytest


# ── 3 proteins, 4 MF terms, 3 BP terms, 2 CC terms — easy to verify by hand ─
TINY_ANNOT_TSV = (
    "### GO-terms (molecular_function)\n"
    "GO:0001\tGO:0002\tGO:0003\tGO:0004\n"
    "### GO-names (molecular_function)\n"
    "mf_one\tmf_two\tmf_three\tmf_four\n"
    "### GO-terms (biological_process)\n"
    "GO:0010\tGO:0011\tGO:0012\n"
    "### GO-names (biological_process)\n"
    "bp_one\tbp_two\tbp_three\n"
    "### GO-terms (cellular_component)\n"
    "GO:0020\tGO:0021\n"
    "### GO-names (cellular_component)\n"
    "cc_one\tcc_two\n"
    "### PDB-chain\tGO-terms (molecular_function)\tGO-terms (biological_process)\tGO-terms (cellular_component)\n"
    "1AAA-A\tGO:0001,GO:0003\tGO:0010\tGO:0020\n"
    "1BBB-A\tGO:0002\tGO:0011,GO:0012\tGO:0021\n"
    "1CCC-A\tGO:0001\t\tGO:0020,GO:0021\n"
)

TINY_TRAIN_TXT = "1AAA-A\n1BBB-A\n"
TINY_VALID_TXT = "1CCC-A\n"
TINY_TEST_CSV = (
    "PDB-chain,<30%,<40%,<50%,<70%,<95%\n"
    "1CCC-A,0,1,1,1,1\n"
    "1AAA-A,1,1,1,1,1\n"
)

# Minimal OBO with the GO terms used above + a parent chain
# GO:0001 and GO:0002 are children of GO:0003; GO:0003 is child of GO:0004 (root for MF subset)
TINY_OBO = """format-version: 1.2

[Term]
id: GO:0001
name: mf one
namespace: molecular_function
is_a: GO:0003 ! mf three

[Term]
id: GO:0002
name: mf two
namespace: molecular_function
is_a: GO:0003 ! mf three

[Term]
id: GO:0003
name: mf three
namespace: molecular_function
is_a: GO:0004 ! mf four

[Term]
id: GO:0004
name: mf four
namespace: molecular_function

[Term]
id: GO:0010
name: bp one
namespace: biological_process

[Term]
id: GO:0011
name: bp two
namespace: biological_process

[Term]
id: GO:0012
name: bp three
namespace: biological_process

[Term]
id: GO:0020
name: cc one
namespace: cellular_component

[Term]
id: GO:0021
name: cc two
namespace: cellular_component
"""


@pytest.fixture
def tiny_pdbch(tmp_path: Path) -> Path:
    """Return a directory populated with tiny PDBch source files."""
    (tmp_path / "annot.tsv").write_text(TINY_ANNOT_TSV, encoding="utf-8")
    (tmp_path / "train.txt").write_text(TINY_TRAIN_TXT, encoding="utf-8")
    (tmp_path / "valid.txt").write_text(TINY_VALID_TXT, encoding="utf-8")
    (tmp_path / "test.csv").write_text(TINY_TEST_CSV, encoding="utf-8")
    (tmp_path / "go-basic.obo").write_text(TINY_OBO, encoding="utf-8")
    return tmp_path
