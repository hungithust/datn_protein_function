from scripts.precompute_go_text import build_go_texts


def test_build_go_texts_name_and_def():
    nodes = {
        "GO:0001": {"name": "catalytic activity",
                    "def": '"Catalysis of a reaction." [GOC:x]'},
        "GO:0002": {"name": "binding"},          # no def
    }
    texts = build_go_texts(["GO:0001", "GO:0002", "GO:9999"], nodes)
    assert texts[0] == "catalytic activity. Catalysis of a reaction."
    assert texts[1] == "binding."
    assert texts[2] == "GO:9999."                # missing node -> id as name
