from scripts.gen_sweep_configs import set_dotted, expand_grid


def test_set_dotted_nested():
    cfg = {'model': {'d_hidden': 512}}
    set_dotted(cfg, 'model.d_hidden', 1024)
    assert cfg['model']['d_hidden'] == 1024


def test_expand_grid_cartesian():
    base = {'model': {'classifier': 'both', 'd_hidden': 512},
            'data': {'go_emb': 'x'}}
    grid = {
        'model.classifier': [('both', 'both'), ('la', 'label_attn')],
        'model.d_hidden':   [('h512', 512), ('h1024', 1024)],
        'data.go_emb':      [('comb', 'go_emb_mf_v2.npy'), ('text', 'go_text_mf.npy')],
    }
    out = expand_grid(base, grid)
    assert len(out) == 8                      # 2*2*2
    names = [n for n, _ in out]
    assert 'both_h512_comb' in names
    assert 'la_h1024_text' in names
    # base is not mutated
    assert base['model']['d_hidden'] == 512
