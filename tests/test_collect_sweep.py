from scripts.collect_sweep import best_fmax_from_log


def test_best_fmax_picks_max():
    log = (
        "[V3] Epoch 1/50: loss=0.4 val_Fmax_raw=0.30 val_Fmax_dag=0.31\n"
        "[V3] Epoch 2/50: loss=0.3 val_Fmax_raw=0.40 val_Fmax_dag=0.42\n"
        "[V3] Epoch 3/50: loss=0.2 val_Fmax_raw=0.39 val_Fmax_dag=0.41\n"
    )
    assert abs(best_fmax_from_log(log) - 0.42) < 1e-9


def test_best_fmax_no_match_returns_none():
    assert best_fmax_from_log("no epochs here") is None
