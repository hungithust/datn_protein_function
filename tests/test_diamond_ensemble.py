import numpy as np
from pathlib import Path
from ampr.evaluation.diamond_ensemble import (
    compute_diamond_scores, ensemble_scores, tune_alpha,
)


def test_diamond_score_max_over_homologs(tmp_path):
    tsv = tmp_path / 'd.tsv'
    tsv.write_text(
        # qseqid sseqid pident length qlen slen
        "Q1\tT1\t40.0\t100\t200\t200\n"
        "Q1\tT2\t60.0\t100\t200\t200\n"
        "Q2\tT1\t30.0\t100\t200\t200\n"
    )
    train_labels = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    train_order = {'T1': 0, 'T2': 1}
    test_ids = ['Q1', 'Q2', 'Q3']  # Q3 không có homolog
    scores = compute_diamond_scores(str(tsv), train_labels, train_order, test_ids, n_terms=3)
    # Q1: max(0.4·[1,0,1], 0.6·[0,1,0]) = [0.4, 0.6, 0.4]
    assert np.allclose(scores[0], [0.4, 0.6, 0.4], atol=1e-4)
    # Q2: chỉ T1 → 0.3·[1,0,1]
    assert np.allclose(scores[1], [0.3, 0.0, 0.3], atol=1e-4)
    # Q3: no homolog → 0
    assert np.all(scores[2] == 0.0)


def test_ensemble_no_homolog_falls_back_to_model():
    model = np.array([[0.8, 0.2], [0.5, 0.5]], dtype=np.float32)
    diamond = np.array([[0.4, 0.6], [0.0, 0.0]], dtype=np.float32)
    out = ensemble_scores(model, diamond, alpha=0.6)
    assert np.allclose(out[0], 0.6 * model[0] + 0.4 * diamond[0])
    assert np.allclose(out[1], model[1])


def test_tune_alpha_picks_better_value():
    np.random.seed(0)
    labels = (np.random.rand(50, 5) > 0.7).astype(np.float32)
    diamond = labels * 0.8 + np.random.rand(50, 5) * 0.05  # diamond gần như đúng
    model = np.random.rand(50, 5) * 0.3                    # model rất tệ
    dag = np.zeros((5, 5))
    a = tune_alpha(model, diamond, labels, dag, alpha_range=np.array([0.1, 0.5, 0.9]))
    assert a == 0.1  # alpha thấp = ưu tiên diamond → đúng hơn
