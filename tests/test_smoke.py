import numpy as np

from towerbench.eval.metrics import per_user_metrics
from towerbench.eval.stats import paired_bootstrap


def test_metrics_single_positive_rank1():
    topk = np.array([[7, 3, 5]])
    m = per_user_metrics(topk, np.array([0]), {0: np.array([7])}, 10)
    assert m["ndcg@10"][0] == 1.0 and m["hit@10"][0] == 1.0 and m["mrr"][0] == 1.0


def test_metrics_multi_positive():
    topk = np.array([[1, 2, 3, 4]])
    m = per_user_metrics(topk, np.array([0]), {0: np.array([2, 9])}, 10)
    assert abs(m["recall@10"][0] - 0.5) < 1e-9
    assert abs(m["ndcg@10"][0] - (1 / np.log2(3)) / (1 + 1 / np.log2(3))) < 1e-9


def test_paired_bootstrap_decisive():
    a, b = np.ones(200), np.zeros(200)
    r = paired_bootstrap(a, b, 200)
    assert r["decisive"] and r["diff"] == 1.0
