"""Single metric implementation shared by every model (metrics parity by construction).

Input: ``topk`` int array [n_eval_users, K_max] of ranked item ids (history already masked)
and a dict user -> positives.  Output: per-user arrays (needed for paired bootstrap).
Definitions match the frozen protocol: NDCG@K binary gain, log2 discount, ideal = min(|pos|,K);
Recall@K = hits/|pos|; HitRate@K = 1[hits>0]; MAP@K; MRR (first hit).
"""
from __future__ import annotations

import numpy as np

KS = (10, 20, 50)


def per_user_metrics(topk: np.ndarray, users: np.ndarray, positives: dict[int, np.ndarray],
                     n_items: int, item_pop: np.ndarray | None = None) -> dict[str, np.ndarray]:
    n, kmax = topk.shape
    out = {f"{m}@{k}": np.zeros(n, dtype=np.float64) for k in KS for m in ("recall", "ndcg", "hit", "map")}
    out["mrr"] = np.zeros(n, dtype=np.float64)
    disc = 1.0 / np.log2(np.arange(2, kmax + 2))
    for r, u in enumerate(users):
        pos = positives[u]
        hits = np.isin(topk[r], pos)
        npos = len(pos)
        if hits.any():
            out["mrr"][r] = 1.0 / (np.argmax(hits) + 1)
        for k in KS:
            h = hits[:k]
            kk = len(h)
            nh = h.sum()
            out[f"recall@{k}"][r] = nh / npos
            out[f"hit@{k}"][r] = float(nh > 0)
            dcg = (h * disc[:kk]).sum()
            idcg = disc[: min(npos, k)].sum()
            out[f"ndcg@{k}"][r] = dcg / idcg
            if nh:
                prec = np.cumsum(h) / np.arange(1, kk + 1)
                out[f"map@{k}"][r] = (prec * h).sum() / min(npos, k)
    return out


def catalog_metrics(topk: np.ndarray, n_items: int, item_pop: np.ndarray, k: int = 10) -> dict:
    """Beyond-accuracy: coverage@k (fraction of catalog recommended) and mean log-popularity."""
    rec = topk[:, :k]
    cov = len(np.unique(rec)) / n_items
    logpop = np.log1p(item_pop[rec]).mean()
    return {f"coverage@{k}": float(cov), f"logpop@{k}": float(logpop)}
