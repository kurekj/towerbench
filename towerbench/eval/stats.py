"""Non-parametric bootstrap over users (percentile CI) and paired bootstrap of differences."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42,
                     alpha: float = 0.05):
    """CI of mean(a-b) over users; users must be aligned. Decisive if CI excludes 0."""
    d = a - b
    mean, lo, hi = bootstrap_ci(d, n_boot, seed, alpha)
    return {"diff": mean, "lo": lo, "hi": hi, "decisive": bool(lo > 0 or hi < 0),
            "p_sign": float(2 * min((d > 0).mean(), (d < 0).mean()))}


def holm(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    order = np.argsort(pvals)
    m = len(pvals)
    reject = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject
