"""Non-parametric bootstrap over users (percentile CI) and paired bootstrap of differences.

Two optional variants surface the fine print of the pooled per-user interval:

* ``paired_bootstrap_seeds`` resamples seeds as well as users, so the interval also carries the
  model-level (seed) variability that the default interval deliberately leaves in the point
  estimate only;
* ``pooled_window_variants`` recomputes a pooled rolling-origin interval under aggregation
  schemes that do not treat window-user pairs as exchangeable: a cluster bootstrap over users
  (a user's pairs from every window are resampled together), a block bootstrap over windows,
  and a DerSimonian-Laird random-effects combination of the per-window estimates.
"""
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


def paired_bootstrap_seeds(A: np.ndarray, B: np.ndarray, n_boot: int = 2000, seed: int = 42,
                           alpha: float = 0.05):
    """Two-level paired bootstrap.  ``A`` and ``B`` are [n_seeds, n_users] matrices of a per-user
    metric on aligned users; every replicate resamples the users with replacement and,
    independently for each model, its seeds with replacement, then averages over the resampled
    seeds before taking the per-user difference.  With one seed per model this reduces to
    ``paired_bootstrap``."""
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    rng = np.random.default_rng(seed)
    n = A.shape[1]
    point = A.mean(0) - B.mean(0)
    means = np.empty(n_boot)
    for r in range(n_boot):
        u = rng.integers(0, n, n)
        sa = rng.integers(0, A.shape[0], A.shape[0])
        sb = rng.integers(0, B.shape[0], B.shape[0])
        means[r] = (A[sa][:, u].mean(0) - B[sb][:, u].mean(0)).mean()
    lo, hi = float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
    return {"diff": float(point.mean()), "lo": lo, "hi": hi, "decisive": bool(lo > 0 or hi < 0)}


def pooled_window_variants(diffs: list[np.ndarray], users: list[np.ndarray], n_boot: int = 2000,
                           seed: int = 42, alpha: float = 0.05) -> dict:
    """Pooled interval of per-user differences over rolling-origin windows under four schemes.

    ``diffs[w]`` are the per-user differences of window ``w`` and ``users[w]`` the matching user ids.
    * ``pairs``: the default, resampling window-user pairs (exchangeable pairs);
    * ``cluster_users``: resampling users, each user carrying its pairs from every window;
    * ``block_windows``: resampling whole windows with replacement;
    * ``random_effects``: DerSimonian-Laird random-effects mean of the per-window estimates with a
      normal interval (between-window variance tau^2 added to each window's variance).
    """
    rng = np.random.default_rng(seed)
    W = len(diffs)
    pooled = np.concatenate(diffs)
    out = {"n_pairs": int(len(pooled)), "n_windows": W}
    mean, lo, hi = bootstrap_ci(pooled, n_boot, seed, alpha)
    out["pairs"] = {"diff": mean, "lo": lo, "hi": hi, "decisive": bool(lo > 0 or hi < 0)}
    # cluster bootstrap over users: per-user sum and count of pairs, ratio-of-sums resampling
    uid = np.concatenate(users)
    uniq, inv = np.unique(uid, return_inverse=True)
    sums = np.bincount(inv, weights=pooled, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
    out["cluster_users"] = {"diff": mean, "lo": lo, "hi": hi, "decisive": bool(lo > 0 or hi < 0),
                            "n_users": int(len(uniq))}
    # block bootstrap over windows (ratio of sums so that large windows keep their weight)
    wsum = np.array([d.sum() for d in diffs])
    wcnt = np.array([len(d) for d in diffs], dtype=float)
    widx = rng.integers(0, W, size=(n_boot, W))
    means = wsum[widx].sum(1) / wcnt[widx].sum(1)
    lo, hi = float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
    out["block_windows"] = {"diff": mean, "lo": lo, "hi": hi, "decisive": bool(lo > 0 or hi < 0)}
    # DerSimonian-Laird random effects over the per-window means
    y = np.array([d.mean() for d in diffs])
    v = np.array([d.var(ddof=1) / len(d) if len(d) > 1 else np.inf for d in diffs])
    w = 1.0 / v
    yf = (w * y).sum() / w.sum()
    Q = (w * (y - yf) ** 2).sum()
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (W - 1)) / C) if C > 0 else 0.0
    ws = 1.0 / (v + tau2)
    yr = (ws * y).sum() / ws.sum()
    se = (1.0 / ws.sum()) ** 0.5
    lo, hi = float(yr - 1.959964 * se), float(yr + 1.959964 * se)
    out["random_effects"] = {"diff": float(yr), "lo": lo, "hi": hi, "decisive": bool(lo > 0 or hi < 0),
                             "tau2": float(tau2), "I2": float(max(0.0, (Q - (W - 1)) / Q)) if Q > 0 else 0.0}
    return out


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
