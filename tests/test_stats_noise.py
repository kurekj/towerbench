"""Seed-aware and window-aggregation variants of the paired bootstrap, and the training-noise switch."""
import numpy as np
import pandas as pd

from towerbench.eval.stats import paired_bootstrap, paired_bootstrap_seeds, pooled_window_variants
from towerbench.run import inject_noise


def test_seed_bootstrap_reduces_to_paired_with_one_seed():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.1, 1, 500), rng.normal(0.0, 1, 500)
    plain = paired_bootstrap(a, b, 500)
    two = paired_bootstrap_seeds(a[None, :], b[None, :], 500)
    assert abs(plain["diff"] - two["diff"]) < 1e-12
    assert abs(plain["lo"] - two["lo"]) < 0.05 and abs(plain["hi"] - two["hi"]) < 0.05


def test_seed_bootstrap_is_wider_with_seed_variance():
    rng = np.random.default_rng(1)
    users = rng.normal(0.05, 0.2, 2000)
    A = users[None, :] + rng.normal(0, 0.05, (5, 1))       # five seeds, each shifted by a seed effect
    B = np.zeros((1, 2000))
    plain = paired_bootstrap(A.mean(0), B[0], 1000)
    two = paired_bootstrap_seeds(A, B, 1000)
    assert (two["hi"] - two["lo"]) > (plain["hi"] - plain["lo"])


def test_pooled_variants_agree_on_independent_windows():
    rng = np.random.default_rng(2)
    diffs, users = [], []
    for w in range(5):
        n = 400
        diffs.append(rng.normal(0.02, 0.1, n))
        users.append(np.arange(w * n, (w + 1) * n))          # disjoint users -> no clustering
    out = pooled_window_variants(diffs, users, 1000)
    assert out["n_pairs"] == 2000 and out["cluster_users"]["n_users"] == 2000
    for k in ("pairs", "cluster_users", "block_windows", "random_effects"):
        assert out[k]["decisive"], k
        assert abs(out[k]["diff"] - out["pairs"]["diff"]) < 1e-9 or k == "random_effects"
    # cluster bootstrap over disjoint users is the pairs bootstrap up to Monte-Carlo noise
    assert abs(out["cluster_users"]["lo"] - out["pairs"]["lo"]) < 0.01


def test_cluster_bootstrap_widens_when_users_repeat():
    rng = np.random.default_rng(3)
    user_effect = rng.normal(0, 0.3, 300)
    diffs = [user_effect + rng.normal(0.01, 0.02, 300) for _ in range(5)]  # same users in every window
    users = [np.arange(300)] * 5
    out = pooled_window_variants(diffs, users, 1000)
    assert out["cluster_users"]["n_users"] == 300
    assert (out["cluster_users"]["hi"] - out["cluster_users"]["lo"]) > (out["pairs"]["hi"] - out["pairs"]["lo"])


def test_inject_noise_replaces_requested_fraction():
    rng = np.random.default_rng(4)
    n_items = 1000
    df = pd.DataFrame({"u": np.repeat(np.arange(200), 10).astype(np.int32), "i": rng.choice(n_items, 2000).astype(np.int32),
                       "ts": np.arange(2000)})  # int32 columns as written by prepare.py
    df = df.drop_duplicates(["u", "i"]).reset_index(drop=True)
    noisy = inject_noise(df, 0.2, n_items, seed=0)
    assert len(noisy) <= len(df)
    changed = (pd.merge(df, noisy, on=["u", "ts"], suffixes=("", "_n")).eval("i != i_n")).mean()
    assert 0.15 < changed < 0.25
    assert inject_noise(df, 0.0, n_items, 0) is df
