"""End-to-end two-tower test on a synthetic dataset (CPU): early-stopping signal must equal the
post-fit validation score (regression test for the stale item-cache bug), and the model must beat
popularity on data with planted structure."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from towerbench.eval.metrics import per_user_metrics
from towerbench.models.simple import Popularity
from towerbench.models.two_tower import TwoTower


def synthetic(n_users=300, n_items=120, n_groups=4, seed=0):
    """Users belong to a group and interact mostly with their group's items, in time order."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        g = u % n_groups
        own = np.arange(g, n_items, n_groups)
        n = rng.integers(6, min(12, len(own) + 1))
        items = rng.choice(own, size=n, replace=False)
        for t, i in enumerate(items):
            rows.append((u, int(i), t))
    df = pd.DataFrame(rows, columns=["u", "i", "ts"])
    pos = df.groupby("u").cumcount(ascending=False)
    df["split"] = np.where(pos == 0, "test", np.where(pos == 1, "val", "train"))
    return df, n_users, n_items


def _csr(df, n_users, n_items):
    return sp.csr_matrix((np.ones(len(df), dtype=np.float32), (df.u.values, df.i.values)), shape=(n_users, n_items))


def test_two_tower_learns_and_validation_is_consistent():
    df, n_u, n_i = synthetic()
    train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    train, val_pos = _csr(train_df, n_u, n_i), {u: g.i.values for u, g in val_df.groupby("u")}
    users = np.arange(n_u)
    cfg = dict(model="two_tower", dim=16, max_len=10, tau=0.1, logq=True, epochs=4, patience=4, batch=64,
               lr=0.01, seed=0, samples_per_user=0)

    def val_eval(m):
        return float(per_user_metrics(m.rank(users, train, k=10), users, val_pos, n_i)["ndcg@10"].mean())

    tt = TwoTower(n_u, n_i, cfg, None, device="cpu")
    info = tt.fit(train, train_df, train, val_eval)
    after = val_eval(tt)
    assert abs(info["best_val_ndcg10"] - after) < 1e-6, (info["best_val_ndcg10"], after)

    pop = Popularity(n_u, n_i, {}, None, device="cpu")
    pop.fit(train, train_df, train, val_eval)
    assert after > val_eval(pop) * 1.5


def test_history_masking_excludes_train_items():
    df, n_u, n_i = synthetic(n_users=50, n_items=40)
    train_df = df[df.split == "train"]
    train = _csr(train_df, n_u, n_i)
    tt = TwoTower(n_u, n_i, dict(dim=8, max_len=5, epochs=1, patience=1, batch=32, seed=0), None, device="cpu")
    tt.fit(train, train_df, train, lambda m: 0.0)
    top = tt.rank(np.arange(n_u), train, k=10)
    for u in range(n_u):
        assert not set(top[u]) & set(train_df[train_df.u == u].i)
