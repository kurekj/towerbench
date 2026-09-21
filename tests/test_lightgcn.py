"""LightGCN on the synthetic grouped dataset (CPU): must beat popularity clearly, the early-stopping
signal must equal the post-fit validation score, and set_history must rebuild the graph."""
import numpy as np
import pandas as pd

from tests.test_two_tower import _csr, synthetic
from towerbench.eval.metrics import per_user_metrics
from towerbench.models.lightgcn import LightGCN, _norm_adj
from towerbench.models.simple import Popularity


def test_norm_adj_is_symmetric_and_normalised():
    df, n_u, n_i = synthetic(n_users=40, n_items=30)
    train = _csr(df[df.split == "train"], n_u, n_i)
    A = _norm_adj(train, "cpu").to_dense().numpy()
    assert np.allclose(A, A.T)
    deg = (A > 0).sum(1)
    # row sums of D^-1/2 A D^-1/2 equal sum_j 1/sqrt(d_i d_j); each entry <= 1/sqrt(d_i)
    assert (A.max(1) <= 1.0 / np.sqrt(np.maximum(deg, 1)) + 1e-6).all()


def test_lightgcn_learns_and_validation_is_consistent():
    df, n_u, n_i = synthetic()
    train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    train, val_pos = _csr(train_df, n_u, n_i), {u: g.i.values for u, g in val_df.groupby("u")}
    users = np.arange(n_u)
    cfg = dict(model="lightgcn", dim=16, layers=2, lr=0.02, reg=1e-4, batch=128, epochs=15, patience=15, seed=0)

    def val_eval(m):
        return float(per_user_metrics(m.rank(users, train, k=10), users, val_pos, n_i)["ndcg@10"].mean())

    lg = LightGCN(n_u, n_i, cfg, None, device="cpu")
    info = lg.fit(train, train_df, train, val_eval)
    after = val_eval(lg)
    assert abs(info["best_val_ndcg10"] - after) < 1e-6, (info["best_val_ndcg10"], after)

    pop = Popularity(n_u, n_i, {}, None, device="cpu")
    pop.fit(train, train_df, train, val_eval)
    assert after > val_eval(pop) * 1.5

    # set_history(train + val) changes the propagated embeddings but keeps the parameters
    w_before = lg.user_emb.weight.detach().clone()
    U_before = lg.U.clone()
    lg.set_history(pd.concat([train_df, val_df]))
    assert not np.allclose(U_before.numpy(), lg.U.numpy())
    assert np.allclose(w_before.numpy(), lg.user_emb.weight.detach().numpy())
    top = lg.rank(users, train, k=10)
    assert top.shape == (n_u, 10)
