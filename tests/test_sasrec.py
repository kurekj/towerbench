"""SASRec on the synthetic grouped dataset: finite loss, consistent validation, beats popularity."""
import numpy as np

from tests.test_two_tower import _csr, synthetic
from towerbench.eval.metrics import per_user_metrics
from towerbench.models.sasrec import SASRec
from towerbench.models.simple import Popularity


def test_sasrec_learns_and_validation_is_consistent():
    df, n_u, n_i = synthetic()
    train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    train, val_pos = _csr(train_df, n_u, n_i), {u: g.i.values for u, g in val_df.groupby("u")}
    users = np.arange(n_u)

    def val_eval(m):
        return float(per_user_metrics(m.rank(users, train, k=10), users, val_pos, n_i)["ndcg@10"].mean())

    sr = SASRec(n_u, n_i, dict(dim=16, max_len=8, layers=1, heads=2, epochs=6, patience=6, batch=32,
                               lr=0.01, seed=0), None, device="cpu")
    info = sr.fit(train, train_df, train, val_eval)
    assert np.isfinite(info["last_loss"])
    after = val_eval(sr)
    assert abs(info["best_val_ndcg10"] - after) < 1e-6
    pop = Popularity(n_u, n_i, {}, None, device="cpu")
    pop.fit(train, train_df, train, val_eval)
    assert after > val_eval(pop) * 1.5
