"""Evaluation protocols.

``loo``      per-user temporal leave-last-out (split column produced by prepare.py):
             train = all but last two events, val = second-to-last, test = last.
``temporal`` frozen global chronological windows (train_days / val_days / test_days ending at an
             anchor); ``window`` steps the anchor back by test_days for rolling-origin
             evaluation (window 0 = most recent).  History for a test positive is train+val only.

Both yield a ``Split`` with CSR-like per-user item lists and never expose test items to fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

DAY = 86400


@dataclass
class Split:
    n_users: int
    n_items: int
    train: pd.DataFrame        # u, i, ts (fit on this)
    val: pd.DataFrame          # u, i, ts (model selection)
    test: pd.DataFrame         # u, i, ts (read once)
    protocol: str
    meta: dict = field(default_factory=dict)

    def csr(self, part: str) -> sp.csr_matrix:
        df = getattr(self, part)
        m = sp.csr_matrix((np.ones(len(df), dtype=np.float32), (df.u.values, df.i.values)),
                          shape=(self.n_users, self.n_items))
        m.data[:] = 1.0
        return m

    def history(self, stage: str) -> sp.csr_matrix:
        """Items a model may use as user history when ranking for ``stage``."""
        return self.csr("train") if stage == "val" else (self.csr("train") + self.csr("val")).tocsr()

    def positives(self, stage: str) -> dict[int, np.ndarray]:
        df = self.val if stage == "val" else self.test
        return {u: g.i.values for u, g in df.groupby("u")}

    def eval_users(self, stage: str) -> np.ndarray:
        """Users with >=1 positive in ``stage`` and >=1 history event (the 'with-history' cohort)."""
        pos = set(self.positives(stage).keys())
        hist = self.history(stage)
        has_hist = np.asarray(hist.sum(axis=1)).ravel() > 0
        return np.array(sorted(u for u in pos if has_hist[u]), dtype=np.int64)


def make_split(inter: pd.DataFrame, n_users: int, n_items: int, protocol: str = "loo",
               train_days: int | None = None, val_days: int = 90, test_days: int = 90,
               window: int = 0) -> Split:
    if protocol == "loo":
        tr, va, te = (inter[inter.split == s][["u", "i", "ts"]] for s in ("train", "val", "test"))
        return Split(n_users, n_items, tr, va, te, "loo")
    if protocol == "temporal":
        # anchor is an exclusive upper bound: window 0 ends one second after the last event
        anchor = int(inter.ts.max()) + 1 - window * test_days * DAY
        t_test0 = anchor - test_days * DAY
        t_val0 = t_test0 - val_days * DAY
        t_train0 = t_val0 - train_days * DAY if train_days else -1
        te = inter[(inter.ts >= t_test0) & (inter.ts < anchor)]
        va = inter[(inter.ts >= t_val0) & (inter.ts < t_test0)]
        tr = inter[(inter.ts >= t_train0) & (inter.ts < t_val0)]
        meta = {"anchor": anchor, "window": window, "train_days": train_days,
                "val_days": val_days, "test_days": test_days}
        return Split(n_users, n_items, tr[["u", "i", "ts"]], va[["u", "i", "ts"]],
                     te[["u", "i", "ts"]], "temporal", meta)
    raise ValueError(protocol)
