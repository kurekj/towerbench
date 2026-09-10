"""Timestamp parsing and the frozen preprocessing rules on tiny in-memory frames."""
import numpy as np
import pandas as pd

from towerbench.data.loaders import _epoch_seconds
from towerbench.data.prepare import k_core
from towerbench.eval.protocol import make_split


def test_epoch_seconds_is_resolution_independent():
    v = _epoch_seconds(pd.Series(["2010-10-19T23:55:27Z", "2024-05-01", "not a date"]))
    assert v.iloc[0] == 1287532527 and v.iloc[1] == 1714521600 and pd.isna(v.iloc[2])


def test_k_core_is_iterative():
    df = pd.DataFrame({"user": list("aaabbbcc"), "item": list("xyzxyzxq"), "ts": range(8)})
    out = k_core(df, k_user=3, k_item=2)
    assert set(out.user) == {"a", "b"} and set(out.item) == {"x", "y", "z"}


def test_temporal_split_history_excludes_test_events():
    day = 86400
    inter = pd.DataFrame({"u": [0] * 6 + [1] * 3, "i": [0, 1, 2, 3, 4, 5, 0, 1, 2],
                          "ts": [d * day for d in (0, 10, 20, 30, 40, 50, 0, 45, 55)], "split": "train"})
    s = make_split(inter, 2, 6, "temporal", train_days=None, val_days=10, test_days=10, window=0)
    assert set(s.test.i) == {5, 2} and set(s.val.i) == {4, 1}
    hist = s.history("test")
    assert hist[0, 5] == 0 and hist[0, 4] == 1 and hist[1, 2] == 0
    assert list(s.eval_users("test")) == [0, 1]
    s1 = make_split(inter, 2, 6, "temporal", train_days=None, val_days=10, test_days=10, window=1)
    assert s1.meta["anchor"] == inter.ts.max() + 1 - 10 * day and set(s1.test.i) == {4, 1}


def test_loo_split_uses_marks():
    inter = pd.DataFrame({"u": [0, 0, 0], "i": [0, 1, 2], "ts": [1, 2, 3], "split": ["train", "val", "test"]})
    s = make_split(inter, 1, 3, "loo")
    assert list(s.test.i) == [2] and list(s.val.i) == [1] and s.history("val")[0, 1] == 0
    assert np.array_equal(s.eval_users("val"), [0])
