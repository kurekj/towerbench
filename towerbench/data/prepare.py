"""Frozen preprocessing protocol.

1. deduplicate (user,item) keeping the earliest event
2. iterative k-core filtering (users >= k_user, items >= k_item events)
3. contiguous integer ids
4. per-user temporal leave-last-out split: last event -> test, second last -> val, rest -> train
   (every user has >= k_user >= 3 events after k-core, so train is never empty)
5. persist parquet + item side features + dataset card (stats.json)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..paths import PREPARED
from .loaders import LOADERS


def k_core(inter: pd.DataFrame, k_user: int, k_item: int) -> pd.DataFrame:
    while True:
        n = len(inter)
        uc = inter.user.value_counts()
        inter = inter[inter.user.isin(uc[uc >= k_user].index)]
        ic = inter.item.value_counts()
        inter = inter[inter.item.isin(ic[ic >= k_item].index)]
        if len(inter) == n:
            return inter


def prepare(name: str, k_user: int = 5, k_item: int = 5, max_users: int | None = None,
            seed: int = 0) -> dict:
    t0 = time.time()
    inter, items = LOADERS[name]()
    raw_n = len(inter)
    inter = inter.dropna()
    inter = inter.assign(ts=inter.ts.astype("int64"))
    inter = (inter.sort_values("ts", kind="stable")
             .drop_duplicates(["user", "item"], keep="first"))
    inter = k_core(inter, k_user, k_item)
    if max_users and inter.user.nunique() > max_users:
        rng = np.random.default_rng(seed)
        keep = rng.choice(inter.user.unique(), size=max_users, replace=False)
        inter = inter[inter.user.isin(keep)]
        inter = k_core(inter, k_user, k_item)
    uids = {u: i for i, u in enumerate(sorted(inter.user.unique()))}
    iids = {v: i for i, v in enumerate(sorted(inter.item.unique()))}
    inter = inter.assign(u=inter.user.map(uids).astype("int32"),
                         i=inter.item.map(iids).astype("int32"))
    inter = inter.sort_values(["u", "ts"], kind="stable").reset_index(drop=True)
    # temporal leave-last-out
    pos = inter.groupby("u").cumcount(ascending=False)  # 0 = last
    inter["split"] = np.where(pos == 0, "test", np.where(pos == 1, "val", "train"))
    # items side table aligned to contiguous ids
    it = pd.DataFrame({"item": list(iids.keys()), "i": list(iids.values())})
    it = it.merge(items, on="item", how="left")
    for col in ("text", "cat", "city"):
        if col not in it:
            it[col] = ""
        it[col] = it[col].fillna("").astype(str)
    for col in ("lat", "lon", "price"):
        if col not in it:
            it[col] = np.nan
        it[col] = pd.to_numeric(it[col], errors="coerce")
    it = it.sort_values("i").reset_index(drop=True)
    out = PREPARED / name
    out.mkdir(parents=True, exist_ok=True)
    inter[["u", "i", "ts", "split"]].to_parquet(out / "interactions.parquet", index=False)
    it.to_parquet(out / "items.parquet", index=False)
    n_u, n_i, n = len(uids), len(iids), len(inter)
    stats = {
        "dataset": name, "raw_events": int(raw_n), "k_user": k_user, "k_item": k_item,
        "n_users": n_u, "n_items": n_i, "n_interactions": int(n),
        "density": n / (n_u * n_i), "avg_len": n / n_u,
        "n_train": int((inter.split == "train").sum()),
        "n_val": int((inter.split == "val").sum()),
        "n_test": int((inter.split == "test").sum()),
        "has_text": bool((it.text.str.len() > 0).mean() > 0.5),
        "has_geo": bool(it.lat.notna().mean() > 0.5),
        "has_price": bool(it.price.notna().mean() > 0.5),
        "ts_min": int(inter.ts.min()), "ts_max": int(inter.ts.max()),
        "prep_seconds": time.time() - t0,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_prepared(name: str):
    out = PREPARED / name
    inter = pd.read_parquet(out / "interactions.parquet")
    items = pd.read_parquet(out / "items.parquet")
    stats = json.loads((out / "stats.json").read_text())
    return inter, items, stats
