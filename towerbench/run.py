"""One experiment = (dataset, protocol[, window], model config, seed) -> results dir with
metrics.json (means + bootstrap CIs), per_user.npz (for paired tests) and a read ledger."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data.prepare import load_prepared
from .eval.metrics import catalog_metrics, per_user_metrics
from .eval.protocol import make_split
from .eval.stats import bootstrap_ci
from .models.multvae import MultVAE
from .models.sasrec import SASRec
from .models.simple import EASE, Popularity
from .models.two_tower import TwoTower
from .paths import PREPARED, RESULTS

MODELS = {"pop": Popularity, "ease": EASE, "multvae": MultVAE, "two_tower": TwoTower, "sasrec": SASRec}

CFG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_cfg(name: str) -> dict:
    p = CFG_DIR / "models" / f"{name}.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {"model": name}


def dataset_cfg(name: str) -> dict:
    allc = yaml.safe_load((CFG_DIR / "datasets.yaml").read_text())
    return allc.get(name, {})


def run(dataset: str, model_cfg: str, seed: int = 0, protocol: str = "loo", window: int = 0,
        device: str = "cuda", n_boot: int = 2000, k: int = 50, overwrite: bool = False) -> dict:
    tag = f"{protocol}" + (f"_w{window}" if protocol == "temporal" else "")
    out = RESULTS / dataset / tag / model_cfg / f"seed{seed}"
    if (out / "metrics.json").exists() and not overwrite:
        return json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = load_cfg(model_cfg)
    cfg["seed"] = seed
    dcfg = dataset_cfg(dataset)
    inter, items, stats = load_prepared(dataset)
    if cfg.get("use_text"):
        emb_path = PREPARED / dataset / "text_emb_bge-base-en-v1.5.npy"
        if emb_path.exists():
            items.attrs["text_emb"] = np.load(emb_path)
    n_u, n_i = stats["n_users"], stats["n_items"]
    split = make_split(inter, n_u, n_i, protocol, dcfg.get("train_days"), dcfg.get("val_days", 90),
                       dcfg.get("test_days", 90), window)
    train = split.csr("train")
    item_pop = np.asarray(train.sum(axis=0)).ravel()
    val_users, val_pos = split.eval_users("val"), split.positives("val")
    val_hist = split.history("val")
    if len(val_users) == 0 or len(split.eval_users("test")) == 0:
        raise RuntimeError(f"{dataset} {tag}: no evaluable users (val={len(val_users)}); window too sparse")
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MODELS[cfg["model"]](n_u, n_i, cfg, items, device)

    sub = val_users if len(val_users) <= 20000 else np.random.default_rng(0).choice(val_users, 20000, replace=False)

    def val_eval(m) -> float:
        topk = m.rank(sub, val_hist, k=10)
        return float(per_user_metrics(topk, sub, val_pos, n_i)["ndcg@10"].mean())

    fit_info = model.fit(train, split.train, val_hist, val_eval)
    # validation (full)
    val_top = model.rank(val_users, val_hist, k=k)
    val_m = per_user_metrics(val_top, val_users, val_pos, n_i)
    # test: read exactly once, history = train + val
    test_users, test_pos, test_hist = split.eval_users("test"), split.positives("test"), split.history("test")
    if hasattr(model, "set_history"):
        model.set_history(pd.concat([split.train, split.val]))
    test_top = model.rank(test_users, test_hist, k=k)
    test_m = per_user_metrics(test_top, test_users, test_pos, n_i)
    res = {"dataset": dataset, "protocol": tag, "model_cfg": model_cfg, "model": cfg["model"], "seed": seed,
           "n_val_users": int(len(val_users)), "n_test_users": int(len(test_users)),
           "n_train": int(train.nnz), "split_meta": split.meta, "fit": fit_info,
           "seconds": time.time() - t0, "device": str(model.device),
           "val": {m: float(v.mean()) for m, v in val_m.items()},
           "test": {}, "test_catalog": catalog_metrics(test_top, n_i, item_pop, 10)}
    for m, v in test_m.items():
        mean, lo, hi = bootstrap_ci(v, n_boot, 42)
        res["test"][m] = {"mean": mean, "lo": lo, "hi": hi}
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
    np.savez_compressed(out / "per_user.npz", users=test_users, **{m: v for m, v in test_m.items()})
    np.save(out / "test_topk.npy", test_top[:, :20])
    with open(RESULTS / dataset / "test_read_ledger.jsonl", "a") as f:
        f.write(json.dumps({"tag": tag, "model_cfg": model_cfg, "seed": seed, "t": time.time()}) + "\n")
    return res
