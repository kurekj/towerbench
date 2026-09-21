"""Statistical fine print of the pooled per-user bootstrap, recomputed from the persisted per-user
vectors without retraining anything (revision of the SoftwareX paper, Reviewer 2 W3 / Reviewer 3 M2a).

1. Window aggregation (W3): the pooled rolling-origin interval of model - reference under four
   schemes: exchangeable window-user pairs (the paper's default), a cluster bootstrap over users,
   a block bootstrap over windows, and a DerSimonian-Laird random-effects combination.
2. Seed variance (M2a): the leave-last-out and pooled intervals with and without resampling seeds.

Usage: python scripts/revision_stats.py [--out ~/towerbench/paper_assets] [--ref ease] [--n-boot 2000]
Writes tables/pooling_variants.tex, tables/seed_variance.tex and revision_stats.json.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from towerbench.eval.stats import paired_bootstrap, paired_bootstrap_seeds, pooled_window_variants
from towerbench.paths import RESULTS

DATASETS = ["ml-1m", "amazon-instruments", "amazon-videogames", "amazon-software", "gowalla", "airbnb", "steam"]
DS_LABEL = {"ml-1m": "MovieLens-1M", "amazon-instruments": "Amazon Instruments", "amazon-videogames": "Amazon Video Games",
            "amazon-software": "Amazon Software", "gowalla": "Gowalla", "airbnb": "Inside Airbnb", "steam": "Steam"}
M_LABEL = {"ease": "EASE", "multvae": "Mult-VAE", "sasrec": "SASRec", "tt_id": "TT-ID", "tt_feat": "TT-Feat",
           "tt_feat_dcn": "TT-Feat + DCN", "lightgcn": "LightGCN"}
COMPARISONS = ["tt_feat", "tt_id", "multvae", "lightgcn"]
METRIC = "ndcg@10"


def load(models):
    """runs[dataset][tag][model] = (users, matrix[n_seeds, n_users]) for the requested models."""
    runs = defaultdict(lambda: defaultdict(dict))
    for d in DATASETS:
        for tagdir in sorted((RESULTS / d).glob("*")):
            if not tagdir.is_dir() or "noise" in tagdir.name:
                continue
            for m in models:
                seeds = sorted(tagdir.glob(f"{m}/seed*/per_user.npz"))
                if not seeds:
                    continue
                mats, users = [], None
                for p in seeds:
                    with np.load(p) as z:
                        u, v = z["users"], z[METRIC]
                    if users is None:
                        users = u
                    assert np.array_equal(users, u)
                    mats.append(v)
                runs[d][tagdir.name][m] = (users, np.stack(mats))
    return runs


def align(a, b):
    (ua, A), (ub, B) = a, b
    _, ia, ib = np.intersect1d(ua, ub, return_indices=True)
    return ua[ia], A[:, ia], B[:, ib]


def ci(x, d=4):
    return f"{x['diff']:+.{d}f} [{x['lo']:+.{d}f}, {x['hi']:+.{d}f}]" + ("$^{\\ast}$" if x["decisive"] else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "towerbench" / "paper_assets"))
    ap.add_argument("--ref", default="ease")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    runs = load([a.ref] + COMPARISONS)
    res = {"pooling": {}, "seeds": {}}

    # ---- 1. window aggregation schemes -----------------------------------------------------
    rows = []
    for m in COMPARISONS:
        for d in DATASETS:
            diffs, users = [], []
            for w in range(5):
                tag = f"temporal_w{w}"
                if tag not in runs[d] or m not in runs[d][tag] or a.ref not in runs[d][tag]:
                    continue
                u, A, B = align(runs[d][tag][m], runs[d][tag][a.ref])
                diffs.append(A.mean(0) - B.mean(0))
                users.append(u)
            if len(diffs) < 2:
                continue
            v = pooled_window_variants(diffs, users, a.n_boot)
            res["pooling"].setdefault(m, {})[d] = v
            rows.append((m, d, v))
    lines = ["\\begin{tabular}{llrrllll}", "\\toprule",
             "Comparison & Dataset & Pairs & Users & Pairs (paper) & Cluster by user & Block by window & Random effects \\\\",
             "\\midrule"]
    for m, d, v in rows:
        lines.append(f"{M_LABEL[m]} $-$ {M_LABEL[a.ref]} & {DS_LABEL[d]} & {v['n_pairs']:,} & {v['cluster_users']['n_users']:,} & "
                     f"{ci(v['pairs'])} & {ci(v['cluster_users'])} & {ci(v['block_windows'])} & {ci(v['random_effects'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / "pooling_variants.tex").write_text("\n".join(lines) + "\n")

    # ---- 2. seed variance ------------------------------------------------------------------
    lines = ["\\begin{tabular}{llrrllr}", "\\toprule",
             "Comparison & Dataset & Protocol & Seeds & Users only (paper) & Users and seeds & Width ratio \\\\", "\\midrule"]
    for m in COMPARISONS:
        for d in DATASETS:
            for proto in ("loo", "pooled"):
                if proto == "loo":
                    if "loo" not in runs[d] or m not in runs[d]["loo"] or a.ref not in runs[d]["loo"]:
                        continue
                    u, A, B = align(runs[d]["loo"][m], runs[d]["loo"][a.ref])
                else:
                    parts = [align(runs[d][f"temporal_w{w}"][m], runs[d][f"temporal_w{w}"][a.ref])
                             for w in range(5) if f"temporal_w{w}" in runs[d] and m in runs[d][f"temporal_w{w}"]
                             and a.ref in runs[d][f"temporal_w{w}"]]
                    if len(parts) < 2 or len({p[1].shape[0] for p in parts}) != 1 or len({p[2].shape[0] for p in parts}) != 1:
                        continue
                    A = np.concatenate([p[1] for p in parts], axis=1)
                    B = np.concatenate([p[2] for p in parts], axis=1)
                if A.shape[0] < 2:
                    continue
                plain = paired_bootstrap(A.mean(0), B.mean(0), a.n_boot)
                both = paired_bootstrap_seeds(A, B, a.n_boot)
                ratio = (both["hi"] - both["lo"]) / (plain["hi"] - plain["lo"])
                entry = {"users_only": plain, "users_and_seeds": both, "width_ratio": ratio,
                         "n_seeds": int(A.shape[0]), "n_ref_seeds": int(B.shape[0]), "n": int(A.shape[1])}
                res["seeds"].setdefault(m, {}).setdefault(d, {})[proto] = entry
                lines.append(f"{M_LABEL[m]} $-$ {M_LABEL[a.ref]} & {DS_LABEL[d]} & {'leave-last-out' if proto == 'loo' else 'pooled windows'} & "
                             f"{A.shape[0]} & {ci(plain)} & {ci(both)} & {ratio:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / "seed_variance.tex").write_text("\n".join(lines) + "\n")
    (out / "revision_stats.json").write_text(json.dumps(res, indent=2, default=float))

    # console digest
    for m in res["pooling"]:
        for d, v in res["pooling"][m].items():
            flags = "".join("Y" if v[k]["decisive"] else "n" for k in ("pairs", "cluster_users", "block_windows", "random_effects"))
            print(f"pooling {m:<9} {d:<20} pairs/cluster/block/RE decisive={flags}  "
                  f"pairs=[{v['pairs']['lo']:+.4f},{v['pairs']['hi']:+.4f}] cluster=[{v['cluster_users']['lo']:+.4f},{v['cluster_users']['hi']:+.4f}] "
                  f"block=[{v['block_windows']['lo']:+.4f},{v['block_windows']['hi']:+.4f}] RE=[{v['random_effects']['lo']:+.4f},{v['random_effects']['hi']:+.4f}] tau2={v['random_effects']['tau2']:.2e}")
    for m in res["seeds"]:
        for d in res["seeds"][m]:
            for proto, e in res["seeds"][m][d].items():
                print(f"seeds   {m:<9} {d:<20} {proto:<7} width ratio {e['width_ratio']:.2f}  "
                      f"users=[{e['users_only']['lo']:+.4f},{e['users_only']['hi']:+.4f}] both=[{e['users_and_seeds']['lo']:+.4f},{e['users_and_seeds']['hi']:+.4f}]")


if __name__ == "__main__":
    main()
