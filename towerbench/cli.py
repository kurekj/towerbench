"""towerbench CLI: prepare -> embed -> run -> report."""
from __future__ import annotations

import json

import numpy as np
import typer
from rich import print as rprint

from .paths import RESULTS

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def prepare(dataset: str, k_user: int = 5, k_item: int = 5, max_users: int = 0):
    """Download-free preprocessing under the frozen protocol (raw files must exist)."""
    from .data.prepare import prepare as _prep
    from .run import dataset_cfg
    d = dataset_cfg(dataset)
    st = _prep(dataset, d.get("k_user", k_user), d.get("k_item", k_item),
               d.get("max_users", max_users) or None)
    rprint(st)


@app.command()
def embed(dataset: str, model: str = "BAAI/bge-base-en-v1.5", device: str = "cuda"):
    """Frozen text embeddings of item side text (GPU)."""
    from .features.text_embed import embed_items
    rprint(embed_items(dataset, model, device=device))


@app.command()
def run(dataset: str, model_cfg: str, seed: int = 0, protocol: str = "loo", window: int = 0,
        device: str = "cuda", n_boot: int = 2000, overwrite: bool = False):
    from .run import run as _run
    res = _run(dataset, model_cfg, seed, protocol, window, device, n_boot, overwrite=overwrite)
    t = res["test"]
    rprint(f"[bold]{dataset} {res['protocol']} {model_cfg} seed{seed}[/bold] "
           f"n={res['n_test_users']} NDCG@10={t['ndcg@10']['mean']:.4f} "
           f"[{t['ndcg@10']['lo']:.4f},{t['ndcg@10']['hi']:.4f}] R@20={t['recall@20']['mean']:.4f} "
           f"HR@10={t['hit@10']['mean']:.4f} ({res['seconds']:.0f}s)")


@app.command()
def report(dataset: str, protocol: str = "loo", ref: str = "ease", metric: str = "ndcg@10",
           n_boot: int = 2000):
    """Aggregate seeds, paired bootstrap vs ``ref`` on the same users, markdown + LaTeX tables."""
    from .eval.stats import paired_bootstrap
    base = RESULTS / dataset
    tags = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith(protocol))
    rows = []
    for tag in tags:
        per_model = {}
        for md in sorted((base / tag).iterdir()):
            if not md.is_dir():
                continue
            seeds = []
            for sd in sorted(md.glob("seed*")):
                if (sd / "metrics.json").exists():
                    with np.load(sd / "per_user.npz") as zf:
                        z = {k: zf[k] for k in zf.files}
                    seeds.append((json.loads((sd / "metrics.json").read_text()), z))
            if seeds:
                per_model[md.name] = seeds
        if ref not in per_model:
            continue
        ref_users = per_model[ref][0][1]["users"]
        ref_vals = np.mean([z[metric] for _, z in per_model[ref]], axis=0)
        for name, seeds in per_model.items():
            vals = np.mean([z[metric] for _, z in seeds], axis=0)
            users = seeds[0][1]["users"]
            _, ia, ib = np.intersect1d(users, ref_users, return_indices=True)
            pb = paired_bootstrap(vals[ia], ref_vals[ib], n_boot)
            per_seed = [m["test"][metric]["mean"] for m, _ in seeds]
            rows.append({"tag": tag, "model": name, "n_seeds": len(seeds), "n_users": len(users),
                         metric: float(np.mean(per_seed)), "seed_sd": float(np.std(per_seed)),
                         "r@20": float(np.mean([m["test"]["recall@20"]["mean"] for m, _ in seeds])),
                         "hr@10": float(np.mean([m["test"]["hit@10"]["mean"] for m, _ in seeds])),
                         "diff_vs_ref": pb["diff"], "lo": pb["lo"], "hi": pb["hi"], "decisive": pb["decisive"],
                         "cov@10": float(np.mean([m["test_catalog"]["coverage@10"] for m, _ in seeds])),
                         "sec": float(np.mean([m["seconds"] for m, _ in seeds]))})
    md = [f"| tag | model | seeds | users | {metric} | sd | R@20 | HR@10 | diff vs {ref} "
          "| 95% CI | decisive | cov@10 | s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['tag']} | {r['model']} | {r['n_seeds']} | {r['n_users']} | {r[metric]:.4f} "
                  f"| {r['seed_sd']:.4f} | {r['r@20']:.4f} | {r['hr@10']:.4f} | {r['diff_vs_ref']:+.4f} "
                  f"| [{r['lo']:+.4f}, {r['hi']:+.4f}] | {'yes' if r['decisive'] else 'no'} "
                  f"| {r['cov@10']:.3f} | {r['sec']:.0f} |")
    text = "\n".join(md)
    (base / f"report_{protocol}_{metric.replace('@', '')}.md").write_text(text)
    (base / f"report_{protocol}_{metric.replace('@', '')}.json").write_text(json.dumps(rows, indent=2))
    print(text)


@app.command()
def status():
    """What has finished so far (for ETA checks)."""
    for p in sorted(RESULTS.rglob("metrics.json")):
        m = json.loads(p.read_text())
        print(f"{m['dataset']:<20} {m['protocol']:<12} {m['model_cfg']:<16} seed{m['seed']} "
              f"NDCG@10={m['test']['ndcg@10']['mean']:.4f} n={m['n_test_users']} {m['seconds']:.0f}s")


if __name__ == "__main__":
    app()
