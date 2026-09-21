"""Build the paper's tables (LaTeX) and figures (PDF+PNG) from results/.  Tolerates partial sweeps.

Usage:  python scripts/make_paper_assets.py [--out ~/towerbench/paper_assets] [--ref ease]
Outputs: tables/datasets.tex, tables/main_loo.tex, tables/ablation_loo.tex, tables/runtime.tex,
         figures/fig_windows.{pdf,png}, figures/fig_ablation.{pdf,png}, figures/fig_runtime.{pdf,png},
         summary.json (everything the manuscript quotes as numbers).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402

from towerbench.eval.stats import paired_bootstrap  # noqa: E402
from towerbench.paths import PREPARED, RESULTS  # noqa: E402

# validated categorical palette (dataviz skill reference instance, light surface)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                            "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GRAY, INK, INK2 = "#b8b7b2", "#0b0b0b", "#52514e"
DATASETS = ["ml-1m", "amazon-instruments", "amazon-videogames", "amazon-software", "gowalla", "airbnb", "steam"]
DS_LABEL = {"ml-1m": "MovieLens-1M", "amazon-instruments": "Amazon Instruments", "amazon-videogames": "Amazon Video Games",
            "amazon-software": "Amazon Software", "gowalla": "Gowalla", "airbnb": "Inside Airbnb", "steam": "Steam"}
MODELS = ["pop", "ease", "multvae", "tt_id", "tt_id_nologq", "tt_id_mns", "tt_id_mixgcf", "tt_feat", "tt_feat_dcn", "tt_feat_all", "lightgcn"]
M_LABEL = {"pop": "Popularity", "ease": "EASE", "multvae": "Mult-VAE", "sasrec": "SASRec", "lightgcn": "LightGCN", "tt_id": "TT-ID",
           "tt_id_nologq": "TT-ID no logQ", "tt_id_mns": "TT-ID + MNS", "tt_id_mixgcf": "TT-ID + MixGCF", "tt_feat": "TT-Feat",
           "tt_feat_dcn": "TT-Feat + DCN", "tt_feat_all": "TT-Feat + all"}
EASE_LAMBDAS = [2, 5, 10, 20, 50, 200, 500, 1000, 2000]
TAUS, DIMS = [0.02, 0.05, 0.1], [64, 128, 256]
SWEEP_DS = ["ml-1m", "amazon-software", "gowalla", "airbnb"]
NOISE_DS, NOISE_LEVELS, NOISE_MODELS = ["ml-1m", "amazon-software"], ["0.1", "0.2"], ["ease", "multvae", "lightgcn", "tt_id", "tt_feat"]


def ease_cfg(lam):
    return "ease" if lam == 500 else f"ease_lam{lam:g}"


def tt_cfg(fam, tau, dim):
    return fam if (tau == 0.05 and dim == 128) else f"{fam}_t{tau:g}_d{dim}"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True})


def load_all():
    """runs[dataset][tag][model] = list of (metrics dict, per_user npz) over seeds."""
    runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in RESULTS.glob("*/*/*/seed*/metrics.json"):
        m = json.loads(p.read_text())
        with np.load(p.parent / "per_user.npz") as zf:  # eager read: lazy NpzFile keeps a descriptor open
            z = {k: zf[k] for k in zf.files}
        runs[m["dataset"]][m["protocol"]][m["model_cfg"]].append((m, z))
    return runs


def seed_mean_per_user(entries, metric):
    users = entries[0][1]["users"]
    return users, np.mean([z[metric] for _, z in entries], axis=0)


def paired(entries_a, entries_b, metric="ndcg@10", n_boot=2000):
    ua, va = seed_mean_per_user(entries_a, metric)
    ub, vb = seed_mean_per_user(entries_b, metric)
    common, ia, ib = np.intersect1d(ua, ub, return_indices=True)
    return paired_bootstrap(va[ia], vb[ib], n_boot), len(common)


def fmt(x, d=4):
    return f"{x:.{d}f}"


def sci(x):
    """Density in scientific notation with a LaTeX exponent, e.g. 3.5$\\times$10$^{-5}$."""
    e = int(np.floor(np.log10(x)))
    return f"{x / 10 ** e:.1f}$\\times$10$^{{{e}}}$"


def table_datasets(out):
    rows = []
    for d in DATASETS:
        p = PREPARED / d / "stats.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        side = ", ".join(k for k, v in (("text", s["has_text"]), ("geo", s["has_geo"]), ("price", s["has_price"])) if v) or "--"
        rows.append(f"{DS_LABEL[d]} & {s['n_users']:,} & {s['n_items']:,} & {s['n_interactions']:,} & "
                    f"{sci(s['density'])} & {s['avg_len']:.1f} & {s['k_user']}/{s['k_item']} & {side} \\\\")
    tex = ("\\begin{tabular}{lrrrrrcl}\n\\toprule\nDataset & Users & Items & Events & Density & Len. & $k$-core & Side info \\\\\n"
           "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (out / "tables" / "datasets.tex").write_text(tex)


def table_main(runs, out, ref, tag="loo", metric="ndcg@10", summary=None):
    cols = ["pop", "ease", "multvae", "sasrec", "lightgcn", "tt_id", "tt_feat", "tt_feat_all"]
    lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             "Dataset & " + " & ".join(M_LABEL[c] for c in cols) + " \\\\", "\\midrule"]
    for d in DATASETS:
        if tag not in runs[d] or ref not in runs[d][tag]:
            continue
        cells = []
        for c in cols:
            e = runs[d][tag].get(c)
            if not e:
                cells.append("--"); continue
            mean = np.mean([m["test"][metric]["mean"] for m, _ in e])
            cell = fmt(mean)
            if c != ref:
                pb, n = paired(e, runs[d][tag][ref], metric)
                mark = "$^{\\ast}$" if pb["decisive"] else ""
                cell = f"{fmt(mean)}{mark}"
                if summary is not None:
                    summary.setdefault("paired", {}).setdefault(d, {}).setdefault(tag, {})[c] = {**pb, "n_users": int(n)}
            if summary is not None:
                summary.setdefault("mean", {}).setdefault(d, {}).setdefault(tag, {})[c] = float(mean)
            cells.append(cell)
        best = max(range(len(cols)), key=lambda i: float(cells[i].split("$")[0]) if cells[i] != "--" else -1)
        cells[best] = "\\textbf{" + cells[best] + "}"
        lines.append(f"{DS_LABEL[d]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / f"main_{tag}.tex").write_text("\n".join(lines) + "\n")


def table_ablation(runs, out, tag="loo", metric="ndcg@10"):
    cols = ["tt_id", "tt_id_nologq", "tt_id_mns", "tt_id_mixgcf", "tt_feat", "tt_feat_dcn", "tt_feat_all"]
    lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             "Dataset & " + " & ".join(M_LABEL[c].replace("TT-", "") for c in cols) + " \\\\", "\\midrule"]
    for d in DATASETS:
        if tag not in runs[d] or "tt_id" not in runs[d][tag]:
            continue
        cells = []
        for c in cols:
            e = runs[d][tag].get(c)
            if not e:
                cells.append("--"); continue
            vals = [m["test"][metric]["mean"] for m, _ in e]
            cells.append(f"{np.mean(vals):.4f}$\\pm${np.std(vals):.4f}")
        lines.append(f"{DS_LABEL[d]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / f"ablation_{tag}.tex").write_text("\n".join(lines) + "\n")


def table_runtime(runs, out):
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "Dataset & Events & EASE & Mult-VAE & Two-tower (TT-Feat) \\\\", "\\midrule"]
    for d in DATASETS:
        if "loo" not in runs[d]:
            continue
        p = PREPARED / d / "stats.json"
        n = json.loads(p.read_text())["n_interactions"] if p.exists() else 0
        cells = []
        for c in ("ease", "multvae", "tt_feat"):
            e = runs[d]["loo"].get(c)
            cells.append(f"{np.mean([m['seconds'] for m, _ in e]):.0f}" if e else "--")
        lines.append(f"{DS_LABEL[d]} & {n:,} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / "runtime.tex").write_text("\n".join(lines) + "\n")


def fig_windows(runs, out, ref, model="tt_feat", metric="ndcg@10", summary=None):
    """Forest plot: paired difference (model - ref) per rolling-origin window and pooled over windows.
    Emphasis form: decisive windows in the accent hue, non-decisive in gray; pooled estimate as a diamond."""
    ds = [d for d in DATASETS if any(t.startswith("temporal") for t in runs[d]) and "loo" in runs[d]]
    if not ds:
        return
    ncol = 2 if len(ds) > 2 else len(ds)
    nrow = int(np.ceil(len(ds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol + 0.9, 2.15 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(ds):]:
        ax.axis("off")
    for ax, d in zip(axes, ds):
        ys, labels = [], []
        diffs_all, users_all = [], []
        ns = {}
        for w in range(5):
            tag = f"temporal_w{w}"
            if tag not in runs[d] or model not in runs[d][tag] or ref not in runs[d][tag]:
                continue
            pb, n = paired(runs[d][tag][model], runs[d][tag][ref], metric)
            ua, va = seed_mean_per_user(runs[d][tag][model], metric)
            ub, vb = seed_mean_per_user(runs[d][tag][ref], metric)
            common, ia, ib = np.intersect1d(ua, ub, return_indices=True)
            diffs_all.append(va[ia] - vb[ib]); users_all.append(n)
            y = w
            col = BLUE if pb["decisive"] else GRAY
            ax.plot([pb["lo"], pb["hi"]], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(pb["diff"], y, "o", color=col, ms=5)
            ys.append(y); labels.append(f"window {w}"); ns[y] = n
            if summary is not None:
                summary.setdefault("windows", {}).setdefault(d, {})[tag] = {**pb, "n_users": int(n)}
        if diffs_all:
            pooled = np.concatenate(diffs_all)
            from towerbench.eval.stats import bootstrap_ci
            mean, lo, hi = bootstrap_ci(pooled, 2000, 42)
            y = -1
            col = BLUE if (lo > 0 or hi < 0) else GRAY
            ax.plot([lo, hi], [y, y], color=col, lw=2.5, solid_capstyle="round")
            ax.plot(mean, y, "D", color=col, ms=6)
            ys.append(y); labels.append("pooled windows"); ns[y] = len(pooled)
            if summary is not None:
                summary.setdefault("windows_pooled", {})[d] = {"diff": mean, "lo": lo, "hi": hi, "n": int(len(pooled))}
        if "loo" in runs[d] and model in runs[d]["loo"] and ref in runs[d]["loo"]:
            pb, n = paired(runs[d]["loo"][model], runs[d]["loo"][ref], metric)
            y = -2
            col = ORANGE if pb["decisive"] else GRAY
            ax.plot([pb["lo"], pb["hi"]], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(pb["diff"], y, "s", color=col, ms=5)
            ys.append(y); labels.append("leave-last-out"); ns[y] = n
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(ys); ax.set_yticklabels(labels)
        ax.set_title(DS_LABEL[d], fontsize=10.5, color=INK)
        ax.set_xlabel(f"{M_LABEL[model]} $-$ {M_LABEL[ref]}, NDCG@10", fontsize=9.5)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.tick_params(axis="y", labelsize=9.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        for y, n in ns.items():  # per-panel user counts (the y labels are shared across panels)
            ax.text(0.99, y + 0.12, f"n={n:,}", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=7.5, color=INK2, bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 0.5})
    if len(axes) > len(ds):  # legend in the spare panel, proxy handles only (axes are y-shared)
        from matplotlib.lines import Line2D
        lg = axes[len(ds)]
        handles = [Line2D([], [], marker="o", color=BLUE, lw=2, label="rolling-origin window, CI excludes 0"),
                   Line2D([], [], marker="o", color=GRAY, lw=2, label="window whose CI includes 0"),
                   Line2D([], [], marker="D", color=BLUE, lw=2.5, ms=6, label="pooled per-user differences of all windows"),
                   Line2D([], [], marker="s", color=ORANGE, lw=2, label="leave-last-out protocol, same models")]
        lg.legend(handles=handles, loc="center left", frameon=False, fontsize=8.5,
                  title="marker = paired mean difference\nbar = 95% bootstrap CI over users\nn = evaluated users",
                  title_fontsize=8.5, alignment="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"fig_windows.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_ablation(runs, out, tag="loo", metric="ndcg@10"):
    """Small multiples of horizontal bars, one hue: test NDCG@10 of every two-tower switch, plus EASE as a reference line."""
    ds = [d for d in DATASETS if tag in runs[d] and "tt_id" in runs[d][tag]]
    if not ds:
        return
    cols = ["tt_id_nologq", "tt_id", "tt_id_mns", "tt_id_mixgcf", "tt_feat", "tt_feat_dcn", "tt_feat_all"]
    ncol = 2 if len(ds) > 2 else len(ds)
    nrow = int(np.ceil(len(ds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol + 0.9, 2.0 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(ds):]:
        ax.axis("off")
    if len(axes) > len(ds):  # legend in the spare panel, proxy handles only (axes are y-shared)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        lg = axes[len(ds)]
        handles = [Patch(color=BLUE, label="test NDCG@10, mean over seeds"),
                   Line2D([], [], color=INK2, lw=0.8, marker="|", label="standard deviation over seeds"),
                   Line2D([], [], color=ORANGE, lw=1.4, ls="--", label="EASE on the same split")]
        lg.legend(handles=handles, loc="center left", frameon=False, fontsize=8.5, alignment="left",
                  title="ID = item-id two-tower\nFeat = + text, geo/price, category\nall = + MNS + MixGCF + DCN-v2",
                  title_fontsize=8.5)
    for ax, d in zip(axes, ds):
        means, sds = [], []
        for c in cols:
            e = runs[d][tag].get(c)
            vals = [m["test"][metric]["mean"] for m, _ in e] if e else [np.nan]
            means.append(np.mean(vals)); sds.append(np.std(vals))
        y = np.arange(len(cols))
        ax.barh(y, means, xerr=sds, color=BLUE, height=0.62, error_kw={"ecolor": INK2, "lw": 0.8, "capsize": 2})
        ez = np.nan
        if "ease" in runs[d][tag]:
            ez = np.mean([m["test"][metric]["mean"] for m, _ in runs[d][tag]["ease"]])
            ax.axvline(ez, color=ORANGE, lw=1.4, ls="--")
            ax.set_ylim(-1.0, len(cols) - 0.4)
            ax.text(ez, -0.95, " EASE", color=ORANGE, fontsize=8, va="bottom", ha="left")
        fmt = (lambda v: f"{v:.3f}") if np.nanmax(means) >= 0.01 else (lambda v: f"{v:.4f}")
        for yi, mv, sd in zip(y, means, sds):
            if np.isfinite(mv):
                ax.text(mv + (sd if np.isfinite(sd) else 0), yi, " " + fmt(mv), va="center", ha="left",
                        fontsize=8, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels([M_LABEL[c].replace("TT-", "") for c in cols])
        ax.set_title(DS_LABEL[d], fontsize=10.5, color=INK)
        ax.set_xlabel("NDCG@10 (test)", fontsize=9.5)
        ax.set_xlim(0, max(np.nanmax([np.nanmax(means), ez]) * 1.35, 1e-3))
        ax.tick_params(axis="x", labelsize=8.5)
        ax.tick_params(axis="y", labelsize=9.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"fig_ablation.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_runtime(runs, out):
    """Wall-clock seconds per experiment (log scale) per dataset, datasets ordered by training events;
    a dot chart because runtime depends on users x items as much as on events."""
    fams = [("ease", "EASE", ORANGE), ("multvae", "Mult-VAE", AQUA), ("sasrec", "SASRec", VIOLET),
            ("tt_feat", "Two-tower (TT-Feat)", BLUE)]
    ds = [d for d in DATASETS if "loo" in runs[d] and "tt_feat" in runs[d]["loo"]]
    ds.sort(key=lambda d: runs[d]["loo"]["tt_feat"][0][0]["n_train"])
    short = {"ml-1m": "MovieLens", "amazon-instruments": "Instruments", "amazon-videogames": "Video Games",
             "amazon-software": "Software", "gowalla": "Gowalla", "airbnb": "Airbnb", "steam": "Steam"}
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(ds))
    for j, (key, label, col) in enumerate(fams):
        ys = [np.mean([m["seconds"] for m, _ in runs[d]["loo"][key]]) if key in runs[d]["loo"] else np.nan for d in ds]
        ax.plot(x + (j - 1.5) * 0.15, ys, "o", color=col, ms=7, label=label)
        for xi, d, yv in zip(x, ds, ys):  # mark runs that fell back to the CPU (e.g. EASE on a 105k-item catalogue)
            if key in runs[d]["loo"] and str(runs[d]["loo"][key][0][0].get("fit", {}).get("device", "")).startswith("cpu"):
                ax.annotate("CPU", (xi + (j - 1.5) * 0.15, yv), textcoords="offset points", xytext=(6, 2), fontsize=7.5, color=col)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{short[d]}\n{runs[d]['loo']['tt_feat'][0][0]['n_train'] / 1e6:.2f}M ev." for d in ds],
                       fontsize=8.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("Seconds per experiment (1 GPU)", fontsize=10)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"fig_runtime.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _mean(runs, d, tag, cfg, metric="ndcg@10"):
    e = runs[d].get(tag, {}).get(cfg)
    return float(np.mean([m["test"][metric]["mean"] for m, _ in e])) if e else np.nan


def table_sensitivity_ease(runs, out, tag="loo", metric="ndcg@10", summary=None):
    """EASE NDCG@10 over the regularisation grid, the best lambda, and the paired verdicts of TT-ID and
    TT-Feat (base configuration) against EASE at its best lambda on the same users."""
    lines = ["\\begin{tabular}{l" + "r" * len(EASE_LAMBDAS) + "rrr}", "\\toprule",
             "Dataset & " + " & ".join(f"$\\lambda={lam}$" for lam in EASE_LAMBDAS)
             + " & best $\\lambda$ & TT-ID $-$ best & TT-Feat $-$ best \\\\", "\\midrule"]
    for d in DATASETS:
        vals = [_mean(runs, d, tag, ease_cfg(lam), metric) for lam in EASE_LAMBDAS]
        if all(np.isnan(v) for v in vals) or "tt_feat" not in runs[d].get(tag, {}):
            continue
        best_i = int(np.nanargmax(vals))
        cells = [("--" if np.isnan(v) else (f"\\textbf{{{v:.4f}}}" if i == best_i else f"{v:.4f}")) for i, v in enumerate(vals)]
        verdicts = []
        for m in ("tt_id", "tt_feat"):
            pb, n = paired(runs[d][tag][m], runs[d][tag][ease_cfg(EASE_LAMBDAS[best_i])], metric)
            verdicts.append(f"{pb['diff']:+.4f}" + ("$^{\\ast}$" if pb["decisive"] else ""))
            if summary is not None:
                summary.setdefault("sens_ease", {}).setdefault(d, {})[m] = {
                    **pb, "best_lambda": EASE_LAMBDAS[best_i], "n_users": int(n), "grid": dict(zip(map(str, EASE_LAMBDAS), vals))}
        lines.append(f"{DS_LABEL[d]} & " + " & ".join(cells) + f" & {EASE_LAMBDAS[best_i]} & " + " & ".join(verdicts) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / f"sens_ease_{tag}.tex").write_text("\n".join(lines) + "\n")


def table_sensitivity_tt(runs, out, tag="loo", metric="ndcg@10", summary=None):
    """Two-tower temperature x width grid (mean over seeds) for TT-ID and TT-Feat on the sweep datasets,
    with EASE at its best lambda for reference; bold = best cell of the family."""
    head = " & ".join(f"$d={dim}$" for dim in DIMS)
    lines = ["\\begin{tabular}{llr" + "r" * len(DIMS) + "}", "\\toprule", f"Dataset & Model & $\\tau$ & {head} \\\\", "\\midrule"]
    for d in SWEEP_DS:
        if "tt_feat" not in runs[d].get(tag, {}):
            continue
        ease_vals = {lam: _mean(runs, d, tag, ease_cfg(lam), metric) for lam in EASE_LAMBDAS}
        best_lam = max(ease_vals, key=lambda k: -1 if np.isnan(ease_vals[k]) else ease_vals[k])
        first = True
        for fam in ("tt_id", "tt_feat"):
            grid = np.array([[_mean(runs, d, tag, tt_cfg(fam, tau, dim), metric) for dim in DIMS] for tau in TAUS])
            if np.isnan(grid).all():
                continue
            bi = np.unravel_index(np.nanargmax(grid), grid.shape)
            for ti, tau in enumerate(TAUS):
                cells = [("--" if np.isnan(v) else (f"\\textbf{{{v:.4f}}}" if (ti, j) == bi else f"{v:.4f}")) for j, v in enumerate(grid[ti])]
                lab = f"{DS_LABEL[d]} (EASE {ease_vals[best_lam]:.4f} at $\\lambda={best_lam}$)" if first else ""
                first = False
                lines.append(f"{lab} & {M_LABEL[fam] if ti == 0 else ''} & {tau:g} & " + " & ".join(cells) + " \\\\")
            if summary is not None:
                summary.setdefault("sens_tt", {}).setdefault(d, {})[fam] = {
                    "grid": {f"tau{tau:g}_d{dim}": float(grid[i, j]) for i, tau in enumerate(TAUS) for j, dim in enumerate(DIMS)},
                    "min": float(np.nanmin(grid)), "max": float(np.nanmax(grid)), "base": float(grid[1, 1]),
                    "ease_best": float(ease_vals[best_lam]), "ease_best_lambda": best_lam}
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / f"sens_tt_{tag}.tex").write_text("\n".join(lines) + "\n")


def table_noise(runs, out, metric="ndcg@10", summary=None):
    """Test NDCG@10 (mean over seeds) with 0 / 10 / 20 % of the training events replaced by random items."""
    lines = ["\\begin{tabular}{ll" + "r" * len(NOISE_MODELS) + "}", "\\toprule",
             "Dataset & Noise & " + " & ".join(M_LABEL[m] for m in NOISE_MODELS) + " \\\\", "\\midrule"]
    for d in NOISE_DS:
        for lvl in ["0"] + NOISE_LEVELS:
            tag = "loo" if lvl == "0" else f"loo_noise{lvl}"
            if tag not in runs[d]:
                continue
            vals = [_mean(runs, d, tag, m, metric) for m in NOISE_MODELS]
            if all(np.isnan(v) for v in vals):
                continue
            order = [NOISE_MODELS[i] for i in np.argsort([-v if not np.isnan(v) else np.inf for v in vals])]
            lines.append(f"{DS_LABEL[d] if lvl == '0' else ''} & {int(float(lvl) * 100)}\\% & "
                         + " & ".join("--" if np.isnan(v) else f"{v:.4f}" for v in vals) + " \\\\")
            if summary is not None:
                summary.setdefault("noise", {}).setdefault(d, {})[lvl] = {"means": dict(zip(NOISE_MODELS, vals)), "order": order}
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / "noise.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "towerbench" / "paper_assets"))
    ap.add_argument("--ref", default="ease")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(parents=True, exist_ok=True)
    runs = load_all()
    n_all = int(sum(len(e) for d in runs.values() for t in d.values() for e in t.values()))
    base_cfgs = (set(MODELS) | {"sasrec"}) - {"lightgcn"}
    n_v1 = int(sum(len(e) for d in runs.values() for t, mm in d.items() if "noise" not in t for c, e in mm.items() if c in base_cfgs))
    summary = {"n_results": n_all, "n_results_v1": n_v1, "n_results_revision": n_all - n_v1}
    table_datasets(out)
    table_main(runs, out, a.ref, "loo", summary=summary)
    table_ablation(runs, out, "loo")
    table_runtime(runs, out)
    table_sensitivity_ease(runs, out, "loo", summary=summary)
    table_sensitivity_tt(runs, out, "loo", summary=summary)
    table_noise(runs, out, summary=summary)
    fig_windows(runs, out, a.ref, summary=summary)
    fig_ablation(runs, out)
    fig_runtime(runs, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("assets written to", out, "from", summary["n_results"], "results")


if __name__ == "__main__":
    main()
