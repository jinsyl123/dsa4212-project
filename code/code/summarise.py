"""Tables and figures for one experiment. Usage:
    python code/summarise.py --config configs/smoke.yaml
Writes results/<experiment>/figures/{convergence_evals,convergence_tokens,test_sets}.{png,pdf}
and results/<experiment>/summary_tables.md. Run evaluate_final.py first for the test-set figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from obpevo import baselines, config as C, core, instances

# Reference categorical palette (validated: CVD-safe adjacent order, light surface). Colour follows the
# method label in config order, never its rank, so a method keeps its colour across every figure.
SLOTS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e9e8e4", "#ffffff"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": "#c9c8c3",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
    }
)


def load_runs(out: Path) -> pd.DataFrame:
    """One row per evaluation, across all runs, with best-so-far training excess and tokens used so far."""
    frames = []
    for summ in sorted(out.glob("*/seed*/summary.json")):
        s, d = json.loads(summ.read_text()), summ.parent
        ev = pd.read_csv(d / "evals.csv").sort_values("eval")
        calls = pd.read_csv(d / "llm_calls.csv").sort_values("t_start")
        ev = pd.merge_asof(
            ev.sort_values("t_start"),
            calls[["t_start", "cum_tokens"]],
            on="t_start",
            direction="backward",
        ).sort_values(
            "eval"
        )  # tokens spent when this eval started
        ev["cum_tokens"] = ev.cum_tokens.fillna(0)
        ev["best_so_far"] = ev.train_excess.cummin().ffill() * 100  # % over lower bound
        frames.append(ev.assign(label=s["label"], seed=s["seed"]))
    return pd.concat(frames, ignore_index=True)


def colours(cfg) -> dict:
    return {m["label"]: SLOTS[i % len(SLOTS)] for i, m in enumerate(cfg["methods"])}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")  # vector copy for LaTeX
    plt.close(fig)


def plot_convergence(
    runs: pd.DataFrame, cfg, bf_train: float, xcol: str, xlabel: str, path: Path
):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    col, ends = colours(cfg), []
    for label in [
        m["label"] for m in cfg["methods"] if m["label"] in set(runs.label)
    ]:  # config order
        g = runs[runs.label == label]
        grid = (
            np.linspace(0, g[xcol].max(), 200)
            if xcol == "cum_tokens"
            else np.arange(1, g["eval"].max() + 1)
        )
        xkey = xcol if xcol == "cum_tokens" else "eval"
        curves = []
        for _, r in g.groupby("seed"):  # step function per run on a common grid
            r = r.dropna(subset=["best_so_far"])
            idx = np.searchsorted(r[xkey].values, grid, side="right") - 1
            curves.append(
                np.where(idx >= 0, r.best_so_far.values[np.clip(idx, 0, None)], np.nan)
            )
        c = np.vstack(curves)
        med, lo, hi = np.nanmedian(c, 0), np.nanmin(c, 0), np.nanmax(c, 0)
        ax.fill_between(grid, lo, hi, color=col[label], alpha=0.10, linewidth=0)
        ax.plot(grid, med, color=col[label], lw=2, label=label)
        ends.append((label, grid[-1], med[-1]))
    ax.axhline(bf_train * 100, color=MUTED, lw=1)
    ax.annotate(
        "Best Fit",
        (ax.get_xlim()[0], bf_train * 100),
        xytext=(4, 4),
        textcoords="offset points",
        color=INK2,
        fontsize=8,
    )
    top = min(
        bf_train * 100 * 2, np.nanmax(runs.best_so_far) * 1.1
    )  # clip early outliers at 2x Best Fit
    ax.set_ylim(
        bottom=max(0, np.nanmin(runs.best_so_far) * 0.9),
        top=max(top, bf_train * 100 * 1.15),
    )
    # Direct end labels (relief for the light hues), nudged apart so converging lines stay readable
    gap, placed = 0.06 * np.diff(ax.get_ylim())[0], []
    for label, x, y in sorted(ends, key=lambda e: e[2]):
        y_lab = max(y, placed[-1] + gap) if placed else y
        placed.append(y_lab)
        ax.text(
            x + 0.015 * np.diff(ax.get_xlim())[0],
            y_lab,
            label,
            va="center",
            color=INK,
            fontsize=8,
            clip_on=False,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Best training excess over\nlower bound so far (%)")
    ax.set_title("Search progress on training instances", loc="left")
    ax.legend(loc="upper right", ncols=min(len(ends), 5), fontsize=8, handlelength=1.5)
    fig.text(
        0.01,
        -0.04,
        "Line: median across seeds. Band: range across seeds. Y-axis clipped at 2x Best Fit.",
        color=MUTED,
        fontsize=7.5,
    )
    _save(fig, path)


def plot_test_sets(final: pd.DataFrame, cfg, path: Path):
    """Small multiples: one panel per held-out set, change in excess vs Best Fit (pp) per method and seed."""
    ev = final[final.kind == "evolved"]
    per_run = (
        ev.groupby(["label", "set", "seed"])
        .agg(excess=("excess", "mean"))
        .reset_index()
    )
    bf = final[final.label == "best_fit"].groupby("set").excess.mean()
    per_run["delta"] = (per_run.excess - per_run.set.map(bf)) * 100
    sets = [s for s in cfg["final_eval"]["sets"] if s in per_run.set.unique()]
    labels = [
        m["label"] for m in cfg["methods"] if m["label"] in per_run.label.unique()
    ]
    col, ncol = colours(cfg), 2
    nrow = int(np.ceil(len(sets) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(6.6, 0.45 * len(labels) * nrow + 0.8 * nrow), squeeze=False
    )
    for ax, s in zip(axes.flat, sets):
        d = per_run[per_run.set == s]
        for yi, label in enumerate(labels):
            r = d[d.label == label]
            ax.scatter(
                r.delta,
                np.full(len(r), yi),
                s=14,
                color=col[label],
                alpha=0.55,
                linewidths=0,
            )
            if len(r):
                ax.scatter(
                    r.delta.mean(),
                    yi,
                    s=46,
                    color=col[label],
                    edgecolors=SURFACE,
                    linewidths=1.5,
                    zorder=3,
                )
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{s}  (Best Fit {bf[s] * 100:.2f}%)", loc="left", fontsize=8.5)
    for ax in axes.flat[len(sets) :]:
        ax.set_visible(False)
    fig.supxlabel(
        "Change in excess over lower bound vs Best Fit (percentage points; left of 0 = better)",
        color=INK2,
        fontsize=8.5,
    )
    fig.suptitle(
        "Held-out performance of each run's best program",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=10,
    )
    fig.text(
        0.01,
        -0.02,
        "Small dots: one per seed. Large dot: mean across seeds.",
        color=MUTED,
        fontsize=7.5,
    )
    fig.tight_layout()
    _save(fig, path)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join(lines)


def tables(out: Path, cfg, final: pd.DataFrame | None) -> str:
    idx = pd.DataFrame(
        [json.loads(p.read_text()) for p in sorted(out.glob("*/seed*/summary.json"))]
    )
    idx["tokens"] = idx.prompt_tokens + idx.completion_tokens
    order = [m["label"] for m in cfg["methods"]]
    idx = idx.assign(
        label=pd.Categorical(idx.label, [l for l in order if l in set(idx.label)])
    )  # config order
    g = idx.groupby("label", sort=True, observed=True)
    runs = pd.DataFrame(
        {
            "runs": g.size(),
            "evals/run": g.n_evals.mean().round(1),
            "ok %": (100 * g.n_ok.sum() / g.n_evals.sum()).round(1),
            "error %": (100 * g.n_error.sum() / g.n_evals.sum()).round(1),
            "timeout %": (100 * g.n_timeout.sum() / g.n_evals.sum()).round(1),
            "LLM calls/run": g.n_llm_calls.mean().round(1),
            "tokens/run": g.tokens.mean().round(0).astype(int),
            "cost USD (total)": g.cost_usd.sum().round(3),
            "minutes/run": g.wall_minutes.mean().round(2),
            "best train excess % (mean ± sd)": [
                f"{100 * v.mean():.3f} ± {100 * (v.std() if len(v) > 1 else 0):.3f}"
                for _, v in g.best_train_excess
            ],
        }
    ).reset_index()
    md = [
        f"# {cfg['experiment']}: summary tables\n",
        f"Budget per run: {cfg['budget']}\n",
        "## Search runs\n",
        md_table(runs),
    ]
    if final is not None:
        fs = pd.read_csv(out / "final_summary.csv")
        fs["cell"] = (fs.mean_excess * 100).map("{:.2f}".format) + np.where(
            fs.runs > 1, " ± " + (fs.sd_excess * 100).map("{:.2f}".format), ""
        )
        order = [m["label"] for m in cfg["methods"]] + list(baselines.BASELINES)
        piv = fs.pivot(index="label", columns="set", values="cell").reindex(
            [l for l in order if l in set(fs.label)]
        )
        piv = piv[
            [s for s in cfg["final_eval"]["sets"] if s in piv.columns]
        ].reset_index()
        md += [
            "\n## Held-out sets: excess over lower bound (%), mean ± sd across seeds\n",
            md_table(piv),
        ]
    text = "\n".join(md) + "\n"
    (out / "summary_tables.md").write_text(text)
    return text


def main(config, overrides=None) -> Path:
    """Write tables and figures; returns the figures folder. Callable from a notebook."""
    cfg = C.load(config, overrides)
    out = C.results_dir(cfg)
    train = instances.load_set(
        C.instance_dir(cfg) / f"{cfg['evaluation']['train_set']}.npz"
    )
    bf_train = -core.score(
        core.evaluate(baselines.best_fit, train, cfg["evaluation"]["mode"])
    )

    runs = load_runs(out)
    plot_convergence(
        runs,
        cfg,
        bf_train,
        "eval",
        "Evaluations used",
        out / "figures" / "convergence_evals",
    )
    plot_convergence(
        runs,
        cfg,
        bf_train,
        "cum_tokens",
        "LLM tokens used (prompt + completion)",
        out / "figures" / "convergence_tokens",
    )
    final = (
        pd.read_csv(out / "final_per_instance.csv")
        if (out / "final_per_instance.csv").exists()
        else None
    )
    if final is not None:
        plot_test_sets(final, cfg, out / "figures" / "test_sets")
    print(tables(out, cfg, final))
    print(f"Figures in {out / 'figures'}")
    return out / "figures"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
