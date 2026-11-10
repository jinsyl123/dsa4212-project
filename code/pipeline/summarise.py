"""Tables and figures for one experiment. Usage:
    python pipeline/summarise.py --config configs/smoke.yaml
Writes results/<experiment>/figures/{convergence_evals,convergence_tokens,test_sets}.{png,pdf}
and results/<experiment>/summary_tables.md. Run evaluate_final.py first for the test-set figure.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluate_final
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
        if not (d / "evals.csv").exists() or not (d / "llm_calls.csv").exists():
            print(
                f"  convergence plots skip {d.relative_to(out)}: evals.csv / llm_calls.csv not present"
            )
            continue
        ev = pd.read_csv(d / "evals.csv")
        if ev.empty:  # stopped before any evaluation
            continue  # (e.g. API quota): nothing to plot
        ev = ev.sort_values("eval")
        calls = pd.read_csv(d / "llm_calls.csv").sort_values("t_start")
        if calls.empty:  # no LLM (e.g. tuned Best Fit)
            ev["cum_tokens"] = 0
        else:
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
    if not frames:
        raise ValueError(f"No run in {out} has any evaluations yet")
    return pd.concat(frames, ignore_index=True)


def colours(cfg) -> dict:
    return {m["label"]: SLOTS[i % len(SLOTS)] for i, m in enumerate(cfg["methods"])}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})  # vector copy for LaTeX; no date, so reruns are byte-identical
    plt.close(fig)


def plot_convergence(
    runs: pd.DataFrame, cfg, bf_train: float, xcol: str, xlabel: str, path: Path
):
    labels = [
        m["label"] for m in cfg["methods"] if m["label"] in set(runs.label)
    ]  # config order
    no_llm = [
        l
        for l in labels
        if xcol == "cum_tokens" and runs.loc[runs.label == l, xcol].max() == 0
    ]
    runs = runs[~runs.label.isin(no_llm)]  # nothing to draw on a token axis
    if runs.empty:
        print(f"  skip {path.name}: no run used LLM tokens")
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    col, ends = colours(cfg), []
    for label in [l for l in labels if l not in no_llm]:
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
        with warnings.catch_warnings():  # grid points before a run's first ok
            warnings.simplefilter(
                "ignore", RuntimeWarning
            )  # candidate are all-NaN: expected, left blank
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
    note = f" Not shown (no LLM tokens): {', '.join(no_llm)}." if no_llm else ""
    fig.text(
        0.01,
        -0.04,
        "Line: median across seeds. Band: range across seeds. Y-axis clipped at 2x Best Fit."
        + note,
        color=MUTED,
        fontsize=7.5,
    )
    _save(fig, path)


def plot_test_sets(final: pd.DataFrame, cfg, path: Path):
    """Small multiples: one panel per held-out set, change in excess vs Best Fit (pp) per method and seed.
    Excess is over L2 when the results have it (#12 decision 7), else over L1."""
    metric = "excess_l2" if "excess_l2" in final else "excess"
    ev = final[final.kind != "baseline"]  # evolved and tuned programs
    per_run = (
        ev.groupby(["label", "set", "seed"])
        .agg(excess=(metric, "mean"))
        .reset_index()
    )
    bf = final[final.label == "best_fit"].groupby("set")[metric].mean()
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
        f"Change in excess over {'L2' if metric == 'excess_l2' else 'L1'} lower bound vs Best Fit "
        "(percentage points; left of 0 = better)",
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


def held_out_table(fs: pd.DataFrame, cfg, sets: list[str], mean: str, sd: str) -> pd.DataFrame:
    """Method x set table of "mean ± sd" excess (%), methods in config order; † marks sets with failed instances."""
    fs = fs.copy()
    fs["cell"] = (fs[mean] * 100).map("{:.2f}".format) + np.where(
        fs.runs > 1, " ± " + (fs[sd] * 100).map("{:.2f}".format), ""
    )
    fs["cell"] += np.where(fs.failed_frac > 0, " †", "")  # some instances failed: scored as Next Fit
    order = [m["label"] for m in cfg["methods"]] + list(baselines.BASELINES)
    piv = fs.pivot(index="label", columns="set", values="cell").reindex(
        [l for l in order if l in set(fs.label)]
    )
    return piv[sets].reset_index()


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
        sets = [s for s in cfg["final_eval"]["sets"] if s in set(fs.set)]
        # Report excess over L2 on every set (#12 decision 7). It equals L1 except where many items exceed half a
        # bin; L1 (the bound FunSearch/EoH use) is shown separately for those sets so the papers' metric is visible
        on_l2 = "mean_excess_l2" in fs
        bound = "L2" if on_l2 else "L1"
        cols = ("mean_excess_l2", "sd_excess_l2") if on_l2 else ("mean_excess", "sd_excess")
        piv = held_out_table(fs, cfg, sets, *cols)
        legend = (
            "† some instances failed (crash/timeout) and were scored as Next Fit\n"
            if (fs.failed_frac > 0).any()
            else ""
        )
        md += [
            f"\n## Held-out sets: excess over {bound} lower bound (%), mean ± sd across seeds\n"
            + legend,
            md_table(piv),
        ]
        # Sets where L2 is tighter than L1 (of ours, test_scale65 and test_control): the only sets where L1 differs
        l2_sets = [s for s in sets if "lb_l2" in final and (final[final.set == s].lb_l2 > final[final.set == s].lb).any()]
        if l2_sets and on_l2:
            md += [
                "\n## For comparison with FunSearch/EoH: excess over L1 (%), sets where it differs from L2\n"
                "Many items exceed half a bin here, so L1 overstates every rule's excess. On every other set the "
                "table above is already L1.\n" + legend,
                md_table(held_out_table(fs, cfg, l2_sets, "mean_excess", "sd_excess")),
            ]
    text = "\n".join(md) + "\n"
    (out / "summary_tables.md").write_text(text)
    return text


def main(config, overrides=None, llm=None, budget=None, experiment=None) -> Path:
    """Write tables and figures; returns the figures folder. Callable from a notebook."""
    cfg = C.load(config, overrides, llm=llm, budget=budget, experiment=experiment)
    out = C.results_dir(cfg)
    train = instances.load_set(
        C.ensure_instances(cfg) / f"{cfg['evaluation']['train_set']}.npz"
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
    if final is not None and "lb_l2" not in final:
        # results from before the L2 column: add it from the saved instances, no need to rerun evaluate_final.py
        idir = C.ensure_instances(cfg)
        sets = {s: instances.load_set(idir / f"{s}.npz") for s in final.set.unique()}
        final = evaluate_final.add_l2(final, sets)
        final.to_csv(out / "final_per_instance.csv", index=False)
        evaluate_final.summarise(final).to_csv(out / "final_summary.csv", index=False)
        print("Added lb_l2 / excess_l2 to final_per_instance.csv and final_summary.csv")
    if final is not None:
        plot_test_sets(final, cfg, out / "figures" / "test_sets")
    print(tables(out, cfg, final))
    print(f"Figures in {out / 'figures'}")
    return out / "figures"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--llm", help="model name from configs/llms.yaml (default: the config's)"
    )
    ap.add_argument(
        "--budget", help="budget name from configs/budgets.yaml (default: the config's)"
    )
    ap.add_argument(
        "--experiment",
        help="results folder name (default: from config, llm and budget)",
    )
    a = ap.parse_args()
    main(a.config, llm=a.llm, budget=a.budget, experiment=a.experiment)
