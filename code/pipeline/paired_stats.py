"""Paired comparison of every rule against reference rules (Best Fit, tuned Best Fit), instance by instance.
Usage:
    python pipeline/paired_stats.py --config configs/eoh_baseline.yaml --llm openrouter-gemini-3.5-flash-lite
Reads results/<experiment>/final_per_instance.csv (written by evaluate_final.py), so run that first.
Outputs: results/<experiment>/paired_stats.csv, one row per (rule, seed, set, reference):
    wins / ties / losses   instances where the rule uses fewer / the same / more bins than the reference
    sign_p                 two-sided sign test on wins vs losses (ties dropped): chance of a split at least this
                           uneven if neither rule were better
    mean_diff_pp           mean of (rule excess - reference excess) in percentage points; negative = rule better
    ci_low_pp, ci_high_pp  95% bootstrap interval for mean_diff_pp (instances resampled with replacement)
    verdict                better / worse when the interval excludes 0 AND sign_p < alpha (0.05); tie when identical
                           on every instance; otherwise unclear. Both are required because a bootstrap interval over
                           a handful of instances is too narrow: with 5 instances, 4-1 can give an interval that
                           excludes 0 while p = 0.125. With 10 instances per test set (#12), 10-0 gives p = 0.002
                           and 9-1 gives 0.021; with 5, nothing could reach p < 0.05 (5-0 gives 0.0625).
Each seed is compared on its own: runs of the same method can differ a lot, and a mean over seeds hides that.
A reference with several seeds (tuned Best Fit) is averaged per instance first, so against it wins/ties/losses
compare with that seed-averaged excess, not with one real packing: a rule identical to one tuned seed rarely shows
as a tie. Against a single-seed reference (Best Fit) they count bins.
Excess is over the L2 bound by default (#12 decision 7), falling back to L1 for results without the L2 columns;
--metric excess compares over L1 instead. Against a single-seed reference only the pp columns depend on the bound.
An instance the reference wasn't scored on is left out of that comparison (n drops), with a warning.
--seeds 0-5 keeps only those seeds of every rule with several seeds (the evolved rule and tuned Best Fit; Best Fit has
one) and writes paired_stats_seeds0-5.csv: the pre-registered baseline comparison is seeds 0-5 against a tuned Best Fit
averaged over seeds 0-5, and averaging over more reference seeds changes verdicts (#19 review).
There is no correction for running many comparisons (rules x seeds x sets x references): treat single verdicts as
descriptive and look for patterns across seeds and sets.
"""
from __future__ import annotations

import argparse
from math import comb

import numpy as np
import pandas as pd

from obpevo import config as C

REFERENCES = ["best_fit", "tuned_bf"]
METRICS = ["excess", "excess_l2"]


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test; ties are dropped before calling. 5-0 gives 0.0625, 10-0 gives 0.00195."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval for the mean of `diffs`. Deterministic for a given seed."""
    if len(diffs) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(n_boot, len(diffs)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _reference(df: pd.DataFrame, label: str, metric: str) -> pd.Series | None:
    """Per-instance metric of a reference rule, averaged over its seeds; None if it isn't in the results."""
    ref = df[df.label == label]
    return None if ref.empty else ref.groupby(["set", "instance"])[metric].mean()


def compare(df: pd.DataFrame, references=REFERENCES, n_boot: int = 10_000, metric: str = "excess",
            alpha: float = 0.05) -> pd.DataFrame:
    """One row per (rule label, seed, set, reference); see the module docstring for the columns."""
    if metric not in df:
        raise KeyError(f"column {metric!r} not in the results (excess_l2 needs evaluate_final.py with the L2 bound)")
    rows = []
    for ref_label in references:
        ref = _reference(df, ref_label, metric)
        if ref is None:
            continue
        for (kind, label, seed, set_name), g in df[df.label != ref_label].groupby(["kind", "label", "seed", "set"]):
            g = g.set_index(["set", "instance"]).sort_index()
            d = (g[metric] - ref.reindex(g.index)).dropna().to_numpy() * 100          # percentage points
            if len(d) < len(g):
                print(f"warning: {ref_label} lacks {len(g) - len(d)} of {len(g)} instances that {label} seed {seed} "
                      f"was scored on in {set_name}; compared on n={len(d)} (rerun evaluate_final.py for both)")
            wins, losses = int((d < 0).sum()), int((d > 0).sum())
            lo, hi = bootstrap_ci(d, n_boot)
            p = sign_test(wins, losses)
            verdict = ("tie" if len(d) and not d.any() else "better" if hi < 0 and p < alpha
                       else "worse" if lo > 0 and p < alpha else "unclear")
            rows.append(dict(kind=kind, label=label, seed=seed, set=set_name, reference=ref_label, metric=metric,
                             n=len(d), wins=wins, ties=len(d) - wins - losses, losses=losses, sign_p=p,
                             mean_diff_pp=float(d.mean()) if len(d) else np.nan, ci_low_pp=lo, ci_high_pp=hi,
                             verdict=verdict, failed=int((g["status"] != "ok").sum()) if "status" in g else 0))
    return pd.DataFrame(rows)


def default_metric(df: pd.DataFrame) -> str:
    """Excess over L2, as the report uses on every set (#12 decision 7); L1 for older results without the column."""
    return "excess_l2" if "excess_l2" in df else "excess"


def main(config, references=None, n_boot=10_000, metric=None, alpha=0.05, overrides=None, llm=None, budget=None,
         experiment=None, seeds=None) -> pd.DataFrame:
    cfg = C.load(config, overrides, llm=llm, budget=budget, experiment=experiment)
    out = C.results_dir(cfg)
    src = out / "final_per_instance.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found: run evaluate_final.py with the same arguments first")
    df = pd.read_csv(src)
    if seeds is not None:  # keep these seeds of every multi-seed rule; single-seed references (Best Fit) stay
        multi = df.groupby("label").seed.transform("nunique") > 1
        df = df[df.seed.isin(seeds) | ~multi]
    if metric is None:
        metric = default_metric(df)
        if metric == "excess":
            print(f"[{cfg['experiment']}] no excess_l2 column: comparing over L1 (rerun evaluate_final.py for L2)")
    res = compare(df, references or REFERENCES, n_boot, metric, alpha).round(5)
    # the default comparison goes to paired_stats.csv; an explicit L1 run next to L2 results gets its own file
    name = "paired_stats.csv" if metric != "excess" or "excess_l2" not in df else "paired_stats_excess.csv"
    if seeds is not None:
        name = name.replace(".csv", f"_seeds{min(seeds)}-{max(seeds)}.csv")
    res.to_csv(out / name, index=False)
    show = res[res.kind != "baseline"]
    print(f"[{cfg['experiment']}] {len(res)} comparisons on {metric} -> {out / name}")
    if not show.empty:
        print(show[["label", "seed", "set", "reference", "wins", "ties", "losses", "sign_p", "mean_diff_pp",
                    "ci_low_pp", "ci_high_pp", "verdict"]].to_string(index=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--references", nargs="*", help=f"reference labels (default: {' '.join(REFERENCES)}, if present)")
    ap.add_argument("--n-boot", type=int, default=10_000, help="bootstrap resamples (default 10000)")
    ap.add_argument("--metric", choices=METRICS, help="excess_l2 (default when present) or excess (over L1)")
    ap.add_argument("--alpha", type=float, default=0.05, help="sign-test level a verdict also needs (default 0.05)")
    ap.add_argument("--llm", help="model name from configs/llms.yaml (default: the config's)")
    ap.add_argument("--budget", help="budget name from configs/budgets.yaml (default: the config's)")
    ap.add_argument("--experiment", help="results folder name (default: from config, llm and budget)")
    ap.add_argument("--seeds", help="only seeds lo-hi of every multi-seed rule, e.g. 0-5 "
                                    "(writes paired_stats_seeds0-5.csv)")
    a = ap.parse_args()
    seeds = None if a.seeds is None else list(range(int(a.seeds.split("-")[0]), int(a.seeds.split("-")[1]) + 1))
    main(a.config, a.references, a.n_boot, a.metric, a.alpha, llm=a.llm, budget=a.budget, experiment=a.experiment,
         seeds=seeds)
