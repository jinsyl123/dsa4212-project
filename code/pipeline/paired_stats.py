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
    verdict                better / worse (interval excludes 0), tie (identical on every instance), unclear
Each seed is compared on its own: runs of the same method can differ a lot, and a mean over seeds hides that.
A reference with several seeds (tuned Best Fit) is averaged per instance first.
"""
from __future__ import annotations

import argparse
from math import comb

import numpy as np
import pandas as pd

from obpevo import config as C

REFERENCES = ["best_fit", "tuned_bf"]


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


def _reference(df: pd.DataFrame, label: str) -> pd.Series | None:
    """Per-instance excess of a reference rule, averaged over its seeds; None if it isn't in the results."""
    ref = df[df.label == label]
    return None if ref.empty else ref.groupby(["set", "instance"])["excess"].mean()


def compare(df: pd.DataFrame, references=REFERENCES, n_boot: int = 10_000) -> pd.DataFrame:
    """One row per (rule label, seed, set, reference); see the module docstring for the columns."""
    rows = []
    for ref_label in references:
        ref = _reference(df, ref_label)
        if ref is None:
            continue
        for (kind, label, seed, set_name), g in df[df.label != ref_label].groupby(["kind", "label", "seed", "set"]):
            g = g.set_index(["set", "instance"]).sort_index()
            d = (g["excess"] - ref.reindex(g.index)).dropna().to_numpy() * 100        # percentage points
            wins, losses = int((d < 0).sum()), int((d > 0).sum())
            lo, hi = bootstrap_ci(d, n_boot)
            verdict = ("tie" if len(d) and not d.any() else "better" if hi < 0 else "worse" if lo > 0 else "unclear")
            rows.append(dict(kind=kind, label=label, seed=seed, set=set_name, reference=ref_label, n=len(d),
                             wins=wins, ties=len(d) - wins - losses, losses=losses, sign_p=sign_test(wins, losses),
                             mean_diff_pp=float(d.mean()) if len(d) else np.nan, ci_low_pp=lo, ci_high_pp=hi,
                             verdict=verdict, failed=int((g["status"] != "ok").sum()) if "status" in g else 0))
    return pd.DataFrame(rows)


def main(config, references=None, n_boot=10_000, overrides=None, llm=None, budget=None, experiment=None) -> pd.DataFrame:
    cfg = C.load(config, overrides, llm=llm, budget=budget, experiment=experiment)
    out = C.results_dir(cfg)
    src = out / "final_per_instance.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found: run evaluate_final.py with the same arguments first")
    res = compare(pd.read_csv(src), references or REFERENCES, n_boot).round(5)
    res.to_csv(out / "paired_stats.csv", index=False)
    show = res[res.kind != "baseline"]
    print(f"[{cfg['experiment']}] {len(res)} comparisons -> {out / 'paired_stats.csv'}")
    if not show.empty:
        print(show[["label", "seed", "set", "reference", "wins", "ties", "losses", "sign_p", "mean_diff_pp",
                    "ci_low_pp", "ci_high_pp", "verdict"]].to_string(index=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--references", nargs="*", help=f"reference labels (default: {' '.join(REFERENCES)}, if present)")
    ap.add_argument("--n-boot", type=int, default=10_000, help="bootstrap resamples (default 10000)")
    ap.add_argument("--llm", help="model name from configs/llms.yaml (default: the config's)")
    ap.add_argument("--budget", help="budget name from configs/budgets.yaml (default: the config's)")
    ap.add_argument("--experiment", help="results folder name (default: from config, llm and budget)")
    a = ap.parse_args()
    main(a.config, a.references, a.n_boot, llm=a.llm, budget=a.budget, experiment=a.experiment)
