"""Setting-level effects: Fisher's exact test on seed counts (claims doc, "Setting-level effect").

Usage, from code/ (after score.sh has written every setting's paired_stats.csv):
    python pipeline/setting_tests.py
Writes results/setting_tests.csv, one row per (setting, reference version, outcome). Each setting is compared with
the setting it changes one thing from (A2b with A1b, the rest with the baseline). Against the baseline there are two
rows: seeds 0-5 against tuned Best Fit seeds 0-5 (pre-registered, paired_stats_seeds0-5.csv) and seeds 0-9 against
tuned Best Fit seeds 0-9 (paired_stats.csv). `claim` is True only if p < 0.05 against every reference version (#21).
Outcomes, counted per seed from the paired verdicts:
    beats_bf_id5k    better than Best Fit on test_id_5k
    beats_tbf_id5k   better than tuned Best Fit on test_id_5k
    worse_bf_shape   worse than Best Fit on test_shape_k1.5 or test_shape_k5
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

R = Path("results")
BASE = "eoh_baseline__openrouter-gemini-3.5-flash-lite"
COMPARE = {"abl_anyfit": BASE, "abl_informed_rules": BASE, "abl_mixed_shapes": BASE,
           "control_randsample": BASE, "abl_mixed_caps_informed": "abl_informed_rules"}
REFS = ["best_fit", "tuned_bf"]


def counts(setting: str, seeds=range(10), stats: str = "paired_stats.csv") -> tuple[pd.Series, int]:
    """Number of seeds with each outcome for the setting's searched rule, and the number of seeds."""
    df = pd.read_csv(R / setting / stats)
    df = df[~df.label.isin(REFS) & (df.kind != "baseline") & df.seed.isin(seeds)]
    assert df.label.nunique() == 1, f"{setting}: expected one searched rule, got {sorted(df.label.unique())}"
    v = lambda s, ref, verdict: df[(df.set == s) & (df.reference == ref)].set_index("seed").verdict.eq(verdict)
    out = pd.DataFrame({"beats_bf_id5k": v("test_id_5k", "best_fit", "better"),
                        "beats_tbf_id5k": v("test_id_5k", "tuned_bf", "better"),
                        "worse_bf_shape": v("test_shape_k1.5", "best_fit", "worse")
                        | v("test_shape_k5", "best_fit", "worse")})
    assert len(out) == len(seeds), f"{setting}: {len(out)} seeds, expected {len(seeds)}"
    return out.sum().astype(int), len(out)


rows = []
for setting, ref in COMPARE.items():
    a, na = counts(setting)
    versions = [("seeds 0-9", *counts(ref))]
    if ref == BASE:  # pre-registered: baseline seeds 0-5 against tuned Best Fit seeds 0-5
        versions.insert(0, ("seeds 0-5", *counts(BASE, range(6), "paired_stats_seeds0-5.csv")))
    for version, b, nb in versions:
        for m in a.index:
            p = fisher_exact([[a[m], na - a[m]], [b[m], nb - b[m]]])[1]   # two-sided
            rows.append(dict(setting=setting, compared_with=ref, reference_seeds=version, outcome=m,
                             count=f"{a[m]}/{na}", reference_count=f"{b[m]}/{nb}", p=round(p, 3)))

out = pd.DataFrame(rows)
out["claim"] = out.groupby(["setting", "outcome"]).p.transform(lambda p: bool((p < 0.05).all()))
out.to_csv(R / "setting_tests.csv", index=False)
print(out.to_string(index=False))