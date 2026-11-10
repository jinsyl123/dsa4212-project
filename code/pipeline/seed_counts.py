"""Setting-level effects (claims.md, shared rules): seeds per category, and a Fisher test against the comparison row.
Usage, from code/, after score.sh has scored every setting:
    python pipeline/seed_counts.py
Reads, per setting: paired_stats.csv (per-seed verdicts; for the baseline's seeds 0-5 row,
paired_stats_seeds0-5.csv), final_per_instance.csv (collapses) and diagnostics.csv (fresh-bin seeds).
No LLM calls, no rescoring; deterministic.
Outputs:
    results/seed_counts.csv   one row per (setting, category, set, comparison row): k of n seeds in the category,
                              the comparison row's k of n, the two-sided Fisher exact p, and significant_vs_all
                              (p < 0.05 against every comparison row of that setting, category and set; for any
                              category and set, so not a claim unless claims.md predicted it)
    results/seed_counts.md    the headline categories (all pre-registered) as a table, significant_vs_all cells in bold

Categories, per seed (claims.md, shared rules and A2):
    better_vs_<ref> / worse_vs_<ref>   paired_stats verdict on a set, ref = best_fit or tuned_bf
    worse_vs_best_fit on "k1.5 or k5"  worse on test_shape_k1.5 or test_shape_k5
    collapse                           mean excess (L2) more than 100 pp above Best Fit's on a set
    fresh_bin                          fresh_given_room > 0.01 on the setting's own training set (train,
                                       train_mixed or train_mixed_cap) and in its own search mode (set "train")
Comparison rows: the baseline's seeds 0-5 (pre-registered, primary) and seeds 0-9 (claims.md, Amendments, "baseline
extended to 10 seeds"); A2b against A1b, seeds 0-9. A result is claimed only if p < 0.05 against both baselines.
Verdicts against tuned Best Fit use tuned Best Fit averaged over the same seeds as the rule: for the baseline's seeds
0-5 row that is seeds 0-5, the pre-registered comparison (claims.md, Amendments, "tuned Best Fit reference over seeds
0-5"; written by score.sh as paired_stats_seeds0-5.csv).
"""
from __future__ import annotations

from math import comb
from pathlib import Path

import pandas as pd

from obpevo import config as C

LLM = "openrouter-gemini-3.5-flash-lite"
# name: (config, method label, compared with); report order
SETTINGS = {
    "baseline": ("eoh_baseline", "eoh", None),
    "control": ("control_randsample", "randsample", "baseline"),
    "A1a": ("abl_anyfit", "eoh", "baseline"),
    "A1b": ("abl_informed_rules", "eoh", "baseline"),
    "A2a": ("abl_mixed_shapes", "eoh", "baseline"),
    "A2b": ("abl_mixed_caps_informed", "eoh", "A1b"),
}
BASELINE_ROWS = {"baseline seeds 0-5": range(6), "baseline seeds 0-9": range(10)}
NOT_CLAIMS = {"val", "test_control"}               # selection only / scoring check (claims.md, shared rules)
SHAPE_SETS = ("test_shape_k1.5", "test_shape_k5")
COLLAPSE_PP = 100.0
FRESH = 0.01


def fisher_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact test on [[k1, n1-k1], [k2, n2-k2]]: sum of tables no more likely than the observed."""
    K, N = k1 + k2, n1 + n2
    prob = lambda a: comb(K, a) * comb(N - K, n1 - a) / comb(N, n1)
    p0 = prob(k1)
    p = sum(prob(a) for a in range(max(0, K - n2), min(K, n1) + 1) if prob(a) <= p0 * (1 + 1e-9))
    return min(1.0, p)


def seed_categories(cfg: dict, label: str, stats: str = "paired_stats.csv") -> pd.DataFrame:
    """One row per (seed, category, set) with in_cat = 0/1."""
    d = C.results_dir(cfg)
    ps = pd.read_csv(d / stats)
    ps = ps[(ps.label == label) & (ps.metric == "excess_l2") & ~ps.set.isin(NOT_CLAIMS)]
    rows = []
    for (seed, s, ref), g in ps.groupby(["seed", "set", "reference"]):
        v = g.verdict.iloc[0]
        rows += [(seed, f"better_vs_{ref}", s, int(v == "better")), (seed, f"worse_vs_{ref}", s, int(v == "worse"))]
    shape = ps[(ps.reference == "best_fit") & ps.set.isin(SHAPE_SETS)].groupby("seed").verdict
    rows += [(seed, "worse_vs_best_fit", "k1.5 or k5", int((v == "worse").any())) for seed, v in shape]

    fp = pd.read_csv(d / "final_per_instance.csv")
    fp = fp[~fp.set.isin(NOT_CLAIMS)]
    bf = fp[fp.label == "best_fit"].groupby("set").excess_l2.mean()
    rule = fp[fp.label == label].groupby(["seed", "set"]).excess_l2.mean()
    rows += [(seed, "collapse", s, int((x - bf[s]) * 100 > COLLAPSE_PP)) for (seed, s), x in rule.items()]

    dg = pd.read_csv(d / "diagnostics.csv")
    ev = cfg["evaluation"]
    fr = dg[(dg.label == label) & (dg.set == ev["train_set"]) & (dg["mode"] == ev["mode"])].groupby("seed").fresh_given_room.mean()
    rows += [(seed, "fresh_bin", "train", int(x > FRESH)) for seed, x in fr.items()]
    out = pd.DataFrame(rows, columns=["seed", "category", "set", "in_cat"])
    return out[out.seed.isin(cfg["seeds"])]


def main() -> pd.DataFrame:
    cats = {}
    for name, (config, label, _) in SETTINGS.items():
        cfg = C.load(C.ROOT / "configs" / f"{config}.yaml", llm=LLM if name == "baseline" else None)
        cats[name] = seed_categories(cfg, label)
    # pre-registered row: baseline seeds 0-5 against tuned Best Fit averaged over seeds 0-5 (#19 review, point 1)
    cfg = C.load(C.ROOT / "configs" / "eoh_baseline.yaml", llm=LLM)
    cats["baseline seeds 0-5"] = seed_categories(cfg, "eoh", "paired_stats_seeds0-5.csv")

    def count(name, seeds=None):
        c = cats[name] if seeds is None else cats[name][cats[name].seed.isin(seeds)]
        return c.groupby(["category", "set"]).in_cat.agg(k="sum", n="count")

    rows = []
    for name, (_, _, against) in SETTINGS.items():
        mine = count(name)
        rows.append(mine.assign(setting=name, compared_with="", k_ref=float("nan"), n_ref=float("nan"), fisher_p=float("nan")))
        if against is None:
            continue
        refs = BASELINE_ROWS if against == "baseline" else {f"{against} seeds 0-9": range(10)}
        for ref_name, seeds in refs.items():
            ref = count(ref_name if ref_name in cats else "baseline" if against == "baseline" else against, seeds)
            j = mine.join(ref, rsuffix="_ref", how="inner")
            j["fisher_p"] = [fisher_p(r.k, r.n, r.k_ref, r.n_ref) for r in j.itertuples()]
            rows.append(j.assign(setting=name, compared_with=ref_name))
    out = pd.concat(rows).reset_index()
    vs = out[out.compared_with != ""].groupby(["setting", "category", "set"]).fisher_p.max() < 0.05
    out["significant_vs_all"] = [bool(vs.get((r.setting, r.category, r.set), False)) for r in out.itertuples()]
    out["setting"] = pd.Categorical(out.setting, list(SETTINGS))
    out = out.sort_values(["setting", "category", "set", "compared_with"])[
        ["setting", "category", "set", "k", "n", "compared_with", "k_ref", "n_ref", "fisher_p", "significant_vs_all"]]
    res = C.ROOT / "results"
    out.round({"fisher_p": 4}).to_csv(res / "seed_counts.csv", index=False)  # the table formats the unrounded p
    (res / "seed_counts.md").write_text(headline(out))
    print(f"Wrote {res / 'seed_counts.csv'} ({len(out)} rows) and seed_counts.md")
    return out


def headline(out: pd.DataFrame) -> str:
    """The readme's results table, with Fisher p against each comparison row."""
    cols = [("better_vs_best_fit", "test_id_5k", "Beats Best Fit on `test_id_5k`"),
            ("better_vs_tuned_bf", "test_id_5k", "Beats its tuned Best Fit on `test_id_5k`"),
            ("worse_vs_best_fit", "k1.5 or k5", "Worse than Best Fit on k1.5 or k5"),
            ("fresh_bin", "train", "Fresh-bin seeds")]
    own = out[out.compared_with == ""].set_index(["setting", "category", "set"])
    vs = out[out.compared_with != ""]
    lines = ["# Seed counts and Fisher tests (`pipeline/seed_counts.py`)", "",
             "Each cell: seeds in the category / seeds, then Fisher p against each comparison row (claims.md, "
             "shared rules). Baseline rows: seeds 0-5 (pre-registered) and seeds 0-9. A2b is compared with A1b. "
             "**Bold**: p < 0.05 against every comparison row, i.e. claimed (claims.md, Amendments).", "",
             "| Setting | " + " | ".join(c[2] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name in SETTINGS:
        if name == "baseline":
            for ref_name, seeds in BASELINE_ROWS.items():
                cells = []
                for cat, s, _ in cols:
                    sub = out[(out.setting == "A1b") & (out.category == cat) & (out.set == s)
                              & (out.compared_with == ref_name)]
                    cells.append(f"{int(sub.k_ref.iloc[0])}/{int(sub.n_ref.iloc[0])}" if len(sub) else "—")
                lines.append(f"| {ref_name} | " + " | ".join(cells) + " |")
            continue
        cells = []
        for cat, s, _ in cols:
            if (name, cat, s) not in own.index:
                cells.append("—")
                continue
            r = own.loc[(name, cat, s)]
            ps = vs[(vs.setting == name) & (vs.category == cat) & (vs.set == s)].fisher_p.tolist()
            cell = f"{int(r.k)}/{int(r.n)} (p = {', '.join(f'{p:.3f}' for p in ps)})"
            cells.append(f"**{cell}**" if r.significant_vs_all else cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
