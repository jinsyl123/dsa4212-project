"""Report §5.5, check 2: masked choice. Empty bins stay visible to the rule but cannot be chosen while a used bin fits.

Usage, from code/:
    python pipeline/masked_choice.py
Rescores A1b's committed best programs (results/abl_informed_rules/eoh/seed<k>/best_program.py) on the frozen test
sets in three packing modes, next to Best Fit, and writes results/abl_informed_rules/masked_choice.csv:
    funsearch  the search's convention: every fitting bin, empty ones included, can be scored and chosen
    anyfit     empty bins withheld: the rule only sees used bins; a new bin opens only when none fits (check 1)
    masked     the rule scores every fitting bin, empty ones included (so a rule reading the capacity from the
               largest bin still sees it), but only used bins can be chosen while one fits
Excess is over the L2 bound, as in the report; delta_pp is the rule minus Best Fit in percentage points (negative =
better). No LLM calls, no randomness. The committed programs run in this process (they passed check_integrity.py).
Exploratory: designed after A1 was scored (code/docs/log.md, 12 Nov).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from obpevo import config as C, core, instances

CONFIG = "configs/abl_informed_rules.yaml"
SEEDS = [0, 2, 3, 4, 9]                                   # A1b's collapsing seeds (report, Table 9)
SETS = ["test_id_5k", "test_cap200", "test_cap500", "test_scale25"]
MODES = ["funsearch", "anyfit", "masked"]


def best_fit(item, bins):
    return item - bins                                    # LLM4AD's template: the tightest fitting bin wins


def pack_masked(priority, items: np.ndarray, capacity: int) -> int:
    """Pre-allocated bins as in core.pack_funsearch, but an empty bin is chosen only when no used bin fits."""
    bins = np.full(len(items), capacity, dtype=int)
    for item in items:
        idx = np.flatnonzero(bins >= item)                # feasible bins, as the rule sees them in the search
        s = core._checked(priority(item, bins[idx]), idx.size)
        used = bins[idx] != capacity
        cand, s = (idx[used], s[used]) if used.any() else (idx, s)
        bins[cand[np.argmax(s)]] -= item
    return int((bins != capacity).sum())


PACK = {"funsearch": core.pack_funsearch, "anyfit": core.pack_anyfit, "masked": pack_masked}


def load_priority(path: Path):
    ns: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    return ns["priority"]


def excess_l2(priority, insts: list[dict], mode: str) -> float:
    """Mean excess over L2 across the set's instances."""
    ex = []
    for i in insts:
        lb = core.lower_bound_l2(i["items"], i["capacity"])
        ex.append((PACK[mode](priority, i["items"], i["capacity"]) - lb) / lb)
    return float(np.mean(ex))


def main() -> pd.DataFrame:
    cfg = C.load(CONFIG)
    data, res = C.ensure_instances(cfg), C.results_dir(cfg)
    sets = {s: instances.load_set(data / f"{s}.npz") for s in SETS}
    bf = {s: excess_l2(best_fit, sets[s], "funsearch") for s in SETS}   # Best Fit packs the same in every mode
    rows = []
    for k in SEEDS:
        prio = load_priority(res / "eoh" / f"seed{k}" / "best_program.py")
        for s in SETS:
            for m in MODES:
                x = excess_l2(prio, sets[s], m)
                rows.append(dict(seed=k, set=s, mode=m, excess_l2_pct=round(100 * x, 4),
                                 best_fit_pct=round(100 * bf[s], 4), delta_pp=round(100 * (x - bf[s]), 4)))
                print(f"seed {k} {s:12} {m:9} {100 * x:9.3f}%  vs Best Fit {100 * (x - bf[s]):+9.3f} pp")
    out = pd.DataFrame(rows)
    out.to_csv(res / "masked_choice.csv", index=False)
    print(f"-> {res / 'masked_choice.csv'}")
    return out


if __name__ == "__main__":
    main()
