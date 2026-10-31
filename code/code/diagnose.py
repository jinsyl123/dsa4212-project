"""Fresh-bin diagnostics: rescore every run's best program, and the priority baselines, under both packing modes.
Usage:
    python code/diagnose.py --config configs/eoh_baseline.yaml [--sets train test_shape_k5] [--workers 4]
Tests the bin-opening explanation: if evolved rules fail because they open fresh bins while a used bin has room,
fresh_bin_rate is high in funsearch mode and the excess drops in anyfit mode (where that choice is impossible).
Outputs: results/<experiment>/diagnostics.csv (per instance) and diagnostics_summary.csv
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from obpevo import baselines, config as C, diagnostics, instances

MODES = ["funsearch", "anyfit"]
REFERENCE = {"best_fit": baselines.best_fit, "first_fit": baselines.first_fit, "worst_fit": baselines.worst_fit}


def _job(j: dict) -> list[dict]:
    meta = {k: j[k] for k in ("kind", "label", "seed", "set")}
    try:
        if j["program"] is None:
            fn = REFERENCE[j["label"]]
        else:
            ns = {}
            exec(j["program"], ns)  # defines priority()
            fn = ns["priority"]
        rows = diagnostics.diagnose(fn, instances.load_set(j["path"]), j["mode"])
        return [dict(**meta, **r, status="ok") for r in rows]
    except Exception as e:
        return [dict(**meta, mode=j["mode"], status=f"error: {type(e).__name__}: {str(e)[:200]}")]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean per (label, set, mode) over instances, then across seeds."""
    ok = df[df.status == "ok"]
    per_run = (ok.groupby(["kind", "label", "set", "mode", "seed"])
               [["excess", "fresh_bin_rate", "fresh_given_room"]].mean().reset_index())
    out = (per_run.groupby(["kind", "label", "set", "mode"])
           .agg(runs=("seed", "nunique"), mean_excess=("excess", "mean"),
                fresh_bin_rate=("fresh_bin_rate", "mean"), fresh_given_room=("fresh_given_room", "mean"))
           .reset_index())
    return out.round(5).sort_values(["set", "label", "mode"])


def main(config, sets=None, workers=None, overrides=None) -> pd.DataFrame:
    """Run the diagnostics and return the summary table. Callable from a notebook."""
    workers = workers or max(mp.cpu_count() - 1, 1)
    cfg = C.load(config, overrides)
    out, idir = C.results_dir(cfg), C.instance_dir(cfg)
    set_names = sets or [cfg["evaluation"]["train_set"]] + cfg["final_eval"]["sets"]
    progs = [dict(kind="baseline", label=name, seed=-1, program=None) for name in REFERENCE]
    for summ in sorted(out.glob("*/seed*/summary.json")):
        s, prog = json.loads(summ.read_text()), summ.parent / "best_program.py"
        if prog.exists():
            progs.append(dict(kind="evolved", label=s["label"], seed=s["seed"], program=prog.read_text()))
    jobs = [dict(**p, set=sn, path=idir / f"{sn}.npz", mode=m) for p in progs for sn in set_names for m in MODES]
    print(f"[{cfg['experiment']}] {len(progs)} programs x {len(set_names)} sets x {len(MODES)} modes")

    with mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn").Pool(workers) as pool:
        df = pd.DataFrame([r for rows in pool.map(_job, jobs) for r in rows])
    df.to_csv(out / "diagnostics.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / "diagnostics_summary.csv", index=False)
    if (bad := df[df.status != "ok"]).size:
        print(bad[["label", "seed", "set", "mode", "status"]].drop_duplicates().to_string(index=False))
    print(summ.to_string(index=False))
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sets", nargs="*", help="default: train + final_eval.sets from the config")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    main(a.config, a.sets, a.workers)
