"""Score every run's best program, and the hand-designed baselines, on held-out sets.
Usage:
    python pipeline/evaluate_final.py --config configs/smoke.yaml [--sets val test_id] [--workers 4]
The search never saw these sets. Evolved programs run in separate processes with a timeout.
Outputs: results/<experiment>/final_per_instance.csv and final_summary.csv
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from obpevo import baselines, config as C, core, instances


def _worker(program: str, insts: list[dict], mode: str, q):
    try:
        ns = {}
        exec(program, ns)                                           # defines priority()
        q.put(("ok", core.evaluate(ns["priority"], insts, mode)))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {str(e)[:200]}"))


def run_programs(jobs: list[dict], sets: dict, mode: str, timeout: float, workers: int) -> list[dict]:
    """Run (program x set) jobs in forked processes, at most `workers` at a time, killing any over `timeout`."""
    # fork on Linux/macOS (fast); spawn on Windows, which has no fork (slower start-up per job, same results)
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx, rows, pending, live = mp.get_context(method), [], list(jobs), []
    while pending or live:
        while pending and len(live) < workers:
            j = pending.pop(0)
            q = ctx.Queue()
            p = ctx.Process(target=_worker, args=(j["program"], sets[j["set"]], mode, q), daemon=True)
            p.start()
            live.append((j, p, q, time.time()))
        time.sleep(0.05)
        for item in list(live):
            j, p, q, t0 = item
            done, status, payload = not q.empty(), None, None
            if done:
                status, payload = q.get()
            elif time.time() - t0 > timeout:
                status, payload = "timeout", f"exceeded {timeout}s"
            elif not p.is_alive():
                status, payload = "crash", "process died"
            if status is None:
                continue
            p.kill() if p.is_alive() else None
            live.remove(item)
            meta = {k: j[k] for k in ("kind", "label", "method", "seed", "set")}
            if status == "ok":
                rows += [dict(**meta, **r, status="ok") for r in payload]
            else:                                                    # one row per instance so failures stay visible
                rows += [dict(**meta, instance=i["id"], n_items=len(i["items"]), capacity=i["capacity"], bins=np.nan,
                              lb=core.lower_bound(i["items"], i["capacity"]), excess=np.nan, time_s=np.nan,
                              status=f"{status}: {payload}") for i in sets[j["set"]]]
            print(f"  {j['label']:>16} seed{j['seed']} {j['set']:<18} {status}")
    return rows


def run_baselines(sets: dict) -> list[dict]:
    rows = []
    for set_name, insts in sets.items():
        for name, pack in baselines.BASELINES.items():
            for i in insts:
                t0 = time.perf_counter()
                n, lb = pack(i["items"], i["capacity"]), core.lower_bound(i["items"], i["capacity"])
                rows.append(dict(kind="baseline", label=name, method=name, seed=-1, set=set_name, instance=i["id"],
                                 n_items=len(i["items"]), capacity=i["capacity"], bins=n, lb=lb, excess=(n - lb) / lb,
                                 time_s=round(time.perf_counter() - t0, 4), status="ok"))
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean excess per (label, set): averaged within each run first, then across seeds; plus wins vs Best Fit."""
    bf = df[df.label == "best_fit"].set_index(["set", "instance"])["bins"]
    d = df.assign(bf_bins=df.set_index(["set", "instance"]).index.map(bf))
    d = d.assign(win=d.bins < d.bf_bins, tie=d.bins == d.bf_bins, loss=d.bins > d.bf_bins)
    per_run = d.groupby(["kind", "label", "set", "seed"]).agg(
        excess=("excess", "mean"), failed=("status", lambda s: (s != "ok").mean()),
        win=("win", "mean"), tie=("tie", "mean"), loss=("loss", "mean")).reset_index()
    out = per_run.groupby(["kind", "label", "set"]).agg(
        runs=("seed", "nunique"), mean_excess=("excess", "mean"), sd_excess=("excess", "std"),
        best_run=("excess", "min"), worst_run=("excess", "max"), failed_frac=("failed", "mean"),
        win_vs_bf=("win", "mean"), tie_vs_bf=("tie", "mean"), loss_vs_bf=("loss", "mean")).reset_index()
    bf_set = out[out.label == "best_fit"].set_index("set")["mean_excess"]
    out["delta_vs_bf_pp"] = (out.mean_excess - out.set.map(bf_set)) * 100      # percentage points; negative = better
    return out.round(5).sort_values(["set", "mean_excess"])


def main(config, sets=None, workers=None, overrides=None, llm=None, budget=None, experiment=None) -> pd.DataFrame:
    """Score best programs + baselines on held-out sets; returns the summary table. Callable from a notebook."""
    workers = workers or max(mp.cpu_count() - 1, 1)
    cfg = C.load(config, overrides, llm=llm, budget=budget, experiment=experiment)
    out, idir = C.results_dir(cfg), C.ensure_instances(cfg)     # builds data/ on a fresh clone
    set_names = sets or cfg["final_eval"]["sets"]
    sets = {s: instances.load_set(idir / f"{s}.npz") for s in set_names}

    jobs = []
    for summ in sorted(out.glob("*/seed*/summary.json")):
        s, prog = json.loads(summ.read_text()), summ.parent / "best_program.py"
        if prog.exists():
            jobs += [dict(kind="evolved", label=s["label"], method=s["method"], seed=s["seed"], set=sn,
                          program=prog.read_text()) for sn in set_names]
    print(f"[{cfg['experiment']}] {len(jobs)} program x set jobs on {set_names}")

    rows = run_baselines(sets) + run_programs(jobs, sets, cfg["evaluation"]["mode"],
                                              cfg["final_eval"]["timeout_seconds"], workers)
    df = pd.DataFrame(rows)
    df.to_csv(out / "final_per_instance.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / "final_summary.csv", index=False)
    print(summ[["label", "set", "runs", "mean_excess", "sd_excess", "delta_vs_bf_pp", "win_vs_bf", "failed_frac"]].to_string(index=False))
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sets", nargs="*", help="override final_eval.sets from the config")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--llm", help="model name from configs/llms.yaml (default: the config's)")
    ap.add_argument("--budget", help="budget name from configs/budgets.yaml (default: the config's)")
    ap.add_argument("--experiment", help="results folder name (default: from config, llm and budget)")
    a = ap.parse_args()
    main(a.config, a.sets, a.workers, llm=a.llm, budget=a.budget, experiment=a.experiment)
