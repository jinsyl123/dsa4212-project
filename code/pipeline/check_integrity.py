"""Integrity check of each run's best program: static rules + determinism.

Usage (from code/):
    python pipeline/check_integrity.py --config configs/eoh_baseline.yaml --experiment eoh_baseline_25oct [--sets train val]

For every results/<experiment>/*/seed*/best_program.py:
  - `flags`: static rule violations from obpevo/integrity.py ('' = clean)
  - rescored three times, each in a fresh process with freshly seeded RNGs:
      A, B: instances in their usual order  -> `repeat_mismatch`  = instances where A and B differ (randomness)
      S:    instances shuffled              -> `shuffle_mismatch` = instances where A and S differ (state kept
                                                                     from one instance to the next)
A clean, deterministic, stateless program has no flags and 0 mismatches. Nothing here changes a score.
Output: results/<experiment>/integrity.csv
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np
import pandas as pd

from obpevo import config as C, core, instances, integrity


def _worker(program: str, insts: list[dict], mode: str, q):
    # Forked children inherit the parent's RNG state, so unseeded randomness would repeat exactly. Reseed from OS entropy.
    np.random.seed(int.from_bytes(os.urandom(4), "little"))
    random.seed(os.urandom(8))
    try:
        ns = {}
        exec(program, ns)
        q.put(("ok", {r["instance"]: r["bins"] for r in core.evaluate(ns["priority"], insts, mode)}))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {str(e)[:200]}"))


def run_isolated(jobs: list[dict], mode: str, timeout: float, workers: int) -> list:
    """Run each job's program on job['insts'] in its own process; returns (status, payload) per job, in order."""
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    results, pending, live = [None] * len(jobs), list(enumerate(jobs)), []
    while pending or live:
        while pending and len(live) < workers:
            k, j = pending.pop(0)
            q = ctx.Queue()
            p = ctx.Process(target=_worker, args=(j["program"], j["insts"], mode, q), daemon=True)
            p.start()
            live.append((k, p, q, time.time()))
        time.sleep(0.05)
        for item in list(live):
            k, p, q, t0 = item
            if not q.empty():
                results[k] = q.get()
            elif time.time() - t0 > timeout:
                results[k] = ("timeout", f"exceeded {timeout}s")
            elif not p.is_alive() and q.empty():
                results[k] = ("crash", "no result from child process")
            else:
                continue
            p.kill()
            live.remove(item)
    return results


def shuffled(insts: list[dict], seed: int = 0) -> list[dict]:
    """Same instances, different order (never the identity order)."""
    order = np.random.default_rng(seed).permutation(len(insts))
    if (order == np.arange(len(insts))).all():
        order = order[::-1]
    return [insts[i] for i in order]


def check_programs(progs: list[dict], sets: dict, mode: str, timeout: float, workers: int) -> pd.DataFrame:
    """progs: dicts with label, seed, program. sets: name -> instances. One row per (program, set)."""
    jobs = [dict(**p, set=sn, run=run, insts=shuffled(insts) if run == "S" else insts)
            for p in progs for sn, insts in sets.items() for run in "ABS"]
    res = run_isolated(jobs, mode, timeout, workers)
    by = {(j["label"], j["seed"], j["set"], j["run"]): r for j, r in zip(jobs, res)}
    rows = []
    for p in progs:
        for sn, insts in sets.items():
            (sa, a), (sb, b), (ss, s) = (by[(p["label"], p["seed"], sn, run)] for run in "ABS")
            ok = sa == sb == ss == "ok"
            rows.append(dict(label=p["label"], seed=p["seed"], flags=integrity.flag_str(p["program"]), set=sn,
                             n_instances=len(insts),
                             status="ok" if ok else "; ".join(f"{r}: {x}" for r, x in zip("ABS", (a, b, s))
                                                              if isinstance(x, str)),
                             repeat_mismatch=sum(a[i] != b[i] for i in a) if ok else None,
                             shuffle_mismatch=sum(a[i] != s[i] for i in a) if ok else None))
    return pd.DataFrame(rows)


def main(config, sets=None, workers=None, overrides=None, llm=None, budget=None, experiment=None) -> pd.DataFrame:
    """Check every run's best program; returns the table and writes integrity.csv. Callable from a notebook."""
    workers = workers or max(mp.cpu_count() - 1, 1)
    cfg = C.load(config, overrides, llm=llm, budget=budget, experiment=experiment)
    out, idir = C.results_dir(cfg), C.ensure_instances(cfg)
    set_names = sets or [cfg["evaluation"]["train_set"], "val"]
    loaded = {s: instances.load_set(idir / f"{s}.npz") for s in set_names}
    progs = []
    for summ in sorted(out.glob("*/seed*/summary.json")):
        s, prog = json.loads(summ.read_text()), summ.parent / "best_program.py"
        if prog.exists():
            progs.append(dict(label=s["label"], seed=s["seed"], program=prog.read_text()))
    print(f"[{cfg['experiment']}] {len(progs)} programs x {len(set_names)} sets x 3 rescorings")
    df = check_programs(progs, loaded, cfg["evaluation"]["mode"], cfg["final_eval"]["timeout_seconds"], workers)
    df.to_csv(out / "integrity.csv", index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sets", nargs="*", help="default: the training set and val")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--llm", help="model name from configs/llms.yaml (default: the config's)")
    ap.add_argument("--budget", help="budget name from configs/budgets.yaml (default: the config's)")
    ap.add_argument("--experiment", help="results folder name (default: from config, llm and budget)")
    a = ap.parse_args()
    main(a.config, a.sets, a.workers, llm=a.llm, budget=a.budget, experiment=a.experiment)
