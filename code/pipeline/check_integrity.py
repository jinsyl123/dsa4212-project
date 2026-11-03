"""Integrity check of each run's best program: static rules + determinism.

Usage (from code/):
    python pipeline/check_integrity.py --config configs/eoh_baseline.yaml --experiment eoh_baseline_25oct [--sets train val]

For every results/<experiment>/*/seed*/best_program.py:
  - `flags`: static rule violations from obpevo/integrity.py ('' = clean)
  - rescored four times, each in a fresh Python process (spawn) with its own hash seed and fresh RNG seeds:
      A, B: instances in their usual order, S, T: the same shuffled order
      `repeat_mismatch`  = instances where A != B or S != T (randomness, hash()/id(), clock)
      `shuffle_mismatch` = instances where A != S (state kept from one instance to the next; only
                           read as state when repeat_mismatch == 0)
  Scripts that call check_programs need the usual `if __name__ == "__main__":` guard (spawn re-imports them).
  - `verdict`: failed / nondeterministic / stateful / flagged / clean
A clean, deterministic, stateless program has no flags and 0 mismatches. Nothing here changes a score.
A program whose randomness rarely changes a bin count can still pass the rescoring; the static flags cover that.
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
    # Each child is a fresh interpreter (spawn) with its own PYTHONHASHSEED, so hash()/id()-based choices vary too;
    # reseed the RNGs from OS entropy as well, in case a library seeded them at import.
    np.random.seed(int.from_bytes(os.urandom(4), "little"))
    random.seed(os.urandom(8))
    try:
        ns = {}
        exec(program, ns)
        q.put(("ok", {r["instance"]: r["bins"] for r in core.evaluate(ns["priority"], insts, mode)}))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {str(e)[:200]}"))


def run_isolated(jobs: list[dict], mode: str, timeout: float, workers: int) -> list:
    """Run each job's program on job['insts'] in its own fresh process; returns (status, payload) per job, in order.
    Always `spawn` (not fork): forked children share the parent's hash seed and memory layout, which would hide
    nondeterminism from hash() or id()."""
    ctx = mp.get_context("spawn")
    results, pending, live = [None] * len(jobs), list(enumerate(jobs)), []
    while pending or live:
        while pending and len(live) < workers:
            k, j = pending.pop(0)
            q = ctx.Queue()
            p = ctx.Process(target=_worker, args=(j["program"], j["insts"], mode, q), daemon=True)
            old = os.environ.get("PYTHONHASHSEED")
            os.environ["PYTHONHASHSEED"] = str(int.from_bytes(os.urandom(4), "little"))   # read by the new child
            try:
                p.start()
            finally:
                if old is None:
                    os.environ.pop("PYTHONHASHSEED", None)
                else:
                    os.environ["PYTHONHASHSEED"] = old
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
    jobs = [dict(**p, set=sn, run=run, insts=shuffled(insts) if run in "ST" else insts)
            for p in progs for sn, insts in sets.items() for run in RUNS]
    res = run_isolated(jobs, mode, timeout, workers)
    by = {(j["label"], j["seed"], j["set"], j["run"]): r for j, r in zip(jobs, res)}
    rows = []
    for p in progs:
        for sn, insts in sets.items():
            got = {run: by[(p["label"], p["seed"], sn, run)] for run in RUNS}
            ok = all(st == "ok" for st, _ in got.values())
            a, b, s, t = (got[run][1] for run in RUNS)
            flags = integrity.flag_str(p["program"])
            # repeat: same order, two fresh processes, for both orders; shuffle: only counted if repeats agree
            rep = sum(a[i] != b[i] or s[i] != t[i] for i in a) if ok else None
            shuf = sum(a[i] != s[i] for i in a) if ok else None
            rows.append(dict(label=p["label"], seed=p["seed"], flags=flags, set=sn, n_instances=len(insts),
                             status="ok" if ok else "; ".join(f"{r}: {x}" for r, (st, x) in got.items() if st != "ok"),
                             repeat_mismatch=rep, shuffle_mismatch=shuf, verdict=_verdict(ok, flags, rep, shuf)))
    return pd.DataFrame(rows)


RUNS = "ABST"   # A, B: usual order; S, T: the same shuffled order


RANDOM_FLAGS = {"random", "call:hash", "call:id", "clock", "star_import", "dynamic_getattr",
                "import:random", "import:time", "import:secrets", "import:datetime", "import:uuid", "import:os"}


def _verdict(ok: bool, flags: str, rep, shuf) -> str:
    """One word per row. Randomness that rarely changes a bin count can show up only in the shuffled runs, so a
    shuffle-only change counts as 'stateful' only when the static check found no source of randomness."""
    if not ok:
        return "failed"
    if rep or (shuf and RANDOM_FLAGS & set(flags.split(";"))):
        return "nondeterministic"
    if shuf:
        return "stateful"
    return "flagged" if flags else "clean"


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
    print(f"[{cfg['experiment']}] {len(progs)} programs x {len(set_names)} sets x {len(RUNS)} rescorings")
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
