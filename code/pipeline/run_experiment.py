"""Run every (method x seed) in a config under identical budgets, with method-independent logging.
Usage:
    python pipeline/run_experiment.py --config configs/smoke.yaml
    python pipeline/run_experiment.py --config configs/eoh_baseline.yaml --methods eoh --seeds 0 1
Each run writes to results/<experiment>/<label>/seed<k>/ (see readme.md for the file list).
Finished runs (with summary.json) are skipped unless --force is given.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from obpevo import config as C, instances, methods, task as T
from obpevo.llm import BudgetExhausted, LoggedLLM, make_backend
from obpevo.recorder import EvalRecorder

warnings.filterwarnings("ignore", category=DeprecationWarning)      # fork-with-threads notice from LLM4AD's sandbox

# Sampler threads stop by raising BudgetExhausted; don't print a traceback for that
_default_hook = threading.excepthook
threading.excepthook = lambda a: None if issubclass(a.exc_type, BudgetExhausted) else _default_hook(a)


ensure_instances = C.ensure_instances                  # kept here so existing notebooks still work


def task_description(cfg) -> str:
    prompt = cfg["evaluation"]["prompt"]
    if prompt not in T.DESCRIPTIONS:
        raise ValueError(f"evaluation.prompt={prompt!r}; choose from {sorted(T.DESCRIPTIONS)}")
    import yaml
    spec = yaml.safe_load((C.ROOT / cfg["instances"]["spec"]).read_text())["sets"][cfg["evaluation"]["train_set"]]
    mix = spec.get("mix") or [{}]
    shape = " or ".join(str(m.get("shape", spec.get("shape", 3.0))) for m in mix)       # mixed sets: "2 or 4"
    return T.DESCRIPTIONS[prompt].format(shape=shape, scale=spec.get("scale", 45), capacity=spec["capacity"])


def versions() -> dict:
    """Library versions for config.json. For LLM4AD installed from git via pip, the exact commit is read from pip's
    install record (direct_url.json); for a source checkout, from git; otherwise it is reported as unknown."""
    import importlib.metadata as md
    import llm4ad
    commit = ""
    try:
        info = json.loads(md.distribution("llm4ad").read_text("direct_url.json") or "{}")
        commit = info.get("vcs_info", {}).get("commit_id", "")
    except Exception:
        pass
    if not commit:
        try:
            src = Path(llm4ad.__file__).resolve().parents[1]
            commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        except Exception:
            commit = ""
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                llm4ad=getattr(llm4ad, "__version__", "?"), llm4ad_commit=commit or "unknown")


def finalize(run_dir: Path, meta: dict) -> dict:
    """Aggregate per-candidate files, pick the best program (by training score) and write summary.json."""
    ev, calls = pd.read_csv(run_dir / "evals.csv"), pd.read_csv(run_dir / "llm_calls.csv")
    rows = []
    this_run = set(ev.code_hash)                                  # ignore any leftovers from an earlier attempt
    for p in sorted((run_dir / "per_instance").glob("*.json")):
        if p.stem in this_run:
            rows += [dict(code_hash=p.stem, **r) for r in json.loads(p.read_text())["rows"]]
    pd.DataFrame(rows).to_csv(run_dir / "per_instance.csv", index=False)        # every candidate x training instance
    shutil.rmtree(run_dir / "per_instance", ignore_errors=True)  # Windows/OneDrive may hold the folder; leftovers are ignored

    ok = ev[ev.status == "ok"].sort_values(["score", "eval"], ascending=[False, True])   # ties -> earliest
    best = ok.iloc[0] if len(ok) else None
    if best is not None:
        cands = [json.loads(l) for l in (run_dir / "candidates.jsonl").read_text().splitlines()]
        prog = next(c["program"] for c in cands if c["eval"] == best["eval"])
        (run_dir / "best_program.py").write_text(f"# eval {best['eval']}, train excess {best['train_excess']:.5f}\n{prog}")
    counts = ev.status.value_counts()
    flags = ev["integrity"].fillna("") if "integrity" in ev else pd.Series("", index=ev.index)   # '' reads as NaN
    summary = dict(**meta,
                   n_llm_calls=len(calls), n_llm_errors=int((calls.status != "ok").sum()),
                   prompt_tokens=int(calls.prompt_tokens.sum()), completion_tokens=int(calls.completion_tokens.sum()),
                   token_source=",".join(sorted(calls.token_source.unique())) if len(calls) else "",
                   cost_usd=round(float(calls.cost_usd.sum()), 4),
                   n_evals=len(ev), n_unique_programs=int(ev.code_hash.nunique()),
                   **{f"n_{s}": int(counts.get(s, 0)) for s in ["ok", "error", "timeout", "crash"]},
                   # LLM calls with no evaluation: unparseable code, ReEvo reflections, or a call cut off by the budget
                   n_calls_not_evaluated=max(len(calls) - len(ev), 0),
                   # candidates breaking an integrity rule (integrity.py); they are still scored, only counted here
                   n_flagged=int((flags != "").sum()),
                   best_integrity=None if best is None else flags[best.name],
                   best_eval=None if best is None else int(best["eval"]),
                   best_train_excess=None if best is None else float(best["train_excess"]),
                   best_code_hash=None if best is None else best["code_hash"])
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def close_run_logs(stdout_log):
    """LLM4AD's profiler attaches a FileHandler (run_log.txt) and a handler on our redirected stdout, and never
    closes them. Windows can't delete open files, so re-running in the same notebook kernel would fail."""
    loggers = [logging.getLogger()] + [l for l in logging.Logger.manager.loggerDict.values() if isinstance(l, logging.Logger)]
    for lg in loggers:
        for h in lg.handlers[:]:
            if isinstance(h, logging.FileHandler) or getattr(h, "stream", None) is stdout_log:
                lg.removeHandler(h)
                h.close()


def reset_run_dir(run_dir: Path) -> Path:
    """Reuse the run folder instead of deleting it (Windows refuses to delete folders that something still holds).
    evals.csv, llm_calls.csv and candidates.jsonl are overwritten when a run starts; remove only what could
    otherwise mix into the new run. Returns a fresh folder for LLM4AD's own logs."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ["summary.json", "error.txt", "best_program.py", "per_instance.csv"]:
        (run_dir / name).unlink(missing_ok=True)
    shutil.rmtree(run_dir / "per_instance", ignore_errors=True)
    for old in run_dir.glob("llm4ad_log*"):
        shutil.rmtree(old, ignore_errors=True)                   # best effort: a locked old log is left in place
    return run_dir / f"llm4ad_log_{time.strftime('%Y%m%d_%H%M%S')}"


def run_one(cfg, m: dict, seed: int, train: list[dict], out: Path, verbose: bool) -> dict:
    run_dir = out / m["label"] / f"seed{seed}"
    llm4ad_log = reset_run_dir(run_dir)
    random.seed(seed)
    np.random.seed(seed)                                         # LLM4AD's selection steps use these RNGs

    b, ev = cfg["budget"], cfg["evaluation"]
    backend = make_backend(cfg["llm"], seed)
    suffix = T.FORMAT_HINT if ev.get("format_hint", True) else ""
    llm = LoggedLLM(backend, run_dir / "llm_calls.csv", b["max_tokens"], b["max_minutes"],
                    cfg["llm"]["price_in_per_1m"], cfg["llm"]["price_out_per_1m"], prompt_suffix=suffix)
    task = T.OBPTask(train, run_dir, ev["mode"], ev["timeout_seconds"], task_description(cfg))

    meta = dict(experiment=cfg["experiment"], label=m["label"], method=m["name"], seed=seed,
                llm_name=cfg["llm"].get("name"), llm_provider=cfg["llm"].get("provider") or cfg["llm"].get("backend"),
                llm_model=cfg["llm"]["model"], budget_name=b.get("name"), mode=ev["mode"],
                train_set=ev["train_set"], n_train=len(train), max_evals=b["max_evals"], max_tokens=b["max_tokens"])
    t0, stop = time.time(), "method_finished"
    with open(run_dir / "stdout.log", "w") as log, \
            EvalRecorder(run_dir, b["max_evals"], ev["timeout_seconds"]) as rec, \
            (contextlib.nullcontext() if verbose else contextlib.redirect_stdout(log)):
        llm.linked_budget = rec                                  # evaluation cap also stops further LLM calls
        try:
            method, used, dropped = methods.build(m["name"], llm, task, b["max_evals"], llm4ad_log,
                                                  m["params"], **cfg["parallel"])
            snapshot = dict(meta=meta, method_kwargs=used, dropped_kwargs=dropped, config=cfg, versions=versions(),
                            task_description=task.task_description, template_program=task.template_program,
                            prompt_suffix=suffix)
            (run_dir / "config.json").write_text(json.dumps(snapshot, indent=2, default=str))
            method.run()
        except BudgetExhausted:
            pass
        except Exception:
            stop = "error"
            (run_dir / "error.txt").write_text(traceback.format_exc())
        finally:
            close_run_logs(log)
    budget_hit = llm.exhausted or rec.exhausted
    meta.update(stop_reason=f"budget:{budget_hit}" if budget_hit else stop, wall_minutes=round((time.time() - t0) / 60, 2))
    return finalize(run_dir, meta)


def main(config, methods=None, seeds=None, force=False, verbose=False, overrides=None,
         llm=None, budget=None, experiment=None) -> pd.DataFrame:
    """Run every (method x seed) in `config`; returns the runs index. Callable from a notebook.
    overrides: nested dict applied on top of the config, e.g. {"experiment": "x", "llm": {"model": "y"}}."""
    cfg = C.load(config, overrides, llm=llm, budget=budget, experiment=experiment)
    out = C.results_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    train = instances.load_set(ensure_instances(cfg) / f"{cfg['evaluation']['train_set']}.npz")
    todo = [(m, s) for m in cfg["methods"] if not methods or m["label"] in methods
            for s in (seeds if seeds is not None else cfg["seeds"])]
    llm = cfg["llm"]
    print(f"LLM: {llm.get('name') or llm['model']} ({llm.get('provider') or llm.get('backend')}, ${llm['price_in_per_1m']}/${llm['price_out_per_1m']} per 1M in/out) | budget: {cfg['budget']} | results: results/{cfg['experiment']}")
    if (llm.get("provider") or llm.get("backend")) != "fake" and not (llm["price_in_per_1m"] or llm["price_out_per_1m"]):
        print("  warning: price_in_per_1m / price_out_per_1m are 0, so cost_usd will read 0. Set them from the provider's pricing page.")
    print(f"[{cfg['experiment']}] {len(todo)} runs | budget {cfg['budget']} | train={cfg['evaluation']['train_set']} ({len(train)} instances)")

    for m, s in todo:
        if (out / m["label"] / f"seed{s}" / "summary.json").exists() and not force:
            print(f"  skip {m['label']} seed{s} (done)")
            continue
        r = run_one(cfg, m, s, train, out, verbose)
        best = "none" if r["best_train_excess"] is None else f"{r['best_train_excess']:.4f}"
        print(f"  {m['label']:>12} seed{s}: {r['n_evals']} evals ({r['n_ok']} ok, {r['n_error']} err, {r['n_timeout']} timeout) "
              f"| {r['n_llm_calls']} calls, {r['prompt_tokens'] + r['completion_tokens']:,} tokens | best train excess {best} "
              f"| {r['stop_reason']} | {r['wall_minutes']} min")

    # index of all finished runs in this experiment
    idx = pd.DataFrame([json.loads(p.read_text()) for p in sorted(out.glob("*/seed*/summary.json"))])
    idx.to_csv(out / "runs_index.csv", index=False)
    print(f"Wrote {out / 'runs_index.csv'} ({len(idx)} runs)")
    return idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", nargs="*", help="labels to run (default: all in config)")
    ap.add_argument("--seeds", nargs="*", type=int, help="seeds to run (default: config)")
    ap.add_argument("--force", action="store_true", help="re-run finished runs")
    ap.add_argument("--verbose", action="store_true", help="show LLM4AD's own console output")
    ap.add_argument("--llm", help="model name from configs/llms.yaml (default: the config's)")
    ap.add_argument("--budget", help="budget name from configs/budgets.yaml (default: the config's)")
    ap.add_argument("--experiment", help="results folder name (default: from config, llm and budget)")
    a = ap.parse_args()
    main(a.config, a.methods, a.seeds, a.force, a.verbose, llm=a.llm, budget=a.budget, experiment=a.experiment)
