# DSA4212 Project 2: LLM-guided evolution of online bin packing heuristics

We use LLM-driven search methods to evolve a heuristic for **online bin packing**, then compare the methods fairly. [LLM4AD](https://github.com/Optima-CityU/LLM4AD) supplies the search methods; **EoH** is our baseline. A thin layer of our own (`code/obpevo/`) wraps every method so they are all measured the same way, following the principles of the BLADE benchmark.

```
                     ┌──────────────── one run = one method × one seed ────────────────┐
LLM API  ◄──────  LoggedLLM  ◄──────  LLM4AD method (EoH / ReEvo / FunSearch / ...)  ──────►  EvalRecorder  ──────►  OBPTask
(Gemini, ...)     logs every call      proposes priority() functions                         logs every candidate     packs the training
                  enforces token /                                                           enforces the eval        instances, returns
                  time budgets                                                               budget                   a score
```

---

## Contents

1. [The problem in one minute](#1-the-problem-in-one-minute)
2. [What the pipeline guarantees](#2-what-the-pipeline-guarantees)
3. [Repository layout](#3-repository-layout)
4. [Setup](#4-setup)
5. [Testing the pipeline (cheapest first)](#5-testing-the-pipeline-cheapest-first)
6. [Running experiments](#6-running-experiments)
7. [Configs](#7-configs)
8. [Choosing the LLM](#8-choosing-the-llm)
9. [Methods and ablations](#9-methods-and-ablations)
10. [Outputs and how to read them](#10-outputs-and-how-to-read-them)
11. [Evaluation details](#11-evaluation-details)
12. [Troubleshooting](#12-troubleshooting)
13. [Results so far](#13-results-so-far)
14. [Known limitations and next steps](#14-known-limitations-and-next-steps)

---

## 1. The problem in one minute

Items arrive one at a time and must be placed immediately into bins of fixed capacity (100 by default). The goal is to use as few bins as possible. A heuristic is a Python function that scores each bin for the arriving item; the item goes into the highest-scoring bin:

```python
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    return item - bins          # Best Fit: the tightest-fitting bin wins
```

This is LLM4AD's template, character for character; the search starts from it. Each method asks the LLM for new versions of this function, scores them on training instances, and keeps the better ones.

**The metric: excess over the lower bound.** For each instance, the lower bound is `ceil(total item size / capacity)`, the fewest bins any packing could possibly use. Excess = (bins used − lower bound) / lower bound, averaged over instances. **Lower is better.** Best Fit scores 4.751% on our training set; that is the number to beat.

---

## 2. What the pipeline guarantees

Each method in LLM4AD has its own loop, logging and stopping rules. Comparing them directly would compare those differences, not the methods. Our layer enforces four rules for every method, without touching LLM4AD's code:

| Guarantee | How | Where |
|---|---|---|
| **1. Same logging for every method** | Every candidate goes through one patched function (`SecureEvaluator.evaluate_program`), and every LLM call through one wrapper (`LoggedLLM`). | `recorder.py`, `llm.py` |
| **2. Equal budgets** | A run stops at the first of `max_evals`, `max_tokens`, `max_minutes`, counted outside the method. Hitting the evaluation cap also blocks further LLM calls. | `recorder.py`, `llm.py` |
| **3. Train/test separation** | Search only ever sees the `train` set. Each run's best program is rescored afterwards on held-out sets it never saw. | `task.py`, `evaluate_final.py` |
| **4. Full records** | Every run saves its resolved config, library versions, every LLM call (tokens, cost, latency), every candidate program with its per-instance results, and the best program. | `run_experiment.py` |

---

## 3. Repository layout

```
code/
  obpevo/core.py         evaluator: packing, lower bound, score (single source of truth)
  obpevo/instances.py    deterministic instance generator (Weibull / uniform)
  obpevo/baselines.py    First / Best / Worst / Almost-Worst / Next Fit
  obpevo/task.py         LLM4AD task wrapper: template, task description, formatting rules for the LLM
  obpevo/llm.py          LLM clients (OpenAI-compatible, Anthropic, fake) + LoggedLLM wrapper
  obpevo/recorder.py     EvalRecorder: logs every candidate, enforces the eval budget
  obpevo/methods.py      registry of LLM4AD methods
  obpevo/config.py       YAML config loading, inheritance (base:) and overrides
  make_instances.py      build instance sets from a spec
  run_experiment.py      step 1: search (every method x seed in a config)
  evaluate_final.py      step 2: score best programs + baselines on held-out sets
  summarise.py           step 3: tables and figures
configs/
  smoke.yaml             fake LLM, plumbing check
  pipecheck.yaml         real LLM, tiny budget
  pipecheck_claude.yaml  example of switching LLM by inheriting a config
  eoh_baseline.yaml      EoH, full budget, 3 seeds (main baseline)
  compare.yaml           EoH vs alternatives and ablations
  instances/v0.yaml      instance sets used for real runs (frozen at G2)
  instances/smoke.yaml   small instance sets for tests
notebooks/run_pipeline.ipynb
tests/test_core.py       evaluator unit tests
docs/log.md              decision log: add an entry for every decision that affects results
results/<experiment>/    run outputs (raw run folders are git-ignored)
```

---

## 4. Setup

**Python 3.10–3.12** (LLM4AD needs `numpy<2`).

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # add `jupyter` too if you use Jupyter in the browser rather than VS Code
python -m unittest discover tests      # 6 evaluator tests should pass
```

**API key.** Put it in a file called `.env` in the repo root (already git-ignored), never in a config or notebook you commit:

```
GEMINI_API_KEY=AIza...
```

Get a Gemini key from Google AI Studio (aistudio.google.com → Get API key). A project without billing uses the free tier.

**Notebook setup** (first cell of any notebook, wherever it sits in the repo):

```python
import os, sys, json
from pathlib import Path
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "code" / "run_experiment.py").exists())
sys.path.insert(0, str(ROOT / "code"))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import run_experiment, evaluate_final, summarise
from obpevo import config as C, core, instances, baselines
```

**Windows.** The pipeline runs natively on Windows (tested), with two differences: each candidate is scored in a freshly started Python process, which adds a few seconds per evaluation, and very small timeouts can kill valid candidates. Keep `timeout_seconds` at 30 or more. WSL (Ubuntu) is faster if you have it set up.

**After editing any file under `code/`, restart the kernel.** Python loads a module once per session; edits are invisible until restart.

---

## 5. Testing the pipeline (cheapest first)

| Step | Command | Checks | Time / cost |
|---|---|---|---|
| 1 | `python -m unittest discover tests` | evaluator correctness | seconds |
| 2 | `configs/smoke.yaml` (fake LLM) | plumbing, budgets, failure logging, outputs | ~2 min, free |
| 3 | `configs/pipecheck.yaml` (Gemini, 12 evals, small instances) | API key, real replies parse, token counts, cost tracking | ~1 min, cents notional |
| 4 | `configs/eoh_baseline.yaml`, one seed | a full run: real time and cost | ~35 min, ~US$0.17 notional |

The fake LLM returns canned heuristics (including deliberately broken ones) without any API calls. It tests plumbing only: its results say nothing about method quality.

**After step 3**, check in `results/pipecheck/eoh/seed0/`:
- `evals.csv`: mostly `ok` rows;
- `llm_calls.csv`: `status = ok` and `token_source = api`;
- `candidates.jsonl`: real heuristics, not just the template.

A single direct API call is the fastest key/model check:

```python
from obpevo.llm import make_backend
print(make_backend(C.load("configs/pipecheck.yaml")["llm"]).complete("Say hi")[0])
```

---

## 6. Running experiments

Every experiment is three steps, each a function (or a script with the same options):

```python
CONFIG = "configs/eoh_baseline.yaml"
idx  = run_experiment.main(CONFIG, seeds=[0])   # 1. search            -> runs index (DataFrame)
summ = evaluate_final.main(CONFIG)               # 2. held-out scoring  -> summary (DataFrame)
figs = summarise.main(CONFIG)                    # 3. tables + figures  -> figures folder
```

```bash
python code/run_experiment.py --config configs/eoh_baseline.yaml --seeds 0
python code/evaluate_final.py --config configs/eoh_baseline.yaml
python code/summarise.py      --config configs/eoh_baseline.yaml
```

- **Finished runs are skipped.** A run with a `summary.json` counts as done. Pass `force=True` (`--force`) to redo it, e.g. after a failed run.
- **Step 2 scores every finished seed** in the experiment together, so run seeds one at a time (e.g. on different days) and evaluate at the end.
- **Long runs block the notebook cell.** Check progress from a second notebook or terminal:

```python
run = ROOT / "results/eoh_baseline/eoh/seed0"
ev, calls = pd.read_csv(run / "evals.csv"), pd.read_csv(run / "llm_calls.csv")
print(f"{len(ev)}/300 evals | {len(calls)} calls ({(calls.status != 'ok').sum()} failed) "
      f"| best so far {ev.train_excess.min():.3%} (Best Fit 4.751%)")
```

**Free-tier limits.** A 300-evaluation EoH run makes about 300 LLM calls; ReEvo makes about 1.8 calls per evaluation. Check your daily request limit on the AI Studio rate-limit page. If the quota runs out mid-run, the run stops within 5 failed calls (`stop_reason: budget:llm_errors`) and keeps everything evaluated so far.

---

## 7. Configs

One YAML file fixes everything about an experiment. Example (`configs/eoh_baseline.yaml`, abridged):

```yaml
experiment: eoh_baseline                     # results folder: results/eoh_baseline/
instances: {spec: configs/instances/v0.yaml, dir: data/v0}

llm:
  provider: gemini
  model: gemini-3.5-flash-lite
  temperature: 1.0
  max_tokens: 8192
  extra: {reasoning_effort: low}
  price_in_per_1m: 0.30                      # USD per 1M tokens; check the provider's pricing page
  price_out_per_1m: 2.50

budget: {max_evals: 300, max_tokens: 1500000, max_minutes: 120}
parallel: {num_samplers: 2, num_evaluators: 2}
evaluation: {mode: funsearch, timeout_seconds: 60, train_set: train, prompt: llm4ad}
seeds: [0, 1, 2]
methods:
  - {name: eoh, params: {pop_size: 10}}
final_eval:
  sets: [val, test_id_5k, test_size_10k, test_cap500, test_cap500_scaled, test_shape_k1.5, test_shape_k5, test_uniform]
  timeout_seconds: 900
```

| Key | Meaning |
|---|---|
| `experiment` | results folder name. **Use a new name for every model or setting you compare**, or runs overwrite each other / get skipped as done. |
| `instances` | instance spec, and where generated sets are stored (built automatically on first run) |
| `llm` | provider, model, sampling settings, prices (see §8) |
| `budget` | per-run limits; the run stops at the first one reached. Keep `max_tokens` and `max_minutes` loose so `max_evals` is normally the binding limit. |
| `parallel` | LLM calls / evaluations in parallel (lower `num_samplers` if you hit rate limits) |
| `evaluation.mode` | `funsearch` (default, matches LLM4AD) or `anyfit` (see §11) |
| `evaluation.timeout_seconds` | per-candidate time limit; ≥ 30 on Windows |
| `evaluation.prompt` | `llm4ad` (default) or `informed` (tells the LLM the item distribution and capacity) |
| `evaluation.format_hint` | `true` (default): append formatting rules to every prompt (§11) |
| `seeds`, `methods` | runs = methods × seeds |
| `final_eval` | held-out sets and timeout for step 2 |

**Inheriting a config.** `base:` copies another config; only what you write replaces it:

```yaml
base: configs/eoh_baseline.yaml
experiment: eoh_baseline_38flash
llm: {provider: gemini, model: gemini-3.8-flash, max_tokens: 8192, extra: {reasoning_effort: low},
      price_in_per_1m: 0.75, price_out_per_1m: 3.75}
```

A new `provider` replaces the whole `llm` section, so nothing carries over from the base model.

**Overrides from a notebook** (no new file). Pass the same overrides to all three steps:

```python
OV = {"experiment": "eoh_pilot", "budget": {"max_evals": 60, "max_tokens": 400000, "max_minutes": 45}}
run_experiment.main("configs/eoh_baseline.yaml", seeds=[0], overrides=OV)
evaluate_final.main("configs/eoh_baseline.yaml", overrides=OV)
summarise.main("configs/eoh_baseline.yaml", overrides=OV)
```

---

## 8. Choosing the LLM

`provider` fills in the API address and which environment variable holds the key:

| `provider` | Key variable | Notes |
|---|---|---|
| `gemini` | `GEMINI_API_KEY` | our default; free tier on Flash / Flash-Lite models |
| `openai` | `OPENAI_API_KEY` | paid only; set `temperature: null` for models that only accept the default |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude; paid only (one-time trial credit for new accounts) |
| `deepseek` | `DEEPSEEK_API_KEY` | paid |
| `openrouter` | `OPENROUTER_API_KEY` | many providers' models with one key |
| `ollama` | none | local models at `localhost:11434` |
| `openai_compat` | your choice | any other OpenAI-style server: also give `base_url` and `api_key_env` |
| `fake` | none | offline stand-in for tests |

Gemini models with a free tier (Oct 2026): `gemini-3.5-flash-lite` (default), `gemini-3.1-flash-lite`, `gemini-3.6-flash`, `gemini-3.8-flash`. On the free tier `cost_usd` shows what the run *would* cost on the paid tier, which is the figure to report. Free-tier prompts may be used by Google to improve its products, so don't send anything confidential.

Pin exact model versions (not `-latest` aliases) so runs stay reproducible. Token counts include hidden reasoning tokens where the provider reports them; failed calls count zero tokens.

---

## 9. Methods and ablations

| Method | `name` | Notes |
|---|---|---|
| EoH | `eoh` | baseline; tested with Gemini |
| ReEvo | `reevo` | also asks the LLM for written reflections (~1.8 calls per evaluation) |
| FunSearch | `funsearch` | |
| (1+1)-EPS hill climbing | `hillclimb` | pure exploitation: always improves on the current best |
| Random sampling | `randsample` | pure exploration, no selection: the control |
| MCTS-AHD, HSEvo | `mcts_ahd`, `hsevo` | registered, untested: experimental |

Only EoH has been run with Gemini so far. **Run each other method once with `pipecheck.yaml` before its full run** to confirm Gemini's replies parse (FunSearch, hill climbing and random sampling use a different prompt style).

**Variants of a method** get their own `label`, results folder, line in plots and row in tables:

```yaml
methods:
  - {name: eoh, params: {pop_size: 10}}
  - {label: eoh_no_m2, name: eoh, params: {pop_size: 10, use_m2_operator: false}}
  - {name: hillclimb}
  - {name: randsample}
```

Unknown params are dropped and listed in each run's `config.json` (`dropped_kwargs`).

**EoH's operators.** EoH creates new heuristics in four ways: **E1** (as different as possible from the parents), **E2** (new heuristic on the parents' shared idea), **M1** (modify one parent), **M2** (keep the idea, change its parameters). Roughly a spectrum from exploration (E1) to exploitation (M2). E2, M1 and M2 can be switched off (`use_e2_operator`, `use_m1_operator`, `use_m2_operator`); E1 cannot.

**Ablations available now:**

| Ablation | Setting |
|---|---|
| EoH operators | `params: {use_m2_operator: false}` etc. |
| Population size | `params: {pop_size: 5 / 10 / 20}` (LLM4AD suggests ~10 for 200+ evals, ~20 for 1000+) |
| Problem information in the prompt | `evaluation: {prompt: informed}` |
| Formatting rules | `evaluation: {format_hint: false}` |
| Model / reasoning / temperature | `llm:` section |
| Empty-bin rule | `evaluation: {mode: anyfit}` |
| Budget | `budget: {max_evals: ...}` |

**Change one thing at a time** against a baseline with the same model, budget and seeds, and use at least 3 seeds per arm: LLM runs vary a lot between seeds.

---

## 10. Outputs and how to read them

**Per run** (`results/<experiment>/<label>/seed<k>/`):

| File | Contents |
|---|---|
| `config.json` | resolved config, method kwargs used/dropped, library versions, exact prompt text added |
| `llm_calls.csv` | one row per LLM call: tokens (`api` or `estimate`), latency, cost, status (with the provider's error message if it failed) |
| `evals.csv` | one row per candidate: status (`ok` / `error` / `timeout` / `crash`), score, train excess, error message |
| `candidates.jsonl` | every evaluated program |
| `per_instance.csv` | every candidate × training instance: bins, lower bound, excess |
| `best_program.py` | best program by training score (ties go to the earliest) |
| `summary.json` | counts, tokens, cost, stop reason, best result |
| `stdout.log`, `llm4ad_log_*/` | LLM4AD's own output |

**Per experiment** (`results/<experiment>/`): `runs_index.csv`, `final_per_instance.csv`, `final_summary.csv`, `summary_tables.md`, and `figures/` (PNG + PDF for LaTeX):
- `convergence_evals` / `convergence_tokens`: best training excess so far vs evaluations / tokens (median and range across seeds, Best Fit line);
- `test_sets`: held-out change vs Best Fit per method and seed, one panel per set.

**Key fields:**

| Field | Meaning |
|---|---|
| `stop_reason` | `budget:evals` is the normal ending. `budget:tokens` / `budget:wall_clock`: another limit bound first (raise it). `budget:llm_errors`: the API kept failing (quota, key). `error`: see `error.txt`. |
| `best_train_excess` | best candidate's excess on training (compare with Best Fit on the same set) |
| `n_calls_not_evaluated` | calls with no evaluation: candidates still being scored when the budget hit, unparseable replies, ReEvo reflections. 1–3 is normal for EoH. |
| `mean_excess` / `sd_excess` | held-out excess, averaged within each run, then mean ± sd across seeds |
| `delta_vs_bf_pp` | difference from Best Fit in percentage points; **negative = better** |
| `win_vs_bf` | share of instances where the program used fewer bins than Best Fit |

**Reading search progress.** Individual candidates go up and down; only the best-so-far matters. Programs that look different can pack identically (e.g. `item - bins` and `-(bins - item)`), so many candidates share exactly the same score. Use the diagnostic in §12 to tell a normal plateau from a search that isn't exploring.

---

## 11. Evaluation details

**How a heuristic is called.** Once per arriving item, with `bins` = the **remaining** capacities of only the bins the item fits in. The item goes into the bin with the highest score (`np.argmax`; ties go to the earliest bin). Output must be the same length as `bins` and contain no NaN (both are errors); `-np.inf` is allowed to rule a bin out.

**Two packing conventions** (`evaluation.mode`):
- `funsearch` (default): one bin per item is set up at the start, all at full capacity, so `bins` always includes empty bins. Identical to LLM4AD, FunSearch and EoH; we verified our scores match LLM4AD's exactly. Consequence: any heuristic that prefers emptier bins always opens a new bin and collapses to about one item per bin (~148% excess on our training set).
- `anyfit`: only open, non-empty bins are offered; a new bin opens only when none fit (Sgall's AnyFit class).

**Lower bound.** `ceil(sum(items) / capacity)` (Martello and Toth's L1), the bound used by FunSearch and EoH. It is not the true optimum, so part of every heuristic's excess is unavoidable.

**Formatting rules.** With `format_hint: true`, every prompt from every method ends with short rules: one complete function, the whole `def` line on one line, no helper functions, numpy only. LLM4AD's parser silently drops candidates with multi-line signatures and evolves the *wrong* function when a helper comes first; the rules prevent both. The text is identical for all methods (fair), recorded in each `config.json`, and is our one deviation from LLM4AD's prompts. Mention it in the report.

**Instance sets** (`configs/instances/v0.yaml`, generated deterministically from seed 2026):

| Set | Instances × items | Capacity | Items | Purpose |
|---|---|---|---|---|
| `train` | 32 × 200–2000 | 100 | Weibull(3) × 45 | the only set the search sees |
| `val` | 10 × 1000 | 100 | Weibull(3) × 45 | choosing between runs and configs |
| `test_id_5k` | 5 × 5000 | 100 | Weibull(3) × 45 | in-distribution (FunSearch/EoH standard) |
| `test_size_10k` | 5 × 10000 | 100 | Weibull(3) × 45 | size shift |
| `test_cap500` | 5 × 5000 | 500 | Weibull(3) × 45 | bigger bins, same items (near-trivial: all heuristics tie) |
| `test_cap500_scaled` | 5 × 5000 | 500 | Weibull(3) × 225 | everything ×5: scale-invariance check |
| `test_shape_k1.5` | 5 × 5000 | 100 | Weibull(1.5) × 45 | wider spread of item sizes |
| `test_shape_k5` | 5 × 5000 | 100 | Weibull(5) × 45 | tightly clustered item sizes |
| `test_uniform` | 5 × 1000 | 150 | Uniform[20, 100] | different distribution and capacity |

Each test set changes one thing from training, so a failure can be attributed to that change. The spec is frozen at G2: changing a set changes its data.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `skip eoh seed0 (done)` and old numbers | a previous run (even a failed one) counts as finished | `force=True` / `--force` |
| `ModuleNotFoundError: run_experiment` | notebook isn't finding the repo | use the `ROOT = next(...)` setup cell (§4) |
| Edits to `code/` have no effect | module already loaded | restart the kernel |
| Many `timeout` rows with `wall_s` ≈ the timeout (Windows) | process start-up per candidate | raise `evaluation.timeout_seconds` (30+ for small sets, 60+ for `train`) |
| `stop_reason: budget:llm_errors` | API kept failing; `llm_calls.csv` `status` has the message | 429 / quota: wait for the daily reset (midnight Pacific, 3–4 pm Singapore) or lower `num_samplers`; 400/403: model name or key |
| Thousands of calls, 0 evals, `budget:tokens` | a broken `llm.py` edit (calls fail instantly) | run the one-call wrapper test below; check indentation inside `LoggedLLM.draw_sample` |
| `UnboundLocalError` in `draw_sample` | a block pasted inside the `except:` branch | `if status == "ok":` must line up with `try:` / `except` |
| `FileExistsError` / `PermissionError` on `results/.../seed0` (Windows) | an old process holds files in the run folder | close Explorer windows there; end stray `python.exe` in Task Manager; restart |
| `best_train_excess` is `None` | no candidate ran successfully | check `stop_reason`, `llm_calls.csv`, `evals.csv`, `error.txt` |

**One-call test through the same wrapper the pipeline uses:**

```python
import tempfile
from obpevo.llm import make_backend, LoggedLLM
cfg = C.load("configs/eoh_baseline.yaml")
log = Path(tempfile.mkdtemp()) / "calls.csv"
w = LoggedLLM(make_backend(cfg["llm"]), log, price_in_per_1m=0.3, price_out_per_1m=2.5)
print(w.draw_sample("Say hi")[:100]); display(pd.read_csv(log))    # one row: status ok, token_source api
```

**Is the search exploring, or stuck?**

```python
ok = ev[ev.status == "ok"]
print(f"ok: {len(ok)} | same score as best: {(abs(ok.train_excess - ok.train_excess.min()) < 1e-9).sum()} "
      f"| distinct scores: {ok.train_excess.round(6).nunique()}")
```

Many candidates tied with Best Fit and few distinct scores mean the LLM keeps rewriting Best Fit (EoH discards same-score candidates as duplicates, so those evaluations are wasted).

---

## 13. Results so far

**EoH baseline, seed 0** (`gemini-3.5-flash-lite`, 300 evals, 34 min, 301 calls, 0 failed, US$0.17 notional):

- Training: **3.788%** excess vs Best Fit 4.751% (−0.96 pp). Stuck at Best Fit for ~150 evaluations (most candidates were Best Fit rewrites), then improved at ~165 and at eval 217.
- Held-out (Δ vs Best Fit, pp; negative = better): `val` −1.00, `test_id_5k` −1.95, `test_size_10k` −2.22, `test_uniform` −2.19, `test_shape_k1.5` −0.42, `test_cap500` −0.10, **`test_cap500_scaled` +1.29**, **`test_shape_k5` +30.69**.
- The two failures are overfitting with a clear mechanism: the evolved rule switches formula at `item / remaining capacity = 0.5`, which suits training's item sizes but sends tightly clustered items (k=5) into fresh bins (43% of bins hold one item, vs 0% for Best Fit); and it uses absolute units of space, so it behaves differently at capacity 500.

One seed only: run seeds 1 and 2 before drawing conclusions. Details in `docs/log.md`.

---

## 14. Known limitations and next steps

- Only EoH has been run with a real LLM; pipecheck the other methods before their full runs.
- Only each run's single best program goes to held-out evaluation.
- No operator attribution yet (which EoH operator produced each candidate) and no diversity metrics (clone rate, collapse rate, behavioural diversity). Both planned before the exploration/exploitation study.
- Planned: an LLM registry (`configs/llms.yaml`) as the single source of model details and prices, with `llm=` and `budget=` arguments on all three steps.
- MCTS-AHD and HSEvo are untested.
- Prices in configs were checked on 2026-10-25; re-check before trusting `cost_usd`.
