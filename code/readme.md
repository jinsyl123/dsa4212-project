# DSA4212 Project 2: LLM-guided evolution of online bin packing heuristics

We use LLM-driven search methods to evolve a heuristic for **online bin packing**, then compare the methods fairly. [LLM4AD](https://github.com/Optima-CityU/LLM4AD) supplies the search methods; **EoH** is our baseline. A thin layer of our own (`pipeline/obpevo/`) wraps every method so they are all measured the same way, following the principles of the BLADE benchmark.

```
                     ┌──────────────── one run = one method × one seed ────────────────┐
LLM API  ◄──────  LoggedLLM  ◄──────  LLM4AD method (EoH / ReEvo / FunSearch / ...)  ──────►  EvalRecorder  ──────►  OBPTask
(Gemini, ...)     logs every call      proposes priority() functions                         logs every candidate     packs the training
                  enforces token /                                                           enforces the eval        instances, returns
                  time budgets                                                               budget                   a score
```

> **Where this lives:** in the team repo, everything below is inside the `code/` folder: paths are relative to it, and commands run from it (`cd code`). The Python scripts are in `code/pipeline/`.
>
> **Reproduce the reported results in 3 commands** (no API key or LLM calls needed; Python 3.10–3.12, see §4), from the `code/` folder:
>
> ```bash
> pip install -r requirements.txt
> python pipeline/evaluate_final.py --config configs/eoh_baseline.yaml   # rescores the committed best programs on every held-out set
> python pipeline/summarise.py      --config configs/eoh_baseline.yaml   # rebuilds the tables and figures
> ```
>
> For runs made with another model, add the same `--llm` (and `--budget`) used for the search, e.g. `--llm openrouter-gemini-3.5-flash-lite`. Instance sets are generated automatically on first use. This reproduces `final_summary.csv`, `summary_tables.md` and the figures exactly. Re-running the _search_ does not reproduce the same programs: LLM sampling is random (see §6).

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

**The metric: excess over the lower bound.** For each instance, the lower bound is `ceil(total item size / capacity)`, the fewest bins any packing could possibly use. Excess = (bins used − lower bound) / lower bound, averaged over instances. **Lower is better.** Plain Best Fit scores 4.751% on our training set: the floor any evolved heuristic must beat. The stronger bar is **tuned Best Fit** (`tuned_bf`, §9): Best Fit with three knobs tuned, without an LLM, under the same 300-evaluation budget.

---

## 2. What the pipeline guarantees

Each method in LLM4AD has its own loop, logging and stopping rules. Comparing them directly would compare those differences, not the methods. Our layer enforces four rules for every method, without touching LLM4AD's code:

| Guarantee                            | How                                                                                                                                                                         | Where                          |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| **1. Same logging for every method** | Every candidate goes through one patched function (`SecureEvaluator.evaluate_program`), and every LLM call through one wrapper (`LoggedLLM`).                               | `recorder.py`, `llm.py`        |
| **2. Equal budgets**                 | A run stops at the first of `max_evals`, `max_tokens`, `max_minutes`, counted outside the method. Hitting the evaluation cap also blocks further LLM calls.                 | `recorder.py`, `llm.py`        |
| **3. Train/test separation**         | Search only ever sees the `train` set. Each run's best program is rescored afterwards on held-out sets it never saw.                                                        | `task.py`, `evaluate_final.py` |
| **4. Full records**                  | Every run saves its resolved config, library versions, every LLM call (tokens, cost, latency), every candidate program with its per-instance results, and the best program. | `run_experiment.py`            |

---

## 3. Repository layout

```
pipeline/
  obpevo/core.py         evaluator: packing, lower bound, score (single source of truth)
  obpevo/instances.py    deterministic instance generator (Weibull / uniform)
  obpevo/baselines.py    First / Best / Worst / Almost-Worst / Next Fit
  obpevo/task.py         LLM4AD task wrapper: template, task description, formatting rules for the LLM
  obpevo/llm.py          LLM clients (OpenAI-compatible, Anthropic, fake) + LoggedLLM wrapper
  obpevo/recorder.py     EvalRecorder: logs every candidate, enforces the eval budget
  obpevo/methods.py      registry of LLM4AD methods
  obpevo/tuned_bf.py     tuned Best Fit: the budget-matched baseline (no LLM)
  obpevo/config.py       YAML config loading: inheritance (base:), LLM/budget lookup, overrides
  make_instances.py      build instance sets from a spec
  run_experiment.py      step 1: search (every method x seed in a config)
  evaluate_final.py      step 2: score best programs + baselines on held-out sets
  summarise.py           step 3: tables and figures
configs/
  llms.yaml              every LLM we can use: provider, model ID, prices, settings (single source of truth)
  budgets.yaml           named budgets (smoke, check, pilot, medium, full, large)
  smoke.yaml             fake LLM, plumbing check
  pipecheck.yaml         real LLM, tiny budget
  pipecheck_claude.yaml  example of switching LLM in a config file
  eoh_baseline.yaml      EoH, full budget, 3 seeds (main baseline)
  compare.yaml           EoH vs alternatives and ablations
  instances/v0.yaml      instance sets used for real runs (frozen at G2)
  instances/smoke.yaml   small instance sets for tests
notebooks/run_pipeline.ipynb
tests/                   unit tests: evaluator, tuned Best Fit, tuned run outputs
docs/log.md              decision log: add an entry for every decision that affects results
results/<experiment>/    run outputs (raw run folders are git-ignored)
```

---

## 4. Setup

**Python 3.10–3.12** (LLM4AD needs `numpy<2`, which has no wheels for 3.13+). Check with `python --version`; if you have another version, install 3.12 alongside it:

| System         | Install Python 3.12                 | Create and activate the environment                                 |
| -------------- | ----------------------------------- | ------------------------------------------------------------------- |
| Windows        | `winget install Python.Python.3.12` | `py -3.12 -m venv .venv` then `.venv\Scripts\activate`              |
| macOS          | `brew install python@3.12`          | `python3.12 -m venv .venv` then `source .venv/bin/activate`         |
| Anaconda (any) | (included)                          | `conda create -n dsa4212 python=3.12` then `conda activate dsa4212` |

Then, with the environment active:

```bash
pip install -r requirements.txt        # installs LLM4AD at the pinned commit; add `jupyter` for browser notebooks
python -m unittest discover tests      # 14 tests should pass
```

Install LLM4AD **only** through `requirements.txt`: `pip install llm4ad` gives an older release (1.0.0) without ReEvo, MCTS-AHD or HSEvo.

**API key.** Put it in a file called `.env` in the `code/` folder (already git-ignored), never in a config or notebook you commit:

```
GEMINI_API_KEY=AIza...
```

Get a Gemini key from Google AI Studio (aistudio.google.com → Get API key). A project without billing uses the free tier.

**Notebook setup** (first cell of any notebook, wherever it sits in the repo):

```python
import os, sys, json
from pathlib import Path
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pipeline" / "run_experiment.py").exists())
sys.path.insert(0, str(ROOT / "pipeline"))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import run_experiment, evaluate_final, summarise
from obpevo import config as C, core, instances, baselines
```

**Windows.** The pipeline runs natively on Windows (tested), with two differences: each candidate is scored in a freshly started Python process, which adds a few seconds per evaluation, and very small timeouts can kill valid candidates. Keep `timeout_seconds` at 30 or more. WSL (Ubuntu) is faster if you have it set up.

**After editing any file under `pipeline/`, restart the kernel.** Python loads a module once per session; edits are invisible until restart.

---

## 5. Testing the pipeline (cheapest first)

| Step | Command                                                      | Checks                                                   | Time / cost                |
| ---- | ------------------------------------------------------------ | -------------------------------------------------------- | -------------------------- |
| 1    | `python -m unittest discover tests`                          | evaluator correctness                                    | seconds                    |
| 2    | `configs/smoke.yaml` (fake LLM)                              | plumbing, budgets, failure logging, outputs              | ~2 min, free               |
| 3    | `configs/pipecheck.yaml` (Gemini, 12 evals, small instances) | API key, real replies parse, token counts, cost tracking | ~1 min, cents notional     |
| 4    | `configs/eoh_baseline.yaml`, one seed                        | a full run: real time and cost                           | ~35 min, ~US$0.17 notional |

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
python pipeline/run_experiment.py --config configs/eoh_baseline.yaml --seeds 0
python pipeline/evaluate_final.py --config configs/eoh_baseline.yaml
python pipeline/summarise.py      --config configs/eoh_baseline.yaml
```

**Trying another model or budget** needs no new config: pass `llm=` and `budget=` (names from `configs/llms.yaml` and `configs/budgets.yaml`) to all three steps. The results folder is named after them, so different choices never overwrite each other:

```python
args = dict(llm="gemini-3.8-flash", budget="pilot")
run_experiment.main(CONFIG, seeds=[0], **args)   # -> results/eoh_baseline__gemini-3.8-flash__pilot/
evaluate_final.main(CONFIG, **args)
summarise.main(CONFIG, **args)
```

**What reproduces exactly and what doesn't.** The LLM samples with temperature 1.0 and is not seeded, so re-running a search gives different programs and scores; `seed` only fixes LLM4AD's own selection steps. That is why we run several seeds and report their spread. Everything _after_ the search is deterministic: given the committed `best_program.py` files, `evaluate_final` and `summarise` reproduce the held-out results, tables and figures exactly (verified on a fresh clone). The instance sets are generated deterministically from `configs/instances/v0.yaml`.

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
experiment: eoh_baseline # results folder: results/eoh_baseline/
instances: { spec: configs/instances/v0.yaml, dir: data/v0 }

llm: gemini-3.5-flash-lite # a name from configs/llms.yaml (model, prices, settings)
budget: full # a name from configs/budgets.yaml (300 evals, 1.5M tokens, 120 min)
parallel: { num_samplers: 2, num_evaluators: 2 }
evaluation:
  { mode: funsearch, timeout_seconds: 60, train_set: train, prompt: llm4ad }
seeds: [0, 1, 2]
methods:
  - { name: eoh, params: { pop_size: 10 } }
final_eval:
  sets:
    [
      val,
      test_id_5k,
      test_size_10k,
      test_cap500,
      test_cap500_scaled,
      test_shape_k1.5,
      test_shape_k5,
      test_uniform,
    ]
  timeout_seconds: 900
```

| Key                          | Meaning                                                                                                                                                                                                                                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment`                 | results folder name. **Use a new name for every model or setting you compare**, or runs overwrite each other / get skipped as done.                                                                                                                              |
| `instances`                  | instance spec, and where generated sets are stored (built automatically on first run)                                                                                                                                                                            |
| `llm`                        | a name from `configs/llms.yaml`, or a dict (see §8)                                                                                                                                                                                                              |
| `budget`                     | a name from `configs/budgets.yaml`, or a dict `{max_evals, max_tokens, max_minutes}`. The run stops at the first limit reached; keep `max_tokens` and `max_minutes` loose so `max_evals` normally binds. A dict replaces the preset entirely, so give all three. |
| `parallel`                   | LLM calls / evaluations in parallel (lower `num_samplers` if you hit rate limits)                                                                                                                                                                                |
| `evaluation.mode`            | `funsearch` (default, matches LLM4AD) or `anyfit` (see §11)                                                                                                                                                                                                      |
| `evaluation.timeout_seconds` | per-candidate time limit; ≥ 30 on Windows                                                                                                                                                                                                                        |
| `evaluation.prompt`          | `llm4ad` (default, LLM4AD's wording), `informed_rules` (how bins are offered: remaining capacities, empty bins included), `informed_dist` (the training distribution and capacity), or `informed` (both)                                                         |
| `evaluation.format_hint`     | `true` (default): append formatting rules to every prompt (§11)                                                                                                                                                                                                  |
| `seeds`, `methods`           | runs = methods × seeds                                                                                                                                                                                                                                           |
| `final_eval`                 | held-out sets and timeout for step 2                                                                                                                                                                                                                             |

**Inheriting a config.** `base:` copies another config; only what you write replaces it:

```yaml
base: configs/eoh_baseline.yaml
experiment: eoh_baseline_38flash
llm: gemini-3.8-flash
```

Naming a different model replaces the whole `llm` section, so nothing carries over from the base model.

**From a notebook** (no new file), every function takes the same optional arguments. Pass the same ones to all three steps:

| Argument     | Example                                                                    | Effect                                                       |
| ------------ | -------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `llm`        | `"gemini-3.8-flash"` or `{"name": "gemini-3.8-flash", "temperature": 0.5}` | model from the registry, optionally adjusted                 |
| `budget`     | `"pilot"` or `{"max_evals": 80, "max_tokens": 500000, "max_minutes": 60}`  | preset or custom budget                                      |
| `experiment` | `"eoh_80"`                                                                 | results folder name (derived from `llm`/`budget` if omitted) |
| `overrides`  | `{"evaluation": {"prompt": "informed"}, "methods": [...]}`                 | any other setting                                            |

`overrides` does **not** rename the results folder: pass `experiment=` with it, or runs overwrite each other.

---

## 8. Choosing the LLM

**`configs/llms.yaml` is the single source of truth** for every model we can use: provider, model ID, prices and settings. Configs and functions refer to a model by its name only:

```yaml
gemini-3.8-flash:
  provider: gemini
  model: gemini-3.8-flash
  price_in_per_1m: 0.75 # USD per 1M tokens, paid tier; doubles from 2027-01-01
  price_out_per_1m: 3.75
  max_tokens: 8192
  extra: { reasoning_effort: low }
  notes: free tier available; stronger model
  checked: 2026-10-25 (ai.google.dev/pricing)
```

| Name                    | Provider  | Price in / out (USD per 1M) | Notes                      |
| ----------------------- | --------- | --------------------------- | -------------------------- |
| `gemini-3.5-flash-lite` | Gemini    | 0.30 / 2.50                 | default; free tier         |
| `gemini-3.1-flash-lite` | Gemini    | 0.25 / 1.50                 | cheapest; free tier        |
| `gemini-3.8-flash`      | Gemini    | 0.75 / 3.75                 | stronger; free tier        |
| `claude-haiku-5-5`      | Anthropic | set before use              | paid only                  |
| `deepseek-chat`         | DeepSeek  | 0.27 / 1.10 (unverified)    | paid only                  |
| `fake`                  | none      | 0 / 0                       | offline stand-in for tests |

See them all from a notebook with `pd.DataFrame(C.llm_registry()).T`. Unknown names fail immediately with the list of valid ones.

**Adding a model:** add an entry to `configs/llms.yaml` with `provider`, `model`, both prices and a `checked` date. Prices change, so re-check the provider's pricing page and update `checked` when you do. Every result records the model name (`llm_name` in `summary.json`), so old results stay attributable.

**Providers** (the `provider` field) fill in the API address and which environment variable holds the key:

| `provider`      | Key variable         | Notes                                                                      |
| --------------- | -------------------- | -------------------------------------------------------------------------- |
| `gemini`        | `GEMINI_API_KEY`     | free tier on Flash / Flash-Lite models                                     |
| `openai`        | `OPENAI_API_KEY`     | paid only; set `temperature: null` for models that only accept the default |
| `anthropic`     | `ANTHROPIC_API_KEY`  | Claude; paid only (one-time trial credit for new accounts)                 |
| `deepseek`      | `DEEPSEEK_API_KEY`   | paid                                                                       |
| `openrouter`    | `OPENROUTER_API_KEY` | many providers' models with one key                                        |
| `ollama`        | none                 | local models at `localhost:11434`                                          |
| `openai_compat` | your choice          | any other OpenAI-style server: also give `base_url` and `api_key_env`      |
| `fake`          | none                 | offline stand-in for tests                                                 |

On the free tier, `cost_usd` shows what the run _would_ cost on the paid tier, which is the figure to report. Free-tier prompts may be used by Google to improve its products, so don't send anything confidential. Pin exact model versions (not `-latest` aliases) so runs stay reproducible. Token counts include hidden reasoning tokens where the provider reports them; failed calls count zero tokens.

## 9. Methods and ablations

| Method                  | `name`              | Notes                                                                 |
| ----------------------- | ------------------- | --------------------------------------------------------------------- |
| EoH                     | `eoh`               | baseline; tested with Gemini                                          |
| ReEvo                   | `reevo`             | also asks the LLM for written reflections (~1.8 calls per evaluation) |
| FunSearch               | `funsearch`         |                                                                       |
| (1+1)-EPS hill climbing | `hillclimb`         | pure exploitation: always improves on the current best                |
| Random sampling         | `randsample`        | pure exploration, no selection: the control                           |
| MCTS-AHD, HSEvo         | `mcts_ahd`, `hsevo` | registered, untested: experimental                                    |
| Tuned Best Fit (no LLM) | `tuned_bf`          | the budget-matched baseline: Best Fit's 3 knobs, random search (below) |

Only EoH has been run with Gemini so far. **Run each other method once with `pipecheck.yaml` before its full run** to confirm Gemini's replies parse (FunSearch, hill climbing and random sampling use a different prompt style).

**Variants of a method** get their own `label`, results folder, line in plots and row in tables:

```yaml
methods:
  - { name: eoh, params: { pop_size: 10 } }
  - {
      label: eoh_no_m2,
      name: eoh,
      params: { pop_size: 10, use_m2_operator: false },
    }
  - { name: hillclimb }
  - { name: randsample }
```

Unknown params are dropped and listed in each run's `config.json` (`dropped_kwargs`).

**Tuned Best Fit (`tuned_bf`): the budget-matched baseline.** Beating plain Best Fit is weak evidence, so every LLM method is also compared with a Best Fit whose knobs get the same number of scored tries. Any score that only falls as the space left `r = bins - item` grows packs exactly like Best Fit, so the variant adds the smallest change that can differ: a penalty for leaving a small gap, the strategy FunSearch reports for its Weibull heuristic.

```python
r = bins - item
return -r - penalty * ((r > lo) & (r < hi))      # lo=0, hi=0, penalty=0 is plain Best Fit
```

The knobs are integers in item units: `lo` 0–10, `hi` lo+2–40, `penalty` 1–100. Like an evolved program, the rule never sees the capacity. The tuner scores Best Fit first (every LLM method also starts from the Best Fit template), then `max_evals − 1` distinct random settings, on `train` only. It makes no LLM calls, takes about a minute per seed, and writes the same run files as an LLM method (with an empty `llm_calls.csv`), so steps 2 and 3 treat it like any other method. Its `summary.json` also records `best_params`. To change the ranges, use `params: { space: { lo: [0, 10], hi_max: 40, penalty: [1, 100] } }`.

**EoH's operators.** EoH creates new heuristics in four ways: **E1** (as different as possible from the parents), **E2** (new heuristic on the parents' shared idea), **M1** (modify one parent), **M2** (keep the idea, change its parameters). Roughly a spectrum from exploration (E1) to exploitation (M2). E2, M1 and M2 can be switched off (`use_e2_operator`, `use_m1_operator`, `use_m2_operator`); E1 cannot.

**Ablations available now:**

| Ablation                          | Setting                                                                                                                                   |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| EoH operators                     | `params: {use_m2_operator: false}` etc.                                                                                                   |
| Population size                   | `params: {pop_size: 5 / 10 / 20}` (LLM4AD suggests ~10 for 200+ evals, ~20 for 1000+)                                                     |
| Problem information in the prompt | `evaluation: {prompt: informed_rules / informed_dist / informed}`: rules only, distribution only, or both, so an effect can be attributed |
| Formatting rules                  | `evaluation: {format_hint: false}`                                                                                                        |
| Model                             | `llm="gemini-3.8-flash"`                                                                                                                  |
| Reasoning / temperature           | `llm={"name": "gemini-3.5-flash-lite", "extra": {"reasoning_effort": "medium"}}` or `{"name": ..., "temperature": 0.5}`                   |
| Empty-bin rule                    | `evaluation: {mode: anyfit}`                                                                                                              |
| Budget                            | `budget="medium"` / `"full"` / `"large"`                                                                                                  |

**Change one thing at a time** against a baseline with the same model, budget and seeds, and use at least 3 seeds per arm: LLM runs vary a lot between seeds.

---

## 10. Outputs and how to read them

**Per run** (`results/<experiment>/<label>/seed<k>/`):

| File                          | Contents                                                                                                                   |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `config.json`                 | resolved config, method kwargs used/dropped, library versions, exact prompt text added                                     |
| `llm_calls.csv`               | one row per LLM call: tokens (`api` or `estimate`), latency, cost, status (with the provider's error message if it failed) |
| `evals.csv`                   | one row per candidate: status (`ok` / `error` / `timeout` / `crash`), score, train excess, error message                   |
| `candidates.jsonl`            | every evaluated program                                                                                                    |
| `per_instance.csv`            | every candidate × training instance: bins, lower bound, excess                                                             |
| `best_program.py`             | best program by training score (ties go to the earliest)                                                                   |
| `summary.json`                | counts, tokens, cost, stop reason, best result                                                                             |
| `stdout.log`, `llm4ad_log_*/` | LLM4AD's own output                                                                                                        |

**In git:** `summary.json`, `best_program.py`, `evals.csv` and `llm_calls.csv` of each run (enough to rebuild every table and figure); the rest of each run folder stays local. Fake-LLM and pipecheck results (`results/smoke/`, `results/pipecheck*/`) are not committed, since they are plumbing checks, not findings.

**Per experiment** (`results/<experiment>/`): `runs_index.csv`, `final_per_instance.csv`, `final_summary.csv`, `summary_tables.md`, and `figures/` (PNG + PDF for LaTeX):

- `convergence_evals` / `convergence_tokens`: best training excess so far vs evaluations / tokens (median and range across seeds, Best Fit line);
- `test_sets`: held-out change vs Best Fit per method and seed, one panel per set.

**Key fields:**

| Field                       | Meaning                                                                                                                                                                                             |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stop_reason`               | `budget:evals` is the normal ending. `budget:tokens` / `budget:wall_clock`: another limit bound first (raise it). `budget:llm_errors`: the API kept failing (quota, key). `error`: see `error.txt`. |
| `best_train_excess`         | best candidate's excess on training (compare with Best Fit on the same set)                                                                                                                         |
| `best_params`               | `tuned_bf` runs only: the winning knob setting                                                                                                                                                      |
| `n_calls_not_evaluated`     | calls with no evaluation: candidates still being scored when the budget hit, unparseable replies, ReEvo reflections. 1–3 is normal for EoH.                                                         |
| `mean_excess` / `sd_excess` | held-out excess, averaged within each run, then mean ± sd across seeds                                                                                                                              |
| `delta_vs_bf_pp`            | difference from Best Fit in percentage points; **negative = better**                                                                                                                                |
| `win_vs_bf`                 | share of instances where the program used fewer bins than Best Fit                                                                                                                                  |

**Reading search progress.** Individual candidates go up and down; only the best-so-far matters. Programs that look different can pack identically (e.g. `item - bins` and `-(bins - item)`), so many candidates share exactly the same score. Use the diagnostic in §12 to tell a normal plateau from a search that isn't exploring.

---

## 11. Evaluation details

**How a heuristic is called.** Once per arriving item, with `bins` = the **remaining** capacities of only the bins the item fits in. The item goes into the bin with the highest score (`np.argmax`; ties go to the earliest bin). Output must be the same length as `bins` and contain no NaN (both are errors); `-np.inf` is allowed to rule a bin out.

**Two packing conventions** (`evaluation.mode`):

- `funsearch` (default): one bin per item is set up at the start, all at full capacity, so `bins` always includes empty bins. Identical to LLM4AD, FunSearch and EoH; we verified our scores match LLM4AD's exactly. Consequence: any heuristic that prefers emptier bins always opens a new bin and collapses to about one item per bin (~148% excess on our training set).
- `anyfit`: only open, non-empty bins are offered; a new bin opens only when none fit (Sgall's AnyFit class).

**Lower bound.** `ceil(sum(items) / capacity)` (Martello and Toth's L1), the bound used by FunSearch and EoH. It is not the true optimum, so part of every heuristic's excess is unavoidable.

**Formatting rules.** With `format_hint: true`, every prompt from every method ends with short rules: one complete function, the whole `def` line on one line, no helper functions, numpy only. LLM4AD's parser silently drops candidates with multi-line signatures and evolves the _wrong_ function when a helper comes first; the rules prevent both. The text is identical for all methods (fair), recorded in each `config.json`, and is our one deviation from LLM4AD's prompts. Mention it in the report.

**Instance sets** (`configs/instances/v0.yaml`, generated deterministically from seed 2026):

| Set                  | Instances × items | Capacity | Items             | Purpose                                                                                                                                                      |
| -------------------- | ----------------- | -------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `train`              | 32 × 200–2000     | 100      | Weibull(3) × 45   | the only set the search sees                                                                                                                                 |
| `val`                | 10 × 1000         | 100      | Weibull(3) × 45   | choosing between configurations (e.g. models, prompts); never seen by the search. Within a run, the best program is picked on training, as in the EoH papers |
| `test_id_5k`         | 5 × 5000          | 100      | Weibull(3) × 45   | in-distribution (FunSearch/EoH standard)                                                                                                                     |
| `test_size_10k`      | 5 × 10000         | 100      | Weibull(3) × 45   | size shift                                                                                                                                                   |
| `test_cap500`        | 5 × 5000          | 500      | Weibull(3) × 45   | bigger bins, same items (near-trivial: all heuristics tie)                                                                                                   |
| `test_cap500_scaled` | 5 × 5000          | 500      | Weibull(3) × 225  | everything ×5: scale-invariance check                                                                                                                        |
| `test_shape_k1.5`    | 5 × 5000          | 100      | Weibull(1.5) × 45 | wider spread of item sizes                                                                                                                                   |
| `test_shape_k5`      | 5 × 5000          | 100      | Weibull(5) × 45   | tightly clustered item sizes                                                                                                                                 |
| `test_uniform`       | 5 × 1000          | 150      | Uniform[20, 100]  | different distribution and capacity                                                                                                                          |

Each test set changes one thing from training, so a failure can be attributed to that change. The spec is frozen at G2: changing a set changes its data.

---

## 12. Troubleshooting

| Symptom                                                                | Cause                                                      | Fix                                                                                                                            |
| ---------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `skip eoh seed0 (done)` and old numbers                                | a previous run (even a failed one) counts as finished      | `force=True` / `--force`                                                                                                       |
| `ModuleNotFoundError: run_experiment`                                  | notebook isn't finding the repo                            | use the `ROOT = next(...)` setup cell (§4)                                                                                     |
| Edits to `pipeline/` have no effect                                    | module already loaded                                      | restart the kernel                                                                                                             |
| Many `timeout` rows with `wall_s` ≈ the timeout (Windows)              | process start-up per candidate                             | raise `evaluation.timeout_seconds` (30+ for small sets, 60+ for `train`)                                                       |
| `stop_reason: budget:llm_errors`                                       | API kept failing; `llm_calls.csv` `status` has the message | 429 / quota: wait for the daily reset (midnight Pacific, 3–4 pm Singapore) or lower `num_samplers`; 400/403: model name or key |
| Thousands of calls, 0 evals, `budget:tokens`                           | a broken `llm.py` edit (calls fail instantly)              | run the one-call wrapper test below; check indentation inside `LoggedLLM.draw_sample`                                          |
| `UnboundLocalError` in `draw_sample`                                   | a block pasted inside the `except:` branch                 | `if status == "ok":` must line up with `try:` / `except`                                                                       |
| `FileExistsError` / `PermissionError` on `results/.../seed0` (Windows) | an old process holds files in the run folder               | close Explorer windows there; end stray `python.exe` in Task Manager; restart                                                  |
| `KeyError: LLM '...' is not in configs/llms.yaml`                      | typo or a model not yet added                              | use a listed name, or add an entry (§8)                                                                                        |
| `best_train_excess` is `None`                                          | no candidate ran successfully                              | check `stop_reason`, `llm_calls.csv`, `evals.csv`, `error.txt`                                                                 |

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

**EoH baseline: 3 seeds** (`openrouter-gemini-3.5-flash-lite`, 300 evals each, pinned LLM4AD, US$0.49 in total; `results/eoh_baseline__openrouter-gemini-3.5-flash-lite/`):

- Training: two seeds never improved on Best Fit (4.751%: their best program was the first one evaluated); one reached 4.507% (−0.24 pp) at eval 66, then plateaued. Mean 4.670 ± 0.141%.
- Held-out: mean difference from Best Fit within ±0.1 pp on every set (`val` −0.10, `test_id_5k` −0.06, `test_uniform` −0.10, `test_shape_k5` +0.02, `test_cap500_scaled` +0.04).
- Reading: with this model and budget, EoH mostly stalls at Best Fit. The likely cause is the LLM repeatedly rewriting Best Fit, which EoH discards as duplicates (to be quantified with the clone-rate diagnostics).

**Tuned Best Fit: 3 seeds** (`tuned_bf`, 300 evaluations each, no LLM, about 2 min per seed, US$0; same results folder):

- Training: **2.427 ± 0.051%** (Best Fit 4.751%, EoH above 4.670 ± 0.141%). All three seeds land on nearly the same knobs: `lo` 3–4, `hi` 31, `penalty` 73–97, i.e. avoid leaving a gap of about 4–30 units, opening a fresh bin instead if needed.
- Held-out (Δ vs Best Fit, pp, mean of 3 seeds): `val` −2.40, `test_id_5k` −2.48, `test_size_10k` −2.45, `test_uniform` −2.01 (fewer bins than Best Fit on every instance of these four sets), `test_cap500` −0.02; **`test_shape_k1.5` +1.19**, **`test_cap500_scaled` +0.77**, **`test_shape_k5` +22.67**.
- Reading: a three-knob Best Fit with the same evaluation budget beats every EoH run so far on training, including the 25 Oct run (3.788%), and on the held-out Weibull sets at capacity 100; the 25 Oct run is slightly better on `test_uniform` (−2.19 vs −2.01 pp). It fails on the same two shifts as the 25 Oct run (`test_shape_k5` +22.7 vs +30.7 pp, `test_cap500_scaled` +0.77 vs +1.29 pp), plus `test_shape_k1.5`, where that run gained. So these failures do not need an LLM: tuning three knobs on the training distribution produces them too.

**EoH baseline, first run (25 Oct)** (`gemini-3.5-flash-lite`, 300 evals, 34 min, 301 calls, 0 failed, US$0.17 notional):

- Training: **3.788%** excess vs Best Fit 4.751% (−0.96 pp). Stuck at Best Fit for ~150 evaluations (most candidates were Best Fit rewrites), then improved at ~165 and at eval 217.
- Held-out (Δ vs Best Fit, pp; negative = better): `val` −1.00, `test_id_5k` −1.95, `test_size_10k` −2.22, `test_uniform` −2.19, `test_shape_k1.5` −0.42, `test_cap500` −0.10, **`test_cap500_scaled` +1.29**, **`test_shape_k5` +30.69**.
- The two failures are _consistent with_ a specific mechanism, which we treat as a hypothesis to test: the evolved rule switches formula at `item / remaining capacity = 0.5`, which suits training's item sizes but may send tightly clustered items (k=5) into fresh bins (43% of bins held one item, vs 0% for Best Fit), and it uses absolute units of space, so it may behave differently at capacity 500.
- Supporting check: rescoring the same program in `anyfit` mode (no empty bins offered while an open bin fits) removes both failures (`test_shape_k5` 8.15%, `test_cap500_scaled` 3.85%) and about half the training gain (4.34%). This points to the rule opening new bins too eagerly, rather than general overfitting.

Reproduce: `python pipeline/evaluate_final.py --config configs/eoh_baseline.yaml --experiment eoh_baseline_25oct` (then `summarise.py` with the same arguments).

This run used LLM4AD 1.0.0 (PyPI). Later runs (pinned LLM4AD, partly cut short by the free-tier quota) found heuristics much closer to Best Fit: small gains, no large failures. Run-to-run variance is large, so conclusions wait for three full-budget seeds (in progress via OpenRouter). Details in `docs/log.md`.

---

## 14. Known limitations and next steps

- Only EoH has been run with a real LLM; pipecheck the other methods before their full runs.
- Only each run's single best program (by training score) goes to held-out evaluation. Selecting among the top programs on `val` instead is a possible extension.
- No operator attribution yet (which EoH operator produced each candidate) and no diversity metrics (clone rate, collapse rate, behavioural diversity). Both planned before the exploration/exploitation study.
- MCTS-AHD and HSEvo are untested.
- `tuned_bf`'s knobs are in absolute item units, tuned at capacity 100 (like the evolved programs). A variant with knobs relative to the bin size, or rescoring in `anyfit` mode, would help separate the two failure hypotheses (hard-coded capacity vs fit to the size distribution).
- Prices in `configs/llms.yaml` were checked on 2026-10-25; re-check before trusting `cost_usd`. `claude-haiku-5-5` has no price set yet.
