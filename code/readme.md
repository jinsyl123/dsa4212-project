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

**Which bound.** The lower bound above is Martello and Toth's L1. It is a bound, not the true minimum: where many items exceed half a bin it falls well short (on `test_scale65`, about 10%). The search scores on L1, as FunSearch and EoH do (on `train` L2 equals L1). Held-out results in the report are excess over the tighter L2 bound on every set (#12 decision 7); see §11, Lower bound.

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
  obpevo/integrity.py    static checks on candidates (imports, randomness, state, exec/eval/open), flag only
  obpevo/methods.py      registry of LLM4AD methods
  obpevo/tuned_bf.py     tuned Best Fit: the budget-matched baseline (no LLM)
  obpevo/config.py       YAML config loading: inheritance (base:), LLM/budget lookup, overrides
  make_instances.py      build instance sets from a spec
  run_experiment.py      step 1: search (every method x seed in a config)
  evaluate_final.py      step 2: score best programs + baselines on held-out sets
  summarise.py           step 3: tables and figures
  check_integrity.py     rescore best programs twice + with shuffled instances; writes integrity.csv
  paired_stats.py        each rule vs Best Fit / tuned Best Fit, instance by instance: sign test + bootstrap CI
configs/
  llms.yaml              every LLM we can use: provider, model ID, prices, settings (single source of truth)
  budgets.yaml           named budgets (smoke, check, pilot, medium, full, large)
  smoke.yaml             fake LLM, plumbing check
  pipecheck.yaml         real LLM, tiny budget
  pipecheck_claude.yaml  example of switching LLM in a config file
  eoh_baseline.yaml      EoH + tuned Best Fit, full budget, seeds 0–5 (main baseline)
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

**Paired statistics** (after step 2): `python pipeline/paired_stats.py --config configs/eoh_baseline.yaml` (same options) compares every rule with Best Fit and tuned Best Fit on the same instances and writes `paired_stats.csv`: wins/ties/losses, a two-sided sign test, the mean difference in percentage points with a 95% bootstrap interval, and a verdict: better / worse only when the interval excludes 0 **and** the sign test gives p < 0.05, tie when identical on every instance, otherwise unclear. Each seed is compared on its own, because runs of the same method differ a lot. Excess is over the L2 bound by default, as in the report (#12 decision 7), falling back to L1 for older results without it; `--metric excess` compares over L1 (written to `paired_stats_excess.csv`). Against Best Fit, wins/ties/losses count bins, so they are the same under either bound. Tuned Best Fit has several seeds, so it is averaged per instance first: against it, wins/ties/losses compare with that average, not with one real packing. An instance the reference lacks is left out with a warning. Every test set has 10 instances (#12), so a rule that beats Best Fit on all 10 gives p = 0.002 (9–1: 0.021); with 5 it could not go below 0.0625. There is no correction for running many comparisons: look for patterns across seeds and sets rather than single verdicts.

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
      test_size_1k,
      test_size_10k,
      test_cap200,
      test_cap500,
      test_cap500_scaled,
      test_scale25,
      test_scale65,
      test_shape_k1.5,
      test_shape_k5,
      test_uniform,
      test_control,
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
| `evaluation.prompt`          | `llm4ad` (default, LLM4AD's wording), `informed_rules` (task framing plus how bins are offered: "online 1D bin packing, minimise bins", remaining capacities, empty bins included; the default prompt says none of this), `informed_dist` (the training distribution and capacity), or `informed` (both)                                                         |
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
| Random sampling         | `randsample`        | no evolution (LLM4AD's fixed FunSearch-style prompt every call); best on train kept: the control |
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

**Tuned Best Fit (`tuned_bf`): the budget-matched baseline.** Beating plain Best Fit is weak evidence, so every LLM method is also compared with a Best Fit whose knobs get the same number of scored tries. Any score that strictly falls as the space left `r = bins - item` grows packs exactly like Best Fit, so the variant adds the smallest change that can differ: a penalty for leaving a small gap. This is the strategy the FunSearch paper describes for several of its discovered heuristics, illustrated with its OR-dataset heuristic (Fig. 6).

```python
r = bins - item
return -r - penalty * ((r > lo) & (r < hi))      # lo=0, hi=0, penalty=0 is plain Best Fit
```

The knobs are integers in item units: `lo` 0–10, `hi` lo+2–40, `penalty` 1–100. Like an evolved program, the rule is never given the capacity. (In `funsearch` mode an evolved program could still read it from `max(bins)`, since an empty bin is always offered; this rule doesn't.) The tuner scores Best Fit first (every LLM method also starts from the Best Fit template), then `max_evals − 1` distinct random settings, on `train` only. It makes no LLM calls, takes about 2.5 minutes per seed (300 tries), and writes the same run files as an LLM method (with an empty `llm_calls.csv`), so steps 2 and 3 treat it like any other method. Its `summary.json` also records `best_params`. To change the ranges, use `params: { space: { lo: [0, 10], hi_max: 40, penalty: [1, 100] } }`. An unknown key in `space`, or a range key put directly under `params`, stops the run with an error before anything is written.

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
| Training-data variety             | `evaluation: {train_set: train_mixed}` (Weibull shapes 2 and 4; a set's `mix:` list alternates specs across its instances)               |
| Budget                            | `budget="medium"` / `"full"` / `"large"`                                                                                                  |

**Change one thing at a time** against a baseline with the same model, budget and seeds, and use at least 3 seeds per arm: LLM runs vary a lot between seeds.

**The planned ablations (#7)** each have a config that inherits `eoh_baseline.yaml` (`base:`) and changes one setting. All use `openrouter-gemini-3.5-flash-lite`, the `full` budget and seeds 0–4, so the main baseline (seeds 0–5) is the comparison row.

| Config                    | Changes                                   | Status (#7)        |
| ------------------------- | ----------------------------------------- | ------------------ |
| `abl_anyfit.yaml`         | empty bins hidden during search           | main               |
| `abl_informed_rules.yaml` | prompt states the task and how bins are offered (incl. empty bins) | main               |
| `abl_mixed_shapes.yaml`   | training on Weibull shapes 2 and 4        | main               |
| `abl_informed_dist.yaml`  | prompt states the training distribution   | optional           |
| `abl_informed.yaml`       | both of the above in the prompt           | optional           |
| `abl_no_m2.yaml`          | EoH without M2                            | fallback           |
| `control_randsample.yaml` | LLM4AD RandSample: same budget, fixed FunSearch-style prompt, no evolution | control            |

Run one with a `check` budget first, then in full, then score it (no `--llm` needed: the model is set in the file):

```bash
python pipeline/run_experiment.py --config configs/abl_anyfit.yaml --budget check
python pipeline/run_experiment.py --config configs/abl_anyfit.yaml
bash pipeline/score.sh abl_anyfit           # all scoring steps, no LLM calls
```

`score.sh` runs every scoring step in a fixed order: tuned Best Fit retuned in the setting, `evaluate_final`, `check_integrity`, `diagnose`, `paired_stats`, `summarise`. All of them are deterministic. To check that committed scores reproduce on your machine, run `bash pipeline/score.sh --check abl_anyfit abl_informed_rules`. It fails, listing the files, if any result differs from the commit. Only CSV row order and measured run times (`time_s`) may differ between runs; every score, bin count and verdict must match exactly (`pipeline/check_outputs.py`).

Once every setting is scored, `python pipeline/seed_counts.py` writes the setting-level effects: seeds per category (better / worse than Best Fit or tuned Best Fit per set, collapses, fresh-bin seeds) and the Fisher test against the baseline's seeds 0–5 and 0–9, or A1b for A2b (`docs/claims.md`, shared rules). Outputs: `results/seed_counts.csv` (every category and set) and `results/seed_counts.md` (the headline table). No LLM calls; it reads the committed `paired_stats.csv`, `final_per_instance.csv` and `diagnostics.csv`. To check it reproduces: `python pipeline/seed_counts.py && python pipeline/check_outputs.py results/seed_counts.csv results/seed_counts.md`.

Before the full runs:

- **Branch:** run from a branch that includes #14, so every run saves each LLM prompt and response (`llm_calls.jsonl`). Commit the whole run folder afterwards.
- **Order:** the three main settings first (`abl_anyfit`, `abl_informed_rules`, `abl_mixed_shapes`), then `control_randsample`. It has not run on OpenRouter yet and uses a different prompt (Best Fit as `priority_v0`, "complete `priority_v1`"), so its `check` run is a gate: look at `n_unique_programs` and `n_calls_not_evaluated` in `summary.json` before the 5 full seeds. The optional settings only if time allows.
- **Credit:** about US$0.16–0.20 per run, 20 runs (3 main settings + control, 5 seeds each) ≈ US$3–4; budget ~US$5 for check runs and headroom. Check the OpenRouter balance first.

---

## 10. Outputs and how to read them

**Per run** (`results/<experiment>/<label>/seed<k>/`):

| File                          | Contents                                                                                                                   |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `config.json`                 | resolved config, method kwargs used/dropped, library versions, exact prompt text added                                     |
| `llm_calls.csv`               | one row per LLM call: tokens (`api` or `estimate`), latency, cost, status (with the provider's error message if it failed) |
| `evals.csv`                   | one row per candidate: status (`ok` / `error` / `timeout` / `crash`), score, train excess, error message                   |
| `llm_calls.jsonl`             | full prompt and response text of every LLM call (runs from 5 Nov on)                                                        |
| `candidates.jsonl`            | every evaluated program                                                                                                    |
| `per_instance.csv`            | every candidate × training instance: bins, lower bound, excess                                                             |
| `best_program.py`             | best program by training score (ties go to the earliest)                                                                   |
| `summary.json`                | counts, tokens, cost, stop reason, best result                                                                             |
| `stdout.log`, `llm4ad_log_*/` | LLM4AD's own output                                                                                                        |

**In git:** every file of each run folder, so every evaluated program and LLM4AD's logs can be audited (`summary.json`, `best_program.py`, `evals.csv` and `llm_calls.csv` alone are enough to rebuild every table and figure). Runs made before 4 Nov only have those four files committed, unless their full folders were added later. Fake-LLM and pipecheck results (`results/smoke/`, `results/pipecheck*/`) are not committed, since they are plumbing checks, not findings.

**Per experiment** (`results/<experiment>/`): `runs_index.csv`, `final_per_instance.csv`, `final_summary.csv`, `paired_stats.csv`, `summary_tables.md`, and `figures/` (PNG + PDF for LaTeX):

- `convergence_evals` / `convergence_tokens`: best training excess so far vs evaluations / tokens (median and range across seeds, Best Fit line);
- `test_sets`: held-out change vs Best Fit per method and seed, one panel per set.

**Key fields:**

| Field                       | Meaning                                                                                                                                                                                             |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stop_reason`               | `budget:evals` is the normal ending. `budget:tokens` / `budget:wall_clock`: another limit bound first (raise it). `budget:llm_errors`: the API kept failing (quota, key). `error`: see `error.txt`. |
| `best_train_excess`         | best candidate's excess on training (compare with Best Fit on the same set)                                                                                                                         |
| `kind`                      | `evolved` (LLM search) or `tuned` (no-LLM tuner such as `tuned_bf`); `final_summary.csv` also has `baseline`                                                                                        |
| `best_params`               | `tuned_bf` runs only: the winning knob setting                                                                                                                                                      |
| `n_calls_not_evaluated`     | calls with no evaluation: candidates still being scored when the budget hit, unparseable replies, ReEvo reflections. 1–3 is normal for EoH.                                                         |
| `mean_excess` / `sd_excess` | held-out excess over **L1**, averaged within each run, then mean ± sd across seeds. L1 is the papers' bound (FunSearch, EoH), kept for comparison |
| `best_run` / `worst_run`    | lowest / highest per-run mean excess across seeds, over **L1** |
| `delta_vs_bf_pp`            | difference from Best Fit in percentage points over **L1**; **negative = better** |
| `*_l2` twins                | `mean_excess_l2`, `sd_excess_l2`, `best_run_l2`, `worst_run_l2`, `delta_vs_bf_pp_l2`: the same over **L2**. **The report uses these** (#12 decision 7); they equal L1 except on `test_scale65` and `test_control` |
| `win_vs_bf`                 | share of instances where the program used fewer bins than Best Fit; counts bins, so the same under L1 and L2                                                                                                                                  |

**Reading search progress.** Individual candidates go up and down; only the best-so-far matters. Programs that look different can pack identically (e.g. `item - bins` and `-(bins - item)`), so many candidates share exactly the same score. Use the diagnostic in §12 to tell a normal plateau from a search that isn't exploring.

---

## 11. Evaluation details

**How a heuristic is called.** Once per arriving item, with `bins` = the **remaining** capacities of only the bins the item fits in. The item goes into the bin with the highest score (`np.argmax`; ties go to the earliest bin). Output must be the same length as `bins` and contain no NaN (both are errors); `-np.inf` is allowed to rule a bin out.

**Two packing conventions** (`evaluation.mode`):

- `funsearch` (default): one bin per item is set up at the start, all at full capacity, so `bins` always includes empty bins. Identical to LLM4AD, FunSearch and EoH; we verified our scores match LLM4AD's exactly. Consequence: any heuristic that prefers emptier bins always opens a new bin and collapses to about one item per bin (~148% excess on our training set).
- `anyfit`: only open, non-empty bins are offered; a new bin opens only when none fit (Sgall's AnyFit class).

**Which convention each row uses.** Evolved programs are always scored in `evaluation.mode` (`funsearch` by default). The baseline rows in `final_summary.csv` use their classical AnyFit definitions, so Worst Fit is ~13% there but 148% in funsearch mode (on train). Best Fit and First Fit never open a fresh bin while a used bin has room, so the two conventions give them identical bins (`tests/test_diagnostics.py` checks the first part; the diagnostics summaries show identical scores on every set), so their rows compare directly. State this in the report. `pipeline/diagnose.py` rescores every program and baseline in both modes and reports `fresh_given_room`: the share of choices where a program put the item in an empty bin although a used bin had room.

**Lower bound.** `ceil(sum(items) / capacity)` (Martello and Toth's L1), the bound used by FunSearch and EoH. It is not the true optimum, so part of every heuristic's excess is unavoidable. The search and `excess` always use L1, so our numbers stay comparable with the papers. `evaluate_final.py` also writes Martello and Toth's tighter L2 bound (`lb_l2`, `excess_l2` per instance; `mean_excess_l2` in `final_summary.csv`). L2 counts that two items over half a bin can never share one. It is there for `test_scale65` (issue #12), where 63% of items exceed half a bin: Best Fit is 10.2% over L1 but 0.4% over L2, so L1 would make every rule look far worse than it is. L2 is also tighter on the control set `test_control`, where every rule must be exactly 0% over L2. On every other test set L2 equals L1. The report uses excess over L2 on every set (#12 decision 7): `summarise.py`'s held-out table and test-set figure are on L2, and a second table shows L1 for the sets where it differs (found from the data), for comparison with the papers. Older results without the L2 columns are filled in by `summarise.py`, no need to rerun `evaluate_final.py`.

**Formatting rules.** With `format_hint: true`, every prompt from every method ends with short rules: one complete function, the whole `def` line on one line, no helper functions, numpy only. LLM4AD's parser silently drops candidates with multi-line signatures and evolves the _wrong_ function when a helper comes first; the rules prevent both. The text is identical for all methods (fair), recorded in each `config.json`, and is our one deviation from LLM4AD's prompts. Mention it in the report.

**Integrity checks** (the brief: keep scoring outside the code being evolved). They flag, never drop, and never change a score:

- Every candidate is parsed (not run) by `obpevo/integrity.py`. Violations go in the `integrity` column of `evals.csv` (`''` = clean), and `summary.json` records `n_flagged` and `best_integrity`. Rules: imports other than numpy, `np.random` / `random` (also via `getattr` or `from numpy import *`), wall-clock time, `hash()` / `id()`, `exec` / `eval` / `compile` / `open` / `__import__`, numpy file I/O (`np.save` / `np.load` / ...), process-wide numpy settings, dunder access such as `__builtins__`, `global`, attribute assignment, mutating a module-level name, mutable default arguments. The full list is in the module docstring.
- **What the static check can see:** LLM4AD keeps only the _body_ of the LLM's function (template imports + `def` line + body); module-level imports, globals and changed signatures in the reply are thrown away before evaluation. So in our runs the body-level rules are the ones that can fire, and an empty column means "no body-level violation", not "the LLM never tried".
- `python pipeline/check_integrity.py --config ... [--experiment ...]` rescores each best program four times, each in a fresh Python process with its own hash seed and RNG seeds: twice in the usual order and twice in one shuffled order. `repeat_mismatch` = randomness (including `hash()`); `shuffle_mismatch` = state carried from one instance to the next. `verdict`: clean / flagged / stateful / nondeterministic / failed. Randomness that rarely changes a bin count can pass the rescoring; the static flags cover that. The rescored best programs (25 Oct seed 0, OpenRouter seeds 0–2) are clean; free-tier seed 11 has no static flags but has not been rescored.
- `tests/test_integrity.py` holds one deliberately bad program per rule, plus legitimate programs that must stay unflagged.

**Instance sets** (`configs/instances/v0.yaml`, generated deterministically from seed 2026):

| Set                  | Instances × items | Capacity | Items             | Purpose                                                                                                                                                      |
| -------------------- | ----------------- | -------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `train`              | 32 × 200–2000     | 100      | Weibull(3) × 45   | the only set the search sees                                                                                                                                 |
| `val`                | 10 × 1000         | 100      | Weibull(3) × 45   | choosing between configurations (e.g. models, prompts); never seen by the search. Within a run, the best program is picked on training, as in the EoH papers |
| `test_id_5k`         | 10 × 5000         | 100      | Weibull(3) × 45   | in-distribution (FunSearch/EoH standard)                                                                                                                     |
| `test_size_1k`       | 10 × 1000         | 100      | Weibull(3) × 45   | size shift (smaller); its own instances, since `val` is used to choose between runs                                                                          |
| `test_size_10k`      | 10 × 10000        | 100      | Weibull(3) × 45   | size shift                                                                                                                                                   |
| `test_cap200`        | 10 × 5000         | 200      | Weibull(3) × 45   | bigger bins, same items (midpoint of the 100 → 200 → 500 series)                                                                                             |
| `test_cap500`        | 10 × 5000         | 500      | Weibull(3) × 45   | bigger bins, same items (BF/FF/EoH seed 0 within 0.1 pp; EoH free-tier seed 11 fails here, see §13)                                                                                                   |
| `test_cap500_scaled` | 10 × 5000         | 500      | Weibull(3) × 225  | everything ×5: scale-invariance check                                                                                                                        |
| `test_scale25`       | 10 × 5000         | 100      | Weibull(3) × 25   | smaller items (~22% of a bin; training ~40%)                                                                                                                 |
| `test_scale65`       | 10 × 5000         | 100      | Weibull(3) × 65   | bigger items (~58% of a bin). 63% exceed half a bin, so read excess over L2 (#10)                                                                            |
| `test_shape_k1.5`    | 10 × 5000         | 100      | Weibull(1.5) × 45 | wider spread of item sizes                                                                                                                                   |
| `test_shape_k5`      | 10 × 5000         | 100      | Weibull(5) × 45   | tightly clustered item sizes                                                                                                                                 |
| `test_uniform`       | 10 × 1000         | 150      | Uniform[20, 100]  | generated like OR-Library's binpack1–4: different distribution, capacity and size                                                                            |
| `test_control`       | 10 × 1000         | 100      | Uniform[51, 100]  | control: every item is over half a bin, so every rule must use one bin per item; any difference is a scoring bug                                             |

Each test set changes one thing from training, so a failure can be attributed to that change. The exceptions are `test_cap500_scaled` (capacity and item sizes ×5 together, on purpose), `test_uniform` (distribution, capacity and size) and the control set. Every test set has 10 instances, so a rule that beats Best Fit on all of them is significant on the sign test (p = 0.002; with 5 it could not go below 0.0625). The spec is frozen (#12): changing a set changes its data.

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

**EoH baseline: 6 seeds** (`openrouter-gemini-3.5-flash-lite`, 300 evals each, pinned LLM4AD, US$1.04 in total; seeds 0–2 run on 1 Nov, seeds 3–5 on 4 Nov with the same setup; `results/eoh_baseline__openrouter-gemini-3.5-flash-lite/`):

- Training excess per seed: 4.751, 4.751, 4.507, 4.602, 3.719, 3.719% (Best Fit 4.751%). Seeds 0–1 are Best Fit itself, seed 2 nearly so. Seeds 4 and 5 found the same rule independently (`-sin(π·r/item) - r/item`, r = space left): it avoids leaving a gap smaller than about one item, and uses only ratios.
- Held-out, Δ vs Best Fit per seed (pp; negative = better):

  | Set | BF % | s0 | s1 | s2 | s3 | s4 | s5 |
  |---|---|---|---|---|---|---|---|
  | val | 4.73 | 0.00 | 0.00 | −0.30 | −0.05 | −1.22 | −1.22 |
  | test_id_5k | 4.17 | 0.00 | 0.00 | −0.18 | −0.04 | −1.30 | −1.30 |
  | test_size_10k | 3.95 | 0.00 | 0.00 | −0.05 | +0.13 | −1.07 | −1.07 |
  | test_cap500 | 0.50 | 0.00 | 0.00 | 0.00 | +0.20 | +0.05 | +0.05 |
  | test_cap500_scaled | 4.44 | 0.00 | 0.00 | +0.12 | +0.17 | −1.33 | −1.33 |
  | test_shape_k1.5 | 1.34 | 0.00 | 0.00 | −0.04 | **+2.58** | −0.12 | −0.12 |
  | test_shape_k5 | 8.25 | 0.00 | 0.00 | +0.06 | −1.94 | **+2.36** | **+2.36** |
  | test_uniform | 5.07 | 0.00 | 0.00 | −0.30 | −0.65 | −1.24 | −1.24 |

- Reading: half the seeds stall at Best Fit (the LLM keeps rewriting it); the other half find rules that open fresh bins, with gains on most sets and a failure on one item-size shift. Mean ± sd (`summary_tables.md`) hides this split, so report results per seed.

**Tuned Best Fit: 6 seeds** (`tuned_bf`, 300 evaluations each, no LLM, about 2.5 min per seed, US$0; seeds 0–2 run on 3 Nov, seeds 3–5 on 5 Nov with the same setup; same results folder):

- Training excess per seed: 2.441, 2.370, 2.468, 2.362, 2.376, 2.469% (**2.414 ± 0.051%**; Best Fit 4.751%, EoH 4.342 ± 0.491%, best EoH seed 3.719%). All six seeds land on nearly the same knobs: `lo` 3–4, `hi` 29–31, `penalty` 64–97, i.e. avoid leaving a gap of about 4–30 units, opening a fresh bin instead if needed.
- Held-out, Δ vs Best Fit per seed (pp; negative = better):

  | Set | BF % | s0 | s1 | s2 | s3 | s4 | s5 | mean |
  |---|---|---|---|---|---|---|---|---|
  | val | 4.73 | −2.41 | −2.44 | −2.36 | −2.19 | −2.31 | −2.24 | −2.32 |
  | test_id_5k | 4.17 | −2.38 | −2.67 | −2.38 | −2.46 | −2.53 | −2.60 | −2.50 |
  | test_size_10k | 3.95 | −2.31 | −2.71 | −2.31 | −2.65 | −2.69 | −2.76 | −2.57 |
  | test_cap500 | 0.50 | −0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | −0.01 |
  | test_cap500_scaled | 4.44 | **+0.71** | **+0.78** | **+0.83** | **+0.70** | **+0.80** | **+0.83** | **+0.77** |
  | test_shape_k1.5 | 1.34 | +0.78 | **+2.21** | +0.58 | −0.32 | −0.27 | +1.38 | +0.73 |
  | test_shape_k5 | 8.25 | **+21.60** | **+24.79** | **+21.60** | **+19.27** | **+19.27** | **+19.27** | **+20.97** |
  | test_uniform | 5.07 | −2.19 | −2.18 | −1.64 | −1.59 | −1.64 | −1.84 | −1.85 |

- Reading: every tuned_bf seed beats every EoH seed on training and on `val`, `test_id_5k`, `test_size_10k` and `test_uniform`, using fewer bins than Best Fit on every instance of those four sets. The 25 Oct run (3.788% train) is better on `test_uniform` (−2.19 vs −1.85 pp). Every seed fails on `test_shape_k5` and `test_cap500_scaled`; on `test_shape_k1.5`, 4 of 6 seeds are worse than Best Fit. EoH runs fail on the same shifts (OpenRouter seeds 3–5 on k5 or k1.5, the 25 Oct run on k5 and cap500_scaled), so these failures do not need an LLM: tuning three knobs on the training distribution produces them too.
- Packing convention: almost all of the gain comes from opening fresh bins on purpose, which `funsearch` mode allows. By `diagnose.py`, tuned_bf puts 33% of items on train (53% on `test_shape_k5`) into a fresh bin while a used bin has room. Rescored in `anyfit` mode, train is 4.62% and `test_shape_k5` 8.20%, close to Best Fit (4.75%, 8.25%). `test_cap500_scaled` is 5.22% in both modes: no fresh bins there, but the sliver zone is in absolute units, so at capacity 500 it penalises what are actually tight fits. State the mode next to every comparison with tuned_bf. An EoH run searched in `anyfit` mode must be compared with tuned_bf **retuned** in `anyfit` mode, not with these programs rescored.
- **Update 7 Nov (Ablation 1):** "almost all of the gain comes from fresh bins" holds only for knobs **tuned in `funsearch` mode**. Retuned in `anyfit` mode (`results/abl_anyfit/tuned_bf/`, seeds 0–9), the same 3 knobs (`lo` 0–2, `hi` 19–23) penalise leaving a small gap in a used bin. That gives 3.90–4.00% on train with no fresh bins (Best Fit 4.75%), and it beats Best Fit on `test_id_5k` in 10/10 seeds by about 1 pp. Checked with an independent packer (no shared code): Best Fit 4.751%, tuned seed 0 3.965% on train. So a tuned Best Fit keeps most of its gain without fresh bins; see #18 and `docs/claims.md` Amendments.

**EoH baseline, first run (25 Oct)** (`gemini-3.5-flash-lite`, 300 evals, 34 min, 301 calls, 0 failed, US$0.17 notional):

- Training: **3.788%** excess vs Best Fit 4.751% (−0.96 pp). Stuck at Best Fit for ~150 evaluations (most candidates were Best Fit rewrites), then improved at ~165 and at eval 217.
- Held-out (Δ vs Best Fit, pp; negative = better): `val` −1.00, `test_id_5k` −1.95, `test_size_10k` −2.22, `test_uniform` −2.19, `test_shape_k1.5` −0.42, `test_cap500` −0.10, **`test_cap500_scaled` +1.29**, **`test_shape_k5` +30.69**.
- The two failures are _consistent with_ a specific mechanism, which we treat as a hypothesis to test: the evolved rule switches formula at `item / remaining capacity = 0.5`, which suits training's item sizes but may send tightly clustered items (k=5) into fresh bins (43% of bins held one item, vs 0% for Best Fit), and it uses absolute units of space, so it may behave differently at capacity 500.
- Supporting check: rescoring the same program in `anyfit` mode (no empty bins offered while an open bin fits) removes both failures (`test_shape_k5` 8.15%, `test_cap500_scaled` 3.85%) and about half the training gain (4.34%). This points to the rule opening new bins too eagerly, rather than general overfitting.

**Fresh-bin diagnostic** (`pipeline/diagnose.py`, `diagnostics_summary.csv` in each folder): OpenRouter seeds 0–2 never open a fresh bin while a used bin has room; seeds 3–5 do (6-seed mean `fresh_given_room` 11% on train, 19% on `test_shape_k5`). 25 Oct seed 0 does in 36% of choices on train (57% on `test_shape_k5`). A second free-tier run, seed 11 (`results/eoh_freetier_seed11/`, separate run: older pipeline, 62 of 363 calls hit rate limits), does in 9%: train 4.48%, no k=5 failure, but **`test_cap500` 6.12%** vs Best Fit 0.50%, because its periodic score `cos(π·remaining/(1.618·item))` also peaks on empty capacity-500 bins. In anyfit mode both failures disappear: seed 11 falls back to about Best Fit on every set, and seed 0 keeps about half its gain. Seeds 4/5 behave the same way: in anyfit mode their k=5 failure disappears (8.18% vs Best Fit 8.25%) and about half their gain remains (`test_id_5k` 3.70% vs 4.17%). Seed 3 is the exception: its `test_shape_k1.5` failure stays in anyfit mode (3.99% vs Best Fit 1.34%), so not every failure is a fresh-bin failure. The anyfit comparison shows what these programs do, not why some runs learn it.

Reproduce: `python pipeline/evaluate_final.py --config configs/eoh_baseline.yaml --experiment eoh_baseline_25oct` (then `summarise.py` with the same arguments).

This run used LLM4AD 1.0.0 (PyPI). Later runs (pinned LLM4AD, partly cut short by the free-tier quota) found heuristics much closer to Best Fit: small gains, no large failures. Run-to-run variance is large: see the 6 OpenRouter seeds above. Details in `docs/log.md`.

---

## 14. Known limitations and next steps

- Only EoH has been run with a real LLM; pipecheck the other methods before their full runs.
- Only each run's single best program (by training score) goes to held-out evaluation. Selecting among the top programs on `val` instead is a possible extension.
- No operator attribution yet (which EoH operator produced each candidate) and no diversity metrics (clone rate, collapse rate, behavioural diversity). Both planned before the exploration/exploitation study.
- MCTS-AHD and HSEvo are untested.
- `tuned_bf`'s knobs are in absolute item units, tuned at capacity 100 (like the evolved programs). A variant with knobs relative to the bin size would test the hard-coded-capacity explanation for `test_cap500_scaled`. Each ablation setting needs its own tuned_bf, retuned under that setting (`anyfit` mode, `train_mixed`; #7).
- Prices in `configs/llms.yaml` were checked on 2026-10-25; re-check before trusting `cost_usd`. `claude-haiku-5-5` has no price set yet.
