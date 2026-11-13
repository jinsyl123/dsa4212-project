# DSA4212 Project: LLM-guided evolution of online bin packing heuristics

We use an LLM-driven search (**EoH**, via [LLM4AD](https://github.com/Optima-CityU/LLM4AD)) to evolve a priority rule for **online bin packing**, and ask what makes the evolved rules beat Best Fit, and why some of them fail on shifted test sets. Two ablations and a control test where the gains and failures come from:

| Setting | What changes (one thing, from the setting it is compared with) | Compared with | Config (`code/configs/`) | Results (`code/results/`) |
|---|---|---|---|---|
| EoH baseline | nothing | — | `eoh_baseline.yaml` | `eoh_baseline__openrouter-gemini-3.5-flash-lite/` |
| Control: no evolution | LLM4AD RandSample: 300 samples of one fixed prompt, best kept | baseline | `control_randsample.yaml` | `control_randsample/` |
| A1a: empty bins hidden | the rule is never offered an empty bin while a used bin fits (`mode: anyfit`) | baseline | `abl_anyfit.yaml` | `abl_anyfit/` |
| A1b: bin rules in the prompt | the prompt states the task and that empty bins are among the options | baseline | `abl_informed_rules.yaml` | `abl_informed_rules/` |
| A2a: mixed training shapes | trained on Weibull shapes 2 and 4 instead of 3 | baseline | `abl_mixed_shapes.yaml` | `abl_mixed_shapes/` |
| A2b: mixed training capacities | A1b, trained on capacities 100 and 300 (item scale 45) instead of 100 only | A1b | `abl_mixed_caps_informed.yaml` | `abl_mixed_caps_informed/` |

A2b builds on A1b because the capacity collapse it tests appears only with A1b's prompt (`code/docs/log.md`, 9 Nov; `code/docs/claims.md`, A2b). Every setting has seeds 0–9. The baseline's seeds 0–5 were run first and are the pre-registered comparison; seeds 6–9 were added on 9 Nov, and a difference in seed counts is claimed only if it is significant against both (`code/docs/log.md`, 9 Nov, "baseline extended to 10 seeds"; `code/docs/claims.md`, Amendments). A third ablation (a stronger LLM, `abl_strong_model.yaml`) was optional and was not run.

Every setting is compared with Best Fit and with **tuned Best Fit** (Best Fit's 3 knobs tuned without an LLM, under the same 300-evaluation budget; §9). Predictions for each setting were frozen before scoring in [`code/docs/claims.md`](code/docs/claims.md); every decision that affects results is in [`code/docs/log.md`](code/docs/log.md).

**To reproduce the results, go to §4: no API key and no LLM calls are needed.** To run new searches with an LLM, see §5.

Assignment guidelines: https://alexxthiery.github.io/teaching/guidelines/guidelines.html
Project brief: https://alexxthiery.github.io/teaching/llm-evolution/llm-evolution.html
Overleaf report: https://www.overleaf.com/project/6ad866a8d72db1d3c73a0dba

## Contents

1. [Results](#1-results)
2. [Repository layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [Reproducing the results (no LLM, no API key)](#4-reproducing-the-results-no-llm-no-api-key)
5. [Running the searches with an LLM (needs an API key)](#5-running-the-searches-with-an-llm-needs-an-api-key)
6. [How the pipeline works](#6-how-the-pipeline-works)
7. [Configs](#7-configs)
8. [Choosing the LLM](#8-choosing-the-llm)
9. [Methods, tuned Best Fit and ablations](#9-methods-tuned-best-fit-and-ablations)
10. [Outputs and audit trail](#10-outputs-and-audit-trail)
11. [Evaluation details](#11-evaluation-details)
12. [Troubleshooting](#12-troubleshooting)
13. [Known limitations](#13-known-limitations)

---

## 1. Results

Frozen test sets (§11), excess over the L2 lower bound, 10 instances per set. Per-seed verdicts: a seed is **better** or **worse** than a reference only if the 95% bootstrap CI excludes 0 and the sign test gives p < 0.05, **tie** if it matches the reference on every instance, and **unclear** otherwise. Full tables and per-seed numbers: `code/results/<setting>/summary_tables.md` and `paired_stats.csv`; the predictions they are checked against: `code/docs/claims.md`. The capacity and scale failures (A1b on `test_cap500`, `test_cap200`, `test_scale25`) were found after A1 was scored; they were not pre-registered.

| Setting | Seeds | Beats Best Fit on `test_id_5k` | Beats its tuned Best Fit on `test_id_5k` | Worse than Best Fit on k1.5 or k5 | Fresh-bin seeds |
|---|---|---|---|---|---|
| EoH baseline, seeds 0–5 (pre-registered) | 6 | 3/6 | 0/6 | 3/6 | 3/6 |
| EoH baseline, seeds 0–9 | 10 | 5/10 | 0/10 | 3/10 | 3/10 |
| Control: no evolution | 10 | **0/10** | 0/10 | 0/10 | 0/10 |
| A1a: empty bins hidden | 10 | 6/10 | 0/10 | 1/10 | 0/10 (by design) |
| A1b: bin rules in the prompt | 10 | **10/10** | 0/10 | 1/10 | 6/10 |
| A2a: mixed training shapes | 10 | 9/10 | 1/10 | 1/10 | 2/10 |
| A2b: mixed training capacities (vs A1b) | 10 | 6/10 | 0/10 | 1/10 | 2/10 |

Bold: claimed, i.e. Fisher's exact test on seed counts gives p < 0.05 against both the 6- and 10-seed baseline. Every cell with its p-values: `code/results/seed_counts.md`; every category and set: `seed_counts.csv` (§4.4).

**What holds up** (significant against both the 6- and 10-seed baseline):
- **The control** never beats Best Fit (0/10; p = 0.036 and 0.033): without evolution, the LLM does not get past Best Fit.
- **A1b** beats Best Fit more often than the baseline (10/10; p = 0.036 and 0.033): stating the task and bin rules in the prompt makes the gain reliable. But 5/10 A1b seeds collapse on `test_cap500` (about 1,140% over the bound, Best Fit 0.5%), which A2b tests.

**Consistent with the predictions but not significant with 10 seeds:** mixed shapes (A2a) cut shape failures from 3/6 to 1/10; mixed capacities (A2b) remove A1b's capacity collapse inside the trained range (0/10 vs 4/10 on `test_cap200` and `test_scale25`, 2/10 vs 5/10 on `test_cap500`).

**Patterns across every setting:** no LLM rule beats its own tuned Best Fit in distribution (1 of 50 evolved seeds, A2a seed 6). In the baseline, the fresh-bin seeds (3, 4, 5; opening an empty bin while a used one has room) account for both its largest in-distribution gains (seeds 4, 5) and all three of its shape failures. Tuned Best Fit fails most consistently under shape shift: every baseline seed is worse than Best Fit on `test_shape_k5` (about 30% over the bound, vs Best Fit's 8%).

Earlier runs (the 25 Oct free-tier run, free-tier seed 11) and how the analysis got here are in `code/docs/log.md`.

---

## 2. Repository layout

```
├── readme.md              # this file
├── authors.md             # name and student ID of every group member
├── report/                # LaTeX source of the submitted report
├── docs/log.md            # early project log (topic and framework, 21 Oct)
├── results/               # pointer only: every result is in code/results/
└── code/                  # every command in this readme runs from here
    ├── requirements.txt
    ├── pipeline/
    │   ├── obpevo/core.py         evaluator: packing, lower bounds, score (single source of truth)
    │   ├── obpevo/instances.py    deterministic instance generator (Weibull / uniform, mixed sets)
    │   ├── obpevo/baselines.py    First / Best / Worst / Almost-Worst / Next Fit
    │   ├── obpevo/task.py         LLM4AD task wrapper: template, task description, formatting rules
    │   ├── obpevo/llm.py          LLM clients (OpenAI-compatible, Anthropic, fake) + LoggedLLM wrapper
    │   ├── obpevo/recorder.py     EvalRecorder: logs every candidate, enforces the eval budget
    │   ├── obpevo/integrity.py    static checks on candidates, flag only
    │   ├── obpevo/methods.py      registry of LLM4AD methods
    │   ├── obpevo/tuned_bf.py     tuned Best Fit: the budget-matched baseline (no LLM)
    │   ├── obpevo/config.py       YAML config loading: inheritance (base:), LLM/budget lookup, overrides
    │   ├── make_instances.py      build instance sets from a spec
    │   ├── run_experiment.py      step 1: search (every method × seed in a config)
    │   ├── evaluate_final.py      step 2: score best programs + baselines on held-out sets
    │   ├── check_integrity.py     rescore best programs twice + with shuffled instances
    │   ├── diagnose.py            fresh-bin diagnostic, both packing modes
    │   ├── paired_stats.py        each rule vs Best Fit / tuned Best Fit, instance by instance
    │   ├── summarise.py           step 3: tables and figures
    │   ├── score.sh               every scoring step for one or more settings; --check compares with the commit
    │   ├── check_outputs.py       the comparison --check uses
    │   ├── seed_counts.py         setting-level seed counts and Fisher tests
    │   ├── masked_choice.py       report §5.5: A1b's rules with empty bins withheld or masked
    │   ├── ratio_sweep.py         report §5.5: A1b's rules at smaller item-to-bin ratios
    │   ├── make_report_assets.py  the report's figures and results tables (writes to report/)
    │   └── verify_report_numbers.py  checks every number quoted in the report against the committed results
    ├── configs/           # one YAML per setting, plus llms.yaml, budgets.yaml and instances/
    ├── tests/             # unit tests
    ├── docs/              # claims.md (frozen predictions), log.md (decision log)
    └── results/           # one folder per setting: every run, its LLM calls, scores and figures
```

The root `docs/log.md` covers the topic choice and framework (from 21 Oct); every decision that affects results, from the v0 pipeline (25 Oct) on, is in `code/docs/log.md`.

---

## 3. Setup

Clone the repository and work from its root:

```bash
git clone https://github.com/jinsyl123/dsa4212-project.git
cd dsa4212-project
```

**Python 3.12** (LLM4AD needs `numpy<2`, which has no wheels for 3.13+). Check with `python --version`. If you have another version, install 3.12 alongside it and use a virtual environment, so the install cannot conflict with other Python packages on your machine:

| System | Install Python 3.12 | Create and activate the environment, from the repo root |
|---|---|---|
| Windows | `winget install Python.Python.3.12` | `py -3.12 -m venv .venv` then `.venv\Scripts\activate` |
| macOS | `brew install python@3.12` | `python3.12 -m venv .venv` then `source .venv/bin/activate` |
| Anaconda (any) | (included) | `conda create -n dsa4212 python=3.12` then `conda activate dsa4212` |

Then, with the environment active, install from the `code/` folder:

```bash
cd code
pip install -r requirements.txt        # installs LLM4AD at the pinned commit; prints "Successfully installed ..."
```

Install LLM4AD **only** through `requirements.txt`: `pip install llm4ad` gives an older release (1.0.0) without ReEvo, MCTS-AHD or HSEvo.

**Windows.** The Python scripts run natively. `score.sh` is a bash script: run it from **Git Bash** (installed with Git for Windows) or WSL. Each candidate is scored in a freshly started Python process, which adds a few seconds per evaluation; keep `timeout_seconds` at 30 or more.

---

## 4. Reproducing the results (no LLM, no API key)

Everything in this section runs from `code/`, makes no LLM calls and needs no API key. Steps 4.1, 4.2 and 4.4 take minutes; 4.3 takes about 15–20 minutes per setting (12 minutes for the baseline and 18 for A2b on a clean checkout), so 1.5–2 hours for all six.

### 4.1 Run the tests

```bash
python -m unittest discover -s tests   # passes if the last line is "OK"; "FAILED (...)" lists the failing tests
```

They cover the evaluator (packing, both lower bounds, score), the instance sets, tuned Best Fit, the integrity checks, the configs of every setting, and the Fisher test against the values quoted in `docs/claims.md`. One `RuntimeWarning: invalid value encountered in sqrt` from `test_core.py` is expected: that test checks that a rule returning NaN is rejected.

### 4.2 Run the pipeline end to end with a fake LLM

`configs/smoke.yaml` runs the whole pipeline (search, held-out scoring, tables and figures) on small instance sets with a **fake LLM** that returns canned heuristics, including deliberately broken ones. It checks the plumbing: budgets, failure logging and every output file. Its scores say nothing about method quality.

```bash
python pipeline/run_experiment.py --config configs/smoke.yaml    # 6 methods x 2 seeds = 12 runs, fake LLM (~7 min on Windows, ~1 min on macOS); ends "Wrote .../runs_index.csv (12 runs)"
python pipeline/evaluate_final.py --config configs/smoke.yaml    # held-out scoring; 40 job lines, each "ok", then a table
python pipeline/summarise.py      --config configs/smoke.yaml    # ends "Figures in .../results/smoke/figures"; the best_fit row of the held-out table reads 4.79, 4.40, 1.25, 8.80
```

The `err` and `timeout` counts and the `RuntimeWarning: invalid value encountered in sqrt` lines come from the deliberately broken candidates and are expected. **The two `funsearch` runs also show `0 evals ... error` (and `nan` in the summary table):** LLM4AD's FunSearch needs `scipy`, which we do not install because FunSearch is not run in this study; nothing else depends on it. A second run skips finished seeds (`skip ... (done)`); add `--force` to rerun them. Outputs go to `results/smoke/`, which is git-ignored. §10 lists what each file holds.

### 4.3 Regenerate every reported result from the committed runs

Every search run is committed in full (§10), so every reported number can be rebuilt from the committed best programs:

```bash
# each command ends with "OK: every committed output reproduced (...)"; RuntimeWarning lines on the way are expected
bash pipeline/score.sh --check --llm openrouter-gemini-3.5-flash-lite eoh_baseline
bash pipeline/score.sh --check abl_anyfit abl_informed_rules control_randsample
bash pipeline/score.sh --check abl_mixed_shapes abl_mixed_caps_informed
```

The baseline needs `--llm` because its results folder is named after the model; the ablation configs set the model in the file. For each setting, `score.sh` runs, in order:

1. tuned Best Fit: the committed runs are reused (finished seeds are skipped), so nothing is retuned. In a folder without them it tunes from scratch, with no LLM, about 2.5 min per seed;
2. `evaluate_final.py`: every committed best program and every baseline on every held-out set;
3. `check_integrity.py`: integrity flags (§11);
4. `diagnose.py`: the fresh-bin diagnostic in both packing modes;
5. `paired_stats.py`: per-seed verdicts against Best Fit and tuned Best Fit; for the baseline, also seeds 0–5 against tuned Best Fit seeds 0–5 (`paired_stats_seeds0-5.csv`), the pre-registered comparison;
6. `summarise.py`: `summary_tables.md` and the figures.

With `--check`, it then prints `OK: every committed output reproduced`, or lists the files that differ and exits with an error. CSV row order and measured run times (`time_s`) are ignored, and changed figures are not compared (their bytes depend on the matplotlib version and fonts; the CSVs they are drawn from are), though a missing or new figure still fails. Every score, bin count and verdict must match exactly. `git status` may then show some CSVs and figures as modified; that is row order, `time_s` and figure bytes only, and `--check` is the verdict.

To score a single setting, name only that one, e.g. `bash pipeline/score.sh --check abl_mixed_caps_informed`.

### 4.4 Setting-level seed counts and Fisher tests

Once every setting is scored:

```bash
python pipeline/seed_counts.py
git status --short results/seed_counts.csv results/seed_counts.md   # empty = both reproduced exactly
```

It counts, per setting, the seeds in each category (better / worse than Best Fit or tuned Best Fit on each set, collapses, fresh-bin seeds) and runs Fisher's exact test against the baseline's seeds 0–5 and 0–9, or against A1b for A2b (`docs/claims.md`, shared rules). It reads the committed `paired_stats.csv`, `final_per_instance.csv` and `diagnostics.csv` and writes `results/seed_counts.csv` (every category and set) and `results/seed_counts.md` (the table in §1, with p-values). It takes seconds.

**Report §5.5's two extra checks** rescore A1b's committed programs, with no LLM calls: empty bins withheld, or visible but not choosable while a used bin fits (`masked_choice.py`), and new instances at smaller item-to-bin ratios and at capacity 1,000 (`ratio_sweep.py`).

```bash
python pipeline/masked_choice.py   # about 10 min; ends "-> .../masked_choice.csv"
python pipeline/ratio_sweep.py     # a few minutes; ends "-> .../ratio_sweep.csv"
git status --short results/abl_informed_rules/masked_choice.csv results/abl_informed_rules/ratio_sweep.csv   # empty = both reproduced
```

### 4.5 Report figures, tables and numbers

```bash
git checkout -- results                                     # if you ran 4.3: restores the run times it re-measured
python pipeline/make_report_assets.py
git status --short ../report/tables ../report/numbers.txt   # empty = tables and numbers reproduced
```

Run it on the committed results. `score.sh` (4.3) re-measures the run times in each `final_per_instance.csv`, and the report's cost table reads them (its Time column, in `report/tables/cost.tex`), so after 4.3 that column differs. `git checkout -- results` restores the committed files.

It reads the committed `final_per_instance.csv`, `final_summary.csv`, `paired_stats.csv`, `diagnostics.csv` and `runs_index.csv` of every setting, the baseline's `evals.csv` files, and `results/abl_informed_rules/ratio_sweep.csv` (written by `pipeline/ratio_sweep.py`, §4.4). It writes the report's five figures to `report/figures/` (`convergence.pdf`: search progress on the training set; `empty_bins.pdf`: each seed with empty bins offered against withheld; `fresh_rate.pdf`: loss on `shape_k5` against how often a rule opens fresh bins there; `collapses.pdf`: each A1b and A2b seed's excess above Best Fit, ordered by item-to-bin ratio; `ratio_sweep.pdf`: the five collapsing A1b rules as the item-to-bin ratio falls), its four results tables to `report/tables/` and the numbers quoted in the text to `report/numbers.txt`. Table means are taken from `final_summary.csv` with the same rounding as `summary_tables.md`, so the two agree. It takes seconds. As in 4.3, figure bytes (and the axes' exact placement) can differ with the matplotlib version, fonts and operating system, so only the tables and numbers are expected to match byte for byte; the plotted data does not change.

**Checking every number in the report.** A second, independent script recomputes each number the report quotes from the committed files and compares it with the value printed in the report:

```bash
python pipeline/verify_report_numbers.py   # about 5-10 minutes; ends "checks run: 2428, failed: 0" and "no failures"
```

It re-packs every final program (instance 0 of every set) against `final_per_instance.csv`, recomputes every verdict from the per-instance rows and requires it to agree with `paired_stats.csv`, checks `runs_index.csv` against each run's `evals.csv`, and then checks the numbers quoted in each report section against `final_per_instance.csv`, `diagnostics.csv`, `runs_index.csv`, `integrity.csv`, tuned Best Fit's `candidates.jsonl` and `summary.json`, and §5.5's `masked_choice.csv` and `ratio_sweep.csv`. For the few numbers that no results file holds, it re-scores the committed programs itself: baseline seed 4's empty-bin placements by item size (§5.2), and the final programs on A2's training sets with EoH's fitness (§5.4). It prints one line per report section with the files it read and the number of checks and failures. It does not read `make_report_assets.py`'s outputs, so it is a second route to the same numbers.

### 4.6 What reproduces exactly, and what doesn't

Everything after the search is deterministic: given the committed `best_program.py` files, steps 4.3 and 4.4 reproduce every held-out score, verdict, table and Fisher test exactly. The instance sets are generated deterministically from `configs/instances/v0.yaml` on first use. The **search itself** does not reproduce: the LLM samples with temperature 1.0 and is not seeded, so rerunning a search gives different programs (`seed` only fixes LLM4AD's own selection steps). That is why we run 10 seeds per setting, report per-seed results, and commit every run folder, including every prompt and response (§10).

---

## 5. Running the searches with an LLM (needs an API key)

This section is only for running **new** searches; it is not needed to reproduce the results (§4). Every reported run used `openrouter-gemini-3.5-flash-lite` (Gemini 3.5 Flash-Lite via OpenRouter), 300 evaluations per run.

### 5.1 API key

Get an OpenRouter key (openrouter.ai → Keys) and add credit. Put it in a file called `.env` in `code/` (already git-ignored), never in a config you commit:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

The scripts read the key from the environment, not from `.env` directly, so load it into the shell before running:

```bash
set -a; source .env; set +a                                   # bash / Git Bash
```

```powershell
Get-Content .env | ForEach-Object { $k, $v = $_ -split '=', 2; Set-Item "env:$k" $v }   # PowerShell
```

Other providers use other variables (§8).

### 5.2 Check the key and the model

```bash
python pipeline/run_experiment.py --config configs/pipecheck.yaml --llm openrouter-gemini-3.5-flash-lite
```

`pipecheck.yaml` is a real LLM on a tiny budget (12 evaluations, small instances): about a minute and a fraction of a cent. Then check in `results/pipecheck__openrouter-gemini-3.5-flash-lite/eoh/seed0/`: `evals.csv` mostly `ok` rows; `llm_calls.csv` with `status = ok` and `token_source = api`; `candidates.jsonl` with real heuristics, not just the template. These results are git-ignored.

### 5.3 Run a setting

Run one seed with the `check` budget first, then every seed in full, then score (§4.3):

```bash
python pipeline/run_experiment.py --config configs/abl_anyfit.yaml --budget check --seeds 0   # a few minutes
python pipeline/run_experiment.py --config configs/abl_anyfit.yaml                           # seeds 0–9
bash pipeline/score.sh abl_anyfit                                                            # no LLM calls
```

For the baseline, add `--llm openrouter-gemini-3.5-flash-lite` to every command, as in §4.3.

- **Time and cost:** roughly 30–40 minutes and US$0.16–0.20 per seed. Actual: A1 about US$4.11 (20 runs, plus one aborted attempt), the control about US$0.90 (10 runs), A2 about US$3.90 (20 runs); per run in `summary.json` (`cost_usd`). Check the OpenRouter balance first.
- **Finished runs are skipped.** A run with a `summary.json` counts as done; pass `--force` to redo it. A run stopped by API errors (`stop_reason: budget:llm_errors`) is incomplete: rerun it with `--force` and don't score it (`docs/log.md`, 7 Nov).
- **Scoring covers every finished seed together**, so seeds can run on different days and be scored at the end.
- **Commit the whole run folder** afterwards (`results/<setting>/<method>/seed<k>/`), so the run can be audited (§10).
- **Check progress** while a run is going, from a second terminal:

  ```python
  import pandas as pd
  run = "results/abl_anyfit/eoh/seed0/"
  ev, calls = pd.read_csv(run + "evals.csv"), pd.read_csv(run + "llm_calls.csv")
  print(f"{len(ev)}/300 evals | {len(calls)} calls ({(calls.status != 'ok').sum()} failed) "
        f"| best so far {ev.train_excess.min():.3%} (Best Fit 4.751%)")
  ```

### 5.4 Another model or budget

No new config is needed: pass `--llm` and `--budget` (names from `configs/llms.yaml` and `configs/budgets.yaml`) to every step. The results folder is named after them, so different choices never overwrite each other:

```bash
python pipeline/run_experiment.py --config configs/eoh_baseline.yaml --llm openrouter-gemini-3.5-flash --budget pilot --seeds 0
python pipeline/evaluate_final.py --config configs/eoh_baseline.yaml --llm openrouter-gemini-3.5-flash --budget pilot
python pipeline/summarise.py      --config configs/eoh_baseline.yaml --llm openrouter-gemini-3.5-flash --budget pilot
# -> results/eoh_baseline__openrouter-gemini-3.5-flash__pilot/
```

Every step is also a Python function with the same options (`run_experiment.main(config, seeds=[0], llm=..., budget=...)`, `evaluate_final.main(...)`, `summarise.main(...)`), importable after `sys.path.insert(0, "pipeline")`.

---

## 6. How the pipeline works

**The problem.** Items arrive one at a time and must be placed immediately into bins of fixed capacity (100 by default). The goal is to use as few bins as possible. A heuristic is a Python function that scores each bin for the arriving item; the item goes into the highest-scoring bin:

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

**The metric: excess over the lower bound.** For each instance, the lower bound is `ceil(total item size / capacity)`, the fewest bins any packing could possibly use. Excess = (bins used − lower bound) / lower bound, averaged over instances. **Lower is better.** Plain Best Fit scores 4.751% on our training set: the floor any evolved heuristic must beat. The stronger bar is **tuned Best Fit** (§9).

**Which bound.** The bound above is Martello and Toth's L1. It is a bound, not the true minimum: where many items exceed half a bin it falls well short (on `test_scale65`, about 10%). The search scores on L1, as FunSearch and EoH do (on `train` L2 equals L1). Held-out results are excess over the tighter L2 bound on every set; see §11, Lower bound.

**The search and our layer.** [LLM4AD](https://github.com/Optima-CityU/LLM4AD) supplies the search methods; EoH is our baseline. A thin layer of our own (`pipeline/obpevo/`) wraps every method so they are all measured the same way, following the principles of the BLADE benchmark:

```
                     ┌──────────────── one run = one method × one seed ────────────────┐
LLM API  ◄──────  LoggedLLM  ◄──────  LLM4AD method (EoH / ReEvo / FunSearch / ...)  ──────►  EvalRecorder  ──────►  OBPTask
(OpenRouter, ...) logs every call      proposes priority() functions                         logs every candidate     packs the training
                  enforces token /                                                           enforces the eval        instances, returns
                  time budgets                                                               budget                   a score
```

Each method in LLM4AD has its own loop, logging and stopping rules. Comparing them directly would compare those differences, not the methods. Our layer enforces four rules for every method, without touching LLM4AD's code:

| Guarantee | How | Where |
|---|---|---|
| **1. Same logging for every method** | Every candidate goes through one patched function (`SecureEvaluator.evaluate_program`), and every LLM call through one wrapper (`LoggedLLM`). | `recorder.py`, `llm.py` |
| **2. Equal budgets** | A run stops at the first of `max_evals`, `max_tokens`, `max_minutes`, counted outside the method. Hitting the evaluation cap also blocks further LLM calls. | `recorder.py`, `llm.py` |
| **3. Train/test separation** | Search only ever sees the training set. Each run's best program is rescored afterwards on held-out sets it never saw. | `task.py`, `evaluate_final.py` |
| **4. Full records** | Every run saves its resolved config, library versions, every LLM call (tokens, cost, latency, full prompt and response), every candidate program with its per-instance results, and the best program. | `run_experiment.py` |

---

## 7. Configs

One YAML file fixes everything about an experiment. Example (`configs/eoh_baseline.yaml`, abridged):

```yaml
experiment: eoh_baseline # results folder (the reported runs add __<llm>: --llm openrouter-gemini-3.5-flash-lite)
instances: { spec: configs/instances/v0.yaml, dir: data/v0 }

llm: gemini-3.5-flash-lite # a name from configs/llms.yaml (model, prices, settings)
budget: full # a name from configs/budgets.yaml (300 evals, 1.5M tokens, 120 min)
parallel: { num_samplers: 2, num_evaluators: 2 }
evaluation: { mode: funsearch, timeout_seconds: 60, train_set: train, prompt: llm4ad }
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
methods:
  - { name: eoh, params: { pop_size: 10 } }
  - { name: tuned_bf }
final_eval:
  sets: [val, test_id_5k, test_size_1k, test_size_10k, test_cap200, test_cap500, test_cap500_scaled,
         test_scale25, test_scale65, test_shape_k1.5, test_shape_k5, test_uniform, test_control]
  timeout_seconds: 900
```

| Key | Meaning |
|---|---|
| `experiment` | results folder name. **Use a new name for every model or setting you compare**, or runs overwrite each other / get skipped as done. |
| `instances` | instance spec, and where generated sets are stored (built automatically on first run) |
| `llm` | a name from `configs/llms.yaml`, or a dict (§8) |
| `budget` | a name from `configs/budgets.yaml`, or a dict `{max_evals, max_tokens, max_minutes}`. The run stops at the first limit reached; keep `max_tokens` and `max_minutes` loose so `max_evals` normally binds. A dict replaces the preset entirely, so give all three. |
| `parallel` | LLM calls / evaluations in parallel (lower `num_samplers` if you hit rate limits) |
| `evaluation.mode` | `funsearch` (default, matches LLM4AD) or `anyfit` (§11) |
| `evaluation.timeout_seconds` | per-candidate time limit; ≥ 30 on Windows |
| `evaluation.train_set` | the only set the search sees: `train`, `train_mixed` (A2a) or `train_mixed_cap` (A2b) |
| `evaluation.prompt` | `llm4ad` (default, LLM4AD's wording), `informed_rules` (task framing plus how bins are offered: "online 1D bin packing, minimise bins", remaining capacities, empty bins included; the default prompt says none of this), `informed_dist` (the training distribution and capacity), or `informed` (both) |
| `evaluation.format_hint` | `true` (default): append formatting rules to every prompt (§11) |
| `seeds`, `methods` | runs = methods × seeds |
| `final_eval` | held-out sets and timeout for step 2 |

**Inheriting a config.** `base:` copies another config; only what you write replaces it. Every ablation config works this way (§9):

```yaml
base: configs/eoh_baseline.yaml
experiment: abl_anyfit
evaluation: { mode: anyfit }
```

Naming a different model replaces the whole `llm` section, so nothing carries over from the base model.

**From Python**, every step takes the same optional arguments. Pass the same ones to all three steps:

| Argument | Example | Effect |
|---|---|---|
| `llm` | `"openrouter-gemini-3.5-flash"` or `{"name": "...", "temperature": 0.5}` | model from the registry, optionally adjusted |
| `budget` | `"pilot"` or `{"max_evals": 80, "max_tokens": 500000, "max_minutes": 60}` | preset or custom budget |
| `experiment` | `"eoh_80"` | results folder name (derived from `llm`/`budget` if omitted) |
| `overrides` | `{"evaluation": {"prompt": "informed"}, "methods": [...]}` | any other setting |

`overrides` does **not** rename the results folder: pass `experiment=` with it, or runs overwrite each other.

---

## 8. Choosing the LLM

**`configs/llms.yaml` is the single source of truth** for every model we can use: provider, model ID, prices and settings. Configs and commands refer to a model by its name only:

```yaml
openrouter-gemini-3.5-flash:
  provider: openrouter
  model: google/gemini-3.5-flash
  price_out_per_1m: 9.00
  max_tokens: 8192
  extra: { reasoning: { effort: low } } # same reasoning setting as the lite model, so only the model changes
  checked: 2026-11-08 (openrouter.ai/api/v1/models)
```

| Name | Provider | Notes |
|---|---|---|
| `openrouter-gemini-3.5-flash-lite` | OpenRouter | **every reported run** |
| `openrouter-gemini-3.5-flash` | OpenRouter | the full model of the same generation (optional A3, not run) |
| `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.8-flash` | Gemini | direct Google API; free tier available (the 25 Oct run) |
| `claude-haiku-5-5` | Anthropic | paid only; no price set yet |
| `deepseek-chat` | DeepSeek | paid only; price unverified |
| `fake` | none | offline stand-in for tests and `smoke.yaml` |

Prices are in `configs/llms.yaml`, each with the date it was checked. List every model with `python -c "import sys; sys.path.insert(0, 'pipeline'); from obpevo import config as C; import pandas as pd; print(pd.DataFrame(C.llm_registry()).T)"`. Unknown names fail immediately with the list of valid ones.

**Adding a model:** add an entry to `configs/llms.yaml` with `provider`, `model`, both prices and a `checked` date. Prices change, so re-check the provider's pricing page and update `checked` when you do. Every result records the model name (`llm_name` in `summary.json`), so old results stay attributable.

**Providers** (the `provider` field) fill in the API address and which environment variable holds the key:

| `provider` | Key variable | Notes |
|---|---|---|
| `openrouter` | `OPENROUTER_API_KEY` | many providers' models with one key; used for every reported run |
| `gemini` | `GEMINI_API_KEY` | free tier on Flash / Flash-Lite models |
| `openai` | `OPENAI_API_KEY` | paid only; set `temperature: null` for models that only accept the default |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude; paid only |
| `deepseek` | `DEEPSEEK_API_KEY` | paid |
| `ollama` | none | local models at `localhost:11434` |
| `openai_compat` | your choice | any other OpenAI-style server: also give `base_url` and `api_key_env` |
| `fake` | none | offline stand-in for tests |

`cost_usd` is computed from the prices in `llms.yaml`; on a free tier it shows what the run _would_ cost on the paid tier. Free-tier prompts may be used by the provider, so don't send anything confidential. Pin exact model versions (not `-latest` aliases) so runs stay reproducible. Token counts include hidden reasoning tokens where the provider reports them; failed calls count zero tokens.

---

## 9. Methods, tuned Best Fit and ablations

| Method | `name` | Notes |
|---|---|---|
| EoH | `eoh` | the baseline and every ablation |
| Random sampling | `randsample` | no evolution (LLM4AD's fixed FunSearch-style prompt every call); best on train kept: the control |
| Tuned Best Fit (no LLM) | `tuned_bf` | the budget-matched baseline: Best Fit's 3 knobs, random search (below) |
| ReEvo | `reevo` | also asks the LLM for written reflections (~1.8 calls per evaluation); not run |
| FunSearch | `funsearch` | not run |
| (1+1)-EPS hill climbing | `hillclimb` | pure exploitation: always improves on the current best; not run |
| MCTS-AHD, HSEvo | `mcts_ahd`, `hsevo` | registered, untested |

**Run a method once with `pipecheck.yaml` before its first full run** (§5.2) to confirm the LLM's replies parse: FunSearch, hill climbing and random sampling use a different prompt style from EoH.

**Variants of a method** get their own `label`, results folder, line in plots and row in tables:

```yaml
methods:
  - { name: eoh, params: { pop_size: 10 } }
  - { label: eoh_no_m2, name: eoh, params: { pop_size: 10, use_m2_operator: false } }
```

Unknown params are dropped and listed in each run's `config.json` (`dropped_kwargs`).

**Tuned Best Fit (`tuned_bf`): the budget-matched baseline.** Beating plain Best Fit is weak evidence, so every LLM method is also compared with a Best Fit whose knobs get the same number of scored tries. Any score that strictly falls as the space left `r = bins - item` grows packs exactly like Best Fit, so the variant adds the smallest change that can differ: a penalty for leaving a small gap. This is the strategy the FunSearch paper describes for several of its discovered heuristics, illustrated with its OR-dataset heuristic (Fig. 6).

```python
r = bins - item
return -r - penalty * ((r > lo) & (r < hi))      # lo=0, hi=0, penalty=0 is plain Best Fit
```

The knobs are integers in item units: `lo` 0–10, `hi` lo+2–40, `penalty` 1–100. Like an evolved program, the rule is never given the capacity. (In `funsearch` mode an evolved program could still read it from `max(bins)`, since an empty bin is always offered; this rule doesn't.) The tuner scores Best Fit first (every LLM method also starts from the Best Fit template), then `max_evals − 1` distinct random settings, on the setting's training set only. It makes no LLM calls, takes about 2.5 minutes per seed (300 tries), and writes the same run files as an LLM method (with an empty `llm_calls.csv`), so scoring treats it like any other method. Its `summary.json` also records `best_params`. It is retuned in every setting, in that setting's packing mode and on its training set, so each ablation is compared with its own tuned Best Fit. To change the ranges, use `params: { space: { lo: [0, 10], hi_max: 40, penalty: [1, 100] } }`. An unknown key in `space`, or a range key put directly under `params`, stops the run with an error before anything is written.

**EoH's operators.** EoH creates new heuristics in four ways: **E1** (as different as possible from the parents), **E2** (new heuristic on the parents' shared idea), **M1** (modify one parent), **M2** (keep the idea, change its parameters). Roughly a spectrum from exploration (E1) to exploitation (M2). E2, M1 and M2 can be switched off (`use_e2_operator`, `use_m1_operator`, `use_m2_operator`); E1 cannot.

**The ablations.** Each has a config that inherits another one (`base:`) and changes one setting. All use `openrouter-gemini-3.5-flash-lite`, the `full` budget and seeds 0–9 (`docs/log.md`, 7 and 9 Nov). The main baseline is the comparison row (seeds 0–5 as pre-registered, 0–9 as a check; `docs/claims.md`, Amendments), except for A2b, which is compared with A1b.

| Config | Changes | Base | Status |
|---|---|---|---|
| `abl_anyfit.yaml` | A1a: empty bins hidden during search | `eoh_baseline.yaml` | run |
| `abl_informed_rules.yaml` | A1b: prompt states the task and how bins are offered (incl. empty bins) | `eoh_baseline.yaml` | run |
| `abl_mixed_shapes.yaml` | A2a: training on Weibull shapes 2 and 4 (`train_mixed`) | `eoh_baseline.yaml` | run |
| `abl_mixed_caps_informed.yaml` | A2b: training on capacities 100 and 300, item scale 45 (`train_mixed_cap`) | `abl_informed_rules.yaml` | run |
| `control_randsample.yaml` | LLM4AD RandSample: same budget, fixed FunSearch-style prompt, no evolution | `eoh_baseline.yaml` | run (control) |
| `abl_strong_model.yaml` | A3: `gemini-3.5-flash` instead of the lite model | `eoh_baseline.yaml` | not run (`docs/claims.md`, Amendments) |
| `abl_informed_dist.yaml` | prompt states the training distribution | `eoh_baseline.yaml` | dropped (7 Nov) |
| `abl_informed.yaml` | both prompt additions | `eoh_baseline.yaml` | dropped (7 Nov) |
| `abl_no_m2.yaml` | EoH without M2 | `eoh_baseline.yaml` | dropped (7 Nov) |

Other ablations the pipeline supports without code changes:

| Ablation | Setting |
|---|---|
| EoH operators | `params: {use_m2_operator: false}` etc. |
| Population size | `params: {pop_size: 5 / 10 / 20}` (LLM4AD suggests ~10 for 200+ evals, ~20 for 1000+) |
| Formatting rules | `evaluation: {format_hint: false}` |
| Reasoning / temperature | `llm={"name": "...", "extra": {...}}` or `{"name": ..., "temperature": 0.5}` |
| Budget | `--budget medium` / `full` / `large` |

**Change one thing at a time** against a baseline with the same model, budget and seeds, and use several seeds per arm: LLM runs vary a lot between seeds.

**The control's prompt** is different (Best Fit as `priority_v0`, "complete `priority_v1`"), so before its full seeds, check `n_unique_programs` and `n_calls_not_evaluated` in a `check` run's `summary.json`.

---

## 10. Outputs and audit trail

Every search run is committed in full, so any result can be traced back to the programs and LLM calls behind it. One folder per run: `code/results/<setting>/<method>/seed<k>/` (e.g. `code/results/abl_informed_rules/eoh/seed0/` holds everything behind A1b seed 0's capacity collapse).

**Per run:**

| File | Contents |
|---|---|
| `config.json` | resolved config, method kwargs used/dropped, library versions, exact prompt text added |
| `llm_calls.jsonl` | full prompt and response text of every LLM call |
| `llm_calls.csv` | one row per LLM call: tokens (`api` or `estimate`), latency, cost, status (with the provider's error message if it failed) |
| `candidates.jsonl` | every evaluated program |
| `evals.csv` | one row per candidate: status (`ok` / `error` / `timeout` / `crash`), score, train excess, integrity flags, error message |
| `per_instance.csv` | every candidate × training instance: bins, lower bound, excess |
| `best_program.py` | best program by training score (ties go to the earliest) |
| `summary.json` | counts, tokens, cost, stop reason, best result |
| `stdout.log`, `llm4ad_log_*/` | LLM4AD's own output, including EoH's population at every generation |

Tuned Best Fit runs (`tuned_bf/seed<k>/`) call no LLM: same files, minus `llm_calls.jsonl` and LLM4AD's logs (`llm_calls.csv` has only its header). Gaps: baseline seeds 0–5 ran before full prompt logging was added, so they have no `llm_calls.jsonl` (their per-call `llm_calls.csv` is there), and seeds 3–5 also have no `stdout.log`. Every other run has every file. Runs made before 4 Nov (the 25 Oct and free-tier runs) have only `summary.json`, `best_program.py`, `evals.csv` and `llm_calls.csv`, which are enough to rebuild their tables and figures. Fake-LLM and pipecheck results (`results/smoke/`, `results/pipecheck*/`) are git-ignored, since they are plumbing checks, not findings.

**Per setting** (`results/<setting>/`):

| File | Contents |
|---|---|
| `runs_index.csv` | one row per run |
| `final_per_instance.csv` | every program and baseline on every held-out instance (bins, L1 and L2 bounds, excess) |
| `final_summary.csv` | per set and method: mean, spread and difference from Best Fit (fields below) |
| `integrity.csv` | integrity rescoring of every best program (§11) |
| `diagnostics.csv`, `diagnostics_summary.csv` | `fresh_given_room` and scores in both packing modes |
| `paired_stats.csv` | per-seed verdicts against Best Fit and tuned Best Fit; the baseline also has `paired_stats_seeds0-5.csv` |
| `summary_tables.md` | the setting's tables |
| `figures/` | PNG + PDF: `convergence_evals` / `convergence_tokens` (best training excess so far vs evaluations / tokens, median and range across seeds, Best Fit line); `test_sets` (held-out change vs Best Fit per method and seed, one panel per set) |

**Across settings** (`results/`): `seed_counts.csv` and `seed_counts.md` (§4.4).

**Key fields:**

| Field | Meaning |
|---|---|
| `stop_reason` | `budget:evals` is the normal ending. `budget:tokens` / `budget:wall_clock`: another limit bound first (raise it). `budget:llm_errors`: the API kept failing (quota, key); the run is incomplete. `error`: see `error.txt`. |
| `best_train_excess` | best candidate's excess on training (compare with Best Fit on the same set) |
| `kind` | `evolved` (LLM search) or `tuned` (no-LLM tuner such as `tuned_bf`); `final_summary.csv` also has `baseline` |
| `best_params` | `tuned_bf` runs only: the winning knob setting |
| `n_calls_not_evaluated` | calls with no evaluation: candidates still being scored when the budget hit, unparseable replies, ReEvo reflections. 1–3 is normal for EoH. |
| `mean_excess` / `sd_excess` | held-out excess over **L1**, averaged within each run, then mean ± sd across seeds. L1 is the papers' bound (FunSearch, EoH), kept for comparison |
| `best_run` / `worst_run` | lowest / highest per-run mean excess across seeds, over **L1** |
| `delta_vs_bf_pp` | difference from Best Fit in percentage points over **L1**; **negative = better** |
| `*_l2` twins | `mean_excess_l2`, `sd_excess_l2`, `best_run_l2`, `worst_run_l2`, `delta_vs_bf_pp_l2`: the same over **L2**. **The report uses these**; they equal L1 except on `test_scale65` and `test_control` |
| `win_vs_bf` | share of instances where the program used fewer bins than Best Fit; counts bins, so the same under L1 and L2 |

**Paired statistics** (`paired_stats.py`, run by `score.sh`) compare every rule with Best Fit and tuned Best Fit on the same instances: wins/ties/losses, a two-sided sign test, the mean difference in percentage points with a 95% bootstrap interval, and a verdict: better / worse only when the interval excludes 0 **and** the sign test gives p < 0.05, tie when identical on every instance, otherwise unclear. Each seed is compared on its own, because runs of the same method differ a lot. Excess is over L2 by default; `--metric excess` compares over L1 (written to `paired_stats_excess.csv`). Against Best Fit, wins/ties/losses count bins, so they are the same under either bound. Tuned Best Fit has several seeds, so it is averaged per instance over the same seeds as the rule; `--seeds 0-5` restricts both to seeds 0–5 (`paired_stats_seeds0-5.csv`). Every test set has 10 instances, so a rule that beats Best Fit on all 10 gives p = 0.002 (9–1: 0.021). There is no correction for running many comparisons: look for patterns across seeds and sets rather than single verdicts.

**Reading search progress.** Individual candidates go up and down; only the best-so-far matters. Programs that look different can pack identically (e.g. `item - bins` and `-(bins - item)`), so many candidates share exactly the same score. Use the diagnostic in §12 to tell a normal plateau from a search that isn't exploring.

---

## 11. Evaluation details

**How a heuristic is called.** Once per arriving item, with `bins` = the **remaining** capacities of only the bins the item fits in. The item goes into the bin with the highest score (`np.argmax`; ties go to the earliest bin). Output must be the same length as `bins` and contain no NaN (both are errors); `-np.inf` is allowed to rule a bin out.

**Two packing conventions** (`evaluation.mode`):

- `funsearch` (default): one bin per item is set up at the start, all at full capacity, so `bins` always includes empty bins. The same packing convention as LLM4AD, FunSearch and EoH, and the same reported metric: our Best Fit excess matches LLM4AD's exactly (4.00% on its Weibull set). The search objective differs: our search maximises minus the mean relative excess over L1 per instance (as EoH-S), LLM4AD's minus the mean number of bins (`docs/log.md`, 9 Nov). Consequence: any heuristic that prefers emptier bins always opens a new bin and collapses to about one item per bin (~148% excess on our training set).
- `anyfit`: only open, non-empty bins are offered; a new bin opens only when none fit (Sgall's AnyFit class).

**Which convention each row uses.** Evolved programs are always scored in `evaluation.mode` (`funsearch` by default). The baseline rows in `final_summary.csv` use their classical AnyFit definitions, so Worst Fit is ~13% there but 148% in funsearch mode (on train). Best Fit and First Fit never open a fresh bin while a used bin has room, so the two conventions give them identical bins (`tests/test_diagnostics.py` checks the first part; the diagnostics summaries show identical scores on every set), so their rows compare directly. `pipeline/diagnose.py` rescores every program and baseline in both modes and reports `fresh_given_room`: the share of choices where a program put the item in an empty bin although a used bin had room. A seed is a **fresh-bin seed** if this is above 0.01 on its training set (`docs/claims.md`, shared rules).

**Lower bound.** `ceil(sum(items) / capacity)` (Martello and Toth's L1), the bound used by FunSearch and EoH. It is not the true optimum, so part of every heuristic's excess is unavoidable. The search and `excess` always use L1, so our numbers stay comparable with the papers. `evaluate_final.py` also writes Martello and Toth's tighter L2 bound (`lb_l2`, `excess_l2` per instance; `mean_excess_l2` in `final_summary.csv`). L2 counts that two items over half a bin can never share one. It matters most on `test_scale65`, where 63% of items exceed half a bin: Best Fit is 10.2% over L1 but 0.4% over L2, so L1 would make every rule look far worse than it is. L2 is also tighter on the control set `test_control`, where every rule must be exactly 0% over L2. On every other test set L2 equals L1. The report uses excess over L2 on every set (`docs/log.md`): `summarise.py`'s held-out table and test-set figure are on L2, and a second table shows L1 for the sets where it differs, for comparison with the papers.

**Formatting rules.** With `format_hint: true`, every prompt from every method ends with short rules: one complete function, the whole `def` line on one line, no helper functions, numpy only. LLM4AD's parser silently drops candidates with multi-line signatures and evolves the _wrong_ function when a helper comes first; the rules prevent both. The text is identical for all methods (fair), recorded in each `config.json`, and is our one deviation from LLM4AD's prompts.

**Integrity checks** (the brief: keep scoring outside the code being evolved). They flag, never drop, and never change a score:

- Every candidate is parsed (not run) by `obpevo/integrity.py`. Violations go in the `integrity` column of `evals.csv` (`''` = clean), and `summary.json` records `n_flagged` and `best_integrity`. Rules: imports other than numpy, `np.random` / `random` (also via `getattr` or `from numpy import *`), wall-clock time, `hash()` / `id()`, `exec` / `eval` / `compile` / `open` / `__import__`, numpy file I/O (`np.save` / `np.load` / ...), process-wide numpy settings, dunder access such as `__builtins__`, `global`, attribute assignment, mutating a module-level name, mutable default arguments. The full list is in the module docstring.
- **What the static check can see:** LLM4AD keeps only the _body_ of the LLM's function (template imports + `def` line + body); module-level imports, globals and changed signatures in the reply are thrown away before evaluation. So in our runs the body-level rules are the ones that can fire, and an empty column means "no body-level violation", not "the LLM never tried".
- `check_integrity.py` (run by `score.sh`) rescores each best program four times, each in a fresh Python process with its own hash seed and RNG seeds: twice in the usual order and twice in one shuffled order. `repeat_mismatch` = randomness (including `hash()`); `shuffle_mismatch` = state carried from one instance to the next. `verdict`: clean / flagged / stateful / nondeterministic / failed, in each setting's `integrity.csv`. Randomness that rarely changes a bin count can pass the rescoring; the static flags cover that.
- `tests/test_integrity.py` holds one deliberately bad program per rule, plus legitimate programs that must stay unflagged.

**Instance sets** (`configs/instances/v0.yaml`, generated deterministically from seed 2026; frozen, so changing a set changes its data):

| Set | Instances × items | Capacity | Items | Purpose |
|---|---|---|---|---|
| `train` | 32 × 200–2000 | 100 | Weibull(3) × 45 | the training set of the baseline, A1 and the control |
| `train_mixed` | 32 × 200–2000 | 100 | Weibull(2 and 4) × 45, alternating | A2a's training set |
| `train_mixed_cap` | 32 × 200–2000 | 100 and 300, alternating | Weibull(3) × 45 | A2b's training set |
| `val` | 10 × 1000 | 100 | Weibull(3) × 45 | choosing between configurations; never seen by the search. Within a run, the best program is picked on training, as in the EoH papers |
| `test_id_5k` | 10 × 5000 | 100 | Weibull(3) × 45 | in-distribution (FunSearch/EoH standard) |
| `test_size_1k` | 10 × 1000 | 100 | Weibull(3) × 45 | size shift (smaller); its own instances, since `val` is used to choose between runs |
| `test_size_10k` | 10 × 10000 | 100 | Weibull(3) × 45 | size shift |
| `test_cap200` | 10 × 5000 | 200 | Weibull(3) × 45 | bigger bins, same items (midpoint of the 100 → 200 → 500 series) |
| `test_cap500` | 10 × 5000 | 500 | Weibull(3) × 45 | bigger bins, same items (5/10 A1b seeds collapse here) |
| `test_cap500_scaled` | 10 × 5000 | 500 | Weibull(3) × 225 | everything ×5: scale-invariance check |
| `test_scale25` | 10 × 5000 | 100 | Weibull(3) × 25 | smaller items (~22% of a bin; training ~40%) |
| `test_scale65` | 10 × 5000 | 100 | Weibull(3) × 65 | bigger items (~58% of a bin). 63% exceed half a bin, so read excess over L2 |
| `test_shape_k1.5` | 10 × 5000 | 100 | Weibull(1.5) × 45 | wider spread of item sizes |
| `test_shape_k5` | 10 × 5000 | 100 | Weibull(5) × 45 | tightly clustered item sizes |
| `test_uniform` | 10 × 1000 | 150 | Uniform[20, 100] | generated like OR-Library's binpack1–4: different distribution, capacity and size |
| `test_control` | 10 × 1000 | 100 | Uniform[51, 100] | control: every item is over half a bin, so every rule must use one bin per item; any difference is a scoring bug |

Each test set changes one thing from training, so a failure can be attributed to that change. The exceptions are `test_cap500_scaled` (capacity and item sizes ×5 together, on purpose), `test_uniform` (distribution, capacity and size) and the control set. Every test set has 10 instances, so a rule that beats Best Fit on all of them is significant on the sign test (p = 0.002; with 5 it could not go below 0.0625).

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `skip eoh seed0 (done)` and old numbers | a previous run (even a failed one) counts as finished | `--force` |
| `score.sh --check` lists files | a committed output changed | look at `git diff` on the listed files; only figures, CSV row order and `time_s` may differ |
| `bash: command not found` / `score.sh` won't run (Windows) | PowerShell has no bash | run it from Git Bash or WSL |
| `RuntimeError: Set the ... environment variable` | the key isn't in the shell's environment | load `.env` into the shell (§5.1) |
| Edits to `pipeline/` have no effect (Python session) | module already loaded | restart Python |
| Many `timeout` rows with `wall_s` ≈ the timeout (Windows) | process start-up per candidate | raise `evaluation.timeout_seconds` (30+ for small sets, 60+ for `train`) |
| `stop_reason: budget:llm_errors` | API kept failing; `llm_calls.csv` `status` has the message | 429 / quota: wait, add credit, or lower `num_samplers`; 400/403: model name or key. Rerun with `--force` |
| Thousands of calls, 0 evals, `budget:tokens` | a broken `llm.py` edit (calls fail instantly) | run the one-call test below; check indentation inside `LoggedLLM.draw_sample` |
| `FileExistsError` / `PermissionError` on `results/.../seed0` (Windows) | an old process holds files in the run folder | close Explorer windows there; end stray `python.exe` in Task Manager; restart |
| `KeyError: LLM '...' is not in configs/llms.yaml` | typo or a model not yet added | use a listed name, or add an entry (§8) |
| `best_train_excess` is `None` | no candidate ran successfully | check `stop_reason`, `llm_calls.csv`, `evals.csv`, `error.txt` |

**One-call test through the same wrapper the pipeline uses** (from `code/`, with the key loaded):

```python
import sys, tempfile
from pathlib import Path
import pandas as pd
sys.path.insert(0, "pipeline")
from obpevo import config as C
from obpevo.llm import make_backend, LoggedLLM
cfg = C.load("configs/eoh_baseline.yaml", llm="openrouter-gemini-3.5-flash-lite")
log = Path(tempfile.mkdtemp()) / "calls.csv"
w = LoggedLLM(make_backend(cfg["llm"]), log, price_in_per_1m=0.3, price_out_per_1m=2.5)
print(w.draw_sample("Say hi")[:100]); print(pd.read_csv(log))    # one row: status ok, token_source api
```

**Is the search exploring, or stuck?** With `ev` from §5.3:

```python
ok = ev[ev.status == "ok"]
print(f"ok: {len(ok)} | same score as best: {(abs(ok.train_excess - ok.train_excess.min()) < 1e-9).sum()} "
      f"| distinct scores: {ok.train_excess.round(6).nunique()}")
```

Many candidates tied with Best Fit and few distinct scores mean the LLM keeps rewriting Best Fit (EoH discards same-score candidates as duplicates, so those evaluations are wasted).

---

## 13. Known limitations

- The searches don't reproduce exactly (§4.6); the analysis rests on 10 seeds per setting and per-seed verdicts.
- Only EoH, RandSample and tuned Best Fit have been run. ReEvo, FunSearch and hill climbing are wired up but not run (FunSearch also needs `scipy`, which we do not install); MCTS-AHD and HSEvo are untested.
- Only each run's single best program (by training score) goes to held-out evaluation. Selecting among the top programs on `val` instead is a possible extension.
- No operator attribution (which EoH operator produced each candidate) and no diversity metrics (clone rate, collapse rate, behavioural diversity).
- `tuned_bf`'s knobs are in absolute item units, tuned at the training capacity (like the evolved programs). A variant with knobs relative to the bin size would test the hard-coded-capacity explanation for the capacity failures.
- Prices in `configs/llms.yaml` are dated (`checked`); re-check them before trusting `cost_usd`. `claude-haiku-5-5` has no price set yet.
