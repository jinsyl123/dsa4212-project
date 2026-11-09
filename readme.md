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

Every setting is compared with Best Fit and with **tuned Best Fit** (Best Fit's 3 knobs tuned without an LLM, under the same 300-evaluation budget). Predictions for each setting were frozen before scoring in [`code/docs/claims.md`](code/docs/claims.md); every decision that affects results is in [`code/docs/log.md`](code/docs/log.md).

Assignment guidelines: https://alexxthiery.github.io/teaching/guidelines/guidelines.html
Project brief: https://alexxthiery.github.io/teaching/llm-evolution/llm-evolution.html

## Team links

- Overleaf report: https://www.overleaf.com/project/6ad866a8d72db1d3c73a0dba

## Repository layout

```
├── readme.md         # this file: overview, setup, how to reproduce
├── authors.md        # name and student ID of every group member
├── report/           # LaTeX source of the submitted report
├── docs/log.md       # early project log (topic and framework, 21 Oct); experiment decisions are in code/docs/log.md
├── results/          # pointer only: every result is in code/results/
└── code/
    ├── readme.md     # full pipeline guide: setup, configs, methods, outputs, troubleshooting
    ├── pipeline/     # evaluator, search wrapper, scoring and plotting scripts
    ├── configs/      # one YAML per setting, plus LLMs, budgets and instance sets
    ├── tests/        # unit tests
    ├── docs/         # claims.md (frozen predictions), log.md (decision log)
    └── results/      # one folder per setting: every run, its LLM calls, scores and figures
```

All numerical results are under `code/results/<setting>/` (see the table above), next to the code that produced them.

## Setup

**Python 3.10–3.12** (LLM4AD needs `numpy<2`). From the `code/` folder:

```bash
pip install -r requirements.txt      # installs LLM4AD at the pinned commit
python -m unittest discover tests    # every test should pass
```

Reproducing the results needs **no API key**: scoring only rescores the committed programs. A key is needed only to rerun a search (see [`code/readme.md`](code/readme.md), §4 and §8).

## Reproducing the main results

From the `code/` folder, with no LLM calls (about 1.5–2 hours in total; on Windows, run from Git Bash):

```bash
bash pipeline/score.sh --check --llm openrouter-gemini-3.5-flash-lite eoh_baseline
bash pipeline/score.sh --check abl_anyfit abl_informed_rules control_randsample
bash pipeline/score.sh --check abl_mixed_shapes abl_mixed_caps_informed
python pipeline/seed_counts.py       # seed counts and Fisher tests across settings (seconds)
```

For each setting, `score.sh` retunes tuned Best Fit, rescores the committed best programs on every held-out set, runs the integrity checks, the fresh-bin diagnostic and the paired statistics, and rebuilds `final_summary.csv`, `paired_stats.csv`, `summary_tables.md` and the figures. With `--check` it then prints `OK: every committed output reproduced`, or lists the files that differ. CSV row order and measured run time are ignored, and changed figures are not compared (their bytes depend on the matplotlib version and fonts; the CSVs they are drawn from are), though a missing or new figure still fails. `seed_counts.py` then rebuilds `code/results/seed_counts.csv` and `seed_counts.md` from those outputs. Rerunning a *search* does not give the same programs, because LLM sampling is random; that is why every run folder, including every LLM call, is committed.

## Results

Frozen test sets, excess over the L2 lower bound, 10 instances per set. Per-seed verdicts: a seed is **better** or **worse** than a reference only if the 95% bootstrap CI excludes 0 and the sign test gives p < 0.05, **tie** if it matches the reference on every instance, and **unclear** otherwise. Full tables and per-seed numbers: `code/results/<setting>/summary_tables.md` and `paired_stats.csv`; the predictions they are checked against: `code/docs/claims.md`. The capacity and scale failures (A1b on `test_cap500`, `test_cap200`, `test_scale25`) were found after A1 was scored; they were not pre-registered.

| Setting | Seeds | Beats Best Fit on `test_id_5k` | Beats its tuned Best Fit on `test_id_5k` | Worse than Best Fit on k1.5 or k5 | Fresh-bin seeds |
|---|---|---|---|---|---|
| EoH baseline, seeds 0–5 (pre-registered) | 6 | 3/6 | 0/6 | 3/6 | 3/6 |
| EoH baseline, seeds 0–9 | 10 | 5/10 | 0/10 | 3/10 | 3/10 |
| Control: no evolution | 10 | **0/10** | 0/10 | 0/10 | 0/10 |
| A1a: empty bins hidden | 10 | 6/10 | 0/10 | 1/10 | 0/10 (by design) |
| A1b: bin rules in the prompt | 10 | **10/10** | 0/10 | 1/10 | 6/10 |
| A2a: mixed training shapes | 10 | 9/10 | 1/10 | 1/10 | 2/10 |
| A2b: mixed training capacities (vs A1b) | 10 | 6/10 | 0/10 | 1/10 | 2/10 |

Bold: claimed, i.e. Fisher's exact test on seed counts gives p < 0.05 against both the 6- and 10-seed baseline. Every cell with its p-values: `code/results/seed_counts.md`; every category and set: `seed_counts.csv`.

**What holds up** (significant against both the 6- and 10-seed baseline):
- **The control** never beats Best Fit (0/10; p = 0.036 and 0.033): without evolution, the LLM does not get past Best Fit.
- **A1b** beats Best Fit more often than the baseline (10/10; p = 0.036 and 0.033): stating the task and bin rules in the prompt makes the gain reliable. But 5/10 A1b seeds collapse on `test_cap500` (about 1,140% over the bound, Best Fit 0.5%), which A2b tests.

**Consistent with the predictions but not significant with 10 seeds:** mixed shapes (A2a) cut shape failures from 3/6 to 1/10; mixed capacities (A2b) remove A1b's capacity collapse inside the trained range (0/10 vs 4/10 on `test_cap200` and `test_scale25`, 2/10 vs 5/10 on `test_cap500`).

**Patterns across every setting:** no LLM rule beats its own tuned Best Fit in distribution (1 of 60 seeds, A2a seed 6). The fresh-bin bet (opening an empty bin while a used one has room) carries both the gains and most of the failures under shift; tuned Best Fit is the most fragile rule under shape and scale shifts (e.g. about 30% over the bound on `test_shape_k5`, vs Best Fit's 8%).

## Audit trail

Every search run is committed in full, so any result can be traced back to the programs and LLM calls behind it. One folder per run: `code/results/<setting>/<method>/seed<k>/` (e.g. `code/results/abl_informed_rules/eoh/seed0/` holds everything behind A1b seed 0's capacity collapse).

| Evidence | File in the run folder |
|---|---|
| Every program the search evaluated | `candidates.jsonl` |
| Every prompt sent to the LLM and its full response | `llm_calls.jsonl` |
| Tokens, cost, latency and errors per LLM call | `llm_calls.csv` |
| Score and status of every candidate (ok / error / timeout) | `evals.csv` |
| Every candidate on every training instance (bins, lower bound, excess) | `per_instance.csv` |
| The best program on the training set | `best_program.py` |
| EoH's population at every generation, and LLM4AD's own log | `llm4ad_log_*/`, `stdout.log` |
| Settings used, and the run's counts, cost and stop reason | `config.json`, `summary.json` |

Tuned Best Fit runs (`tuned_bf/seed<k>/`) call no LLM: same files, minus `llm_calls.jsonl` and LLM4AD's logs (`llm_calls.csv` has only its header). Gaps: baseline seeds 0–5 ran before full prompt logging was added, so they have no `llm_calls.jsonl` (their per-call `llm_calls.csv` is there), and seeds 3–5 also have no `stdout.log`. Every other run has every file. What each file contains, column by column: [`code/readme.md`](code/readme.md), §10.
