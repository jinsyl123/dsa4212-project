# DSA4212 Project: LLM-guided evolution of online bin packing heuristics

We use an LLM-driven search (**EoH**, via [LLM4AD](https://github.com/Optima-CityU/LLM4AD)) to evolve a priority rule for **online bin packing**, and ask what makes the evolved rules beat Best Fit, and why some of them fail on shifted test sets. Two ablations and a control test where the gains and failures come from:

| Setting | What changes from the EoH baseline | Config (`code/configs/`) | Results (`code/results/`) |
|---|---|---|---|
| EoH baseline | nothing | `eoh_baseline.yaml` | `eoh_baseline__openrouter-gemini-3.5-flash-lite/` |
| A1a: empty bins hidden | the rule is never offered an empty bin while a used bin fits (`mode: anyfit`) | `abl_anyfit.yaml` | `abl_anyfit/` |
| A1b: bin rules in the prompt | the prompt states the task and that empty bins are among the options | `abl_informed_rules.yaml` | `abl_informed_rules/` |
| A2: mixed training shapes | trained on Weibull shapes 2 and 4 instead of 3 | `abl_mixed_shapes.yaml` | `abl_mixed_shapes/` |
| Control: no evolution | LLM4AD RandSample: 300 samples of one fixed prompt, best kept | `control_randsample.yaml` | `control_randsample/` |

Every setting is compared with Best Fit and with **tuned Best Fit** (Best Fit's 3 knobs tuned without an LLM, under the same 300-evaluation budget). Predictions for each setting were frozen before scoring in [`code/docs/claims.md`](code/docs/claims.md); every decision that affects results is in [`code/docs/log.md`](code/docs/log.md).

Assignment guidelines: https://alexxthiery.github.io/teaching/guidelines/guidelines.html
Project brief: https://alexxthiery.github.io/teaching/llm-evolution/llm-evolution.html

## Team links

- Overleaf report: https://www.overleaf.com/project/6ad866a8d72db1d3c73a0dba
- Google Docs: https://docs.google.com/document/d/1Y8FMyBoCH7kqoFPx2sO1rX7DdENuaLY6vYlsOpZdIk0/edit?usp=sharing
- Report LaTeX template (Overleaf, read-only): https://www.overleaf.com/read/hrstqvkjhzff#a35f95

## Repository layout

```
├── readme.md         # this file: overview, setup, how to reproduce
├── authors.md        # name and student ID of every group member
├── report/           # LaTeX source of the submitted report
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

From the `code/` folder, with no LLM calls:

```bash
python pipeline/evaluate_final.py --config configs/eoh_baseline.yaml --llm openrouter-gemini-3.5-flash-lite
python pipeline/summarise.py      --config configs/eoh_baseline.yaml --llm openrouter-gemini-3.5-flash-lite
```

This rescores the committed best programs on every held-out set and rebuilds `final_summary.csv`, `summary_tables.md` and the figures. Rerunning a *search* does not give the same programs, because LLM sampling is random; that is why every run folder, including every LLM call, is committed.

_To be finalised once the ablation scoring (#18) is merged:_
- _#18 commits the baseline rescored on the frozen test sets (#16). Until then, the baseline results committed here are on the earlier sets, so the commands above give the frozen-set numbers, not the committed ones._
- _#18 adds one command per setting, `bash pipeline/score.sh --check <setting>`, which rescores it and fails if any committed result differs._

## Results

_To be added once Ablation 2 is scored: headline numbers per setting, with links to the report's figures and tables._
