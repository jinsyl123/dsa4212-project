# Frozen claims for the ablations (#7)

Written **before** any ablation result is scored. For each setting: what we expect, how we will measure it,
and what we will conclude whichever way it goes. Once this file is merged, it is frozen: later changes go in a
dated **Amendments** entry at the bottom, never as edits above. The report checks its claims against this file.

Team agreement: "Freeze test sets **and the claim we'll make for each outcome** before the main searches"
(decisions doc, "Agreed by default"). Owner: Jia Kang (claims doc). Written by: ___. Frozen on: ___.

## What the course asks for (why this file exists)

- **Experimental design, 35%:** "breadth and depth, controlled comparisons, and follow-up on findings"
  (Assignment guidelines, Marking). Each setting below changes one thing from the baseline.
- **Analysis, 20%:** "evidence-based conclusions, discussion of limitations". Fill in **What remains
  uncertain** for every setting.
- **Negative results count:** "A carefully investigated negative result can earn high marks" (guidelines),
  and "rediscovering a known heuristic or failing to improve a baseline can also provide evidence about the
  problem and the search procedure" (project brief, Experimental considerations). So **every** outcome below
  needs a conclusion, not only the one we hope for.
- **Explain success and failure:** investigate "the features of the problem that explain success or failure"
  and "construct a case where the rule fails" (project brief). The predictions should name the mechanism
  (e.g. opening fresh bins), not only the direction.
- **Comparable budgets:** compare "under comparable execution budgets" (brief). Every setting uses the `full`
  budget (300 evals) and `openrouter-gemini-3.5-flash-lite`.

## Shared rules (apply to every setting)

| Item | Rule |
|---|---|
| Comparison row | EoH baseline, seeds 0–5 (`eoh_baseline.yaml`) |
| Seeds per setting | 5 (seeds 0–4) |
| References | Best Fit, and tuned Best Fit retuned in the same setting (#8) |
| Sets | train, val (selection only, never reported as a test), and the test sets frozen in #12 |
| Metric | excess over L1 (%), over L2 on `test_scale65` (#10); fewer bins = better |
| Per-seed verdict vs a reference | `paired_stats.py` (#13): **better / worse** only if the 95% bootstrap CI excludes 0 **and** sign-test p < 0.05; **tie** if identical on every instance; else **unclear** |
| Setting-level effect | ___ (e.g. "at least 3 of 5 seeds better than the baseline on ___"; decide before the runs) |
| Mechanism measure | `fresh_given_room` from `diagnose.py` (#5): share of placements that open an empty bin while a used one has room |
| Integrity | `check_integrity.py` (#6): flagged programs reported, not dropped |

## Template (copy for each setting)

- **Changes from baseline:** one thing only.
- **Question:** what this setting tells us that the baseline cannot.
- **Hypothesis and why:** the mechanism we expect, and the baseline evidence behind it (#5, `baseline_findings.md`).
- **Prediction:** direction and rough size, on which sets, for which measure.
- **What counts as an effect:** using the shared rules above.
- **If confirmed, we claim:**
- **If not confirmed, we claim:**
- **What remains uncertain either way:** e.g. 5 seeds, one LLM, one problem family.

---

## A1a. Empty bins hidden during search (`abl_anyfit.yaml`)

- **Changes from baseline:** `evaluation: {mode: anyfit}`; the rule is never offered an empty bin while a used bin fits.
- **Question:** can EoH still beat Best Fit without the fresh-bin bet?
- **Hypothesis and why:**
- **Prediction:**
- **What counts as an effect:**
- **If confirmed, we claim:**
- **If not confirmed, we claim:**
- **What remains uncertain either way:**

## A1b. Bin rules explained in the prompt (`abl_informed_rules.yaml`)

- **Changes from baseline:** `evaluation: {prompt: informed_rules}`; the prompt says how bins are offered, including empty bins.
- **Question:** does knowing the bin rules change how often evolved rules open fresh bins?
- **Hypothesis and why:**
- **Prediction:**
- **What counts as an effect:**
- **If confirmed, we claim:**
- **If not confirmed, we claim:**
- **What remains uncertain either way:**

## A2. Training-data variety (`abl_mixed_shapes.yaml`)

- **Changes from baseline:** `evaluation: {train_set: train_mixed}`; Weibull shapes 2 and 4 (test shapes 1.5 and 5 stay out).
- **Question:** do the k1.5 / k5 failures come from training on a single item-size shape?
- **Hypothesis and why:**
- **Prediction:**
- **What counts as an effect:**
- **If confirmed, we claim:**
- **If not confirmed, we claim:**
- **What remains uncertain either way:**

## Control. Random sampling, no search (`control_randsample.yaml`)

- **Changes from baseline:** `randsample`: the same number of LLM proposals, no selection or evolution.
- **Question:** does the evolutionary search add anything over sampling the LLM 300 times and keeping the best?
- **Hypothesis and why:**
- **Prediction:**
- **What counts as an effect:**
- **If confirmed, we claim:**
- **If not confirmed, we claim:**
- **What remains uncertain either way:**

## Optional settings (fill in only if they will be run)

`abl_informed_dist.yaml`, `abl_informed.yaml`, fallback `abl_no_m2.yaml`.

---

## Amendments

Dated entries only, each with the reason. Nothing above changes after the freeze.
