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
| Setting-level effect | Descriptive only: the number of seeds (of 5) in each category, next to the baseline's (of 6). Even the most extreme split, 3/6 vs 0/5 or 5/5, gives Fisher p = 0.18, so **seed counts are never called significant**. Significance comes only from the per-seed paired verdicts above. |
| Mechanism measure | `fresh_given_room` from `diagnose.py` (#5): share of placements that open an empty bin while a used one has room. A seed is a **fresh-bin seed** if it is > 0.01 on train |
| Held-out sets used for claims | **in-distribution:** `test_id_5k`; **shift:** `test_shape_k1.5`, `test_shape_k5`, plus the other #12 sets as secondary. `train` is reported but not a claim (it is the set searched on) |
| Integrity | `check_integrity.py` (#6): flagged programs reported, not dropped |

**Baseline to compare against** (EoH, 6 seeds, `baseline_findings.md`): 3/6 fresh-bin seeds (3, 4, 5). Seeds 0–2
behave like Best Fit (2 exactly; seed 2 Best Fit for small gaps, else First Fit). Seeds 4/5 beat Best Fit on train
(3.72 vs 4.75) but lose on k5 (10.62 vs 8.25) through fresh bins. Seed 3 loses on k1.5 (3.92 vs 1.34) for a different
reason (an accidental skip window). Tuned Best Fit: train 2.4 under `funsearch`, 4.65 with empty bins blocked.

## Template (copy for each setting)

- **Changes from baseline:** one thing only.
- **Question:** what this setting tells us that the baseline cannot.
- **Hypothesis and why:** the mechanism we expect, and the baseline evidence behind it.
- **Prediction:** direction and rough size, on which sets, for which measure.
- **What counts as an effect:** using the shared rules above.
- **If confirmed, we claim:**
- **If not confirmed, we claim:**
- **What remains uncertain either way:**

---

## A1a. Empty bins hidden during search (`abl_anyfit.yaml`)

- **Changes from baseline:** `evaluation: {mode: anyfit}`. The rule is never offered an empty bin while a used bin fits, so the fresh-bin bet is impossible by construction (`fresh_given_room` = 0).
- **Question:** can EoH still beat Best Fit without the fresh-bin bet?
- **Hypothesis and why:** part of EoH's gain does not need fresh bins. When seeds 4/5 are rescored with empty bins blocked, about half the train gain remains (4.31 vs Best Fit 4.75), and the k5 failure disappears (8.18 vs 8.25). Tuned Best Fit keeps almost none of its gain (4.65), so a 3-knob family cannot find this part, but an LLM might.
- **Prediction:**
  - At least 2 of 5 seeds are **better** than Best Fit on `test_id_5k`, by about 0.3–0.5 pp (half the baseline's ~1 pp).
  - **No** seed is worse than Best Fit on `test_shape_k5` (the fresh-bin failure cannot happen).
  - Failures of seed 3's kind (k1.5) can still appear: they do not come from fresh bins.
  - The anyfit-retuned tuned Best Fit is within ~0.1 pp of Best Fit.
- **What counts as an effect:** per-seed **better** verdicts vs Best Fit on `test_id_5k`, and vs anyfit tuned Best Fit.
- **If confirmed, we claim:** EoH finds gains that do not rely on opening empty bins, which a tuned 3-knob Best Fit does not. Removing the option also removes the k5 failure, so the option drives the failure, not the gain as a whole.
- **If not confirmed, we claim** (all seeds tie with or lose to Best Fit): the fresh-bin bet was the whole of EoH's advantage on this distribution. Without it, EoH rediscovers Best Fit. The brief counts this as evidence too ("rediscovering a known heuristic … can also provide evidence").
- **What remains uncertain either way:** whether hiding empty bins changes what the LLM proposes or only what gets selected. Read a sample of `candidates.jsonl` to tell them apart.

## A1b. Bin rules explained in the prompt (`abl_informed_rules.yaml`)

- **Changes from baseline:** `evaluation: {prompt: informed_rules}`. The prompt states the task ("online 1D bin packing … minimise the number of bins") **and** that the function scores "the bins the current item fits into (remaining capacities, including empty bins)". The default LLM4AD prompt says neither, so this setting changes the **task framing and the bin rules together**.
- **Question:** does knowing that empty bins are among the options change how often evolved rules open them?
- **Hypothesis and why:** the default LLM4AD prompt never mentions empty bins, so the baseline found the bet by accident in 3/6 seeds. Saying so explicitly makes it easier to target: **more** fresh-bin seeds, found **earlier**. (The opposite is possible: an LLM told about empty bins may treat them as a last resort, as Best Fit does.)
- **Prediction:**
  - At least 4 of 5 fresh-bin seeds (baseline 3/6).
  - The best rule is found earlier than the baseline's eval ~200.
  - Hence more seeds better on train and `test_id_5k`, **and** more seeds worse on `test_shape_k5`.
- **What counts as an effect:** the descriptive seed counts and median discovery eval, and per-seed verdicts on `test_id_5k` and `test_shape_k5`.
- **If confirmed, we claim:** the fresh-bin habit comes partly from the prompt: the LLM exploits options it is told about, and that gain and that failure come together.
- **If not confirmed, we claim:** the bet comes from selection pressure on this training set, not from what the LLM is told. Tuned Best Fit supports this reading, since it finds the bet with no prompt at all.
- **What remains uncertain either way:**
  - **Which addition caused it:** an effect can't be pinned on the empty-bin clause alone, since the task framing changed too. Claim "task framing + bin rules", not "telling it about empty bins".
  - **Seed counts:** with 5 seeds, a 3/6 vs 4/5 difference can be chance (see the shared rules). This setting can only give strong evidence if the counts end up at 0/5 or 5/5.

## A2. Training-data variety (`abl_mixed_shapes.yaml`)

- **Changes from baseline:** `evaluation: {train_set: train_mixed}`. Same sizes and capacity, with half the instances at Weibull shape 2 and half at shape 4. Test shapes 1.5 and 5 stay out.
- **Question:** do the k1.5 / k5 failures come from training on a single item-size shape?
- **Hypothesis and why:** the fresh-bin bet pays off only when later small items fill the gaps, which holds for shape 3 but not for clustered sizes (k5: 9% single-item bins). Shape-4 training instances are clustered enough to punish the bet during search, so selection should prefer rules that hold up across shapes. Seed 3's skip window fitted shape 3 by accident and would be scored on two shapes instead.
- **Prediction:**
  - Fewer seeds worse than Best Fit on `test_shape_k5` or `test_shape_k1.5` (baseline: 3 of 6 lose on one of them).
  - Smaller gains on `test_id_5k` than the baseline's best seeds: about 0–0.5 pp, against ~1 pp.
  - `fresh_given_room` lower on average.
- **What counts as an effect:** per-seed **worse** verdicts on k1.5 and k5 vs Best Fit and vs `train_mixed`-retuned tuned Best Fit, compared with the baseline's.
- **If confirmed, we claim:** the shape-shift failures come from training on one shape. A more varied training set trades some in-distribution gain for robustness.
- **If not confirmed, we claim:** the failures don't come from the training shape alone. The rule family or the search picks fragile rules even when the training data covers more than one shape.
- **What remains uncertain either way:** shapes 2 and 4 bracket the training shape but not the test shapes (1.5, 5), so this tests interpolation, not extrapolation. Mixing also changes the average training difficulty.

## Control. Random sampling, no evolution (`control_randsample.yaml`)

- **Changes from baseline:** LLM4AD `randsample`. The same 300-evaluation budget, no evolution: every call sends the **same fixed FunSearch-style prompt** (Best Fit as `priority_v0`, "complete `priority_v1`"), with no task description and no parents. The best on train is kept, as in EoH. So the prompt differs from EoH's as well as the search.
- **Question:** does the evolutionary search add anything beyond sampling the LLM many times and keeping the best?
- **Hypothesis and why:** evolution helps. In the baseline, the rules that beat Best Fit appeared late (eval ~200 of 300, seeds 3–5), after many generations. Seeds 0 and 1 found Best Fit at eval 1 and never improved, so the LLM's first guess is Best Fit.
- **Prediction:**
  - Most seeds end at Best Fit or a near copy.
  - At most 1 of 5 fresh-bin seeds (baseline 3/6).
  - No seed better than the best EoH seeds on `test_id_5k`.
- **What counts as an effect:** seed counts and per-seed verdicts vs Best Fit, and each seed's paired difference vs EoH's best seeds on `test_id_5k`.
- **If confirmed, we claim:** evolution, not the LLM's prior alone, finds the rules that beat Best Fit (including the fresh-bin bet).
- **If not confirmed, we claim:** best-of-300 sampling matches EoH, so its gains come from the LLM's prior plus selection on train, not from evolution. That is a negative result worth reporting.
- **What remains uncertain either way:**
  - **Search vs prompt:** randsample changes the prompt too, so "no evolution" and "different prompt" are confounded. Report it as "LLM4AD RandSample", not "EoH without search".
  - **Unique programs:** a fixed prompt may return many duplicates (EoH's 300 evals hold 160–220 unique ones). Report unique programs per run next to the result.
  - **The check run is a gate:** if its replies mostly fail to parse or repeat, fix that before the 5 seeds.

## Optional settings (fill in only if they will be run)

`abl_informed_dist.yaml`, `abl_informed.yaml`, fallback `abl_no_m2.yaml`.

---

## Amendments

Dated entries only, each with the reason. Nothing above changes after the freeze.
