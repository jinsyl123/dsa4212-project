# Frozen claims for the ablations (#7)

Written **before** any ablation result is scored. For each setting: what we expect, how we will measure it,
and what we will conclude whichever way it goes. Once this file is merged, it is frozen: later changes go in a
dated **Amendments** entry at the bottom, never as edits above. The report checks its claims against this file.

Team agreement: "Freeze test sets **and the claim we'll make for each outcome** before the main searches"
(decisions doc, "Agreed by default"). Owner: Jia Kang (claims doc). Written by: Jin. Frozen on: ___ (the day #15 merges).
Ablation 1 predictions (A1a, A1b) committed on 7 Nov 2026, before any Ablation 1 run was scored; the commit timestamp is the record. A1 was then scored before this file was frozen: see Amendments, 7 Nov.

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
| Seeds per setting | **Ablation 1 (A1a, A1b) and the control: 10 (seeds 0–9)**, as run (#7). A2: 5 (seeds 0–4). Every seed run in full is scored; none is dropped |
| References | Best Fit, and tuned Best Fit retuned in the same setting (#8), with the same seeds as the setting. A setting that changes neither `mode` nor the training set (A1b, control) leaves tuned Best Fit unchanged: it is still retuned in its own folder, and seeds 0–5 must match the baseline's exactly (a determinism check) |
| Sets | train, val (selection only, never reported as a test), and the test sets frozen in #12 |
| Metric | excess over L2 (%) on every set (#12 decision 7, #10); fewer bins = better. Same as L1 except on `test_scale65` and `test_control`, so numbers stay comparable with FunSearch/EoH |
| Per-seed verdict vs a reference | `paired_stats.py` (#13, scored with its code as of this freeze; if #13 changes before merging, rescore and note it in Amendments): **better / worse** only if the 95% bootstrap CI excludes 0 **and** sign-test p < 0.05; **tie** if identical on every instance; else **unclear** |
| Setting-level effect | The number of seeds in each category (of 10 for Ablation 1 and the control, of 5 for A2), next to the baseline's (of 6), tested with a two-sided Fisher exact test vs the baseline (2×2 table: seeds in that category vs the rest, this setting vs the baseline). Against the baseline's 3/6, **only 0/10 or 10/10 reach p < 0.05** (p = 0.036; 1/10 or 9/10 give 0.12). With 5 seeds nothing can (0/5 or 5/5 gives 0.18), so A2's seed counts are descriptive only. Otherwise significance comes only from the per-seed paired verdicts above. |
| Mechanism measure | `fresh_given_room` from `diagnose.py` (#5): share of placements that open an empty bin while a used one has room. A seed is a **fresh-bin seed** if it is > 0.01 on train |
| Held-out sets used for claims | **in-distribution:** `test_id_5k`; **shift:** `test_shape_k1.5`, `test_shape_k5`, plus the other #12 sets as secondary. `train` is reported but not a claim (it is the set searched on). `test_control` (#16) is a scoring check, not a claim: every rule must be exactly 0% over L2 there |
| Integrity | `check_integrity.py` (#6): flagged programs reported, not dropped. Runs stopped early by an error are rerun in full, not scored (`docs/log.md`); a run that hit the budget mid-evaluation (`abl_anyfit` seed 4, 299 of 300) counts as complete |

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
  - At least 4 of 10 seeds are **better** than Best Fit on `test_id_5k`, by about 0.3–0.5 pp (half the baseline's ~1 pp).
  - **No** seed is worse than Best Fit on `test_shape_k5` (the fresh-bin failure cannot happen).
  - Failures of seed 3's kind (k1.5) can still appear: they do not come from fresh bins.
  - The anyfit-retuned tuned Best Fit is within ~0.1 pp of Best Fit.
- **What counts as an effect:** per-seed **better** verdicts vs Best Fit on `test_id_5k`, and vs anyfit tuned Best Fit.
- **If confirmed, we claim:** EoH finds gains that do not rely on opening empty bins, which a tuned 3-knob Best Fit does not. Removing the option also removes the k5 failure, so the option drives the failure, not the gain as a whole.
- **If partly confirmed** (1–3 of 10 seeds better on `test_id_5k`): gains without fresh bins exist but are not reliable across seeds; report which seeds and what their rules do (`baseline_findings.md` style), and do not claim a setting-level gain.
- **If not confirmed, we claim** (all seeds tie with or lose to Best Fit): the fresh-bin bet was the whole of EoH's advantage on this distribution. Without it, EoH rediscovers Best Fit. The brief counts this as evidence too ("rediscovering a known heuristic … can also provide evidence").
- **What remains uncertain either way:** whether hiding empty bins changes what the LLM proposes or only what gets selected. Read a sample of `candidates.jsonl` to tell them apart.

## A1b. Bin rules explained in the prompt (`abl_informed_rules.yaml`)

- **Changes from baseline:** `evaluation: {prompt: informed_rules}`. The prompt states the task ("online 1D bin packing … minimise the number of bins") **and** that the function scores "the bins the current item fits into (remaining capacities, including empty bins)". The default LLM4AD prompt says neither, so this setting changes the **task framing and the bin rules together**.
- **Question:** does knowing that empty bins are among the options change how often evolved rules open them?
- **Hypothesis and why:** the default LLM4AD prompt never mentions empty bins, so the baseline found the bet by accident in 3/6 seeds. Saying so explicitly makes it easier to target: **more** fresh-bin seeds, found **earlier**. (The opposite is possible: an LLM told about empty bins may treat them as a last resort, as Best Fit does.)
- **Prediction:**
  - At least 8 of 10 fresh-bin seeds (baseline 3/6).
  - The best rule is found earlier than the baseline's eval ~200.
  - Hence more seeds better on train and `test_id_5k`, **and** more seeds worse on `test_shape_k5`.
- **What counts as an effect:** the descriptive seed counts and median discovery eval, and per-seed verdicts on `test_id_5k` and `test_shape_k5`.
- **If confirmed, we claim:** the fresh-bin habit comes partly from the prompt: the LLM exploits options it is told about, and that gain and that failure come together.
- **If not confirmed, we claim** (fresh-bin seeds about the baseline's rate, 2–7 of 10): the bet comes from selection pressure on this training set, not from what the LLM is told. Tuned Best Fit supports this reading, since it finds the bet with no prompt at all.
- **If the opposite happens** (0–1 of 10 fresh-bin seeds): telling the LLM about empty bins makes it treat them as a last resort, as Best Fit does, so the baseline's bet needed the LLM *not* to know the rule. Fewer seeds should then be worse on `test_shape_k5`.
- **What remains uncertain either way:**
  - **Which addition caused it:** an effect can't be pinned on the empty-bin clause alone, since the task framing changed too. Claim "task framing + bin rules", not "telling it about empty bins".
  - **Seed counts:** with 10 seeds, 3/6 vs 8/10 or 9/10 can still be chance (p ≥ 0.12, see the shared rules). Only 0/10 or 10/10 counts as significant on its own.

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
  - At most 2 of 10 fresh-bin seeds (baseline 3/6). Only 0/10 is significant on its own (p = 0.036); 1/10 or 2/10 is **consistent with** the prediction.
  - No seed better than the best EoH seeds on `test_id_5k` (compared descriptively: mean excess side by side).
- **What counts as an effect:** seed counts and per-seed verdicts vs Best Fit and tuned Best Fit; EoH's best seeds descriptively (mean excess on `test_id_5k` side by side), since `paired_stats.py` can't pair across settings (#13).
- **If confirmed, we claim:** evolution, not the LLM's prior alone, finds the rules that beat Best Fit (including the fresh-bin bet).
- **If not confirmed, we claim:** best-of-300 sampling matches EoH, so its gains come from the LLM's prior plus selection on train, not from evolution. That is a negative result worth reporting.
- **What remains uncertain either way:**
  - **Search vs prompt:** randsample changes the prompt too, so "no evolution" and "different prompt" are confounded. Report it as "LLM4AD RandSample", not "EoH without search".
  - **Unique programs:** a fixed prompt may return many duplicates (EoH's 300 evals hold 160–220 unique ones). Report unique programs per run next to the result.
  - **The check run is a gate:** if its replies mostly fail to parse or repeat, fix that before the 10 seeds.

## Optional settings (fill in only if they will be run)

`abl_informed_dist.yaml`, `abl_informed.yaml`, fallback `abl_no_m2.yaml`.

---

## Amendments

Dated entries only, each with the reason. Nothing above changes after the freeze.

### 2026-11-07, after Ablation 1 was scored (Jia Kang's review of #15 and #13)

Written after the A1 results were seen. None of these changes a prediction, or which outcome a result maps to.

1. **Seed count (line 32).** Ablation 1 went from 5 to 10 seeds on 7 Nov: @lottomonster ran seeds 0–9 (#7), and the team scores every complete run. The configs and tests changed in `e3009aa` (on `ablation-runs`, reaching `pipeline` with #19); the control followed in `b14e98b`, which also adds its `docs/log.md` entry ("2026-11-07: control runs 10 seeds"). `docs/log.md` has no matching entry for A1 yet, so "2026-11-04: ablation settings locked" still reads seeds 0–4 until one is added on `ablation-runs`. The A1 predictions were rescaled from 5 to 10 seeds on 7 Nov (commit `87516b5`), after the runs but before any scoring.
2. **Fresh-bin seed (line 38).** It is the measured definition: `fresh_given_room` > 0.01 on train. By that definition the baseline has 3/6 (seeds 3, 4, 5). `baseline_findings.md` explains seed 3's behaviour by an accidental skip window, so only seeds 4/5 place the bet deliberately (2/6). Seed counts in this file use the measured 3/6.
3. **A1b wording (line 86).** "If confirmed, we claim" reads as a significant result, but under the shared rules only 10/10 fresh-bin seeds would be significant vs 3/6 (8/10 or 9/10 give p ≥ 0.12). Read 8/10 or 9/10 as **consistent with** that claim. Moot for A1b's result (6/10 fresh-bin seeds: not confirmed).
4. **Control comparison (line 116).** `paired_stats.py` can't compare a seed with another setting's per-seed results (#13). The control's verdicts are vs Best Fit and tuned Best Fit only, and EoH's best seeds are compared descriptively (means side by side). Edited above, since the control has not run.
5. **Baseline numbers on the frozen sets.** The baseline paragraph quotes the pre-#16 test sets. On the frozen sets (excess over L2, `results/eoh_baseline__openrouter-gemini-3.5-flash-lite/`): Best Fit is 4.08% on `test_id_5k`, 1.35% on `test_shape_k1.5` and 8.32% on `test_shape_k5`. Seeds 4/5 are 2.88% on `test_id_5k` but 10.81% on k5; seed 3 is 3.90% on k1.5 (6.32% on k5). Train numbers are unchanged. The tuned Best Fit "4.65% with empty bins blocked" came from rescoring the funsearch-tuned knobs; retuned in anyfit (A1a), tuned Best Fit gets 3.90–4.00% on train.
6. **Tuned Best Fit reference (#13 review, point 2).** `paired_stats.py` compares each rule with tuned Best Fit averaged per instance over its seeds, not with each tuned seed. This is how A1 was scored. If #13 changes it, A1 is rescored and noted here.

### 2026-11-08, before the control was scored (@lottomonster's review of #15)

The control has run (seeds 0–9, `b14e98b`, `99452e2`, `f3dfa94`) but has not been scored. Point 1 changes text above; none of the points changes an Ablation 1 prediction or which outcome an A1 result maps to.

1. **Control: 10 seeds (lines 32, 37, 114, 122).** The control runs seeds 0–9 (`docs/log.md`, "2026-11-07: control runs 10 seeds"). Edited above, as in Amendment 4, since it has not been scored: the seeds and Fisher rows, "at most 1 of 5" → "at most 2 of 10" fresh-bin seeds, and "the 5 seeds" → "the 10 seeds". Against the baseline's 3/6, 0/10 gives p = 0.036, 1/10 gives 0.12 and 2/10 gives 0.30, so only 0/10 is significant on its own.
2. **A1b "found earlier" (line 83).** No definition of "found" was written before scoring. It is reported descriptively, and the definition used is stated next to the result as one chosen after scoring.
3. **What "better" needs with 10 instances.** Under #13's rule, **better** or **worse** needs the sign test at p < 0.05, so with 10 instances per set: at least 9 wins if all 10 differ (8 of 10 gives p = 0.11), 8 of 9, or every instance when only 6–8 differ; with 5 or fewer non-tied instances no verdict can be better or worse. So **unclear** often means a small effect, not no effect.
4. **No multiple-comparisons correction.** Verdicts are not corrected across rules × seeds × sets × references (#13's caveat). A single per-seed verdict is descriptive; claims rest on seed counts and on patterns that repeat across seeds and sets.
5. **A1a tuned Best Fit (line 69).** Retuned in anyfit, tuned Best Fit gets 3.90–4.00% on train (Amendment 5), against the prediction "within ~0.1 pp of Best Fit". The prediction is left as written; if held-out agrees, the report shows it as a miss.
6. **Scoring start time (Amendment 1).** Commit times show when results were saved, not when scoring ran. Checked: the scoring script (`score.sh`) was first committed at 14:02 on 7 Nov, and the earliest scoring output on disk (`abl_anyfit/tuned_bf/seed0`) is from 13:57, both after the A1 predictions were committed (`67b478c`, 13:55). No A1 scoring ran before 13:55.
