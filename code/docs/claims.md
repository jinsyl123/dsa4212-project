# Frozen claims for the ablations (#7)

Written **before** any ablation result is scored. For each setting: what we expect, how we will measure it,
and what we will conclude whichever way it goes. Once this file is merged, it is frozen: later changes go in a
dated **Amendments** entry at the bottom, never as edits above. The report checks its claims against this file.

Team agreement: "Freeze test sets **and the claim we'll make for each outcome** before the main searches"
(decisions doc, "Agreed by default"). Owner: Jia Kang (claims doc). Written by: Jin. Frozen on: ___ (the day #15 merges).
Ablation 1 predictions (A1a, A1b) committed on 7 Nov 2026, before any Ablation 1 run was scored; the commit timestamp is the record. A1 was then scored before this file was frozen: see Amendments, 7 Nov.

## What the course asks for (why this file exists)

- **Experimental design, 35%:** "breadth and depth, controlled comparisons, and follow-up on findings"
  (Assignment guidelines, Marking). Each setting below changes one thing from its comparison row (the baseline, except A2b: A1b).
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
| Comparison row | EoH baseline, seeds 0–5 (`eoh_baseline.yaml`). Exception: A2b compares against A1b, seeds 0–9 (`abl_informed_rules.yaml`) |
| Seeds per setting | **Every setting: 10 (seeds 0–9)**: Ablation 1 (A1a, A1b) and the control as run (#7), Ablation 2 (A2a, A2b) as agreed on #7. Every seed run in full is scored; none is dropped |
| References | Best Fit, and tuned Best Fit retuned in the same setting (#8), with the same seeds as the setting. A setting that changes neither `mode` nor the training set (A1b, control) leaves tuned Best Fit unchanged: it is still retuned in its own folder, and seeds 0–5 must match the baseline's exactly (a determinism check) |
| Sets | train, val (selection only, never reported as a test), and the test sets frozen in #12 |
| Metric | excess over L2 (%) on every set (#12 decision 7, #10); fewer bins = better. Same as L1 except on `test_scale65` and `test_control`, so numbers stay comparable with FunSearch/EoH |
| Per-seed verdict vs a reference | `paired_stats.py` (#13, scored with its code as of this freeze; if #13 changes before merging, rescore and note it in Amendments): **better / worse** only if the 95% bootstrap CI excludes 0 **and** sign-test p < 0.05; **tie** if identical on every instance; else **unclear** |
| Setting-level effect | The number of seeds in each category (of 10), next to the baseline's (of 6), tested with a two-sided Fisher exact test vs the baseline (2×2 table: seeds in that category vs the rest, this setting vs the baseline). Against the baseline's 3/6, **only 0/10 or 10/10 reach p < 0.05** (p = 0.036; 1/10 or 9/10 give 0.12). A2b is tested against A1b (of 10): against 5/10, only 0/10 reaches p < 0.05 (p = 0.033; 1/10 gives 0.14), and against 4/10 no count below it does (0/10 gives 0.087). Otherwise significance comes only from the per-seed paired verdicts above. |
| Mechanism measure | `fresh_given_room` from `diagnose.py` (#5): share of placements that open an empty bin while a used one has room. A seed is a **fresh-bin seed** if it is > 0.01 on train |
| Held-out sets used for claims | **in-distribution:** `test_id_5k`; **shift:** `test_shape_k1.5`, `test_shape_k5`, plus the other #12 sets as secondary. For Ablation 2 (A2a, A2b), every held-out set is a claim set. `train` is reported but not a claim (it is the set searched on). `test_control` (#16) is a scoring check, not a claim: every rule must be exactly 0% over L2 there |
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

## Ablation 2. Training-data variety

One ablation, two settings (#7). Both change only the training data, on the lite model with 300 evaluations and 10 seeds. **A2a** starts from the baseline setup and compares against the EoH baseline, because its target failures (k1.5, k5) are baseline failures. **A2b** starts from the A1b setup (informed prompt) and compares against A1b, seeds 0–9, because the capacity collapse appears only there. Both were written after A1 and the control were scored and before any A2 run, so A1's results informed them. Counts are per-seed verdicts against Best Fit.

Counts on the frozen sets (`paired_stats.csv` in each setting's results folder):

| Set | EoH baseline better / worse (of 6) | A1b better / worse (of 10) | Tuned Best Fit better / worse (of 6) |
|---|---|---|---|
| `test_id_5k` | 3 / 0 | 10 / 0 | 6 / 0 |
| `test_size_1k` | 2 / 0 | 3 / 0 | 6 / 0 |
| `test_size_10k` | 3 / 0 | 9 / 0 | 6 / 0 |
| `test_uniform` | 3 / 0 | 4 / 0 | 6 / 0 |
| `test_cap200` | 0 / 1 | 0 / 5 | 3 / 0 |
| `test_cap500` | 0 / 1 | 0 / 5 | 0 / 0 |
| `test_cap500_scaled` | 2 / 1 | 5 / 1 | 0 / 6 |
| `test_scale25` | 0 / 1 | 1 / 5 | 2 / 2 |
| `test_scale65` | 0 / 1 | 0 / 0 | 0 / 0 |
| `test_shape_k1.5` | 2 / 1 | 2 / 0 | 2 / 4 |
| `test_shape_k5` | 1 / 2 | 2 / 1 | 0 / 6 |

A **collapse** is a mean excess more than 100 pp above Best Fit on a set. In A1b that is 5 of 10 seeds on `test_cap500` (0, 2, 3, 4, 9) and 4 of 10 on `test_cap200` and `test_scale25` (0, 2, 3, 9). Seed 4 is worse on those two by only about 0.7 pp.

## A2a. Mixed shapes (`abl_mixed_shapes.yaml`)

- **Changes from baseline:** `evaluation: {train_set: train_mixed}`. Same sizes and capacity, with half the instances at Weibull shape 2 and half at shape 4. Test shapes 1.5 and 5 stay out.
- **Question:** do the k1.5 / k5 failures come from training on a single item-size shape?
- **Hypothesis and why:** the fresh-bin bet pays off only when later small items fill the gaps, which holds for shape 3 but not for clustered sizes (k5: 9% single-item bins). Shape-4 training instances are clustered enough to punish the bet during search, so selection should prefer rules that hold up across shapes. Seed 3's skip window fitted shape 3 by accident and would be scored on two shapes instead.
- **Prediction:**
  - Fewer seeds worse than Best Fit on `test_shape_k5` or `test_shape_k1.5`: at most 3 of 10 (baseline: 3 of 6 lose on one of them).
  - Smaller gains on `test_id_5k` than the baseline's best seeds: about 0–0.5 pp, against ~1 pp, with at most 5 of 10 seeds better.
  - `fresh_given_room` lower on average.
  - On the other held-out sets:

    | Sets | Expectation (of 10) | Why |
    |---|---|---|
    | `test_size_1k`, `test_size_10k` | At most 5 better, at most 1 worse | Same item distribution as `test_id_5k`, so the same smaller gain |
    | `test_uniform` | At most 1 worse | Mixing shapes adds variety in item size, and the baseline has none worse |
    | `test_cap200`, `test_cap500`, `test_cap500_scaled` | At most 3 worse | The training capacity is unchanged, so no reason to move far from the baseline's 1 of 6 |
    | `test_scale25`, `test_scale65` | At most 3 worse | The training scale is unchanged, so no reason to move far from the baseline's 1 of 6 |
- **What counts as an effect:** per-seed **worse** verdicts on k1.5 and k5 vs Best Fit and vs `train_mixed`-retuned tuned Best Fit, compared with the baseline's, and the same verdicts on the other held-out sets.
- **If confirmed, we claim:** the shape-shift failures come from training on one shape. A more varied training set trades some in-distribution gain for robustness.
- **If not confirmed, we claim:** the failures don't come from the training shape alone. The rule family or the search picks fragile rules even when the training data covers more than one shape.
- **What remains uncertain either way:** shapes 2 and 4 bracket the original training shape (3) but not the test shapes (1.5, 5), so the shape sets test extrapolation beyond the training range. Mixing also changes the average training difficulty.

## A2b. Mixed capacities (`abl_mixed_capacities.yaml`, to be built)

- **Changes from A1b:** the training set only, with the informed prompt kept (`evaluation: {prompt: informed_rules}`). Half the instances are at capacity 100 and half at capacity 300, otherwise as `train` (Weibull shape 3, scale 45, so the same item sizes in both halves). The test sets are unchanged. This is one change from A1b and two from the baseline.
- **Question:** does A1b's capacity collapse come from training on a single capacity?
- **Hypothesis and why:** all five collapsing A1b seeds read the capacity through `max(bins)` and score empty bins separately from used ones. Calibrated at capacity 100, the empty bin then wins almost every time at larger capacities, so each item opens a new bin. The search scores mean relative excess per instance (`core.py`), so a rule that collapses at capacity 300 is heavily penalised in training and selection should remove it. Capacity 300 also moves the mean item from ~40% to ~13% of a bin, so `test_cap200` and `test_scale25` (~20–22%) sit inside the training range, while `test_cap500` (~8%) and `test_scale65` (~58%) sit outside it.
- **Prediction:**
  - **Collapses:** 0 of 10 on `test_cap200` and `test_scale25` (A1b: 4 of 10 each), and at most 2 of 10 on `test_cap500` (A1b: 5 of 10).
  - **Fresh-bin seeds:** at most 6 of 10 (A1b: 6 of 10).
  - **Per-seed verdicts against Best Fit:**

    | Sets | Position | Expectation (of 10) | Why |
    |---|---|---|---|
    | `test_id_5k`, `test_size_10k` | Capacity 100 | At least 5 better, none worse, with a smaller mean gain than A1b | Only half of training is at this capacity, so gains fitted to it shrink |
    | `test_size_1k` | Capacity 100 | At most 1 worse | Same item distribution, but small differences are harder to detect |
    | `test_uniform` | Item ~40% of a bin | At most 1 worse | Inside the training range |
    | `test_cap200`, `test_scale25` | Inside the range | At most 1 worse (A1b: 5) | Training covers this item-to-bin range |
    | `test_cap500` | Outside the range | At most 2 worse (A1b: 5) | Outside what training covers, so some seeds may still fail |
    | `test_scale65` | Outside the range | At most 2 worse (A1b: 0) | The mean item is a larger share of the bin than in either half of training |
    | `test_cap500_scaled` | Capacity outside, ratio as at 100 | At most 2 worse (A1b: 1) | A check: items scale with the bin, which training does not cover |
    | `test_shape_k1.5`, `test_shape_k5` | Shape unchanged | At most 3 worse on one of them (A1b: 1) | Mixing capacities does not touch shape |

  - **Tuned Best Fit** (retuned on this training set): still worse than Best Fit on `test_cap500_scaled` in at least 8 of 10 seeds (10 of 10 with A1b's copy). Its knobs are absolute and training has no variety in item size, so that failure should stay.
- **What counts as an effect:** collapse counts and per-seed verdicts on every held-out set, vs Best Fit and vs the tuned Best Fit retuned in this setting, next to A1b's. Report what each best program does with empty bins (`baseline_findings.md` style).
- **If confirmed, we claim:** under the informed prompt, A1b's capacity collapse comes from training on one capacity. Training on two removes it inside the trained range, at some cost in gain at capacity 100.
- **If not confirmed, we claim:**
  - **Collapses remain on `test_cap200` or `test_scale25`:** the collapse does not come from the training capacity alone. The prompt or the rule family produces fragile rules even when training punishes them.
  - **Collapses are gone inside the range but stay on `test_cap500`:** the rules fit the trained capacities, not capacity in general.
  - **Seeds are worse on capacity 100 sets than predicted:** the search trades away the in-distribution gain to survive capacity 300.
- **What remains uncertain either way:**
  - **Conditional on the prompt:** the result holds for the informed prompt only. If time allows, A2b on the baseline prompt would give a 2×2 (prompt × training capacities) and show whether the fix depends on the prompt.
  - **Item sizes are fixed at scale 45:** training has no variety in item size, so `test_cap500_scaled` is only a check.
  - **Tuned Best Fit's ranges:** its knob ranges assume capacity 100 (`tuned_bf.py`), so at capacity 300 its penalty cannot make an empty bin win.

## A2 (proposed alternative, 8 Nov). Stronger LLM, same budget (`abl_strong_model.yaml`)

**Status: draft, proposed on #7 after A1 and the control were scored. It replaces the mixed-shapes A2 above only if @lottomonster and @JiaKangL agree, and must be committed before any `abl_strong_model` run.** Every prediction below is written before any A2 run; the only A2 run so far is a 12-evaluation `pipecheck` (setup check, small instances, not scored).

- **Changes from baseline:** the LLM only: `google/gemini-3.5-flash` instead of `google/gemini-3.5-flash-lite` (same generation, same reasoning effort `low`, same 8192-token cap). Everything else as `eoh_baseline.yaml`: `funsearch` mode, default LLM4AD prompt, 300 evaluations, train on `train`, the frozen test sets. 10 seeds (0–9).
- **Question:** is "no LLM rule beats tuned Best Fit on `test_id_5k`" (0 of 36 seeds across the baseline, A1a, A1b and the control, #18) a limit of the model, or a ceiling for this search and rule family within 300 evaluations?
- **Hypothesis and why:** mostly a ceiling. Tuned Best Fit is 1.67% over L2 on `test_id_5k`; the best LLM seeds are 2.88% (baseline 4/5) and ~1.8% (A1b), and every gain found so far comes from the same few ideas (fresh-bin bet, sliver avoidance) that tuned Best Fit already tunes directly. A stronger model should find those ideas more reliably, so more seeds beat plain Best Fit, but not a new kind of rule. A1b showed the fresh-bin bet comes from selection on `train`, so a better optimiser of `train` should place it at least as often, with the same failures under shift.
- **Prediction:**
  - **At most 2 of 10 seeds better than tuned Best Fit** (same setting, i.e. funsearch-tuned) on `test_id_5k`.
  - **At least 6 of 10 seeds better than Best Fit** on `test_id_5k` (baseline 3/6).
  - **At least 5 of 10 fresh-bin seeds** (baseline 3/6).
  - **Failures follow the fresh-bin seeds:** at least 2 seeds worse than Best Fit on `test_shape_k1.5` or `test_shape_k5`, and at least 2 worse on a capacity or scale set (`test_cap200`, `test_cap500`, `test_scale25`), as in A1b.
- **What counts as an effect:** the shared rules: per-seed verdicts vs Best Fit and vs tuned Best Fit; seed counts vs the baseline's with Fisher's test (only 0/10 or 10/10 against 3/6 reaches p < 0.05, so the other counts are descriptive). For this setting the capacity and scale sets above are **claim sets**, pre-registered here, not secondary. Tuned Best Fit is retuned in this folder; it needs no LLM, so seeds 0–5 must match the baseline's exactly.
- **If confirmed, we claim:** the ceiling is not the lite model. A stronger LLM finds the same kind of rules more reliably, with the same trade: gains in distribution, failures under shift.
- **If not confirmed:**
  - **3 or more seeds beat tuned Best Fit:** model capability limited the lite runs. Report what those rules do (`baseline_findings.md` style) and whether they still fail under shift.
  - **Fewer than 6 seeds beat Best Fit, or fewer than 5 fresh-bin seeds:** a stronger model does not optimise `train` more reliably here; the search budget, not the model, sets how often EoH escapes Best Fit.
  - **No capacity or shape failures among the fresh-bin seeds:** the stronger model writes the bet in a capacity-relative form (e.g. reading the capacity from `max(bins)`). Check the programs.
- **What remains uncertain either way:**
  - **One stronger model:** this is one data point ("a stronger LLM under the same evaluation budget"), not a trend over model size.
  - **More than capability changes:** a larger model also writes longer, differently styled code and uses more tokens per call. Report tokens and cost per run next to the result; the evaluation budget (300) is the same.

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

1. **Seed count (line 32).** Ablation 1 went from 5 to 10 seeds on 7 Nov: @lottomonster ran seeds 0–9 (#7), and the team scores every complete run. The configs and tests changed in `e3009aa` (on `ablation-runs`, reaching `pipeline` with #19); the control followed in `b14e98b`, which also adds its `docs/log.md` entry ("2026-11-07: control runs 10 seeds"). The A1 entry in `docs/log.md` ("2026-11-07: Ablation 1 runs 10 seeds; optional rows dropped") was added later, in `a2b668e` on `ablation-runs`; it supersedes the seed count in "2026-11-04: ablation settings locked" for A1. The A1 predictions were rescaled from 5 to 10 seeds on 7 Nov (commit `87516b5`), after the runs but before any scoring.
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

### 2026-11-08, A2 proposal (after A1 and the control were scored)

1. **A2 alternative (section above).** Proposed on #7: a stronger LLM instead of mixed shapes, because A1 and the control left "no LLM rule beats tuned Best Fit in distribution" as the main open question, and the new failures are capacity failures, which mixed shapes (capacity 100 throughout) cannot test. Written before any A2 run; adopted only if the team agrees on #7, then logged in `docs/log.md`.
2. **Optional settings.** `abl_informed_dist` and `abl_informed` were dropped on 7 Nov (`docs/log.md`, `a2b668e`); `abl_no_m2` is not planned.
3. **Amendment 1** now cites the A1 log entry (`a2b668e`).

### 2026-11-09, A1 predictions for the secondary sets (in retrospect)

Added 9 Nov, in retrospect. Predictions for the secondary sets were left out by mistake when A1 was registered. These continue the hypotheses above. They were written after A1 and the control were scored, so they are not pre-registered and do not count as confirmations. They follow from the A1a and A1b hypotheses, `baseline_findings.md` and the set definitions in `configs/instances/v0.yaml`, and were not adjusted to match the results.

Counts are per-seed verdicts against Best Fit, out of 10 seeds. "Mostly unclear or tie" means at most 2 seeds better.

**A1a (empty bins hidden).** The fresh-bin failure cannot happen, so any loss comes from how a rule chooses among used bins (seed 3's kind).

| Sets | Expectation | Why |
|---|---|---|
| `test_size_1k`, `test_size_10k` | At least 4 better, none worse | Same item distribution as `test_id_5k`; only the length changes |
| `test_uniform` | Fewer better than on `test_id_5k`; at most 2 worse | Different item distribution and capacity, so gains fitted to Weibull items carry over only partly |
| `test_cap200`, `test_cap500` | Mostly unclear or tie; at most 2 worse | Bigger bins hold more items, so gaps fill easily and rules differ less |
| `test_cap500_scaled` | Fewer better than on `test_id_5k`; at most 2 worse | Same item-to-bin ratio, but rules with fixed constants (baseline seeds 2 and 3) do not scale |
| `test_scale25` | Mostly unclear or tie; at most 2 worse | Small items fill gaps easily |
| `test_scale65` | Mostly unclear or tie; at most 2 worse | 63% of items exceed half a bin, so few placements are real choices |

**A1b (bin rules in the prompt).** The hypothesis is more fresh-bin seeds, with the gain and the failure coming together. The bet pays off only when later small items fill the gaps.

| Sets | Expectation | Why |
|---|---|---|
| `test_size_1k`, `test_size_10k` | More than half better, as on `test_id_5k` | Same item distribution; only the length changes |
| `test_uniform` | More seeds better than worse | Spread-out sizes, so later small items fill the gaps, as in training |
| `test_cap200`, `test_cap500` | Mostly unclear or tie; fewer worse than on `test_shape_k5` | Many items per bin, so gaps fill and an extra opened bin costs little |
| `test_cap500_scaled` | Fewer better than on `test_id_5k`; at most as many worse as on `test_shape_k5` | Same item-to-bin ratio, so rules based on ratios keep the gain and rules with fixed constants lose it |
| `test_scale25` | Mostly unclear or tie; fewer worse than on `test_shape_k5` | Small items fill the gaps, so the bet costs little |
| `test_scale65` | More seeds worse than on `test_id_5k` | Few small items are left to fill the gaps, as on `test_shape_k5` |

### 2026-11-09, Ablation 2 predictions (before any A2 run)

Edited above, as in Amendment 4, since A2 has not run.

1. **Seeds and Fisher rows.** A2 runs 10 seeds (#7), so A2's seed counts are no longer descriptive only.
2. **Ablation 2 is one ablation with two settings** (#7): A2a mixed shapes, and the new A2b mixed capacities. A2a uses the baseline setup and compares against the baseline. A2b uses the A1b setup (informed prompt) and compares against A1b, seeds 0–9, because the capacity collapse appears only there. This changes the earlier rule that every setting is one change from the baseline: A2b is one change from A1b. Every held-out set is a claim set for both.
3. **A2a wording.** The sets at shapes 1.5 and 5 are outside the training range of shapes 2 to 4, so they test extrapolation. The earlier text said interpolation.
