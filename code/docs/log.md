# Decision log

One entry per decision that affects results. Newest at the top. Link the config and the results folder.

## Template

### YYYY-MM-DD: short title (who)

- **Decision:**
- **Why:**
- **Alternatives considered:**
- **Affects:** configs / results folders / report sections

---

### 2026-11-05: test sets frozen (Jia Kang)

- **Decision:** every test set has 10 instances (was 5). Added `test_size_1k`, `test_cap200`, `test_scale25`, `test_scale65` and the control set `test_control`, and listed all of them in `final_eval.sets`. `train`, `train_mixed`, `val` and the seed are unchanged, and instances 0–4 of the existing test sets are identical to before (checked against the old data).
- **Why:** with 5 instances no rule can beat Best Fit significantly on the sign test (best case p = 0.0625; with 10, p = 0.002). `test_cap200` fills the capacity series 100 → 200 → 500; `test_scale25`/`test_scale65` shift item size relative to the bin in both directions. `test_size_1k` needs its own instances because `val` chooses between runs. The control set catches scoring bugs: every item is over half a bin, so every rule must use one bin per item (all six OpenRouter seeds' best programs and Best Fit do). On `test_scale65` L1 is ~10% too low, so its excess is read over L2 (#10); on every other test set L2 = L1.
- **Alternatives considered:** the real OR-Library files (binpack5–8) and Beyond the Hype's datasets (skipped); renaming `test_uniform` to `test_orlib_like` (kept: a set's seed comes from its name, so renaming changes its data).
- **Affects:** `configs/instances/v0.yaml`; `final_eval.sets` in `eoh_baseline.yaml` and `compare.yaml` (the ablation configs inherit it); every `final_*.csv` once re-scored. Discussion: #12.

### 2026-11-04: ablation settings locked (Nic)

- **Decision:** run the main settings `abl_anyfit`, `abl_informed_rules` and `abl_mixed_shapes`, plus the control `control_randsample`. Seeds 0–4 each, `openrouter-gemini-3.5-flash-lite`, budget `full` (300 evals). The comparison row is the EoH baseline, seeds 0–5. Optional if time allows: `abl_informed_dist`, `abl_informed`. Fallback: `abl_no_m2`.
- **Why:** the baseline shows the gain and the k5 failure both come from opening fresh bins (seeds 4/5), plus one failure from training on a single shape (seed 3, k1.5). Ablation 1 tests the empty-bin option; Ablation 2 tests training variety; randsample tests whether search adds anything. 5 seeds because half the baseline seeds collapse to Best Fit.
- **Alternatives considered:** no-M2 as a main setting (no run points to M2), thoughts on/off (EoH Table 6 already covers it), a budget sweep (would be a third ablation).
- **Affects:** `configs/abl_*.yaml`, `configs/control_randsample.yaml`, `results/<setting>/`. Plan and evidence: #7. Claims frozen before scoring: #15.

### 2026-11-01: integrity checks on candidates, flag only (Jin)

- **Decision:** `obpevo/integrity.py` parses every candidate with `ast` and records rule violations (non-numpy imports, randomness incl. `hash()`/`id()`/clock, `exec`/`eval`/`open`, numpy file I/O, dunder access, `global`, attribute assignment, module-level state, mutable defaults) in a new `integrity` column of `evals.csv`; `summary.json` gets `n_flagged` and `best_integrity`. `check_integrity.py` rescores each best program four times (usual order twice, one shuffled order twice), each in a fresh spawned process with its own hash and RNG seeds. Flagged programs are still scored and can still win.
- **Why:** the brief asks to keep scoring outside the code being evolved. Flagging instead of dropping keeps every method's search unchanged (fair) and lets us report how often each rule is hit. All best programs committed at the time (25 Oct seed 0, OpenRouter seeds 0–2) are clean: no flags, 0 repeat and 0 shuffle mismatches on train and val.
- **Limits (from an independent review):** LLM4AD keeps only the function body of each reply, so module-level rules rarely fire in our runs; an empty column means "no body-level violation". Static checks can be bypassed (e.g. a helper function mutating shared state); the rescoring catches such cases only if they change a bin count. Fresh spawned processes are needed because forked children share the parent's hash seed, which hid `hash()`-based nondeterminism in the first version.
- **Alternatives considered:** rejecting flagged candidates (changes the search and LLM4AD's behaviour); a runtime sandbox (heavier; LLM4AD already runs candidates in a child process with a timeout).
- **Affects:** `evals.csv` (new column, at the end), `summary.json`, `results/*/integrity.csv`. No score changes.

### 2026-10-31: fresh-bin diagnostic, scores unchanged (Jin; updated 2026-11-04 after Nic's review)

- **Decision:** `obpevo/diagnostics.py` + `diagnose.py` rescore best programs and the BF/FF/WF baselines in both packing modes and add `fresh_bin_rate` (share of items put in an empty bin while a used bin had room). Separate from `core.py`; tests check it packs identically to `core.py` in both modes.
- **Why:** tests the explanation for EoH seed 0's k=5 failure. Seed 0 (`eoh_baseline_25oct`) opens a fresh bin in ~36% of choices on train (57% on k=5); BF/FF never do. In anyfit mode k=5 excess falls 38.9% → 8.1%, but EoH also loses most of its in-distribution lead (train 3.79% → 4.34%, BF 4.75%). Seed 11 (`eoh_freetier_seed11`, a separate run: free Gemini, different date, older pipeline, 62 of 363 calls hit 429s but all 300 evals ran) opens fresh bins less (~9%), gains less (train 4.48%), has no k=5 failure but fails on `test_cap500` (6.12% vs BF 0.50%); in anyfit it matches BF on every set. The OpenRouter seeds 0–2 never open a fresh bin (`fresh_given_room` = 0 everywhere), so funsearch = anyfit for them, and they have no failure. In our five runs, the failures come only from programs that open fresh bins; the OpenRouter programs never do. The anyfit comparison shows what these programs do, not why some runs learn it (the two groups also differ in provider, pipeline version and date).
- **Seed 11's mechanism (Nic):** its score is `cos(π · remaining / (1.618 · item))`. Cosine repeats, so the score peaks at an exact fit and again whenever ~3.2×, 6.5×, … the item is left over. For a 45-unit item: exact fit +1.00, 146 left +1.00, an empty capacity-500 bin (455 left) +0.71, above many partly filled bins. At capacity 100 this rarely matters; at 500 empty bins often hit a peak. An example of an LLM heuristic that scores well in training for reasons unrelated to bin packing (the hard-coded golden ratio).
- **Follow-ups:** `test_cap500` is not trivial (seed 11 fails there), so it stays in the final sets and the "near-trivial" note is dropped. Baseline rows in `final_summary.csv` use their classical AnyFit definitions (Worst Fit ~13%), while evolved programs are scored in funsearch mode (Worst Fit 148% there); for BF/FF the two conventions give identical bins (readme §11).
- **Alternatives considered:** adding the column inside `core.evaluate` (touches the search hot path and Nic's scorer).
- **Affects:** `results/{eoh_baseline_25oct,eoh_freetier_seed11,eoh_baseline__openrouter-gemini-3.5-flash-lite}/diagnostics_summary.csv`, readme §11, report analysis section.
- **Update 2026-11-04:** OpenRouter seeds 3–5 (same setup) do open fresh bins, and seed 3 has a failure that is not a fresh-bin failure. See "ablation plan revised after seeds 3–5" below and readme §13.

### 2026-10-31: failed held-out instances scored as Next Fit (Jin)

- **Decision:** in `evaluate_final`, an instance where a program errors or times out gets Next Fit's excess instead of `NaN`; `status` and `failed_frac` still record the failure.
- **Why:** `NaN` drops out of the mean, so a program that fails often would look better than one that runs everywhere. Next Fit is worse than every sensible heuristic (it never beats Best Fit on any v0 instance), so a failure is not rewarded relative to any program worth reporting. It is not a strict floor: in funsearch mode a valid program that always opens a new bin uses more bins than Next Fit. No current run has failures, so no reported number changes.
- **Alternatives considered:** leave `NaN` and report `failed_frac` only (mean still flattering); a fixed penalty (arbitrary scale).
- **Affects:** `evaluate_final.py`, `final_summary.csv` only when a program fails.

### 2026-10-25: v0 pipeline design (Nic)

- **Decision:** LLM4AD runs the search methods; a thin BLADE-style layer (`code/obpevo/`) handles logging, budgets, train/test separation and run records. EoH is the baseline method.
- **Why:** reuses tested reference implementations of EoH, ReEvo, FunSearch, (1+1)-EPS and random sampling, while keeping the comparison fair in one place that no method controls.
- **Alternatives considered:** BLADE itself (fewer of our target methods built in); writing each method ourselves (too slow, harder to defend).
- **Affects:** everything.

### 2026-10-25: evaluation conventions (Nic)

- **Decision:** score = minus mean excess over the L1 lower bound ceil(sum/C), "funsearch" packing convention (scores all n preallocated bins), identical to LLM4AD's OBP task. "anyfit" (open bins only) kept as an ablation.
- **Why:** verified bit-for-bit equal to LLM4AD's evaluator (Best Fit 4.00% on their Weibull set), so our numbers are comparable with the EoH/FunSearch papers.
- **Affects:** `core.py`, all configs.

### 2026-10-25: budgets count evaluations and tokens (Nic)

- **Decision:** every run stops at whichever comes first of max_evals, max_tokens, max_minutes. The evaluation cap also blocks further LLM calls.
- **Why:** found that LLM4AD's RandSample/HillClimb keep calling the LLM after the evaluation budget ends (one sampler thread shuts the shared pool, the other loops). Linking the budgets fixed it.
- **Affects:** `llm.py`, `recorder.py`, `run_experiment.py`.

### 2026-10-31: LLM4AD version and main baseline (Nic)

- **Decision:** all runs use LLM4AD at commit 73ae214 (installed via requirements.txt); the EoH baseline is the 3 OpenRouter seeds (gemini-3.5-flash-lite, 300 evals each).
- **Why:** PyPI llm4ad 1.0.0 lacks ReEvo/MCTS-AHD/HSEvo; Google's free tier cut runs short (~200 calls/day), giving unequal budgets.
- **Affects:** results/eoh_baseline\_\_openrouter-gemini-3.5-flash-lite (main), results/eoh_baseline_25oct (exploratory, 1.0.0).

### 2026-11-01: review changes (Nic)

- **Decision:** inner folder renamed code/code → code/pipeline; informed prompt split into informed_rules / informed_dist / informed; best program still selected on training (val chooses between configurations).
- **Why:** PR review (Jin): attribution of prompt effects, comparability with the EoH papers.
- **Affects:** pipeline/, task.py, readme §6, §11, §13.

### 2026-11-03: tuned Best Fit baseline (Jia Kang)

- **Decision:** the budget-matched baseline `tuned_bf` is Best Fit plus a penalty for leaving a small gap: score = −r − penalty·[lo < r < hi], with r the space left in a bin after packing. Three integer knobs (`lo` 0–10, `hi` lo+2–40, `penalty` 1–100) in item units, tuned by random search on `train`: Best Fit first, then 299 distinct random settings, i.e. the same 300 scored tries as an EoH run. Seeds 0–5, like EoH (3–5 added on 5 Nov, after EoH's).
- **Why:** beating plain Best Fit is weak evidence (decision #13). Any score strictly decreasing in r packs exactly like Best Fit, so the sliver penalty is the smallest change that can differ. It is the strategy the FunSearch paper describes for several of its discovered heuristics (illustrated with its OR-dataset heuristic, Fig. 6). The knobs are in absolute units, and neither the rule nor an evolved program is given the capacity (in funsearch mode an evolved program could read it from max(bins)). Random search is the plainest tuner, so its result is a floor for what tuning alone achieves.
- **Alternatives considered:** a 4th knob gating the penalty by item size (no better in an 80-try pilot: 2.68% vs 2.55% train excess); matching EoH on wall-clock time (unfair: EoH waits on the API) or dollars (tuning is free); a stronger tuner (TPE, grid), which would only raise the bar.
- **Affects:** `obpevo/tuned_bf.py`, `run_experiment.py`, `evaluate_final.py`, `summarise.py`, `configs/eoh_baseline.yaml`, `configs/smoke.yaml`; results in results/eoh_baseline\_\_openrouter-gemini-3.5-flash-lite/tuned_bf (readme §13).

### 2026-11-04: ablation plan revised after seeds 3–5 (Jin)

- **Decision:** proposed replacing "prompt information + no-M2" with "empty-bin option (anyfit search, informed_rules) + training-data variety", 5 seeds per setting (Nic to confirm, #7).
- **Why:** 3 of 6 OpenRouter seeds converge to Best Fit; the other 3 open fresh bins. For seeds 4/5 that carries both the gain and the k5 failure (blocking fresh bins removes it); seed 3's k1.5 failure stays with fresh bins blocked, so it has a different cause. No run points to M2.
- **Affects:** #7, ablation configs.

### 2026-11-04: LLM for the ablations; full run logs committed (Jin)

- **Decision:** proposed keeping `openrouter-gemini-3.5-flash-lite` for every ablation run (Nic to confirm, #7). Run folders are now committed in full, and each run also saves every LLM prompt and response (`llm_calls.jsonl`).
- **Why:** same model as the 6 baseline seeds (comparability); 0/1806 LLM errors vs 62/363 on the free tier; ~$0.17 per run. Full logs so every result can be traced to the programs and calls behind it (Nic, #7). Evidence for the per-seed explanations: `docs/baseline_findings.md`.
- **Affects:** .gitignore, obpevo/llm.py, readme §10, #7.

### 2026-11-04: L2 lower bound as an extra column (Jin)

- **Decision:** `evaluate_final.py` adds Martello and Toth's L2 bound (`lb_l2`, `excess_l2`; `mean_excess_l2` in `final_summary.csv`). The search score and `excess` stay on L1.
- **Why:** for the planned `test_scale65` (#12). L1 ignores that two items over half a bin can't share one; on `test_scale65` 63% of items do, so Best Fit is 10.2% over L1 but 0.4% over L2, and L1 would make every rule look far worse than it is. L2 is also tighter on the control set `test_control` (#16), where every item is over half a bin and every rule is exactly 0% over L2. On every other test set L2 equals L1, so no existing number changes. Keeping L1 for the search keeps scores comparable with LLM4AD, FunSearch and EoH.
- **Update 2026-11-05 (#12 decision 7):** the report uses excess over L2 on every set. `summarise.py`'s held-out table and test-set figure are now on L2; L1 is a second table for the sets where it differs. Same numbers as L1 on every other set.
- **Update 2026-11-07 (#10 review, Jia Kang):** every excess-based column in `final_summary.csv` has an `_l2` twin (`best_run_l2`, `worst_run_l2`, `delta_vs_bf_pp_l2` added). Unsuffixed columns stay L1, so older files keep their meaning; `win_vs_bf` counts bins and needs none. The `evaluate_final.py` printout shows L2.
- **Affects:** `core.py`, `evaluate_final.py`, `summarise.py` (held-out table and figure on L2, L1 table where it differs; backfills older results), tests, readme (lower bound).

### 2026-11-07: incomplete runs are rerun, not scored (Nic)

- **Decision:** a run that ends early for an infrastructure reason (`budget:llm_errors`, `error`, interrupted) is rerun in full with `--force`; only complete runs (the evaluation budget reached) are scored. The failed attempt stays in git history.
- **Why:** the stop has nothing to do with the method; the rule is applied whatever the rerun scores.
- **Applied to:** `abl_informed_rules` seed 6 (first attempt: 54 evals, network failure; kept in commit `1ef5ac5`).
- **Affects:** `results/abl_informed_rules/eoh/seed6/`.

### 2026-11-07: control runs 10 seeds (Nic)

- **Decision:** `control_randsample` runs seeds 0–9 (10), like Ablation 1's main settings.
- **Why:** the control is compared with both ablations; with 10 seeds its seed counts are on the same footing as A1a/A1b, and the Fisher rule in #15 can reach p < 0.05 (only at 0/10 or 10/10 vs the baseline's 3/6).
- **Affects:** `configs/control_randsample.yaml`, `tests/test_ablations.py`; results in `results/control_randsample/`.

### 2026-11-07: Ablation 1 runs 10 seeds; optional rows dropped (Nic)

- **Decision:** Ablation 1 runs `abl_anyfit` and `abl_informed_rules` with seeds 0–9 (10 each), instead of 0–4. `abl_informed_dist` and `abl_informed` are not run. Supersedes the seed count in "2026-11-04: ablation settings locked" for A1.
- **Why:** about half of the baseline seeds converge to Best Fit (3 of 6), so 5 seeds per setting leaves 2–3 informative runs, too few to compare how often runs open fresh bins. Dropping the optional rows keeps the cost about the same (20 runs, ~$4) while doubling the seeds where the questions are.
- **When:** decided and committed on 7 Nov before the A1 runs started (`e3009aa`, configs and test); this entry was meant to go in with that commit and was added later.
- **Alternatives considered:** 5 seeds per setting plus both optional rows (same cost, less power on the main question).
- **Affects:** `configs/abl_anyfit.yaml`, `configs/abl_informed_rules.yaml`, `tests/test_ablations.py`; results in `results/abl_anyfit/`, `results/abl_informed_rules/`.

### 2026-11-09: Ablation 2 = A2a mixed shapes + A2b mixed capacities, 10 seeds each; A3 optional (Jin)

- **Decision:** Ablation 2 has two settings, both changing only the training data, 10 seeds (0–9) each. **A2a** `abl_mixed_shapes` (baseline setup, compared with the baseline). **A2b** `abl_mixed_capacities` (A1b's informed prompt, trained on `train_mixed_cap`: capacities 100 and 300, item scale 45 in both halves; compared with `abl_informed_rules`). The stronger model (`abl_strong_model`) becomes optional A3. Agreed on #7 (Jia Kang, 9 Nov); predictions in #15 before any A2 run.
- **Why A2b builds on A1b:** the capacity collapse it tests appears only there (5/10 seeds on cap500, 4/10 on cap200 and scale25); the baseline has none, so on the baseline setup A2b could only show "nothing breaks". Cost: the result holds for the informed prompt only (#15).
- **Tuned Best Fit's penalty range in A2b: 1..300 (default 1..100).** An empty bin at capacity 300 leaves ~260, so with penalty ≤ 100 the tuner can never place the fresh-bin bet in half the training set, while the LLM rules can (they read the capacity from `max(bins)`). `lo` and `hi` stay: they measure gaps in item units, and item sizes don't change. Same 300 tries over a 3× larger space (112,200 settings vs 37,400).
- **Alternatives considered:** keep 1..100 (handicaps the reference in half the training set); scale every knob ×3, as the `tuned_bf.py` docstring suggests for a new capacity (that rule assumes items scale with the bin, which they don't here).
- **Check:** pipecheck of `abl_mixed_capacities` (seed 99, `check` budget, eoh only): 12/12 evals ok, informed prompt confirmed, 32 training instances. Not scored.
- **Affects:** `configs/abl_mixed_shapes.yaml`, `configs/abl_mixed_capacities.yaml`, `configs/instances/v0.yaml` (`train_mixed_cap`; no other set changes), `tests/test_ablations.py`; #15.
