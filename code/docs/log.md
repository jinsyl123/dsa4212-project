# Decision log

One entry per decision that affects results. Newest at the top. Link the config and the results folder.

## Template

### YYYY-MM-DD: short title (who)

- **Decision:**
- **Why:**
- **Alternatives considered:**
- **Affects:** configs / results folders / report sections

---

### 2026-11-01: integrity checks on candidates, flag only (Jin)
- **Decision:** `obpevo/integrity.py` parses every candidate with `ast` and records rule violations (non-numpy imports, randomness incl. `hash()`/`id()`/clock, `exec`/`eval`/`open`, numpy file I/O, dunder access, `global`, attribute assignment, module-level state, mutable defaults) in a new `integrity` column of `evals.csv`; `summary.json` gets `n_flagged` and `best_integrity`. `check_integrity.py` rescores each best program four times (usual order twice, one shuffled order twice), each in a fresh spawned process with its own hash and RNG seeds. Flagged programs are still scored and can still win.
- **Why:** the brief asks to keep scoring outside the code being evolved. Flagging instead of dropping keeps every method's search unchanged (fair) and lets us report how often each rule is hit. All committed best programs (25 Oct seed 0, OpenRouter seeds 0–2) are clean: no flags, 0 repeat and 0 shuffle mismatches on train and val.
- **Limits (from an independent review):** LLM4AD keeps only the function body of each reply, so module-level rules rarely fire in our runs; an empty column means "no body-level violation". Static checks can be bypassed (e.g. a helper function mutating shared state); the rescoring catches such cases only if they change a bin count. Fresh spawned processes are needed because forked children share the parent's hash seed, which hid `hash()`-based nondeterminism in the first version.
- **Alternatives considered:** rejecting flagged candidates (changes the search and LLM4AD's behaviour); a runtime sandbox (heavier; LLM4AD already runs candidates in a child process with a timeout).
- **Affects:** `evals.csv` (new column, at the end), `summary.json`, `results/*/integrity.csv`. No score changes.

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

### 2026-11-04: paired statistics per seed (Jin)

- **Decision:** `paired_stats.py` compares every rule with Best Fit and tuned Best Fit instance by instance: wins/ties/losses, two-sided sign test, mean difference with a 95% bootstrap interval, per seed. A verdict (better / worse) needs both the interval to exclude 0 and p < 0.05, since an interval over 5 values is too narrow (seed 2's 4–1 on `test_id_5k` gives an interval excluding 0 at p = 0.125). Reference rules with several seeds (tuned Best Fit) are averaged per instance. Excess is over the L2 bound by default (#12 decision 7, #10), `--metric excess` for L1.
- **Why:** the plan's "instance-by-instance sign test vs best fit" and win rates (decision 14). Per seed, because 3 of 6 OpenRouter seeds tie Best Fit everywhere and the other 3 differ by ±1–2.6 pp, which a mean over seeds hides. On the baseline, seed 4 beats Best Fit on all 5 `test_id_5k` instances but the sign test can't go below p = 0.0625 with 5 instances (#12 proposes 10).
- **Affects:** new `pipeline/paired_stats.py`, `results/*/paired_stats.csv`, readme §3 and §6.
