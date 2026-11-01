# Decision log

One entry per decision that affects results. Newest at the top. Link the config and the results folder.

## Template

### YYYY-MM-DD: short title (who)

- **Decision:**
- **Why:**
- **Alternatives considered:**
- **Affects:** configs / results folders / report sections

---

### 2026-10-31: fresh-bin diagnostic, scores unchanged (Jin)
- **Decision:** `obpevo/diagnostics.py` + `diagnose.py` rescore best programs and the BF/FF/WF baselines in both packing modes and add `fresh_bin_rate` (share of items put in an empty bin while a used bin had room). Separate from `core.py`; tests check it packs identically to `core.py` in both modes.
- **Why:** tests the explanation for EoH seed 0's k=5 failure. Result: EoH opens a fresh bin in ~36% of choices on train (57% on k=5); BF/FF never do. In anyfit mode k=5 excess falls 38.9% → 8.1%, but EoH also loses most of its in-distribution lead (train 3.79% → 4.34%, BF 4.75%). The same bet drives both the gain and the failure. Seed 11 (Jin, free Gemini; 62 calls hit 429 rate limits but all 300 evals ran): opens fresh bins far less (~9%), gains less (train 4.48%), no k=5 failure but fails on `test_cap500` (6.12% vs BF 0.50%). In anyfit it matches BF on every set (train 4.77%): again both its gain and its failure disappear.
- **Alternatives considered:** adding the column inside `core.evaluate` (touches the search hot path and Nic's scorer).
- **Affects:** `results/eoh_baseline_25oct/diagnostics_summary.csv`, report analysis section.

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
