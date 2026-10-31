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
- **Why:** tests the explanation for EoH seed 0's k=5 failure. Result: EoH opens a fresh bin in ~36% of choices on train (57% on k=5); BF/FF never do. In anyfit mode k=5 excess falls 38.9% → 8.1%, but EoH also loses most of its in-distribution lead (train 3.79% → 4.34%, BF 4.75%). The same bet drives both the gain and the failure. One seed so far.
- **Alternatives considered:** adding the column inside `core.evaluate` (touches the search hot path and Nic's scorer).
- **Affects:** `results/eoh_baseline/diagnostics_summary.csv`, report analysis section.

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
