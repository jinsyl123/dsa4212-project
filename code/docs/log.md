# Decision log

One entry per decision that affects results. Newest at the top. Link the config and the results folder.

## Template

### YYYY-MM-DD: short title (who)

- **Decision:**
- **Why:**
- **Alternatives considered:**
- **Affects:** configs / results folders / report sections

---

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

- **Decision:** the budget-matched baseline `tuned_bf` is Best Fit plus a penalty for leaving a small gap: score = −r − penalty·[lo < r < hi], with r the space left in a bin after packing. Three integer knobs (`lo` 0–10, `hi` lo+2–40, `penalty` 1–100) in item units, tuned by random search on `train`: Best Fit first, then 299 distinct random settings, i.e. the same 300 scored tries as an EoH run. Seeds 0–2, like EoH.
- **Why:** beating plain Best Fit is weak evidence (decision #13). Any score strictly decreasing in r packs exactly like Best Fit, so the sliver penalty is the smallest change that can differ. It is the strategy the FunSearch paper describes for several of its discovered heuristics (illustrated with its OR-dataset heuristic, Fig. 6). The knobs are in absolute units, and neither the rule nor an evolved program is given the capacity (in funsearch mode an evolved program could read it from max(bins)). Random search is the plainest tuner, so its result is a floor for what tuning alone achieves.
- **Alternatives considered:** a 4th knob gating the penalty by item size (no better in an 80-try pilot: 2.68% vs 2.55% train excess); matching EoH on wall-clock time (unfair: EoH waits on the API) or dollars (tuning is free); a stronger tuner (TPE, grid), which would only raise the bar.
- **Affects:** `obpevo/tuned_bf.py`, `run_experiment.py`, `evaluate_final.py`, `summarise.py`, `configs/eoh_baseline.yaml`, `configs/smoke.yaml`; results in results/eoh_baseline\_\_openrouter-gemini-3.5-flash-lite/tuned_bf (readme §13).
