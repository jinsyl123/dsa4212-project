# Seed counts and Fisher tests (`pipeline/seed_counts.py`)

Each cell: seeds in the category / seeds, then Fisher p against each comparison row (claims.md, shared rules). Baseline rows: seeds 0-5 (pre-registered) and seeds 0-9. A2b is compared with A1b. **Bold**: claimed, p < 0.05 against every comparison row.

| Setting | Beats Best Fit on `test_id_5k` | Beats its tuned Best Fit on `test_id_5k` | Worse than Best Fit on k1.5 or k5 | Fresh-bin seeds |
|---|---|---|---|---|
| baseline seeds 0-5 | 3/6 | 0/6 | 3/6 | 3/6 |
| baseline seeds 0-9 | 5/10 | 0/10 | 3/10 | 3/10 |
| control | **0/10 (p = 0.036, 0.033)** | 0/10 (p = 1.000, 1.000) | 0/10 (p = 0.036, 0.211) | 0/10 (p = 0.036, 0.211) |
| A1a | 6/10 (p = 1.000, 1.000) | 0/10 (p = 1.000, 1.000) | 1/10 (p = 0.118, 0.582) | 0/10 (p = 0.036, 0.211) |
| A1b | **10/10 (p = 0.036, 0.033)** | 0/10 (p = 1.000, 1.000) | 1/10 (p = 0.118, 0.582) | 6/10 (p = 1.000, 0.370) |
| A2a | 9/10 (p = 0.118, 0.141) | 1/10 (p = 1.000, 1.000) | 1/10 (p = 0.118, 0.582) | 2/10 (p = 0.299, 1.000) |
| A2b | 6/10 (p = 0.087) | 0/10 (p = 1.000) | 1/10 (p = 1.000) | 2/10 (p = 0.170) |
