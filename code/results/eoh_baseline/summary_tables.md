# eoh_baseline: summary tables

Budget per run: {'max_evals': 300, 'max_tokens': 1500000, 'max_minutes': 120}

## Search runs

| label | runs | evals/run | ok % | error % | timeout % | LLM calls/run | tokens/run | cost USD (total) | minutes/run | best train excess % (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| eoh | 1 | 300.0 | 99.7 | 0.3 | 0.0 | 301.0 | 206658 | 0.174 | 34.22 | 3.788 ± 0.000 |

## Held-out sets: excess over lower bound (%), mean ± sd across seeds

| label | val | test_id_5k | test_size_10k | test_cap500 | test_cap500_scaled | test_shape_k1.5 | test_shape_k5 | test_uniform |
|---|---|---|---|---|---|---|---|---|
| eoh | 3.73 | 2.22 | 1.73 | 0.40 | 5.73 | 0.92 | 38.95 | 2.88 |
| first_fit | 5.23 | 4.55 | 4.27 | 0.50 | 4.80 | 1.58 | 8.45 | 5.57 |
| best_fit | 4.73 | 4.17 | 3.95 | 0.50 | 4.44 | 1.34 | 8.25 | 5.07 |
| worst_fit | 13.34 | 13.14 | 13.01 | 2.34 | 13.72 | 12.27 | 12.95 | 14.46 |
| almost_worst_fit | 8.24 | 8.44 | 8.34 | 0.90 | 8.82 | 6.63 | 10.47 | 8.30 |
| next_fit | 28.63 | 29.20 | 28.96 | 4.62 | 29.43 | 30.05 | 24.32 | 31.71 |
