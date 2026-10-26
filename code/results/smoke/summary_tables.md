# smoke: summary tables

Budget per run: {'max_evals': 24, 'max_tokens': 200000, 'max_minutes': 10}

## Search runs

| label | runs | evals/run | ok % | error % | timeout % | LLM calls/run | tokens/run | cost USD (total) | minutes/run | best train excess % (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| eoh | 2 | 24.0 | 91.7 | 8.3 | 0.0 | 25.0 | 8160 | 0.0 | 0.01 | 4.891 ± 0.000 |
| randsample | 2 | 24.0 | 89.6 | 8.3 | 2.1 | 24.0 | 4326 | 0.0 | 0.08 | 4.891 ± 0.000 |
| hillclimb | 2 | 24.0 | 89.6 | 8.3 | 2.1 | 24.0 | 4326 | 0.0 | 0.08 | 4.891 ± 0.000 |
| reevo | 2 | 24.0 | 89.6 | 10.4 | 0.0 | 43.0 | 17335 | 0.0 | 0.01 | 4.891 ± 0.000 |
| funsearch | 2 | 24.0 | 89.6 | 8.3 | 2.1 | 28.0 | 5127 | 0.0 | 0.08 | 4.891 ± 0.000 |

## Held-out sets: excess over lower bound (%), mean ± sd across seeds

| label | val | test_id | test_cap500 | test_shape_k5 |
|---|---|---|---|---|
| eoh | 4.79 ± 0.00 | 4.40 ± 0.00 | 1.25 ± 0.00 | 8.80 ± 0.00 |
| randsample | 4.79 ± 0.00 | 4.40 ± 0.00 | 1.25 ± 0.00 | 8.80 ± 0.00 |
| hillclimb | 4.79 ± 0.00 | 4.40 ± 0.00 | 1.25 ± 0.00 | 8.80 ± 0.00 |
| reevo | 4.79 ± 0.00 | 4.40 ± 0.00 | 1.25 ± 0.00 | 8.80 ± 0.00 |
| funsearch | 4.79 ± 0.00 | 4.40 ± 0.00 | 1.25 ± 0.00 | 8.80 ± 0.00 |
| first_fit | 5.17 | 4.90 | 1.25 | 9.36 |
| best_fit | 4.79 | 4.40 | 1.25 | 8.80 |
| worst_fit | 13.49 | 13.04 | 2.50 | 12.80 |
| almost_worst_fit | 8.07 | 8.80 | 1.25 | 10.56 |
| next_fit | 28.37 | 29.40 | 5.00 | 24.48 |
