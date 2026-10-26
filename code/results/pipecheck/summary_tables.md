# pipecheck: summary tables

Budget per run: {'max_evals': 12, 'max_tokens': 60000, 'max_minutes': 8}

## Search runs

| label | runs | evals/run | ok % | error % | timeout % | LLM calls/run | tokens/run | cost USD (total) | minutes/run | best train excess % (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| eoh | 1 | 12.0 | 100.0 | 0.0 | 0.0 | 15.0 | 8743 | 0.008 | 0.77 | 4.891 ± 0.000 |

## Held-out sets: excess over lower bound (%), mean ± sd across seeds

| label | val | test_id |
|---|---|---|
| eoh | 4.79 | 4.40 |
| first_fit | 5.17 | 4.90 |
| best_fit | 4.79 | 4.40 |
| worst_fit | 13.49 | 13.04 |
| almost_worst_fit | 8.07 | 8.80 |
| next_fit | 28.37 | 29.40 |
