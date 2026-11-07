# eoh_baseline__openrouter-gemini-3.5-flash-lite: summary tables

Budget per run: {'max_evals': 300, 'max_tokens': 1500000, 'max_minutes': 120, 'name': 'full'}

## Search runs

| label | runs | evals/run | ok % | error % | timeout % | LLM calls/run | tokens/run | cost USD (total) | minutes/run | best train excess % (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| eoh | 6 | 300.0 | 99.1 | 0.9 | 0.0 | 301.0 | 203849 | 1.037 | 11.67 | 4.342 ± 0.491 |
| tuned_bf | 6 | 300.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 2.78 | 2.414 ± 0.051 |

## Held-out sets: excess over L2 lower bound (%), mean ± sd across seeds

| label | val | test_id_5k | test_size_1k | test_size_10k | test_cap200 | test_cap500 | test_cap500_scaled | test_scale25 | test_scale65 | test_shape_k1.5 | test_shape_k5 | test_uniform | test_control |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eoh | 4.26 ± 0.59 | 3.66 ± 0.61 | 4.14 ± 0.48 | 3.61 ± 0.58 | 1.57 ± 0.15 | 0.56 ± 0.08 | 4.01 ± 0.66 | 1.88 ± 0.54 | 0.49 ± 0.14 | 1.73 ± 1.07 | 8.82 ± 1.73 | 4.47 ± 0.60 | 0.00 ± 0.00 |
| tuned_bf | 2.40 ± 0.10 | 1.67 ± 0.13 | 2.27 ± 0.15 | 1.39 ± 0.20 | 1.45 ± 0.03 | 0.49 ± 0.01 | 5.17 ± 0.06 | 1.81 ± 0.35 | 0.41 ± 0.00 | 2.12 ± 1.03 | 29.45 ± 2.20 | 3.23 ± 0.23 | 0.00 ± 0.00 |
| first_fit | 5.23 | 4.43 | 5.06 | 4.29 | 1.66 | 0.52 | 4.77 | 1.77 | 0.79 | 1.60 | 8.48 | 5.47 | 0.00 |
| best_fit | 4.73 | 4.08 | 4.44 | 3.99 | 1.52 | 0.52 | 4.38 | 1.68 | 0.43 | 1.35 | 8.32 | 5.08 | 0.00 |
| worst_fit | 13.34 | 13.08 | 13.24 | 13.04 | 6.38 | 2.44 | 13.61 | 6.92 | 10.93 | 12.36 | 12.93 | 14.56 | 0.00 |
| almost_worst_fit | 8.24 | 8.37 | 8.23 | 8.33 | 3.20 | 0.94 | 8.83 | 3.49 | 6.51 | 6.61 | 10.50 | 8.20 | 0.00 |
| next_fit | 28.63 | 29.33 | 29.07 | 29.08 | 12.40 | 4.60 | 29.51 | 13.78 | 25.40 | 29.98 | 24.33 | 31.47 | 0.00 |

## For comparison with FunSearch/EoH: excess over L1 (%), sets where it differs from L2
Many items exceed half a bin here, so L1 overstates every rule's excess. On every other set the table above is already L1.

| label | test_scale65 | test_control |
|---|---|---|
| eoh | 10.76 ± 0.16 | 32.37 ± 0.00 |
| tuned_bf | 10.66 ± 0.00 | 32.37 ± 0.00 |
| first_fit | 11.08 | 32.37 |
| best_fit | 10.69 | 32.37 |
| worst_fit | 22.25 | 32.37 |
| almost_worst_fit | 17.39 | 32.37 |
| next_fit | 38.20 | 32.37 |
