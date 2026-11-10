# abl_mixed_shapes: summary tables

Budget per run: {'max_evals': 300, 'max_tokens': 1500000, 'max_minutes': 120, 'name': 'full'}

## Search runs

| label | runs | evals/run | ok % | error % | timeout % | LLM calls/run | tokens/run | cost USD (total) | minutes/run | best train excess % (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| eoh | 10 | 300.0 | 99.3 | 0.6 | 0.0 | 301.1 | 202843 | 1.664 | 14.24 | 4.594 ± 0.347 |
| tuned_bf | 10 | 300.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 3.26 | 3.295 ± 0.069 |

## Held-out sets: excess over L2 lower bound (%), mean ± sd across seeds

| label | val | test_id_5k | test_size_1k | test_size_10k | test_cap200 | test_cap500 | test_cap500_scaled | test_scale25 | test_scale65 | test_shape_k1.5 | test_shape_k5 | test_uniform | test_control |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eoh | 4.27 ± 0.45 | 3.64 ± 0.64 | 4.21 ± 0.49 | 3.60 ± 0.62 | 1.44 ± 0.25 | 0.51 ± 0.03 | 4.16 ± 0.62 | 1.56 ± 0.30 | 0.50 ± 0.21 | 1.25 ± 0.19 | 8.43 ± 0.19 | 4.53 ± 0.54 | 0.00 ± 0.00 |
| tuned_bf | 3.14 ± 0.21 | 2.63 ± 0.25 | 3.04 ± 0.20 | 2.54 ± 0.28 | 1.47 ± 0.02 | 0.50 ± 0.01 | 4.85 ± 0.05 | 2.34 ± 0.46 | 0.42 ± 0.00 | 1.70 ± 0.21 | 7.68 ± 0.84 | 3.37 ± 0.32 | 0.00 ± 0.00 |
| first_fit | 5.23 | 4.43 | 5.06 | 4.29 | 1.66 | 0.52 | 4.77 | 1.77 | 0.79 | 1.60 | 8.48 | 5.47 | 0.00 |
| best_fit | 4.73 | 4.08 | 4.44 | 3.99 | 1.52 | 0.52 | 4.38 | 1.68 | 0.43 | 1.35 | 8.32 | 5.08 | 0.00 |
| worst_fit | 13.34 | 13.08 | 13.24 | 13.04 | 6.38 | 2.44 | 13.61 | 6.92 | 10.93 | 12.36 | 12.93 | 14.56 | 0.00 |
| almost_worst_fit | 8.24 | 8.37 | 8.23 | 8.33 | 3.20 | 0.94 | 8.83 | 3.49 | 6.51 | 6.61 | 10.50 | 8.20 | 0.00 |
| next_fit | 28.63 | 29.33 | 29.07 | 29.08 | 12.40 | 4.60 | 29.51 | 13.78 | 25.40 | 29.98 | 24.33 | 31.47 | 0.00 |

## For comparison with FunSearch/EoH: excess over L1 (%), sets where it differs from L2
Many items exceed half a bin here, so L1 overstates every rule's excess. On every other set the table above is already L1.

| label | test_scale65 | test_control |
|---|---|---|
| eoh | 10.76 ± 0.23 | 32.37 ± 0.00 |
| tuned_bf | 10.68 ± 0.00 | 32.37 ± 0.00 |
| first_fit | 11.08 | 32.37 |
| best_fit | 10.69 | 32.37 |
| worst_fit | 22.25 | 32.37 |
| almost_worst_fit | 17.39 | 32.37 |
| next_fit | 38.20 | 32.37 |
