# Baseline findings: what the evolved rules do (EoH, OpenRouter seeds 0–5)

Evidence behind the answers on #7. Numbers: excess over the L1 bound (%, lower is better), rescored from each run's
committed `best_program.py` with `core.evaluate` on `configs/instances/v0.yaml`. They match `final_summary.csv` and
`diagnostics_summary.csv` in `results/eoh_baseline__openrouter-gemini-3.5-flash-lite/`.

## Summary

| Rule | Found at eval | train | k5 | k1.5 | `fresh_given_room` (train) | What it is |
|---|---|---|---|---|---|---|
| Best Fit | – | 4.75 | 8.25 | 1.34 | 0 | |
| First Fit | – | 5.29 | 8.45 | 1.58 | 0 | |
| seeds 0, 1 | 1 | 4.75 | 8.25 | 1.34 | 0 | Best Fit |
| seed 2 | 66 | 4.51 | 8.31 | 1.30 | 0 | Best Fit for small gaps, First Fit otherwise |
| seed 3 | 222 | 4.60 | 6.32 | 3.92 | 0.13 | First Fit with an accidental skip window |
| seeds 4, 5 | 212 / 196 | 3.72 | 10.62 | 1.22 | 0.25 | the fresh-bin bet (same rule, written two ways) |
| tuned Best Fit (#8, 3 seeds) | – | 2.37–2.47 | 29.86–33.05 | 1.92–3.55 | > 0 | the fresh-bin bet, found without an LLM |

Main points:
1. **The rules that clearly beat Best Fit open empty bins on purpose** (seeds 4/5, and tuned Best Fit). That habit gives
   about half of EoH's gain and nearly all of tuned Best Fit's, and it causes the k5 failure.
2. **Seed 3's failure has another cause:** its formula underflows to 0, so it is First Fit plus an arbitrary skip window.
3. **The mode is constant** (`funsearch` everywhere). Whether a rule opens empty bins is an outcome, measured by
   `fresh_given_room`.

Notation: `bins` is each bin's remaining room, `remaining = bins - item` is the gap left after placing the item.

## Seeds 0, 1: Best Fit

```python
return np.where(remaining >= 0, -remaining, -np.inf)              # seed 0
return np.where(residual >= 0, 1.0 / (residual + 1e-5), -np.inf)  # seed 1: same ordering
```

Both rank bins by smallest gap, so they pack exactly like Best Fit. The search never improved on its first program.

## Seed 2: Best Fit for small gaps, First Fit for large ones

```python
return np.where(remaining >= 0, np.exp(-((remaining - 0.1) ** 2) / (2 * 0.2 ** 2)), -1.0)
```

`exp(-r²/0.08)` underflows to exactly 0 for gaps above ~7.7. If some bin leaves a gap ≤ 7, the tightest wins (Best Fit).
Otherwise all bins tie at 0 and `np.argmax` takes the earliest-opened one (First Fit). Empty bins come after the
used ones, so they never win.

## Seeds 4, 5: the fresh-bin bet

```python
scores = -np.sin(np.pi * remaining / item) - remaining / item      # seed 4
scores = -remaining / item + np.sin(np.pi * bins / item)           # seed 5
```

Since `bins = remaining + item`, sin(π·bins/item) = −sin(π·remaining/item): the two are equal, and they give identical
packings. The score depends only on x = gap / item size, f(x) = −x − sin(πx):

| gap left, in items (x) | 0 | 0.25 | 0.5 | 1.0 | 1.4 | 2.0 |
|---|---|---|---|---|---|---|
| score | **0** (exact fit) | −0.96 | −1.50 | −1.00 | **−0.45** | −2.00 |

- An empty bin with an average item (~40) leaves x ≈ 1.5 (score −0.50), close to the second peak. A used bin only
  beats it if the gap left is under ~0.1 item or around 1.2–1.6 items. So the rule opens a new bin rather than leave
  a gap smaller than one item.
- **Train** (spread-out sizes): later small items fill those gaps, so −1.03 pp vs Best Fit.
- **k5** (items all ~41): a 2nd item in a bin leaves x ≈ 0.44 (−1.42), so it opens a fresh bin instead, and the ~59 gaps
  rarely fill. **9% of k5 bins hold a single item, vs 0% for Best Fit** (train: 1% vs 1%). Result: +2.37 pp.
- With empty bins blocked (`anyfit`): k5 8.18 (below Best Fit), train 4.31. The failure disappears and about half the
  gain remains.

## Seed 3: First Fit plus an accidental skip window

```python
optimal_remaining = 0.3 * bins
score = -np.exp(-((remaining - optimal_remaining) ** 2) / (2 * (0.1 ** 2) + 1e-8))
```

The denominator is 0.02, so `exp` underflows to exactly 0 whenever |gap − 0.3·room| > ~3.8. **For 97% of placements,
the chosen bin scores exactly 0** and wins the tie by coming first, i.e. First Fit. Bins inside the narrow window get a
tiny negative score and are skipped.

Check: "First Fit, but skip bins where |gap − 0.3·room| < 3.77" gives 4.61 / 3.94 / 6.36 vs seed 3's 4.60 / 3.92 / 6.32
(train / k1.5 / k5). Its k1.5 loss (worse than plain First Fit's 1.58) comes from that window, not from fresh bins, so
it stays when empty bins are blocked (3.99). Its k5 win is the same window happening to help on that distribution.

## Tuned Best Fit (#8): the same bet, without an LLM

```python
return -r - 97 * ((r > 4) & (r < 31))     # seed 0: tightest bin, but a gap of 5–30 costs 97
```

An empty bin (gap ~60) beats any bin that would be left with a 5–30 gap, so it opens bins eagerly.

| tuned Best Fit seed | train `funsearch` / `anyfit` | k5 | k1.5 |
|---|---|---|---|
| 0 | 2.44 / 4.65 | 29.86 / 8.22 | 2.12 / 1.29 |
| 1 | 2.37 / 4.64 | 33.05 / 8.21 | 3.55 / 1.23 |
| 2 | 2.47 / 4.65 | 29.86 / 8.22 | 1.92 / 1.29 |

Under `anyfit`, tuned Best Fit is 4.65 vs Best Fit's 4.75: almost all of its gain comes from opening empty bins. Any
search on this training set finds the bet, not just the LLM.

## What this means for the ablations (#7)

- **Ablation 1 (empty bins hidden during search):** tests whether EoH can find the non-fresh-bin half of the gain on
  purpose. Bar: tuned Best Fit re-tuned in `anyfit` mode.
- **Ablation 2 (mixed training shapes):** seed 3 shows a rule tuned to the training distribution by accident. Tests
  whether variety removes such failures. Bar: tuned Best Fit re-tuned on `train_mixed`.
- The good rules appeared late (eval ~200 of 300), so keep the `full` budget.
- Report the `anyfit` rescoring and `fresh_given_room` next to every rule.
