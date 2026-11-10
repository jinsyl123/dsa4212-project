# Project log

> **Early project log (topic, framework, key dates, from 21 Oct).** Every decision that affects results, from the v0 pipeline (25 Oct) on, is in [`code/docs/log.md`](../code/docs/log.md).

One place for decisions, failed attempts and surprises. Add one dated line per entry; keep it short.
Full plan and discussion: [issue #1](https://github.com/jinsyl123/dsa4212-project/issues/1).

## Key dates

G1 loop runs end to end: Sun 25 Oct · G2 design frozen: Sun 1 Nov · G3 experiments frozen: Sun 8 Nov · submit: Thu 12 Nov · due: Fri 13 Nov

## Decisions

| Date       | Decision                                                                           | Why                                                       | Who |
| ---------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------- | --- |
| 2026-10-21 | Topic: LLM-guided evolution on online bin packing; fallback BO vs ES if G1 fails   | Team vote; CPU only, fits $15 OpenRouter credits          | All |
| 2026-10-21 | Framework: try ReEvo and EoH first, OpenEvolve as backup                           | Both ship an online bin-packing task; OpenEvolve does not | A   |
| 2026-10-21 | G1 = a search runs end to end and logs scored candidates (not "beats best fit")    | Brief: failing to improve a baseline is valid evidence    | All |
| 2026-10-21 | Add a no-feedback control (same LLM calls, no best-program feedback)               | Tests whether evolution itself helps                      | C   |
| 2026-10-21 | Headline comparison vs first fit / best fit; FunSearch heuristic as extra baseline | Brief asks to compare with established methods            | All |

## Failed attempts and surprises

| Date | What happened | What we learned / did next | Who |
| ---- | ------------- | -------------------------- | --- |

## Links

- Brief: https://alexxthiery.github.io/teaching/llm-evolution/llm-evolution.html
- Guidelines: https://alexxthiery.github.io/teaching/guidelines/guidelines.html
- ReEvo: https://github.com/ai4co/reevo (`problems/bpp_online`)
- EoH: https://github.com/FeiLiu36/EoH (`examples/bp_online`)
- FunSearch: https://github.com/google-deepmind/funsearch (`bin_packing/`)
- OpenEvolve: https://github.com/algorithmicsuperintelligence/openevolve
