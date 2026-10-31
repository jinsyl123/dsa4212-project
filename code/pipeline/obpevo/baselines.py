"""Hand-designed online bin-packing baselines (Johnson 1973; as benchmarked by Sim et al. 2025).

All baselines are AnyFit-style algorithms over *open* bins. First Fit and Best Fit give
identical bin counts under the "funsearch" and "anyfit" packing conventions (tested in
tests/test_core.py), so they are valid reference points for evolved heuristics in either mode.
"""
from __future__ import annotations

import numpy as np

from .core import pack_anyfit


# --- priority-function baselines (same signature as evolved heuristics) ---
def first_fit(item, bins):
    return -np.arange(len(bins))          # earliest feasible bin wins


def best_fit(item, bins):
    return item - bins                    # tightest fit wins (the LLM4AD seed/template)


def worst_fit(item, bins):
    return bins - item                    # emptiest open bin wins


# --- baselines that need their own packing loop ---
def pack_next_fit(items, capacity) -> int:
    """Keep a single open bin; close it when the next item does not fit."""
    used, room = 0, 0
    for item in items:
        if item > room:
            used, room = used + 1, capacity
        room -= item
    return used


def pack_almost_worst_fit(items, capacity) -> int:
    """Try the second-emptiest open bin, then the emptiest, else open a new bin."""
    bins = np.empty(len(items), dtype=int)
    used = 0
    for item in items:
        order = np.argsort(-bins[:used], kind="stable")        # emptiest first
        for b in order[1:2].tolist() + order[:1].tolist():
            if bins[b] >= item:
                bins[b] -= item
                break
        else:
            bins[used] = capacity - item
            used += 1
    return used


BASELINES = {
    "first_fit": lambda items, cap: pack_anyfit(first_fit, items, cap),
    "best_fit": lambda items, cap: pack_anyfit(best_fit, items, cap),
    "worst_fit": lambda items, cap: pack_anyfit(worst_fit, items, cap),
    "almost_worst_fit": pack_almost_worst_fit,
    "next_fit": pack_next_fit,
}
