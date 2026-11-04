"""Framework-free online bin packing (OBP): packing, lower bound and per-instance scoring.

This module is the single source of truth for scores. Search methods only ever see the
mean score through `task.OBPTask`; analysis scripts use the per-instance rows directly.

Two packing conventions are supported:
  - "funsearch": n empty bins are pre-allocated and the heuristic scores *every* bin the
    item fits into, including empty ones (FunSearch / EoH / LLM4AD convention, so results
    are comparable with published numbers).
  - "anyfit": the heuristic only scores open bins; a new bin opens only if nothing fits
    (the AnyFit class analysed by Sgall, 2014).
"""
from __future__ import annotations

import time
import numpy as np


class InvalidHeuristic(Exception):
    """Raised when a priority function returns something we cannot use."""


def lower_bound(items: np.ndarray, capacity: int) -> int:
    """L1 lower bound on the optimal number of bins: ceil(total size / capacity)."""
    return int(np.ceil(np.sum(items) / capacity))


def lower_bound_l2(items: np.ndarray, capacity: int) -> int:
    """Martello and Toth's L2 lower bound (1990): never below L1, and much tighter when many items exceed half a bin.

    Two items larger than half a bin can never share one, which L1 ignores. For each threshold K in [0, C/2]:
    J1 = items > C - K, J2 = items in (C/2, C - K], J3 = items in [K, C/2]. Every J1 and J2 item needs its own bin,
    and J3 items that don't fit in the space left in J2 bins need extra bins:
        L(K) = |J1| + |J2| + max(0, ceil((sum(J3) - (|J2| * C - sum(J2))) / C)).
    L2 = max over K; the maximum is reached at K = 0 or at an item size <= C/2.
    """
    s, c = np.asarray(items), capacity
    best = 0
    for k in np.unique(np.concatenate(([0], s[s <= c / 2]))):
        n1 = int(np.count_nonzero(s > c - k))
        j2 = s[(s > c / 2) & (s <= c - k)]
        j3_sum = int(s[(s >= k) & (s <= c / 2)].sum())
        spare = len(j2) * c - int(j2.sum())
        best = max(best, n1 + len(j2) + max(0, int(np.ceil((j3_sum - spare) / c))))
    return best


def _checked(scores, n: int) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if s.shape != (n,):
        raise InvalidHeuristic(f"priority returned shape {s.shape}, expected ({n},)")
    if np.isnan(s).any():
        raise InvalidHeuristic("priority returned NaN")  # +/-inf is allowed (e.g. exact-fit bonus)
    return s


def pack_funsearch(priority, items: np.ndarray, capacity: int) -> int:
    """FunSearch convention: score all feasible bins, including pre-allocated empty ones."""
    bins = np.full(len(items), capacity, dtype=int)            # remaining capacity per bin
    for item in items:
        idx = np.flatnonzero(bins >= item)                       # feasible bins (always >= 1 empty bin)
        best = idx[np.argmax(_checked(priority(item, bins[idx]), idx.size))]
        bins[best] -= item
    return int((bins != capacity).sum())                         # bins actually used


def pack_anyfit(priority, items: np.ndarray, capacity: int) -> int:
    """AnyFit convention: score open bins only; open a new bin only when nothing fits."""
    bins = np.empty(len(items), dtype=int)                       # remaining capacity, first `used` entries live
    used = 0
    for item in items:
        idx = np.flatnonzero(bins[:used] >= item)
        if idx.size:
            best = idx[np.argmax(_checked(priority(item, bins[idx]), idx.size))]
            bins[best] -= item
        else:
            bins[used] = capacity - item                         # open a new bin
            used += 1
    return used


PACKERS = {"funsearch": pack_funsearch, "anyfit": pack_anyfit}


def evaluate(priority, instances: list[dict], mode: str = "funsearch") -> list[dict]:
    """Pack every instance and return one row per instance (bins used, lower bound, excess, time)."""
    pack = PACKERS[mode]
    rows = []
    for inst in instances:
        t0 = time.perf_counter()
        n_bins = pack(priority, inst["items"], inst["capacity"])
        lb = lower_bound(inst["items"], inst["capacity"])
        rows.append(dict(instance=inst["id"], n_items=len(inst["items"]), capacity=inst["capacity"],
                         bins=n_bins, lb=lb, excess=(n_bins - lb) / lb,
                         time_s=round(time.perf_counter() - t0, 4)))
    return rows


def score(rows: list[dict]) -> float:
    """Fitness passed to the search: higher is better, so it is minus the mean excess over the lower bound."""
    return -float(np.mean([r["excess"] for r in rows]))
