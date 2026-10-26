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
