"""Behaviour diagnostics for packing rules. Adds columns only: core.py stays the single source of truth for scores.

fresh_bin_rate: share of items placed in an empty bin while some already-used bin had room for them.
Only the "funsearch" convention allows this (empty bins are scored too); under "anyfit" it is 0 by construction.
Best Fit and First Fit never do it (rate 0), Worst Fit always does when it can.
"""
from __future__ import annotations

import numpy as np

from .core import _checked, lower_bound


def pack_trace(priority, items: np.ndarray, capacity: int, mode: str = "funsearch") -> dict:
    """Same packing as core.pack_funsearch / core.pack_anyfit, plus counts of fresh-bin choices."""
    n_room = n_fresh = 0          # items with a used bin that fits; of those, items put in an empty bin anyway
    if mode == "funsearch":
        bins = np.full(len(items), capacity, dtype=int)
        for item in items:
            idx = np.flatnonzero(bins >= item)
            best = idx[np.argmax(_checked(priority(item, bins[idx]), idx.size))]
            if (bins[idx] < capacity).any():                     # a used bin had room
                n_room += 1
                n_fresh += bins[best] == capacity
            bins[best] -= item
        used = int((bins != capacity).sum())
    elif mode == "anyfit":
        bins, used = np.empty(len(items), dtype=int), 0
        for item in items:
            idx = np.flatnonzero(bins[:used] >= item)
            if idx.size:
                n_room += 1
                best = idx[np.argmax(_checked(priority(item, bins[idx]), idx.size))]
                bins[best] -= item
            else:
                bins[used] = capacity - item
                used += 1
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return dict(bins=used, n_room=n_room, n_fresh=int(n_fresh))


def diagnose(priority, instances: list[dict], mode: str = "funsearch") -> list[dict]:
    """One row per instance: bins and excess (as core.evaluate) plus the fresh-bin columns."""
    rows = []
    for inst in instances:
        t = pack_trace(priority, inst["items"], inst["capacity"], mode)
        n, lb = len(inst["items"]), lower_bound(inst["items"], inst["capacity"])
        rows.append(dict(instance=inst["id"], mode=mode, n_items=n, capacity=inst["capacity"],
                         bins=t["bins"], lb=lb, excess=(t["bins"] - lb) / lb,
                         n_room=t["n_room"], n_fresh=t["n_fresh"],
                         fresh_bin_rate=t["n_fresh"] / n,                                  # share of all items
                         fresh_given_room=t["n_fresh"] / t["n_room"] if t["n_room"] else np.nan))  # share of items that had a choice
    return rows
