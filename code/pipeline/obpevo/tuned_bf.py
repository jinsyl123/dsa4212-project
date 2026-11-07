"""Tuned Best Fit: the budget-matched baseline an LLM search has to beat (decision #13).

Best Fit puts each item in the bin it fills most tightly. Any score that strictly falls as the space left,
r = bins - item, grows packs exactly like Best Fit, so a variant can only differ by not strictly falling in r.
The simplest such variant is the strategy the FunSearch paper describes for several of its discovered heuristics
(illustrated with its OR-dataset heuristic, Fig. 6): pack tightly when the fit is very tight, otherwise avoid
leaving a small gap that later items are unlikely to fill.

    score = -r - penalty * (lo < r < hi)

Three integer knobs, in the same absolute units as the items. Like an evolved program it gets only `item` and
`bins`, never the capacity. (In funsearch mode an evolved program could still read it from max(bins), since an
empty bin is always offered; this rule doesn't.) It is tuned at capacity 100 and faces the same shifted test sets.

    lo       gaps of at most `lo` count as a tight fit and are not penalised             0 .. 10
    hi       gaps from lo+1 to hi-1 are slivers                                          lo+2 .. 40
    penalty  what leaving a sliver costs. At capacity 100, from 99 up even opening an    1 .. 100
             empty bin beats leaving a sliver, so larger values change nothing

lo=0, hi=0, penalty=0 is plain Best Fit.

These ranges (and the 99 above) assume capacity 100, the training capacity. To tune at another capacity,
rescale SPACE with it (config: params: {space: ...}). Test sets at another capacity (e.g. test_cap500) are only
scored, never tuned on, so the tuned knobs keep their capacity-100 values there.

Same budget means the same number of scored tries: `tune` scores Best Fit first (every LLM method also starts
from the Best Fit template), then max_evals - 1 distinct random knob settings, on the training set only. Same
wall-clock time would be unfair (an LLM search waits on the API) and same dollars is meaningless (tuning is free).
Random search is the plainest tuner, so its result is a floor for what tuning alone achieves.
"""
from __future__ import annotations

import time

import numpy as np

from . import core

SPACE = dict(lo=(0, 10), hi_max=40, penalty=(1, 100))     # inclusive ranges; hi is drawn from lo+2 .. hi_max.
                                                          # Sized for capacity 100: rescale to tune at another
BEST_FIT = dict(lo=0, hi=0, penalty=0)

TEMPLATE = '''import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo={lo}, hi={hi}, penalty={penalty}): the tightest bin wins, but leaving a gap
    strictly between {lo} and {hi} units costs {penalty}. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - {penalty} * ((r > {lo}) & (r < {hi}))
'''


def program(lo: int, hi: int, penalty: int) -> str:
    """Source of the priority function for one knob setting (what gets scored is exactly what gets saved)."""
    return TEMPLATE.format(lo=int(lo), hi=int(hi), penalty=int(penalty))


def space_size(space: dict = SPACE) -> int:
    (lo0, lo1), (p0, p1) = space["lo"], space["penalty"]
    return sum(max(space["hi_max"] - lo - 1, 0) for lo in range(lo0, lo1 + 1)) * (p1 - p0 + 1)


def sample(n: int, seed: int, space: dict = SPACE) -> list[dict]:
    """Best Fit, then n - 1 distinct knob settings drawn at random. Deterministic for a given seed."""
    if n - 1 > space_size(space):
        raise ValueError(f"asked for {n} settings but the knob space only has {space_size(space)} (+ Best Fit)")
    rng = np.random.default_rng(seed)
    out, seen = [dict(BEST_FIT)], set()
    while len(out) < n:
        lo = int(rng.integers(space["lo"][0], space["lo"][1] + 1))
        if lo + 2 > space["hi_max"]:
            continue
        hi = int(rng.integers(lo + 2, space["hi_max"] + 1))
        penalty = int(rng.integers(space["penalty"][0], space["penalty"][1] + 1))
        if (lo, hi, penalty) not in seen:
            seen.add((lo, hi, penalty))
            out.append(dict(lo=lo, hi=hi, penalty=penalty))
    return out


def evaluate_program(src: str, insts: list[dict], mode: str = "funsearch") -> list[dict]:
    ns = {}
    exec(src, ns)                                     # our own template, so no sandbox needed
    return core.evaluate(ns["priority"], insts, mode)


def check_space(space: dict | None = None) -> dict:
    """SPACE with `space`'s overrides applied. An unknown key raises, so a typo can't silently keep the defaults."""
    unknown = sorted(set(space or {}) - set(SPACE))
    if unknown:
        raise ValueError(f"unknown tuned_bf space key(s) {unknown}; valid keys: {sorted(SPACE)}")
    return {**SPACE, **(space or {})}


def tune(train: list[dict], seed: int, max_evals: int, mode: str = "funsearch", space: dict | None = None):
    """Score each knob setting on the training set, in order. Returns an iterator with one record per try:
    eval (1-based), t_start, wall_s, params, program, status ("ok" / "error"), rows (per instance), error.
    `space` overrides any of SPACE's ranges (config: params: {space: {hi_max: 30}}).
    Bad arguments raise here, when tune() is called, before anything is scored."""
    return _score_each(train, sample(max_evals, seed, check_space(space)), mode)


def _score_each(train: list[dict], settings: list[dict], mode: str):
    for i, params in enumerate(settings, start=1):
        src, t0 = program(**params), time.time()
        try:
            rows, status, err = evaluate_program(src, train, mode), "ok", ""
        except Exception as e:                        # not expected for this template; logged like any failure
            rows, status, err = [], "error", f"{type(e).__name__}: {str(e)[:300]}"
        yield dict(eval=i, t_start=t0, wall_s=time.time() - t0, params=params, program=src,
                   status=status, rows=rows, error=err)
