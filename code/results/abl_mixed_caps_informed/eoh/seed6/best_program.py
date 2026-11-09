# eval 232, train excess 0.02950

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    rem = bins - item
    scores = np.full_like(bins, -np.inf, dtype=float)
    fit = rem >= 0
    if not np.any(fit):
        return scores
    
    empty_mask = np.isclose(bins, np.max(bins)) & np.isclose(rem, np.max(bins) - item)
    tightness = rem
    penalty = np.where(rem < 0.05 * bins, 5.0, 1.0)
    
    scores[fit] = - (tightness[fit] ** 1.5) * penalty[fit]
    scores[fit & empty_mask] += 1.5
    return scores

