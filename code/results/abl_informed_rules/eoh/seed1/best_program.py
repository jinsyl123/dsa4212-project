# eval 211, train excess 0.02799

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
    valid = rem >= 0
    scores = np.full_like(bins, -np.inf, dtype=float)
    if not np.any(valid):
        return scores
    
    orig = bins[valid] - rem[valid] + item
    frac = rem[valid] / (orig + 1e-7)
    
    scores[valid] = -np.abs(frac - 0.5) + (1.5 / (rem[valid] + 1e-5))
    return scores

