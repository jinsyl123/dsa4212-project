# eval 141, train excess 0.04444

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    valid = remaining >= 0
    if not np.any(valid):
        return np.full_like(bins, -np.inf, dtype=float)
    utilization = 1.0 - (remaining / bins)
    mean_util = np.mean(utilization[valid])
    score = np.exp(-remaining) - np.exp(-((utilization - mean_util) ** 2))
    return np.where(valid, score, -np.inf)

