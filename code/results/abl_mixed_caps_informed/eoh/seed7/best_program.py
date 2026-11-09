# eval 247, train excess 0.02897

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
    target_residual = 0.1
    fit_score = np.exp(-((remaining - target_residual) ** 2) / 0.05)
    indices = np.arange(len(bins), dtype=float)
    pos_factor = np.exp(-0.1 * indices)
    scores = np.where(valid, fit_score * pos_factor, -np.inf)
    return scores

