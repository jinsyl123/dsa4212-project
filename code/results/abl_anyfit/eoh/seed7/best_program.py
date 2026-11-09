# eval 228, train excess 0.04064

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
    ratio = np.divide(remaining, bins, out=np.zeros_like(bins, dtype=float), where=bins > 0)
    score = -np.exp(-((ratio - 0.33) ** 2) / 0.05) - 1.0 / (1.0 + np.exp(-20.0 * (ratio - 0.1)))
    return np.where(remaining >= 0, score, -np.inf)

