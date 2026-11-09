# eval 237, train excess 0.04775

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
    target = 0.02 * bins
    width = 0.05 * bins + 1e-5
    normalized = (remaining - target) / width
    score = np.where(np.abs(normalized) <= np.pi, 0.5 * (np.cos(normalized) + 1.0), 0.0)
    overflow_penalty = np.where(remaining < 0, -1.0 / (1.0 + remaining ** 2), 0.0)
    return score + 10000.0 * overflow_penalty

