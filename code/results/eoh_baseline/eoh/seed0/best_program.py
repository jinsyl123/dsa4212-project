# eval 217, train excess 0.03788

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
    density = item / bins
    score = np.where(density < 0.5, np.exp(density) - np.log1p(remaining), np.power(density, 1.5) - remaining)
    return np.where(remaining >= 0, score, -np.inf)

