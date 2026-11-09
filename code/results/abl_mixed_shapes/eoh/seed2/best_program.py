# eval 140, train excess 0.04719

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
    target = 0.2
    sigma = 0.15
    score = np.exp(-((remaining - target) ** 2) / (2 * sigma ** 2))
    return np.where(remaining < 0, -np.inf, score)

