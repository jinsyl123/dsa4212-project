# eval 222, train excess 0.04602

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
    optimal_remaining = 0.3 * bins
    score = -np.exp(-((remaining - optimal_remaining) ** 2) / (2 * (0.1 ** 2) + 1e-8))
    return np.where((bins < item) | (bins <= 0), -np.inf, score)

