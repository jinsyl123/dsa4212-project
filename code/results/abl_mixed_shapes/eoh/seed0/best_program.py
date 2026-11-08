# eval 169, train excess 0.04737

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
    optimal_residual = 0.2
    score = np.exp(-((remaining - optimal_residual) ** 2) / 0.1)
    return np.where(remaining < 0, -np.inf, score)

