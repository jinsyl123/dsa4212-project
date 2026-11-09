# eval 196, train excess 0.04706

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
    optimal_buffer = 0.05
    deviation = remaining - optimal_buffer
    scores = np.exp(-(deviation ** 2) / (2 * (0.2 ** 2)))
    return np.where(remaining < 0, -np.inf, scores)

