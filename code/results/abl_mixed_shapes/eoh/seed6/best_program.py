# eval 195, train excess 0.04375

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
    fraction = remaining / np.maximum(bins, 1e-8)
    score = np.where(fraction < 0, -np.inf, np.where(fraction <= 0.3, 1.2 - 4.0 * fraction, 0.9 - 0.6 * (fraction - 0.3)**2))
    return score

