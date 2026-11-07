# eval 275, train excess 0.04512

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    rem = bins - item
    valid = rem >= 0
    indices = np.arange(len(bins), dtype=float) + 1.0
    used = 1.0 - (rem / np.maximum(bins, 1e-8))
    scores = np.where(valid, (used ** 2) / (rem + 0.05) - 0.05 * indices, -np.inf)
    return scores

