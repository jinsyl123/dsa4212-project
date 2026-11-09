# eval 186, train excess 0.04715

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
    fill_ratio = item / np.maximum(bins, 1e-9)
    score = -10.0 * np.maximum(0.0, remaining) - 1.0 / (np.abs(fill_ratio - 0.6) + 1e-3)
    return np.where(remaining < 0, -np.inf, score)

