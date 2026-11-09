# eval 223, train excess 0.04618

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
    ratio = remaining / (bins + 1e-8)
    return np.where(remaining >= 0, np.cos(2 * np.pi * ratio), -np.inf)

