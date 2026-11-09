# eval 231, train excess 0.03666

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
    phi = 1.618033988749895
    ratio = np.divide(remaining, bins + 1e-8)
    score = -np.abs(np.sin(ratio * np.pi * phi)) - (remaining / (bins + 1e-8))
    return np.where(remaining < 0, -np.inf, score)

