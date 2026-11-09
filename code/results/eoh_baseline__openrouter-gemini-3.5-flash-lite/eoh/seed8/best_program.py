# eval 292, train excess 0.04486

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
    target = bins * 0.2
    ratio = np.minimum(1.0, np.maximum(0.0, remaining / (target + 1e-5)))
    score = -1.0 + np.cos(ratio * np.pi)
    return np.where(remaining < 0, -np.inf, score)

