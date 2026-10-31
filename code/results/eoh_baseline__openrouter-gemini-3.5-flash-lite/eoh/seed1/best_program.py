# eval 1, train excess 0.04751

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    residual = bins - item
    return np.where(residual >= 0, 1.0 / (residual + 1e-5), -np.inf)

