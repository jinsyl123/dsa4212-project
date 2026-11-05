# eval 196, train excess 0.03719

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        remaining = bins - item
        penalty = -remaining / (item + 1e-8)
        modulation = np.sin(np.pi * bins / (item + 1e-8))
        scores = np.where(bins >= item, penalty + modulation, -np.inf)
    return scores

