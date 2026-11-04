# eval 212, train excess 0.03719

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    valid = bins >= item
    remaining = bins - item
    scores = np.where(valid, -np.sin(np.pi * remaining / (item + 1e-8)) - (remaining / (item + 1e-8)), -np.inf)
    return scores

