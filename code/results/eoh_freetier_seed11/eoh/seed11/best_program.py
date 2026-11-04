# eval 277, train excess 0.04475

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
    valid = remaining >= 0
    target = 1.618
    priorities = np.where(valid, np.cos(np.pi * remaining / (target * np.maximum(item, 1e-5))), -np.inf)
    return priorities

