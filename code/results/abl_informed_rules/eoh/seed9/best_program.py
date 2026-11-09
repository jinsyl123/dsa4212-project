# eval 151, train excess 0.04466

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    max_cap = np.max(bins)
    rem = bins - item
    feasible = rem >= 0
    ratio = rem / max_cap
    scores = np.where(feasible, 1.0 / (1.0 + ratio ** 2 * 16.0), -np.inf)
    empty_mask = (bins == max_cap) & feasible
    if np.any(empty_mask):
        first_empty = np.argmax(empty_mask)
        scores[first_empty] = 0.5 - 0.5 * (item / max_cap)
    return scores

