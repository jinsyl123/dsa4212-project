# eval 143, train excess 0.04516

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
    max_cap = np.max(bins)
    scores = np.where(rem >= 0, -(1.0 + rem / max_cap) ** 3, -np.inf)
    empty_mask = (bins == max_cap) & (rem >= 0)
    if np.any(empty_mask):
        first_empty = np.argmax(empty_mask)
        scores[first_empty] = -2.5
    return scores

