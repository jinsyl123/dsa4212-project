# eval 192, train excess 0.02666

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
    valid = rem >= 0
    max_cap = np.max(bins)
    is_empty = (bins == max_cap)
    scores = np.where(valid, np.exp(-2.0 * rem) - np.where(is_empty, 0.8, 0.2 * (1.0 / (1.0 + rem))), -1.0)
    return scores

