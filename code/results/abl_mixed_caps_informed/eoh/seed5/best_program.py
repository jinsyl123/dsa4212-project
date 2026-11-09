# eval 1, train excess 0.02957

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
    is_empty = (bins == 1.0) & valid
    score = np.where(valid, 1.0 / (rem + 1e-5), -1.0)
    score = np.where(is_empty, score - 0.5, score)
    return score

