# eval 295, train excess 0.02897

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
    scores = np.where(rem >= 0, np.exp(-((rem - 0.1) ** 2) / 0.05), -np.inf)
    empty_idx = np.where((bins == 1.0) & (rem >= 0))[0]
    if empty_idx.size > 0:
        first_empty = empty_idx[0]
        non_empty_valid = np.any((bins < 1.0) & (rem >= 0))
        if not non_empty_valid:
            scores[first_empty] = 1.0
        else:
            scores[first_empty] = -0.5
    return scores

