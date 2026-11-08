# eval 127, train excess 0.02848

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
    optimal_rem = 0.35
    scores = np.where(rem >= 0, np.exp(-((rem - optimal_rem) ** 2) / 0.12), -1.0)
    return scores

