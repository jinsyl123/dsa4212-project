# eval 282, train excess 0.03939

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
    utilization = 1.0 - (remaining / (bins + 1e-8))
    scores = np.where(valid, utilization ** 2 - 2.0 * utilization + 1.0 / (remaining + 1.0), -np.inf)
    return scores

