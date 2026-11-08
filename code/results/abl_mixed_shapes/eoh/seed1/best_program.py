# eval 54, train excess 0.04706

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
    target_slack = 0.1
    sigma = 0.2
    valid_score = np.exp(-((remaining - target_slack) ** 2) / (2 * sigma ** 2))
    invalid_score = -10.0 * (remaining ** 2)
    return np.where(remaining >= 0, valid_score, invalid_score)

