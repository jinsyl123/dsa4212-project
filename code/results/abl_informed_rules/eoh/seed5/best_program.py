# eval 229, train excess 0.04507

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
    ideal_remaining = 0.2
    scale = 0.2
    score = np.exp(-((remaining - ideal_remaining) ** 2) / (2 * scale ** 2))
    return np.where(remaining >= 0, score, -np.inf)

