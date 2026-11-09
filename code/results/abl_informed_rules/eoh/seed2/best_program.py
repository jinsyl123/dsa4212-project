# eval 250, train excess 0.04579

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
    max_cap = np.max(bins) if len(bins) > 0 and np.max(bins) > 0 else 1.0
    scores = np.where(valid, -np.power(rem + 1.0, 1.5), -np.inf)
    empty_mask = (bins == max_cap)
    empty_score = -2.0 * max_cap - 5.0
    scores = np.where(empty_mask, np.maximum(scores, empty_score), scores)
    return scores

