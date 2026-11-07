# eval 47, train excess 0.04629

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
    scores = np.where(rem >= 0, 1.0 / (1.0 + rem / item), -np.inf)
    max_cap = np.max(bins) if len(bins) > 0 and np.max(bins) > 0 else 1.0
    empty_mask = (bins == max_cap) & (rem >= 0)
    if np.any(empty_mask):
        scores[np.argmax(empty_mask)] = 0.5
    return scores

