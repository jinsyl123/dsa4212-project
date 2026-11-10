# eval 224, train excess 0.03305

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
    if len(bins) == 0:
        return np.array([], dtype=float)
    max_cap = np.max(bins)
    is_empty = (bins == max_cap) & valid
    norm_rem = rem / np.maximum(max_cap, 1e-9)
    scores = np.where(valid, 1.0 - (norm_rem ** 1.5), -np.inf)
    interference = 0.3 * (1.0 - np.cos(2 * np.pi * norm_rem / 0.25)) * (norm_rem < 0.25)
    scores = np.where(valid & ~is_empty, scores - np.maximum(interference, 0.0), scores)
    scores = np.where(is_empty, 0.48, scores)
    return scores

