# eval 223, train excess 0.04109

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
    max_cap = np.max(bins) if bins.size > 0 else 1.0
    is_empty = (bins == max_cap)
    
    ratio = rem / max_cap
    harmonic_score = np.where(ratio < 0.2, 10.0 / (1.0 + 50.0 * ratio), 1.0 / (0.1 + ratio))
    
    scores = np.where(valid, harmonic_score, -999.0)
    scores = np.where(is_empty & valid, -15.0 - 10.0 * (item / max_cap), scores)
    
    return scores

