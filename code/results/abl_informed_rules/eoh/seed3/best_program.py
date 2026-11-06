# eval 173, train excess 0.04519

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
    scores = np.full_like(bins, -np.inf, dtype=float)
    fit = rem >= 0
    if not np.any(fit):
        return scores
    
    max_cap = np.max(bins) if len(bins) > 0 else 1.0
    empty_mask = (bins == max_cap) & fit
    non_empty_fit = fit & ~empty_mask
    
    if np.any(non_empty_fit):
        scores[non_empty_fit] = np.exp(-rem[non_empty_fit] / (0.05 * max_cap))
        
    if np.any(empty_mask):
        first_empty = np.argmax(empty_mask)
        num_used = np.sum(bins < max_cap)
        scores[first_empty] = 0.05 if num_used == 0 else 0.001
        
    return scores

