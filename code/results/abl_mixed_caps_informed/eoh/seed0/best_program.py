# eval 94, train excess 0.02932

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
    scores = np.full_like(bins, -np.inf, dtype=float)
    if not np.any(valid):
        return scores
    
    max_cap = np.max(bins) if len(bins) > 0 and np.max(bins) > 0 else 1.0
    is_empty = (bins == max_cap) & valid
    is_non_empty = valid & ~is_empty
    
    if np.any(is_non_empty):
        ratio = remaining[is_non_empty] / max_cap
        scores[is_non_empty] = 1.0 / (1.0 + ratio ** 2)
        
    if np.any(is_empty):
        num_active = np.sum(bins < max_cap)
        open_factor = 1.0 / (1.0 + 0.2 * num_active)
        empty_indices = np.where(is_empty)[0]
        scores[empty_indices[0]] = open_factor * (1.0 - (item / max_cap) ** 0.5)
        
    return scores

