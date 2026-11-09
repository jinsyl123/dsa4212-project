# eval 70, train excess 0.02814

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
    if not np.any(valid):
        return np.full_like(bins, -np.inf, dtype=float)
    
    # Harmonic-inspired weighting based on remaining capacity ratio relative to original bin size implied by max bin
    max_cap = np.max(bins)
    if max_cap == 0:
        return np.where(valid, 0.0, -np.inf)
        
    ratio = rem / max_cap
    # Prefer ratios close to certain harmonic fractions (e.g., 1/3, 1/2, 2/3) to reduce fragmentation
    harmonic_targets = np.array([0.0, 0.33, 0.5, 0.67, 1.0])
    
    # Find distance to the closest harmonic target for each valid bin
    distances = np.min(np.abs(ratio[:, None] - harmonic_targets[None, :]), axis=1)
    
    # Score penalizes deviation from harmonic targets, with a slight tie-breaker for smaller remaining capacities
    scores = np.where(valid, -distances - 0.01 * (rem / max_cap), -np.inf)
    return scores

