# eval 250, train excess 0.02766

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
    
    # Hyperbolic penalty for leftover space to favor tight packing
    tightness_score = np.where(valid, 1.0 / (1.0 + 4.0 * remaining), -np.inf)
    
    # Trigonometric resonance reward targeting optimal fill states
    capacity = np.max(bins)
    current_load = capacity - bins
    new_load = current_load + item
    fill_ratio = new_load / (capacity + 1e-9)
    balance_reward = np.cos(np.pi * (fill_ratio - 0.75))
    
    scores = tightness_score + 0.4 * balance_reward
    scores = np.where(valid, scores, -np.inf)
    
    # Penalize opening a brand new bin using a dampening factor
    empty_mask = (bins == capacity) & valid
    if np.any(empty_mask):
        first_empty = np.argmax(empty_mask)
        scores[first_empty] *= 0.3
        
    return scores

