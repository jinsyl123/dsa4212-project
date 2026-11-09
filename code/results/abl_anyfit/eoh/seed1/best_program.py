# eval 235, train excess 0.04023

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
    if not np.any(valid):
        return np.full_like(bins, -np.inf, dtype=float)
    
    utilization = 1.0 - (remaining / (bins + 1e-8))
    power_reward = 1.0 / (remaining + 1.0)
    trig_penalty = np.sin(utilization * np.pi / 2.0)
    
    score = power_reward - trig_penalty
    return np.where(valid, score, -np.inf)

