# eval 221, train excess 0.04148

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    valid = bins >= item
    remaining = bins - item
    sigmoid_term = 1.0 / (1.0 + np.exp(2.0 * remaining - 5.0))
    utilization = 1.0 - np.divide(remaining, bins, out=np.zeros_like(bins, dtype=float), where=bins>0)
    quadratic_bonus = np.where(utilization >= 0.8, 15.0 * (utilization ** 2), 0.0)
    priorities = np.where(valid, -sigmoid_term + quadratic_bonus, -np.inf)
    return priorities

