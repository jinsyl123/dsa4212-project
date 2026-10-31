# eval 135, train excess 0.04532

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    import numpy as np
    remaining = bins - item
    initial_capacity = bins
    fill_used = 1.0 - (remaining / initial_capacity)
    tanh_penalty = -np.tanh(np.maximum(0.0, remaining) * 2.0)
    efficiency_bonus = np.where((fill_used >= 0.8) & (fill_used <= 1.0), 5.0 * fill_used, 0.0)
    score = tanh_penalty + efficiency_bonus
    return np.where(remaining < 0, -np.inf, score)

