# eval 261, train excess 0.04750

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    residuals = bins - item
    valid = residuals >= 0
    if not np.any(valid):
        return np.full_like(bins, -np.inf, dtype=float)
    optimal_residual = 0.05 * np.mean(residuals[valid])
    scores = -((residuals - optimal_residual) ** 2) / (np.var(residuals[valid]) + 1.0)
    return np.where(valid, scores, -np.inf)

