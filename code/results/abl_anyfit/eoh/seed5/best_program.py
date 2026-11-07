# eval 299, train excess 0.04394

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
    priorities = np.where(
        valid,
        1.0 / (1.0 + (residuals / (0.2 * bins + 1e-8)) ** 3) + 0.2 * np.sqrt(bins + 1e-8),
        -2e8 - np.exp(1.5 * np.abs(residuals))
    )
    return priorities

