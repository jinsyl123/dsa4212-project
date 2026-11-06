# eval 213, train excess 0.04511

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
    return np.where(residuals >= 0, np.exp(-(residuals ** 2) / 0.1), -np.inf)

