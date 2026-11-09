# eval 35, train excess 0.01851
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=4, hi=24, penalty=74): the tightest bin wins, but leaving a gap
    strictly between 4 and 24 units costs 74. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 74 * ((r > 4) & (r < 24))
