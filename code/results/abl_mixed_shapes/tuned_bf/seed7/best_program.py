# eval 113, train excess 0.03298
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=6, hi=23, penalty=66): the tightest bin wins, but leaving a gap
    strictly between 6 and 23 units costs 66. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 66 * ((r > 6) & (r < 23))
