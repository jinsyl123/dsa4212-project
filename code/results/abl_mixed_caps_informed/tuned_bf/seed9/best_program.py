# eval 184, train excess 0.01875
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=3, hi=31, penalty=87): the tightest bin wins, but leaving a gap
    strictly between 3 and 31 units costs 87. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 87 * ((r > 3) & (r < 31))
