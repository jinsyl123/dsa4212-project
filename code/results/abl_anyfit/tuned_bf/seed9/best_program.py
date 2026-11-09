# eval 264, train excess 0.03938
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=2, hi=20, penalty=40): the tightest bin wins, but leaving a gap
    strictly between 2 and 20 units costs 40. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 40 * ((r > 2) & (r < 20))
