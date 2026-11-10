# eval 237, train excess 0.03900
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=0, hi=22, penalty=38): the tightest bin wins, but leaving a gap
    strictly between 0 and 22 units costs 38. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 38 * ((r > 0) & (r < 22))
