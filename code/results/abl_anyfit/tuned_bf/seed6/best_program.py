# eval 60, train excess 0.03975
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=1, hi=21, penalty=79): the tightest bin wins, but leaving a gap
    strictly between 1 and 21 units costs 79. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 79 * ((r > 1) & (r < 21))
