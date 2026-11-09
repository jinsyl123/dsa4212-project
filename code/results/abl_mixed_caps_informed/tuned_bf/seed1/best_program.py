# eval 112, train excess 0.01792
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=4, hi=30, penalty=95): the tightest bin wins, but leaving a gap
    strictly between 4 and 30 units costs 95. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 95 * ((r > 4) & (r < 30))
