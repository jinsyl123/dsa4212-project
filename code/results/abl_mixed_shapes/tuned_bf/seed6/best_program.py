# eval 165, train excess 0.03367
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=7, hi=25, penalty=95): the tightest bin wins, but leaving a gap
    strictly between 7 and 25 units costs 95. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 95 * ((r > 7) & (r < 25))
