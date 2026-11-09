# eval 201, train excess 0.02585
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=3, hi=34, penalty=90): the tightest bin wins, but leaving a gap
    strictly between 3 and 34 units costs 90. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 90 * ((r > 3) & (r < 34))
