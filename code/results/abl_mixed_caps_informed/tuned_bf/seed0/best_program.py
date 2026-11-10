# eval 230, train excess 0.01849
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=4, hi=31, penalty=97): the tightest bin wins, but leaving a gap
    strictly between 4 and 31 units costs 97. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 97 * ((r > 4) & (r < 31))
