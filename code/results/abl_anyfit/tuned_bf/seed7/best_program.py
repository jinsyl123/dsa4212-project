# eval 281, train excess 0.04004
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=0, hi=23, penalty=54): the tightest bin wins, but leaving a gap
    strictly between 0 and 23 units costs 54. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 54 * ((r > 0) & (r < 23))
