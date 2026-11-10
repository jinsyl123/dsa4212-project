# eval 66, train excess 0.03211
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=5, hi=24, penalty=59): the tightest bin wins, but leaving a gap
    strictly between 5 and 24 units costs 59. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 59 * ((r > 5) & (r < 24))
