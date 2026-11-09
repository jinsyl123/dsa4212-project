# eval 288, train excess 0.03231
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=6, hi=25, penalty=60): the tightest bin wins, but leaving a gap
    strictly between 6 and 25 units costs 60. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 60 * ((r > 6) & (r < 25))
