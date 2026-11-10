# eval 203, train excess 0.01798
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=4, hi=31, penalty=73): the tightest bin wins, but leaving a gap
    strictly between 4 and 31 units costs 73. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 73 * ((r > 4) & (r < 31))
