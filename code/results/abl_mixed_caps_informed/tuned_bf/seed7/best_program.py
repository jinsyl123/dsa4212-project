# eval 181, train excess 0.01912
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=3, hi=32, penalty=84): the tightest bin wins, but leaving a gap
    strictly between 3 and 32 units costs 84. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 84 * ((r > 3) & (r < 32))
