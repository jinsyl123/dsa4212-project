# eval 193, train excess 0.01858
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=4, hi=33, penalty=84): the tightest bin wins, but leaving a gap
    strictly between 4 and 33 units costs 84. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 84 * ((r > 4) & (r < 33))
