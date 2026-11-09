# eval 131, train excess 0.03996
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=1, hi=19, penalty=47): the tightest bin wins, but leaving a gap
    strictly between 1 and 19 units costs 47. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 47 * ((r > 1) & (r < 19))
