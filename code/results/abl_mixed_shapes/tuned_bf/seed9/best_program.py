# eval 195, train excess 0.03451
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=4, hi=25, penalty=49): the tightest bin wins, but leaving a gap
    strictly between 4 and 25 units costs 49. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 49 * ((r > 4) & (r < 25))
