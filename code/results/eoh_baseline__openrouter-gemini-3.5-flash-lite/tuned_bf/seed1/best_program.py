# eval 224, train excess 0.02370
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=3, hi=31, penalty=85): the tightest bin wins, but leaving a gap
    strictly between 3 and 31 units costs 85. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 85 * ((r > 3) & (r < 31))
