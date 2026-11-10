# eval 242, train excess 0.02376
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=3, hi=29, penalty=68): the tightest bin wins, but leaving a gap
    strictly between 3 and 29 units costs 68. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 68 * ((r > 3) & (r < 29))
